"""
Nanotron training script example using a custom dataloader.

Usage:
```
export CUDA_DEVICE_MAX_CONNECTIONS=1 # important for some distributed operations
torchrun --nproc_per_node=2 examples/vqa/run_train.py --config-file examples/vqa/config_vqa.yaml
```
"""
import argparse
import dataclasses
from typing import Dict, List, Union, cast

import datasets
import numpy as np
import torch
from torch.utils.data import DataLoader

from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import (
    DataArgs,
    DatasetStageArgs,
    PretrainDatasetsArgs,
)
from nanotron.dataloader import (
    get_datasets,
    get_train_dataloader,
)
from nanotron.helpers import (
    compute_remain_train_steps_of_a_data_stage_from_ckp,
    get_consumed_train_samples_of_a_data_stage_from_ckp,
)
from nanotron.logging import log_rank
from nanotron.parallel.context import ParallelContext
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from nanotron.parallel.pipeline_parallel.utils import get_input_output_pp_ranks
from nanotron.trainer import DistributedTrainer
from nanotron.utils import main_rank_first

try:
    from huggingface_hub import __version__ as hf_hub_version
    from transformers import AutoProcessor, AutoTokenizer
    from transformers import __version__ as tf_version
except ImportError:
    hf_hub_version = None
    tf_version = None

logger = logging.get_logger(__name__)


def vqa_process(
    raw_dataset: datasets.Dataset,
    processor: AutoProcessor,
    dataset_processing_num_proc_per_process: int,
    dataset_overwrite_cache: bool,
    sequence_length: int,
):
    def format_example(example):
        messages = []
        for i, x in enumerate(example["en"]):
            user_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": x["question"]},
                ]
            }

            if i == 0:
                user_message["content"].append(
                    {"type": "image"},
                )

            messages.append(user_message)
            assistant_message = {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": x["answer"]},
                ]
            }

            messages.append(assistant_message)
        return messages

    def _process_examples(examples: Dict, images) -> Dict[str, List[np.ndarray]]:
        inputs = [
            processor(
                text=processor.apply_chat_template(format_example(ex), add_generation_prompt=True),
                images = [img],
                return_tensors="np", max_length=sequence_length + 1, padding="longest", truncation=True
            )
            for ex, img in zip(examples, images)
        ]

        inputs = {
            k: [v[k] for v in inputs] for k in ["input_ids", "pixel_values"]
        }

        return inputs

    train_dataset = raw_dataset.map(
        _process_examples,
        input_columns=["qa", "image"],
        remove_columns=raw_dataset.column_names,
        batched=True,
        num_proc=dataset_processing_num_proc_per_process,
        load_from_cache_file=not dataset_overwrite_cache,
    )

    return train_dataset


@dataclasses.dataclass
class DataCollatorForVQA:
    sequence_length: int
    input_pp_rank: int
    output_pp_rank: int
    parallel_context: ParallelContext
    padding_idx: int = 128_002

    def __call__(self, examples: List[Dict[str, List[np.ndarray]]]) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        # Process the case when current rank doesn't require data. We return `TensorPointer` that points to ranks having the data.
        current_pp_rank = dist.get_rank(self.parallel_context.pp_pg)
        if current_pp_rank not in [
            self.input_pp_rank,
            self.output_pp_rank,
        ]:
            assert all(len(example) == 0 for example in examples)
            return {
                "input_ids": TensorPointer(group_rank=self.input_pp_rank),
                "input_mask": TensorPointer(group_rank=self.input_pp_rank),
                "label_ids": TensorPointer(group_rank=self.output_pp_rank),
                "label_mask": TensorPointer(group_rank=self.output_pp_rank),
                "pixel_values": TensorPointer(group_rank=self.input_pp_rank),
            }

        # Make sure we load only what's necessary, ie we only load `input_ids` and `pixel_values` column.
        assert all(list(example.keys()) == ["input_ids", "pixel_values"] for example in examples)

        max_n_patches = max([len(examples[i]["pixel_values"][0]) for i in range(len(examples))])

        padded_pixel_values = []

        for example in examples:
            pixel_values = example["pixel_values"]
            current_patches = len(pixel_values[0])

            # Pad the pixel_values to have max_n_patches along dimension 1 (patches)
            padding = ((0, 0), (0, max_n_patches - current_patches), (0, 0), (0, 0), (0, 0))  # Only pad the patches dimension
            padded_values = np.pad(pixel_values, pad_width=padding, mode='constant', constant_values=0)
            padded_pixel_values.append(padded_values)


        # Step 3: Stack padded pixel_values and pixel_attention_masks
        pixel_values = np.vstack(padded_pixel_values)  # Stacked pixel_values
        result: Dict[str, Union[np.ndarray, TensorPointer]] = {}

        result["input_ids"] = TensorPointer(group_rank=self.input_pp_rank)
        result["input_mask"] = TensorPointer(group_rank=self.input_pp_rank)
        result["label_ids"] = TensorPointer(group_rank=self.output_pp_rank)
        result["label_mask"] = TensorPointer(group_rank=self.output_pp_rank)
        result["pixel_values"] = TensorPointer(group_rank=self.input_pp_rank)

        def pad_tokens(inputs = True):
            padded_tokens = []
            token_masks = []

            max_seq_length = max([len(examples[i]["input_ids"][0]) for i in range(len(examples))]) - 1
            # make it divisible by 4 for tp
            max_seq_length = max_seq_length + (4 - max_seq_length % 4) % 4

            for example in examples:
                input_ids = example["input_ids"]
                if isinstance(input_ids, list):
                    input_ids = np.array(input_ids)

                if inputs:
                    input_ids = input_ids[:, :-1]
                else:
                    input_ids = input_ids[:, 1:]

                current_length = input_ids.shape[1]

                padding = ((0, 0), (0, max_seq_length - current_length))
                input_ids = np.pad(input_ids, pad_width=padding, mode='constant', constant_values=self.padding_idx)
                padded_tokens.append(input_ids)

                mask = np.ones((1, current_length), dtype=np.bool_)
                mask = np.pad(mask, pad_width=padding, mode='constant', constant_values=0)
                token_masks.append(mask)

            padded_tokens = np.vstack(padded_tokens)
            token_masks = np.vstack(token_masks)

            return padded_tokens, token_masks

        if current_pp_rank == self.input_pp_rank:
            result["input_ids"], result["input_mask"] = pad_tokens(inputs=True)
            result["pixel_values"] = pixel_values

        if current_pp_rank == self.output_pp_rank:
            result["label_ids"], result["label_mask"] = pad_tokens(inputs=False)

        result = {k: v if isinstance(v, TensorPointer) else torch.from_numpy(v) for k, v in result.items()}
        return result


def get_dataloader_from_data_stage(
    trainer: DistributedTrainer,
    data: DataArgs,
    consumed_train_samples: int,
    num_remaining_train_steps: int,
):
    """
    Returns a dataloader for a given data stage.

    data: The data configuration for the current stage.
    consumed_train_samples: The number of samples consumed by the model in the this stage (each stage starts from zero).
    num_remaining_train_steps: The number of remaining training steps for this stage.
    """
    assert consumed_train_samples >= 0, "consumed_train_samples should be greater than 0"
    assert num_remaining_train_steps >= 0, "num_remaining_train_steps should be greater than 0"

    # First, we need to know which ranks to feed the dataloader to
    input_pp_rank, output_pp_rank = get_input_output_pp_ranks(model=trainer.model)

    if isinstance(data.dataset, PretrainDatasetsArgs):
        log_rank("Using `datasets` library", logger=logger, level=logging.INFO, rank=0)
        tokenizer_path = trainer.config.tokenizer.tokenizer_name_or_path
        log_rank(
            f"Loading tokenizer from {tokenizer_path} and transformers/hf_hub versions {tf_version, hf_hub_version}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        # We need to the 1st device to process dataset and cache it, then other devices load from cache
        with main_rank_first(trainer.parallel_context.world_pg):
            # We load the raw dataset
            raw_dataset = get_datasets(
                hf_dataset_or_datasets=data.dataset.hf_dataset_or_datasets,
                hf_dataset_config_name=data.dataset.hf_dataset_config_name,
                splits=data.dataset.hf_dataset_splits,
            )["train"]

            raw_dataset = raw_dataset.select(range(1000))

            processor = AutoProcessor.from_pretrained(tokenizer_path, size= {"longest_edge": 2*364})
            train_dataset = vqa_process(
                raw_dataset=raw_dataset,
                processor=processor,
                dataset_processing_num_proc_per_process=data.dataset.dataset_processing_num_proc_per_process,
                dataset_overwrite_cache=data.dataset.dataset_overwrite_cache,
                sequence_length=trainer.sequence_length,
            )


            # We load the processed dataset on the ranks requiring it
            dataloader = get_train_dataloader(
                train_dataset=train_dataset,
                sequence_length=trainer.sequence_length,
                parallel_context=trainer.parallel_context,
                input_pp_rank=input_pp_rank,
                output_pp_rank=output_pp_rank,
                micro_batch_size=trainer.micro_batch_size,
                consumed_train_samples=consumed_train_samples,
                dataloader_num_workers=data.num_loading_workers,
                seed_worker=data.seed,
                dataloader_drop_last=True,
                dataset_columns=["input_ids", "pixel_values"],
                collator_builder=DataCollatorForVQA
            )

            # Check if we have enough samples for train_steps
            total_tokens_dataset = len(dataloader.dataset) * trainer.sequence_length
            num_tokens_needed_for_training = (
                num_remaining_train_steps * trainer.global_batch_size * trainer.sequence_length
            )
            assert num_tokens_needed_for_training <= total_tokens_dataset, (
                f"Dataset is too small for steps ({total_tokens_dataset} < {num_tokens_needed_for_training}), "
                f"Try train_steps<={len(dataloader.dataset) // trainer.global_batch_size + trainer.iteration_step}"
            )
    else:
        raise ValueError(f"Unhandled case of `self.config.data.dataset`. Got: {data.dataset}")

    return dataloader


def get_dataloader(trainer: DistributedTrainer) -> Dict[str, DataLoader]:
    dataloaders = {}

    for stage_idx, stage in enumerate(trainer.config.data_stages):
        # NOTE: we only create the dataloader for the first stage,
        # then we lazy initialize the dataloader for the other stages
        stage = cast(DatasetStageArgs, stage)
        consumed_train_samples = get_consumed_train_samples_of_a_data_stage_from_ckp(stage, trainer.metadata)
        assert (
            consumed_train_samples is not None
        ), f"Cannot find consumed_train_samples for stage {stage.start_training_step} in the checkpoint"

        num_remaining_train_steps = compute_remain_train_steps_of_a_data_stage_from_ckp(
            stage, trainer.config, trainer.metadata
        )
        log_rank(
            f"[Training Plan] Stage {stage.name} has {num_remaining_train_steps} remaining training steps and has consumed {consumed_train_samples} samples",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        dataloader = (
            get_dataloader_from_data_stage(
                trainer,
                stage.data,
                consumed_train_samples=consumed_train_samples,
                num_remaining_train_steps=num_remaining_train_steps,
            )
            if stage_idx == 0
            else lambda stage=stage: get_dataloader_from_data_stage(
                trainer,
                stage.data,
                consumed_train_samples=consumed_train_samples,
                num_remaining_train_steps=num_remaining_train_steps,
            )
        )
        dataloaders[stage.name] = dataloader
    return dataloaders


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True, help="Path to the YAML or python config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config_file = args.config_file

    # Load trainer and data
    trainer = DistributedTrainer(config_file)
    dataloader = get_dataloader(trainer)

    # Train
    trainer.train(dataloader)

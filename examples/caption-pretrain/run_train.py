"""
Nanotron training script example using a custom dataloader.

Usage:
```
export CUDA_DEVICE_MAX_CONNECTIONS=1 # important for some distributed operations
torchrun --nproc_per_node=4 examples/caption-pretrain/run_train.py --config-file examples/caption-pretrain/pretrain.yaml
```

For preprocessed dataset:
torchrun --nproc_per_node=4 examples/caption-pretrain/run_train.py --config-file examples/caption-pretrain/pretrain_preprocessed.yaml
"""

import argparse
from typing import Dict, cast

import datasets
from torch.utils.data import DataLoader

from nanotron import logging
from nanotron.config import (
    DataArgs,
    DatasetStageArgs,
)
from nanotron.config.config import ImageDatasetsArgs
from nanotron.helpers import get_consumed_train_samples_of_a_data_stage_from_ckp
from nanotron.logging import log_rank
from nanotron.modular_dataloader import BATCH_ENCODERS, SAMPLE_ENCODERS
from nanotron.modular_dataloader.iterable import get_train_dataloader
from nanotron.parallel.pipeline_parallel.utils import get_input_output_pp_ranks
from nanotron.trainer import DistributedTrainer

try:
    from huggingface_hub import __version__ as hf_hub_version
    from transformers import AutoProcessor, AutoTokenizer
    from transformers import __version__ as tf_version
except ImportError:
    hf_hub_version = None
    tf_version = None

logger = logging.get_logger(__name__)


def get_dataloader_from_data_stage(
    trainer: DistributedTrainer,
    data: DataArgs,
    consumed_train_samples: int
):
    """
    Returns a dataloader for a given data stage.

    data: The data configuration for the current stage.
    """

    # First, we need to know which ranks to feed the dataloader to
    input_pp_rank, output_pp_rank = get_input_output_pp_ranks(model=trainer.model)

    if isinstance(data.dataset, ImageDatasetsArgs):
        log_rank("Using iterable dataset from `datasets` library", logger=logger, level=logging.INFO, rank=0)
        tokenizer_path = trainer.config.tokenizer.tokenizer_name_or_path
        log_rank(
            f"Loading tokenizer from {tokenizer_path} with HF version {hf_hub_version} and Transformers version {tf_version}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        dataset = datasets.load_dataset(
            data.dataset.hf_dataset_name_or_type,
            data_dir=data.dataset.hf_dataset_data_dir,
            split=data.dataset.hf_dataset_splits,
            streaming=True
        )

        processor = AutoProcessor.from_pretrained(
            tokenizer_path,
            size={"longest_edge": data.dataset.image_size * data.dataset.image_scale_factor},
        )

        sample_encoder = SAMPLE_ENCODERS[data.dataset.sample_encoder](
            processor=processor,
            sequence_length=trainer.sequence_length,
            **data.dataset.sample_encoder_args
        )

        batch_encoder = BATCH_ENCODERS[data.dataset.batch_encoder](
            input_pp_rank=input_pp_rank,
            output_pp_rank=output_pp_rank,
            parallel_context=trainer.parallel_context,
            processor=processor,
            sequence_length=trainer.sequence_length,
            **data.dataset.batch_encoder_args
        )

        dataloader = get_train_dataloader(
            train_dataset=dataset,
            sample_encoder=sample_encoder,
            batch_encoder=batch_encoder,
            parallel_context=trainer.parallel_context,
            input_pp_rank=input_pp_rank,
            output_pp_rank=output_pp_rank,
            micro_batch_size=trainer.micro_batch_size,
            sample_encoding_batch=data.dataset.sample_encoding_batch,
            batch_encoding_batch=data.dataset.batch_encoding_batch,
            seed_worker=data.seed,
            sample_encoding_workers=data.dataset.sample_encoding_workers,
            batch_encoding_workers=data.dataset.batch_encoding_workers,
            consumed_train_samples=consumed_train_samples,
            drop_last=True,
        )
    else:
        raise ValueError(f"Unsupported dataset case: {data.dataset}")

    return dataloader


def get_dataloader(trainer: DistributedTrainer) -> Dict[str, DataLoader]:
    dataloaders = {}

    for stage_idx, stage in enumerate(trainer.config.data_stages):
        # NOTE: we only create the dataloader for the first stage,
        # then we lazy initialize the dataloader for the other stages
        stage = cast(DatasetStageArgs, stage)
        consumed_train_samples = get_consumed_train_samples_of_a_data_stage_from_ckp(stage, trainer.metadata)

        log_rank(
            f"[Training Plan] Stage {stage.name}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        dataloader = (
            get_dataloader_from_data_stage(
                trainer,
                stage.data,
                consumed_train_samples
            )
            if stage_idx == 0
            else lambda stage=stage: get_dataloader_from_data_stage(
                trainer,
                stage.data,
                consumed_train_samples
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

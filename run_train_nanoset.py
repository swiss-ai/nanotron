"""
Nanotron training script.

Usage:
```
export CUDA_DEVICE_MAX_CONNECTIONS=1 # important for some distributed operations
torchrun --nproc_per_node=8 run_train.py --config-file examples/config_tiny_llama.yaml
```
"""
import argparse

from nanotron import logging
from nanotron.data.dataloader_builder import build_nanoset_dataloader
from nanotron.data.dataset_builder import NanosetBuilder
from nanotron.data.nanoset import NanosetConfig
from nanotron.parallel.pipeline_parallel.utils import get_input_output_pp_ranks
from nanotron.trainer import DistributedTrainer

logger = logging.get_logger(__name__)


def get_dataloaders(trainer: DistributedTrainer):
    """Returns train, valid and test dataloaders"""
    assert (
        len(trainer.config.data_stages) == 1
    ), "Nanosets currently don't support loading DataLoaders based on training stages"

    # First, we need to know which ranks to feed the dataloader to
    input_pp_rank, output_pp_rank = get_input_output_pp_ranks(model=trainer.model)

    # Create Nanoset config
    nanoset_config = NanosetConfig(
        random_seed=trainer.config.data_stages[0].data.seed,
        sequence_length=trainer.sequence_length,
        data_path=trainer.config.data_stages[0].data.dataset.data_path,
        split=trainer.config.data_stages[0].data.dataset.split,
        train_split_samples=trainer.config.tokens.train_steps * trainer.global_batch_size,
        path_to_cache=trainer.config.data_stages[0].data.dataset.path_to_cache,
    )

    # Build Nanoset datasets
    train_dataset, valid_dataset, test_dataset = NanosetBuilder(nanoset_config).build()

    # Prepare train, valid and test dataloaders
    train_dataloader = build_nanoset_dataloader(
        train_dataset,
        trainer.sequence_length,
        parallel_context=trainer.parallel_context,
        input_pp_rank=input_pp_rank,
        output_pp_rank=output_pp_rank,
        micro_batch_size=trainer.micro_batch_size,
        consumed_train_samples=trainer.consumed_train_samples,
        dataloader_num_workers=trainer.config.data_stages[0].data.num_loading_workers,
        dataloader_drop_last=True,
    )

    valid_dataloader = build_nanoset_dataloader(
        valid_dataset,
        trainer.sequence_length,
        parallel_context=trainer.parallel_context,
        input_pp_rank=input_pp_rank,
        output_pp_rank=output_pp_rank,
        micro_batch_size=trainer.micro_batch_size,
        dataloader_num_workers=trainer.config.data_stages[0].data.num_loading_workers,
        dataloader_drop_last=True,
    )

    test_dataloader = build_nanoset_dataloader(
        test_dataset,
        trainer.sequence_length,
        parallel_context=trainer.parallel_context,
        input_pp_rank=input_pp_rank,
        output_pp_rank=output_pp_rank,
        micro_batch_size=trainer.micro_batch_size,
        dataloader_num_workers=trainer.config.data_stages[0].data.num_loading_workers,
        dataloader_drop_last=True,
    )

    return train_dataloader, valid_dataloader, test_dataloader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True, help="Path to the YAML or python config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config_file = args.config_file

    # Load trainer and data
    trainer = DistributedTrainer(config_file)

    train_dataloader, valid_dataloader, test_dataloader = get_dataloaders(trainer)

    # Train
    trainer.train([train_dataloader])

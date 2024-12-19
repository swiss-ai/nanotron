# Inspired by https://github.com/NVIDIA/Megatron-Energon
from abc import ABC
from typing import Any, Dict, Generic, List, TypeVar, Union

import torch
from transformers import AutoProcessor

from nanotron.parallel.context import ParallelContext
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer

T_encoded_sample = TypeVar("T_encoded_sample")


class SampleEncoder(ABC, Generic[T_encoded_sample]):
    """
    Processes a single sample. E.g. formats caption text.
    """

    processor: AutoProcessor
    sequence_length: int

    def encode(self, sample: Dict[str, Any]) -> T_encoded_sample:
        """
        Encode a sample.
        """
        raise NotImplementedError


class BatchEncoder(ABC, Generic[T_encoded_sample]):
    """
    Collates and encodes a batch of samples.
    """
    input_pp_rank: int
    output_pp_rank: int
    parallel_context: ParallelContext
    processor: AutoProcessor
    sequence_length: int

    def encode(self, batch: List[T_encoded_sample]) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        """
        Collate and encode a batch of samples.
        """
        raise NotImplementedError

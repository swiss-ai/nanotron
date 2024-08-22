from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np
import torch
from nanotron import distributed as dist
from nanotron.parallel.context import ParallelContext
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer

LLAMA3_EOS_TOKEN = 128001  # NOTE(tj.solergibert) Currently, we hardcode this value as we only support Llama3 for removing the document cross attention


def build_position_ids_and_label_mask(input_ids, sequence_length):
    """
    For each sample in the batch, create:
        1. Position ids for each document
        2. Mask eos token. Both the previous token generating the eos token and the token generated from the eos token
    """
    position_ids_list = []
    label_mask_list = []

    for sample in input_ids:
        # Position ids
        document_ends = (sample == LLAMA3_EOS_TOKEN).nonzero().flatten().tolist()
        document_ends.append(sequence_length)
        lengths = [end - start for start, end in zip([0] + document_ends[:-1], document_ends)]
        position_ids_list.append(build_position_ids(lengths))

        # Label ids
        label_mask = torch.ones(sequence_length, dtype=torch.bool)
        for eos_token in document_ends[:-1]:
            label_mask[eos_token - 1] = False
            label_mask[eos_token] = False

        label_mask_list.append(label_mask)
    return torch.tensor(np.stack((position_ids_list))), torch.stack(label_mask_list)


@dataclass
class NanosetDataCollatorForCLM:
    """
    Data collator used for causal language modeling with Nanosets dataset.

    - input_pp_rank: Discards last input id token
    - output_pp_rank: Discards first label id token
    - other pp ranks: Don't have data. Instead, we use `TensorPointer` to point to the rank having the data.
    """

    sequence_length: int
    input_pp_rank: int
    output_pp_rank: int
    parallel_context: ParallelContext
    remove_document_xattention: bool

    def __call__(self, examples: List[Dict[str, List[np.ndarray]]]) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        # Process the case when current rank doesn't require data. We return `TensorPointer` that points to ranks having the data.
        current_pp_rank = dist.get_rank(self.parallel_context.pp_pg)
        if current_pp_rank not in [
            self.input_pp_rank,
            self.output_pp_rank,
        ]:
            assert all(len(example) == 0 for example in examples)
            result = {
                "input_ids": TensorPointer(group_rank=self.input_pp_rank),
                "label_ids": TensorPointer(group_rank=self.output_pp_rank),
                "label_mask": TensorPointer(group_rank=self.output_pp_rank),
            }

            if self.remove_document_xattention:
                result["position_ids"] = TensorPointer(group_rank=self.input_pp_rank)
            else:
                result["input_mask"] = TensorPointer(group_rank=self.input_pp_rank)

            return result

        # Make sure we load only what's necessary, ie we only load a `input_ids` column.
        assert all(list(example.keys()) == ["input_ids"] for example in examples)

        # TODO @nouamanetazi: Is it better to have examples as np.array or torch.Tensor?
        input_ids = torch.vstack([examples[i]["input_ids"] for i in range(len(examples))])  # (b, s)
        batch_size, expanded_input_length = input_ids.shape

        result: Dict[str, Union[torch.LongTensor, TensorPointer]] = {}

        result["input_ids"] = TensorPointer(group_rank=self.input_pp_rank)
        result["label_ids"] = TensorPointer(group_rank=self.output_pp_rank)
        result["label_mask"] = TensorPointer(group_rank=self.output_pp_rank)

        assert (
            expanded_input_length == self.sequence_length + 1
        ), f"Samples should be of length {self.sequence_length + 1} (seq_len+1), but got {expanded_input_length}"

        if self.remove_document_xattention:
            # LlamaForTraining requires input_mask while LlamaForSFT requires position_ids
            result["position_ids"] = TensorPointer(group_rank=self.input_pp_rank)
            position_ids, label_mask = build_position_ids_and_label_mask(input_ids, self.sequence_length)
            # TODO(tj.solergibert) assert shape of this 2 new tensors
            # Process inputs: last token is the label
            if current_pp_rank == self.input_pp_rank:
                result["input_ids"] = input_ids[:, :-1]
                result["position_ids"] = position_ids

            # Process labels: shift them to the left
            if current_pp_rank == self.output_pp_rank:
                result["label_ids"] = input_ids[:, 1:]
                result["label_mask"] = label_mask

        else:
            # LlamaForTraining requires input_mask while LlamaForSFT requires position_ids
            result["input_mask"] = TensorPointer(group_rank=self.input_pp_rank)
            # Process inputs: last token is the label
            if current_pp_rank == self.input_pp_rank:
                result["input_ids"] = input_ids[:, :-1]
                result["input_mask"] = torch.ones((batch_size, self.sequence_length), dtype=torch.bool)

            # Process labels: shift them to the left
            if current_pp_rank == self.output_pp_rank:
                result["label_ids"] = input_ids[:, 1:]
                result["label_mask"] = torch.ones((batch_size, self.sequence_length), dtype=torch.bool)

        if isinstance(result["input_ids"], torch.Tensor) and result["input_ids"].shape[-1] != self.sequence_length:
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {result['input_ids'].shape[-1]}, but should be"
                f" {self.sequence_length}."
            )
        if isinstance(result["label_ids"], torch.Tensor) and result["label_ids"].shape[-1] != self.sequence_length:
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {result['label_ids'].shape[-1]}, but should be"
                f" {self.sequence_length}."
            )

        return result


# TODO(tj.solergibert) After "Beta", delete all the functs except `build_position_ids` and move `build_position_ids` to chat_dataset.py
def build_position_ids(lengths) -> np.array:
    position_ids = [list(range(length)) for length in lengths]  # Create position ids list
    return np.array([x for xs in position_ids for x in xs], dtype=np.int32)  # Flatten list of position ids


# TODO(tj.solergibert) Delete (debug), just 4 switching the remove cross-attention setting
def build_position_ids_dummy(lengths) -> np.array:
    return np.array(list(range(sum(lengths))), dtype=np.int32)  # TODO numpy arange


# TODO(tj.solergibert) Delete (debug), just 4 switching the training only on completitions setting.
def build_labels_completions_only(is_completitions):
    return is_completitions


# TODO(tj.solergibert) Delete (debug), just 4 switching the training only on completitions setting
def build_labels(is_completitions):
    return [True for _ in range(len(is_completitions))]


@dataclass
class DataCollatorForSFT:
    """
    Data collator used with Chat Dataset.
    - input_pp_rank: Discards last input id token
    - output_pp_rank: Discards first label id token
    - other pp ranks: Don't have data. Instead, we use `TensorPointer` to point to the rank having the data.
    """

    input_pp_rank: int
    output_pp_rank: int
    parallel_context: ParallelContext

    def __call__(self, examples: List[Dict[str, List[int]]]) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        # Process the case when current rank doesn't require data. We return `TensorPointer` that points to ranks having the data.

        current_pp_rank = dist.get_rank(self.parallel_context.pp_pg)
        if current_pp_rank not in [
            self.input_pp_rank,
            self.output_pp_rank,
        ]:
            assert all(len(example) == 0 for example in examples)
            return {
                "input_ids": TensorPointer(group_rank=self.input_pp_rank),
                "label_ids": TensorPointer(group_rank=self.output_pp_rank),
                "position_ids": TensorPointer(group_rank=self.input_pp_rank),
                "label_mask": TensorPointer(group_rank=self.output_pp_rank),
            }

        input_ids = np.vstack([examples[i]["input_ids"] for i in range(len(examples))])  # (b, s)
        is_completitions = np.vstack([examples[i]["is_completitions"] for i in range(len(examples))])  # (b, s)
        position_ids = np.vstack([examples[i]["position_ids"] for i in range(len(examples))])  # (b, s)

        result: Dict[str, Union[np.ndarray, TensorPointer]] = {}

        result["input_ids"] = TensorPointer(group_rank=self.input_pp_rank)
        result["position_ids"] = TensorPointer(group_rank=self.input_pp_rank)
        result["label_ids"] = TensorPointer(group_rank=self.output_pp_rank)
        result["label_mask"] = TensorPointer(group_rank=self.output_pp_rank)

        # Process inputs
        if current_pp_rank == self.input_pp_rank:
            result["input_ids"] = input_ids[:, :-1]
            result["position_ids"] = position_ids[:, :-1]

        # Process labels: shift them to the left.
        if current_pp_rank == self.output_pp_rank:
            result["label_ids"] = input_ids[:, 1:]
            result["label_mask"] = is_completitions[:, 1:]

        # Cast np.array to torch.Tensor
        result = {k: v if isinstance(v, TensorPointer) else torch.from_numpy(v) for k, v in result.items()}
        return result

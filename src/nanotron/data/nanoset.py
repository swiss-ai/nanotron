import hashlib
import json
import os
import time
from collections import OrderedDict
from typing import Dict, Tuple

import numpy
import torch

from nanotron import logging
from nanotron.data.indexed_dataset import MMapIndexedDataset
from nanotron.data.nanoset_configs import NanosetConfig
from nanotron.data.utils import Split
from nanotron.logging import log_rank

logger = logging.get_logger(__name__)


class Nanoset(torch.utils.data.Dataset):
    """Adapted from https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/datasets/gpt_dataset.py

    The base Nanoset dataset

    Args:
        indexed_dataset (MMapIndexedDataset): The MMapIndexedDataset around which to build the
        Nanoset

        indexed_indices (numpy.ndarray): The set of the documents indices to expose

        num_samples (int): The number of samples to draw from the indexed dataset

        index_split (Split): The indexed_indices Split (train, valid, test)

        config (NanosetConfig): The Nanoset-specific container for all config sourced parameters
    """

    def __init__(
        self,
        indexed_dataset: MMapIndexedDataset,
        indexed_indices: numpy.ndarray,
        num_samples: int,
        index_split: Split,
        config: NanosetConfig,
    ) -> None:

        assert indexed_indices.size > 0
        assert num_samples > 0

        self.indexed_dataset = indexed_dataset
        self.indexed_indices = indexed_indices
        self.num_samples = num_samples
        self.index_split = index_split
        self.config = config

        # Create unique identifier

        self.unique_identifiers = OrderedDict()
        self.unique_identifiers["class"] = type(self).__name__
        self.unique_identifiers["path_prefix"] = self.indexed_dataset.path_prefix
        self.unique_identifiers["num_samples"] = self.num_samples
        self.unique_identifiers["index_split"] = self.index_split.name
        self.unique_identifiers["split"] = self.config.split
        self.unique_identifiers["random_seed"] = self.config.random_seed
        self.unique_identifiers["sequence_length"] = self.config.sequence_length

        self.unique_description = json.dumps(self.unique_identifiers, indent=4)
        self.unique_description_hash = hashlib.md5(self.unique_description.encode("utf-8")).hexdigest()

        # Load or build/cache the document, sample, and shuffle indices

        (
            self.document_index,
            self.sample_index,
            self.shuffle_index,
        ) = self._build_document_sample_shuffle_indices()

    def __len__(self) -> int:
        """Abstract method implementation

        Returns:
            int: The length of the dataset
        """
        return self.sample_index.shape[0] - 1

    def __getitem__(self, idx: int) -> Dict[str, numpy.ndarray]:
        """Get the text (token ids) for a given index

        Args:
            idx (int): The index into the dataset

        Returns:
            Dict[str, numpy.ndarray]: The token ids wrapped in a dictionary
        """

        # Do the shuffle mapping
        idx = self.shuffle_index[idx]

        # Get the beginning and end documents and offsets
        doc_index_beg, doc_index_beg_offset = self.sample_index[idx]
        doc_index_end, doc_index_end_offset = self.sample_index[idx + 1]

        tokens = []

        # Sample spans a single document
        if doc_index_beg == doc_index_end:

            # Add the entire sample
            tokens.append(
                self.indexed_dataset.get(
                    self.document_index[doc_index_beg],
                    offset=doc_index_beg_offset,
                    length=doc_index_end_offset - doc_index_beg_offset + 1,
                )
            )

        # Sample spans multiple documents
        else:
            for i in range(doc_index_beg, doc_index_end + 1):

                # Add the sample part
                offset = 0 if i > doc_index_beg else doc_index_beg_offset
                length = None if i < doc_index_end else doc_index_end_offset + 1
                tokens.append(self.indexed_dataset.get(self.document_index[i], offset=offset, length=length))

        return {"text": numpy.array(numpy.concatenate(tokens), dtype=numpy.int64)}

    def _build_document_sample_shuffle_indices(
        self,
    ) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """Build the document index, the sample index, and the shuffle index

        The document index:
            -- 1-D
            -- An ordered array of document ids

        The sample index:
            -- 2-D
            -- The document indices and offsets which mark the start of every sample

        The shuffle index:
            -- 1-D
            -- A random permutation of index range of the sample index

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: The document index, the sample index, and the
            shuffle index

        TODO: Explain the 80% threshold
        """
        path_to_cache = getattr(self.config, "path_to_cache")
        if path_to_cache is None:
            path_to_cache = os.path.join(self.indexed_dataset.path_prefix, "cache", f"{type(self).__name__}_indices")

        def get_path_to(suffix):
            return os.path.join(path_to_cache, f"{self.unique_description_hash}-{type(self).__name__}-{suffix}")

        path_to_description = get_path_to("description.txt")
        path_to_document_index = get_path_to("document_index.npy")
        path_to_sample_index = get_path_to("sample_index.npy")
        path_to_shuffle_index = get_path_to("shuffle_index.npy")
        cache_hit = all(
            map(
                os.path.isfile,
                [
                    path_to_description,
                    path_to_document_index,
                    path_to_sample_index,
                    path_to_shuffle_index,
                ],
            )
        )

        num_tokens_per_epoch = _get_num_tokens_per_epoch(self.indexed_dataset, self.indexed_indices)

        sequence_length = getattr(self.config, "sequence_length")

        num_epochs = _get_num_epochs(num_tokens_per_epoch, sequence_length, self.num_samples)

        if not cache_hit and torch.distributed.get_rank() == 0:
            log_rank(
                f"Build and save the {type(self).__name__} {self.index_split.name} indices",
                logger=logger,
                level=logging.INFO,
                rank=0,
            )

            if num_epochs == 1:
                separate_final_epoch = False
            else:
                # Get the number of samples for the last epoch
                num_samples_sans_final_epoch = ((num_epochs - 1) * num_tokens_per_epoch - 1) // sequence_length
                num_samples_from_final_epoch = self.num_samples - num_samples_sans_final_epoch
                num_samples_per_epoch = (num_tokens_per_epoch - 1) // sequence_length

                # num_samples_from_final_epoch should be non-negative
                assert num_samples_from_final_epoch >= 0

                # num_samples_from_final_epoch should not exceed max value
                assert num_samples_from_final_epoch <= num_samples_per_epoch + 1

                # Separate the final epoch if it falls below the threshold
                threshold = 0.80
                separate_final_epoch = num_samples_from_final_epoch < int(threshold * num_samples_per_epoch)

                log_rank(
                    f"> num_samples_from_final_epoch: {num_samples_from_final_epoch}",
                    logger=logger,
                    level=logging.DEBUG,
                    rank=0,
                )
                log_rank(f"> Threshold: {threshold}", logger=logger, level=logging.DEBUG, rank=0)
                log_rank(
                    f"> num_samples_per_epoch: {num_samples_per_epoch}", logger=logger, level=logging.DEBUG, rank=0
                )

            log_rank(f"> separate_final_epoch: {separate_final_epoch}", logger=logger, level=logging.DEBUG, rank=0)

            numpy_random_state = numpy.random.RandomState(getattr(self.config, "random_seed"))

            os.makedirs(path_to_cache, exist_ok=True)

            # Write the description
            with open(path_to_description, "wt") as writer:
                writer.write(self.unique_description)

            # Build the document index
            log_rank(
                f"\tBuild and save the document index to {os.path.basename(path_to_document_index)}",
                logger=logger,
                level=logging.INFO,
                rank=0,
            )
            t_beg = time.time()
            document_index = _build_document_index(
                self.indexed_indices, num_epochs, numpy_random_state, separate_final_epoch
            )
            numpy.save(path_to_document_index, document_index, allow_pickle=True)
            t_end = time.time()
            log_rank(f"\t> Time elapsed: {t_end - t_beg:4f} seconds", logger=logger, level=logging.DEBUG, rank=0)

            # Build the sample index
            log_rank(
                f"\tBuild and save the sample index to {os.path.basename(path_to_sample_index)}",
                logger=logger,
                level=logging.INFO,
                rank=0,
            )
            t_beg = time.time()
            from nanotron.data import helpers

            assert document_index.dtype == numpy.int32
            assert self.indexed_dataset.sequence_lengths.dtype == numpy.int32
            sample_index = helpers.build_sample_idx(
                self.indexed_dataset.sequence_lengths,
                document_index,
                sequence_length,
                num_epochs,
                num_tokens_per_epoch,
            )
            numpy.save(path_to_sample_index, sample_index, allow_pickle=True)
            t_end = time.time()
            log_rank(f"\t> Time elapsed: {t_end - t_beg:4f} seconds", logger=logger, level=logging.DEBUG, rank=0)

            # Build the shuffle index
            log_rank(
                f"\tBuild and save the shuffle index to {os.path.basename(path_to_shuffle_index)}",
                logger=logger,
                level=logging.INFO,
                rank=0,
            )
            t_beg = time.time()
            if separate_final_epoch:
                shuffle_index = _build_shuffle_index(
                    num_samples_sans_final_epoch, sample_index.shape[0] - 1, numpy_random_state
                )
            else:
                shuffle_index = _build_shuffle_index(
                    sample_index.shape[0] - 1, sample_index.shape[0] - 1, numpy_random_state
                )
            numpy.save(path_to_shuffle_index, shuffle_index, allow_pickle=True)
            t_end = time.time()
            log_rank(f"\t> Time elapsed: {t_end - t_beg:4f} seconds", logger=logger, level=logging.DEBUG, rank=0)

        log_rank(
            f"Load the {type(self).__name__} {self.index_split.name} indices",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        log_rank(
            f"\tLoad the document index from {os.path.basename(path_to_document_index)}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )
        t_beg = time.time()
        document_index = numpy.load(path_to_document_index, allow_pickle=True, mmap_mode="r")
        t_end = time.time()
        log_rank(f"\t> time elapsed: {t_end - t_beg:4f} seconds", logger=logger, level=logging.DEBUG, rank=0)

        log_rank(
            f"\tLoad the sample index from {os.path.basename(path_to_sample_index)}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )
        t_beg = time.time()
        sample_index = numpy.load(path_to_sample_index, allow_pickle=True, mmap_mode="r")
        t_end = time.time()
        log_rank(f"\t> Time elapsed: {t_end - t_beg:4f} seconds", logger=logger, level=logging.DEBUG, rank=0)

        log_rank(
            f"\tLoad the shuffle index from {os.path.basename(path_to_shuffle_index)}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )
        t_beg = time.time()
        shuffle_index = numpy.load(path_to_shuffle_index, allow_pickle=True, mmap_mode="r")
        t_end = time.time()
        log_rank(f"\t> Time elapsed: {t_end - t_beg:4f} seconds", logger=logger, level=logging.DEBUG, rank=0)

        log_rank(f"> Total number of samples: {sample_index.shape[0] - 1}", logger=logger, level=logging.INFO, rank=0)
        log_rank(f"> Total number of epochs: {num_epochs}", logger=logger, level=logging.INFO, rank=0)

        return document_index, sample_index, shuffle_index


def _get_num_tokens_per_epoch(indexed_dataset: MMapIndexedDataset, indices: numpy.ndarray) -> int:
    """Calculate the number of tokens in a single epoch

    Args:
        indexed_dataset (MMapIndexedDataset): The underlying MMapIndexedDataset

        indices (numpy.ndarray): The subset of indices into the underlying MMapIndexedDataset

    Returns:
        int: The number of tokens in a single epoch
    """
    return numpy.sum(indexed_dataset.sequence_lengths[indices])


def _get_num_epochs(num_tokens_per_epoch: int, seq_length: int, num_samples: int) -> int:
    """Calculate the number of epochs

    Args:
        num_tokens_per_epoch (int): The number of tokens in a single epoch

        seq_length (int): The sequence length in tokens

        num_samples (int): The total number of samples

    Returns:
        int: The number of epochs
    """
    num_epochs = 0
    num_tokens = 0
    while True:
        num_epochs += 1
        num_tokens += num_tokens_per_epoch
        # -1 is because we need to retrieve seq_length + 1 token each time
        # but the last token will overlap with the first token of the next
        # sample except for the last sample.
        if ((num_tokens - 1) // seq_length) >= num_samples:
            return num_epochs


def _build_document_index(
    documents: numpy.ndarray,
    num_epochs: int,
    numpy_random_state: numpy.random.RandomState,
    separate_final_epoch: bool,
) -> numpy.ndarray:
    """Build an array with length = num epochs * num documents

    Args:
        documents (numpy.ndarray): the subset of exposed document indices

        num_epochs (int): The number of epochs

        numpy_random_state (numpy.random.RandomState): The NumPy random state

        separate_final_epoch (bool): Whether to exclude the last epoch from the global shuffle

    Returns:
        numpy.ndarray: The document index

    TODO: Explain separate_final_epoch
    """
    if not separate_final_epoch or num_epochs == 1:
        document_index = numpy.mgrid[0:num_epochs, 0 : len(documents)][1]
        document_index[:] = documents
        document_index = document_index.reshape(-1)
        document_index = document_index.astype(numpy.int32)
        numpy_random_state.shuffle(document_index)
        return document_index

    doc_idx_first = _build_document_index(documents, num_epochs - 1, numpy_random_state, False)
    doc_idx_last = _build_document_index(documents, 1, numpy_random_state, False)
    return numpy.concatenate((doc_idx_first, doc_idx_last))


def _build_shuffle_index(
    num_samples: int, total_size: int, numpy_random_state: numpy.random.RandomState
) -> numpy.ndarray:
    """Build the range [0, size) and shuffle

    Args:
        num_samples (int): The size of the first shuffle range [0, num_samples)

        total_size (int): The size of the entire index. If larger than 'num_samples', it defines

        the second shuffle range [num_samples, total_size)

        numpy_random_state (numpy.random.RandomState): The NumPy random state

    Returns:
        numpy.ndarray: The shuffle index

    TODO: Explain [0, num_samples) [num_samples, total_size) split
    """
    dtype_ = numpy.uint32
    if total_size >= (numpy.iinfo(numpy.uint32).max - 1):
        dtype_ = numpy.int64

    shuffle_idx_first = numpy.arange(start=0, stop=num_samples, step=1, dtype=dtype_)
    numpy_random_state.shuffle(shuffle_idx_first)
    if num_samples == total_size:
        return shuffle_idx_first

    shuffle_idx_last = numpy.arange(start=num_samples, stop=total_size, step=1, dtype=dtype_)
    numpy_random_state.shuffle(shuffle_idx_last)

    return numpy.concatenate((shuffle_idx_first, shuffle_idx_last))
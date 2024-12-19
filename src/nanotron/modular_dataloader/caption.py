import io
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

from nanotron import distributed as dist
from nanotron.modular_dataloader.base import BatchEncoder, SampleEncoder
from nanotron.parallel.context import ParallelContext
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer


@dataclass
class FormattedTextSample:
    """
    A sample with formatted text for processing.
    :param text: Text should include a single <image> instance.
    """
    text: str
    image: bytes

@dataclass
class ProcessedSample:
    """
    Tokenized text and image.
    """
    input_ids: torch.Tensor
    pixel_values: torch.Tensor

@dataclass
class CaptionSampleEncoder(SampleEncoder[FormattedTextSample]):
    """
    Sample encoder for caption samples.
    """

    text_field: str = "caption"
    image_field: str = "jpg"
    image_token: str = "<image>"

    def encode(self, sample: Dict[str, Any]) -> FormattedTextSample:
        """
        Encode a caption sample.
        """
        return FormattedTextSample(text= f"{self.image_token}{sample[self.text_field]}", image=sample[self.image_field])


@dataclass
class ProcessSampleEncoder(SampleEncoder[ProcessedSample]):
    """
    Sample encoder for caption samples that also applies processor to it.
    """

    processor: AutoProcessor
    sequence_length: int
    text_field: str = "caption"
    image_field: str = "jpg"

    def encode(self, sample: Dict[str, Any]) -> ProcessedSample:
        """
        Encode a caption sample.
        """
        image_token = self.processor.image_token.content

        text = f"{image_token}{sample[self.text_field]}"
        image = sample[self.image_field]
        image_arr = byte_img_to_array(image)
        inputs = self.processor(text=text, images=[image_arr], return_tensors="pt", padding="longest", max_length=self.sequence_length + 1, truncation=True)

        return ProcessedSample(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"]
    )

@dataclass
class PreprocessedSampleEncoder(SampleEncoder[ProcessedSample]):
    """
    Sample encoder for caption samples that also applies processor to it.
    """

    processor: AutoProcessor
    sequence_length: int

    def encode(self, sample: Dict[str, Any]) -> ProcessedSample:
        """
        Encode a caption sample.
        """
        input_ids = torch.tensor(sample["input_ids"], dtype=torch.long)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        pixel_values = torch.tensor(sample["pixel_values"], dtype=torch.float32)
        pixel_shape = sample["pixel_shape"]

        pixel_values = pixel_values.reshape(pixel_shape)

        if len(input_ids.shape) == 4:
            input_ids = input_ids.unsqueeze(0)

        return ProcessedSample(
            input_ids=input_ids,
            pixel_values=pixel_values
        )


def byte_img_to_array(bimg):
    imageStream = io.BytesIO(bimg)
    imageFile = Image.open(imageStream)
    img_arr = np.array(imageFile)
    if len(img_arr.shape) == 2:  # Grayscale image
        img_arr = np.expand_dims(img_arr, axis=-1)  # Add a channel dimension
    # imageFile.show()
    return img_arr

@dataclass
class SingleImageBatchEncoder(BatchEncoder[FormattedTextSample]):
    """
    Expects an Idefics3 compatible processor. Pads texts and splits images.
    Only works for a single image per caption. 
    - input_pp_rank: Discards last input id token. Returns input_ids, input_mask, pixel_values
    - output_pp_rank: Discards first label id token. Returns label_ids, label_mask
    - other pp ranks: Don't have data. Instead, we use `TensorPointer` to point to the rank having the data.
    """
    input_pp_rank: int
    output_pp_rank: int
    parallel_context: ParallelContext
    processor: AutoProcessor
    sequence_length: int

    def encode(self, batch: List[FormattedTextSample]) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        """
        Encode a batch of caption samples.
        """
        current_pp_rank = dist.get_rank(self.parallel_context.pp_pg)
        if current_pp_rank not in [
            self.input_pp_rank,
            self.output_pp_rank,
        ]:
            assert len(batch) == 0
            return {
                "input_ids": TensorPointer(group_rank=self.input_pp_rank),
                "input_mask": TensorPointer(group_rank=self.input_pp_rank),
                "label_ids": TensorPointer(group_rank=self.output_pp_rank),
                "label_mask": TensorPointer(group_rank=self.output_pp_rank),
                "pixel_values": TensorPointer(group_rank=self.input_pp_rank),
            }

        result = {}

        result["input_ids"] = TensorPointer(group_rank=self.input_pp_rank)
        result["input_mask"] = TensorPointer(group_rank=self.input_pp_rank)
        result["label_ids"] = TensorPointer(group_rank=self.output_pp_rank)
        result["label_mask"] = TensorPointer(group_rank=self.output_pp_rank)
        result["pixel_values"] = TensorPointer(group_rank=self.input_pp_rank)

        texts = [sample.text for sample in batch]
        images = [[byte_img_to_array(sample.image)] for sample in batch]

        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding="longest", max_length=self.sequence_length + 1, truncation=True)


        if current_pp_rank == self.input_pp_rank:
            result["input_ids"] = inputs["input_ids"][:, :-1]
            result["input_mask"] = inputs["attention_mask"]
            result["pixel_values"] = inputs["pixel_values"]

        if current_pp_rank == self.output_pp_rank:
            result["label_ids"] = inputs["input_ids"][:, 1:]
            result["label_mask"] = inputs["input_ids"][:, 1:] < self.processor.tokenizer.vocab_size

        return result


@dataclass
class PreprocessedCollator(BatchEncoder[ProcessedSample]):
    """
    Collates and encodes a batch of samples.
    """
    input_pp_rank: int
    output_pp_rank: int
    parallel_context: ParallelContext
    processor: AutoProcessor
    sequence_length: int
    padding_side: str = "right"

    def encode(self, batch: List[ProcessedSample]) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        """
        Collate and encode a batch of samples.
        """
        current_pp_rank = dist.get_rank(self.parallel_context.pp_pg)
        if current_pp_rank not in [
            self.input_pp_rank,
            self.output_pp_rank,
        ]:
            assert len(batch) == 0
            return {
                "input_ids": TensorPointer(group_rank=self.input_pp_rank),
                "input_mask": TensorPointer(group_rank=self.input_pp_rank),
                "label_ids": TensorPointer(group_rank=self.output_pp_rank),
                "label_mask": TensorPointer(group_rank=self.output_pp_rank),
                "pixel_values": TensorPointer(group_rank=self.input_pp_rank),
            }

        result = {}

        result["input_ids"] = TensorPointer(group_rank=self.input_pp_rank)
        result["input_mask"] = TensorPointer(group_rank=self.input_pp_rank)
        result["label_ids"] = TensorPointer(group_rank=self.output_pp_rank)
        result["label_mask"] = TensorPointer(group_rank=self.output_pp_rank)
        result["pixel_values"] = TensorPointer(group_rank=self.input_pp_rank)

        def pad_tokens(inputs = True):
            max_seq_len = max(x.input_ids.shape[1] for x in batch) - 1
            # Make it divisible by tp group size
            gs = self.parallel_context.tp_pg.size()
            max_seq_len = max_seq_len + (gs - max_seq_len % gs) % gs

            padded_token_ids = []
            padded_attention_mask = []

            for example in batch:
                token_ids = example.input_ids
                if inputs:
                    token_ids = token_ids[:, :-1]
                else:
                    token_ids = token_ids[:, 1:]

                current_seq_len = token_ids.shape[1]
                padding_length = max_seq_len - current_seq_len

                if inputs:
                    attention_mask = torch.ones_like(token_ids)
                else:
                    attention_mask = token_ids < self.processor.tokenizer.vocab_size

                padding = torch.zeros((token_ids.shape[0], padding_length), dtype=token_ids.dtype, device=token_ids.device)

                if self.padding_side == "right":
                    padded_token_ids.append(torch.cat([token_ids, padding], dim=1))
                    padded_attention_mask.append(torch.cat([attention_mask, padding], dim=1))

                elif self.padding_side == "left":
                    padded_token_ids.append(torch.cat([padding, token_ids], dim=1))
                    padded_attention_mask.append(torch.cat([padding, attention_mask], dim=1))

            padded_token_ids = torch.cat(padded_token_ids, dim=0)
            padded_attention_mask = torch.cat(padded_attention_mask, dim=0)

            return padded_token_ids, padded_attention_mask



        if current_pp_rank == self.input_pp_rank:
            max_n_patches = max(x.pixel_values.shape[1] for x in batch)
            padded_pixel_values = []

            for example in batch:
                pixel_values = example.pixel_values
                current_patches = pixel_values.shape[1]

                # Pad the pixel_values to have max_n_patches along dimension 1 (patches)
                padding = torch.zeros((1, max_n_patches - current_patches) + pixel_values.shape[2:], dtype=pixel_values.dtype, device=pixel_values.device)
                padded_pixel_values.append(torch.cat([pixel_values, padding], dim=1))

            padded_pixel_values = torch.cat(padded_pixel_values, dim=0)
            result["pixel_values"] = padded_pixel_values
            result["input_ids"], result["input_mask"] = pad_tokens(inputs=True)
        if current_pp_rank == self.output_pp_rank:
            result["label_ids"], result["label_mask"] = pad_tokens(inputs=False)

        return result

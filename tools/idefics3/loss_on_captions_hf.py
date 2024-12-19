"""
torchrun --nproc-per-node 1 tools/idefics3/loss_on_captions_hf.py --pretrained-model-name-or-path HuggingFaceM4/Idefics3-8B-Llama3
"""

import argparse
import os
from typing import List, Optional
from PIL import Image

import numpy as np
import requests
import torch

from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
from transformers.modeling_flash_attention_utils import _flash_attention_forward

from torch.utils.data import DataLoader

from datasets import load_dataset

DEVICE = torch.device("cuda")
TORCH_DTYPE = torch.bfloat16


def caption_to_messages(caption):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What do we see in this image?"},
            ]
        },     
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "This image shows: " + caption},
            ]
        },
    ]   

    return messages 

def check_image(image):
    image = np.array(image)
    if image.ndim == 2:
        image = image[:, :, None]
    return image

def collate_fn(examples, processor):
    captions = [
        processor.apply_chat_template(caption_to_messages(example["image_description"])) for example in examples
    ]
    images = [[check_image(example["image"])] for example in examples]

    inputs = processor(text=captions, images=images, return_tensors="pt", padding="max_length", max_length=2049, truncation=True, padding_side="right")

    input_ids = inputs["input_ids"][:, :-1]
    attention_mask = inputs["attention_mask"][:, :-1] == 1
    label_ids = inputs["input_ids"][:, 1:]
    label_mask = label_ids < processor.tokenizer.vocab_size
    pixel_values = inputs["pixel_values"]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": label_ids, "label_mask": label_mask, "pixel_values": pixel_values}


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="HuggingFace Model")
    group.add_argument(
        "--pretrained-model-name-or-path",
        type=str,
        required=True,
        help="A path to a directory containing model weights saved using save_pretrained() or the model id of a pretrained model hosted inside a model repo on the Hugging Face Hub",
    )

    args = parser.parse_args()

    return args


def main(args):

    model = AutoModelForVision2Seq.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=TORCH_DTYPE,
        attn_implementation="flash_attention_2",
        device_map="auto",
    ).eval()


    dataset = load_dataset("jmhessel/newyorker_caption_contest", 'explanation', split="validation[:100]")
    processor = AutoProcessor.from_pretrained("HuggingFaceM4/Idefics3-8B-Llama3", size={"longest_edge": 2 * 364})

    dataloader = DataLoader(dataset, batch_size=16, num_workers=16, collate_fn=lambda x: collate_fn(x, processor))

    total_loss = 0
    total_acc = 0 

    for batch in dataloader:
        inputs = {k: v.to(DEVICE) for k, v in batch.items()}

        with torch.no_grad():
            output = model(
                use_cache=False,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"],
            )
        
        logits = output[0]
        label_mask = inputs["label_mask"]
        labels = inputs["labels"]

        loss = torch.nn.functional.cross_entropy(logits[label_mask], labels[label_mask])
        total_loss += loss.item()

        acc = (logits.argmax(dim=-1)[label_mask] == labels[label_mask]).float().mean().item()

        total_acc += acc

    print(f"Average Loss: {total_loss / len(dataloader)}")
    print(f"Average Accuracy: {total_acc / len(dataloader)}")
    
    # Average Loss: 2.1875
    # Average Accuracy: 0.6541961346353803


if __name__ == "__main__":
    _args = get_args()
    main(_args)

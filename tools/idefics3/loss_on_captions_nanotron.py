"""
torchrun --nproc-per-node 1 tools/idefics3/loss_on_captions_nanotron.py --tp 1 --nanotron-checkpoint-path nanotron_checkpoints/Nanotron-Idefics3-8B-Llama3 --dataset-path "../datasets/ny_captions.hf"
"""
import argparse
import os
from pathlib import Path

import nanotron.distributed as dist
import torch.distributed as torch_dist
import numpy as np
import torch
from nanotron.config import Config, ParallelismArgs, get_config_from_file
from nanotron.models import build_model
from nanotron.models.idefics import Idefics3ForTraining
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import sanity_check
from nanotron.parallel.pipeline_parallel.engine import AllForwardAllBackwardPipelineEngine
from nanotron.parallel.tensor_parallel.nn import TensorParallelLinearMode
from nanotron.serialize import load_weights
from nanotron.trainer import mark_tied_parameters
from transformers import AutoProcessor
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

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
    group = parser.add_argument_group(title="Nanotron Model")
    group.add_argument(
        "--nanotron-checkpoint-path",
        type=str,
        required=True,
        help="A path to a directory containing a Nanotron Checkpoint",
    )
    group = parser.add_argument_group(title="Dataset")
    group.add_argument(
        "--dataset-path",
        type=str,
        required=False,
        help="A path to a directory containing the dataset",
    )

    group = parser.add_argument_group(title="Nanotron Parallelism")
    group.add_argument("--tp", type=int, required=True, help="Tensor Parallelism Degree of the Nanotron Checkpoint")

    args = parser.parse_args()

    return args    

def main(args):
    # Init Nanotron Parallel Utilities
    parallel_config = ParallelismArgs(
        dp=1,
        pp=1,
        tp=args.tp,
        pp_engine=AllForwardAllBackwardPipelineEngine(),
        tp_mode=TensorParallelLinearMode.ALL_REDUCE,
        tp_linear_async_communication=False,
    )
    assert (
        parallel_config.tp_mode == TensorParallelLinearMode.ALL_REDUCE
        and parallel_config.tp_linear_async_communication is False
    )

    parallel_context = ParallelContext(
        data_parallel_size=parallel_config.dp,
        pipeline_parallel_size=parallel_config.pp,
        tensor_parallel_size=parallel_config.tp,
    )

    RANK = dist.get_rank(parallel_context.world_pg)

    nanotron_config = get_config_from_file(
        os.path.join(args.nanotron_checkpoint_path, "config.yaml"), config_class=Config, model_config_class=None
    )

    model = build_model(
        model_builder=lambda: Idefics3ForTraining(
            config=nanotron_config.model.model_config,
            parallel_context=parallel_context,
            parallel_config=parallel_config,
        ),
        parallel_context=parallel_context,
        dtype=TORCH_DTYPE,
        device=DEVICE,  # TODO Check with different parallelism if cpu is available
    )

    mark_tied_parameters(model=model, parallel_context=parallel_context)
    sanity_check(root_module=model)

    # Load checkpoint directly in memory and then only keep the state dictionary
    load_weights(model=model, parallel_context=parallel_context, root_folder=Path(args.nanotron_checkpoint_path))

    if args.dataset_path is not None:
        dataset = load_from_disk(args.dataset_path)
    else:
        dataset = load_dataset("jmhessel/newyorker_caption_contest", 'explanation', split="validation[:100]")
    
    processor = AutoProcessor.from_pretrained("HuggingFaceM4/Idefics3-8B-Llama3", size={"longest_edge": 2 * 364})

    dataloader = DataLoader(dataset, batch_size=4, num_workers=16, collate_fn=lambda x: collate_fn(x, processor))

    total_loss = 0
    total_acc = 0
    n_samples = 0

    def gather_logits(logits, parallel_context):
        tp_pg = parallel_context.tp_pg
        if tp_pg.size() == 1:
            return logits

        sharded_shape = logits.shape
        
        tensor_list = [torch.empty(sharded_shape, device=logits.device, dtype=logits.dtype) for _ in range(tp_pg.size())]

        torch_dist.all_gather(tensor_list, logits, group=tp_pg)

        logits = torch.cat(tensor_list, dim=-1)

        return logits

    for batch in tqdm(dataloader):
        inputs = {k: v.to(DEVICE) for k, v in batch.items()}

        with torch.no_grad():
            output = model.model(
                input_ids=inputs["input_ids"],
                input_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"],
            )

        logits = gather_logits(output, parallel_context).transpose(0, 1)

        label_mask = inputs["label_mask"]
        labels = inputs["labels"]
        
        loss = torch.nn.functional.cross_entropy(logits[label_mask], labels[label_mask])

        acc = (logits.argmax(dim=-1)[label_mask] == labels[label_mask]).float().mean().item()

        if RANK == 0:
            total_acc += acc
            total_loss += loss.item()


    if RANK == 0:
        print(f"Average Loss: {total_loss / len(dataloader)}")
        print(f"Average Accuracy: {total_acc / len(dataloader)}")
    
    # Average Loss: 2.1875 (HF)
    # Average Loss: 2.112454278128488 (Nanotron TP=1)
    # Average Loss: 2.112218448093959 (Nanotron TP=2)

    # Average Accuracy: 0.6541961346353803 (HF)
    # Average Loss:  0.6715155754770551 (Nanotron TP=1)
    # Average Loss:  0.6702071257999965 (Nanotron TP=2)

if __name__ == "__main__":
    _args = get_args()
    main(_args)

"""
torchrun --nproc-per-node 2 tools/idefics3/generate_nanotron_predictions.py --tp 2 --nanotron-checkpoint-path nanotron_checkpoints/Nanotron-Idefics3-8B-Llama3
"""
import argparse
import os
from pathlib import Path

import requests

import nanotron.distributed as dist
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
# from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoProcessor
from PIL import Image


messages = [{
    "role": "user",
    "content": [
        {"type": "text", "text": "What’s the difference between these two images?"},
        {"type": "image"},
        {"type": "image"},
    ],
},
{
    "role": "assistant",
    "content": [
        {"type": "text", "text": "The difference is that one image is about dogs and the other one about cats."},
    ],
}]


url_1 = "http://images.cocodataset.org/val2017/000000039769.jpg"
url_2 = "http://images.cocodataset.org/val2017/000000219578.jpg"

SEQ_LENGTH = 512  # For truncating the TXT if GPU can't fit too many tokens

DEVICE = torch.device("cuda")
TORCH_DTYPE = torch.bfloat16


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="Nanotron Model")
    group.add_argument(
        "--nanotron-checkpoint-path",
        type=str,
        required=True,
        help="A path to a directory containing a Nanotron Checkpoint",
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

    
    #torch.Size([484, 26, 768])

    mark_tied_parameters(model=model, parallel_context=parallel_context)
    sanity_check(root_module=model)

    # Load checkpoint directly in memory and then only keep the state dictionary
    load_weights(model=model, parallel_context=parallel_context, root_folder=Path(args.nanotron_checkpoint_path))


    image_1 = Image.open(requests.get(url_1, stream=True).raw)
    image_2 = Image.open(requests.get(url_2, stream=True).raw)
    images = [image_1, image_2]

    # Using non-Idefics3 image size may break the pixel shuffle
    # For example, instead of 384 you should use either 364 or 404
    image_size = nanotron_config.model.model_config.vision_config.image_size
    
    image_size = 364

    target_image_seq_len = int(((image_size // nanotron_config.model.model_config.vision_config.patch_size) ** 2) / (nanotron_config.model.model_config.scale_factor**2))

    processor = AutoProcessor.from_pretrained("HuggingFaceM4/Idefics3-8B-Llama3", image_seq_len=target_image_seq_len, size= {"longest_edge": 4*image_size}, max_image_size = {"longest_edge": image_size})

    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(images=images, text=text, return_tensors="pt", image_seq_len=target_image_seq_len).to(DEVICE)

    inputs = {
        "input_ids": inputs['input_ids'],
        "input_mask": inputs['attention_mask'],
        "pixel_values": inputs['pixel_values'].bfloat16(),
        "pixel_attention_mask": inputs['pixel_attention_mask'],
    }

    model.eval()

    with torch.no_grad():
        output = model.model(**inputs)

    if not RANK:
        print(output.shape)

        predicted_tokens = [5, 27, 34]  # Index of the predictions to compare across models
        term_cols = int(os.get_terminal_size().columns / 3)

        for predicted_token in predicted_tokens:

            print("\n", "=" * term_cols, f"Predictions of token {predicted_token}", "=" * term_cols)
            next_tokens = torch.softmax(output.transpose(0, 1)[0, predicted_token, :], -1)
            
            topk_next_tokens = torch.topk(next_tokens, 10)

            print(
                *[
                    f"[Nanotron Model] Next token: {idx.item()}, probability: {prob}"
                    for idx, prob in zip(topk_next_tokens.indices, topk_next_tokens.values)
                ],
                sep="\n",
            )

if __name__ == "__main__":
    _args = get_args()
    main(_args)

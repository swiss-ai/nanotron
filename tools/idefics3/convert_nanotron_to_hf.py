"""
torchrun --nproc-per-node 1 tools/idefics3/convert_nanotron_to_hf.py --huggingface-checkpoint-path idefics3_ckpt --pretrained-model-name-or-path nanotron_checkpoints/Nanotron-Idefics3-8B-Llama3
"""

import sys
sys.path.insert(0, '/capstor/scratch/cscs/eguo/vlm_convert/nanotron')

import argparse
import os
from dataclasses import asdict
import json
from pathlib import Path
import torch
from tqdm import tqdm
import yaml
from nanotron import logging
from nanotron.config import Config, LoggingArgs, ParallelismArgs, get_config_from_file
from nanotron.config.models_config import ExistingCheckpointInit, Idefics3VisionConfig, Idefics3Config
from nanotron.config.parallelism_config import ParallelismArgs
from nanotron.logging import log_rank, set_ranks_logging_level
from nanotron.config.models_config import LlamaConfig as LlamaConfigNanotron
from nanotron.models.base import build_model
from nanotron.models.llama import LlamaForTraining
from nanotron.models.idefics import Idefics3ForTraining, Idefics3Model, VisionTransformer
from nanotron.parallel.context import ParallelContext
from nanotron.trainer import mark_tied_parameters
from nanotron.parallel.parameters import sanity_check
from nanotron.serialize import load_weights
from nanotron.serialize.weights import save_weights

from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig
from transformers.models.llama import LlamaConfig as LlamaConfigHF
from transformers import Idefics3Config as Idefics3ConfigHF
from accelerate import init_empty_weights


logger = logging.get_logger(__name__)

DEVICE = torch.device("cpu")
TORCH_DTYPE = torch.bfloat16

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="Path to Save Converted HuggingFace Idefic3 Model")
    group.add_argument(
        "--huggingface-checkpoint-path",
        type=str,
        required=True,
        help="A path to a directory to store the converted HF Checkpoint",
    )

    group = parser.add_argument_group(title="Nanotron Idefic3 Model")
    group.add_argument(
        "--pretrained-model-name-or-path",
        type=str,
        required=True,
        help="A path to a directory containing model weights saved using save_pretrained() or the model id of a pretrained model hosted inside a model repo in nanotron",
    )

    args = parser.parse_args()

    return args

def copy_weights_from_nanotron_to_hf_llama(nanotron_model, hf_model, nanotron_llama_config, additional_vocab_size):
    # Copy params from Nanotron to HF
    log_rank("Copying weights from Nanotron model to HF model...", logger=logger, level=logging.INFO, rank=0)

    # Decoder layers
    for i in tqdm(
        range(nanotron_llama_config.num_hidden_layers),
        desc="Copying Hidden Layers",
        total=nanotron_llama_config.num_hidden_layers,
    ):
        # Input layer norm
        assert (
            hf_model.layers[i].input_layernorm.weight.shape
            == nanotron_model.decoder[i].pp_block.input_layernorm.weight.shape
        )
        with torch.no_grad():
            hf_model.layers[i].input_layernorm.weight.copy_(
                nanotron_model.decoder[i].pp_block.input_layernorm.weight
            )

        # Self-attention QKV split
        qkv_proj = nanotron_model.decoder[i].pp_block.attn.qkv_proj.weight
        q_size = nanotron_llama_config.num_attention_heads * nanotron_llama_config.hidden_size // nanotron_llama_config.num_attention_heads
        k_size = nanotron_llama_config.num_key_value_heads * nanotron_llama_config.hidden_size // nanotron_llama_config.num_attention_heads
        v_size = nanotron_llama_config.num_key_value_heads * nanotron_llama_config.hidden_size // nanotron_llama_config.num_attention_heads

        q, k, v = torch.split(qkv_proj, [q_size, k_size, v_size], dim=0)

        assert q.shape == hf_model.layers[i].self_attn.q_proj.weight.shape
        assert k.shape == hf_model.layers[i].self_attn.k_proj.weight.shape
        assert v.shape == hf_model.layers[i].self_attn.v_proj.weight.shape

        with torch.no_grad():
            hf_model.layers[i].self_attn.q_proj.weight.copy_(q)
            hf_model.layers[i].self_attn.k_proj.weight.copy_(k)
            hf_model.layers[i].self_attn.v_proj.weight.copy_(v)

        # Output projection (O)
        assert (
            hf_model.layers[i].self_attn.o_proj.weight.shape
            == nanotron_model.decoder[i].pp_block.attn.o_proj.weight.shape
        )
        with torch.no_grad():
            hf_model.layers[i].self_attn.o_proj.weight.copy_(
                nanotron_model.decoder[i].pp_block.attn.o_proj.weight
            )

        # MLP: Gate and Up Proj
        gate_up_proj = nanotron_model.decoder[i].pp_block.mlp.gate_up_proj.weight
        split_size = nanotron_llama_config.intermediate_size
        gate_proj, up_proj = torch.split(gate_up_proj, [split_size, split_size], dim=0)

        assert gate_proj.shape == hf_model.layers[i].mlp.gate_proj.weight.shape
        assert up_proj.shape == hf_model.layers[i].mlp.up_proj.weight.shape

        with torch.no_grad():
            hf_model.layers[i].mlp.gate_proj.weight.copy_(gate_proj)
            hf_model.layers[i].mlp.up_proj.weight.copy_(up_proj)

        # MLP: Down Proj
        assert (
            hf_model.layers[i].mlp.down_proj.weight.shape
            == nanotron_model.decoder[i].pp_block.mlp.down_proj.weight.shape
        )
        with torch.no_grad():
            hf_model.layers[i].mlp.down_proj.weight.copy_(
                nanotron_model.decoder[i].pp_block.mlp.down_proj.weight
            )

        # Post-attention Layer Norm
        assert (
            hf_model.layers[i].post_attention_layernorm.weight.shape
            == nanotron_model.decoder[i].pp_block.post_attention_layernorm.weight.shape
        )
        with torch.no_grad():
            hf_model.layers[i].post_attention_layernorm.weight.copy_(
                nanotron_model.decoder[i].pp_block.post_attention_layernorm.weight
            )

    # Final Layer Norm
    log_rank("Copying Final Layer Norm...", logger=logger, level=logging.INFO, rank=0)
    assert nanotron_model.final_layer_norm.pp_block.weight.shape == hf_model.norm.weight.shape
    with torch.no_grad():
        hf_model.norm.weight.copy_(nanotron_model.final_layer_norm.pp_block.weight)

    log_rank("Llama weight copying completed successfully!", logger=logger, level=logging.INFO, rank=0)


def copy_weights_from_nanotron_to_hf_vision(
    nanotron_model: VisionTransformer,
    hf_model: AutoModel,
    nanotron_vision_config: Idefics3VisionConfig
):
    log_rank("Copying weights from Nanotron model to HF model...", logger=logger, level=logging.INFO, rank=0)

    # Vision Embeddings
    log_rank("Copying Vision Embeddings...", logger=logger, level=logging.INFO, rank=0)

    assert (
        nanotron_model.embeddings.patch_embedding.weight.shape == hf_model.embeddings.patch_embedding.weight.shape
    )
    assert (
        nanotron_model.embeddings.patch_embedding.bias.shape == hf_model.embeddings.patch_embedding.bias.shape
    )
    assert (
        nanotron_model.embeddings.position_embedding.weight.shape
        == hf_model.embeddings.position_embedding.weight.shape
    )

    with torch.no_grad():
        hf_model.embeddings.patch_embedding.weight.copy_(
            nanotron_model.embeddings.patch_embedding.weight
        )
        hf_model.embeddings.patch_embedding.bias.copy_(
            nanotron_model.embeddings.patch_embedding.bias
        )
        hf_model.embeddings.position_embedding.weight.copy_(
            nanotron_model.embeddings.position_embedding.weight
        )

    log_rank("Copied Vision Embeddings", logger=logger, level=logging.INFO, rank=0)

    for i in tqdm(
        range(nanotron_vision_config.num_hidden_layers),
        desc="Copying Vision Layers",
        total=nanotron_vision_config.num_hidden_layers,
    ):
        # Layer Norm 1
        assert (
            nanotron_model.encoder[i].layer_norm1.weight.shape == hf_model.encoder.layers[i].layer_norm1.weight.shape
        )
        assert (
            nanotron_model.encoder[i].layer_norm1.bias.shape == hf_model.encoder.layers[i].layer_norm1.bias.shape
        )

        with torch.no_grad():
            hf_model.encoder.layers[i].layer_norm1.weight.copy_(
                nanotron_model.encoder[i].layer_norm1.weight
            )
            hf_model.encoder.layers[i].layer_norm1.bias.copy_(
                nanotron_model.encoder[i].layer_norm1.bias
            )

        # QKV Projections
        tmp_qkv_proj = nanotron_model.encoder[i].self_attn.qkv_proj.weight.chunk(3, dim=0)

        assert (
            tmp_qkv_proj[0].shape == hf_model.encoder.layers[i].self_attn.q_proj.weight.shape
        )
        assert (
            tmp_qkv_proj[1].shape == hf_model.encoder.layers[i].self_attn.k_proj.weight.shape
        )
        assert (
            tmp_qkv_proj[2].shape == hf_model.encoder.layers[i].self_attn.v_proj.weight.shape
        )

        with torch.no_grad():
            hf_model.encoder.layers[i].self_attn.q_proj.weight.copy_(tmp_qkv_proj[0])
            hf_model.encoder.layers[i].self_attn.k_proj.weight.copy_(tmp_qkv_proj[1])
            hf_model.encoder.layers[i].self_attn.v_proj.weight.copy_(tmp_qkv_proj[2])

        # QKV Biases
        tmp_qkv_proj_bias = nanotron_model.encoder[i].self_attn.qkv_proj.bias.chunk(3, dim=0)

        assert (
            tmp_qkv_proj_bias[0].shape == hf_model.encoder.layers[i].self_attn.q_proj.bias.shape
        )
        assert (
            tmp_qkv_proj_bias[1].shape == hf_model.encoder.layers[i].self_attn.k_proj.bias.shape
        )
        assert (
            tmp_qkv_proj_bias[2].shape == hf_model.encoder.layers[i].self_attn.v_proj.bias.shape
        )

        with torch.no_grad():
            hf_model.encoder.layers[i].self_attn.q_proj.bias.copy_(tmp_qkv_proj_bias[0])
            hf_model.encoder.layers[i].self_attn.k_proj.bias.copy_(tmp_qkv_proj_bias[1])
            hf_model.encoder.layers[i].self_attn.v_proj.bias.copy_(tmp_qkv_proj_bias[2])

        # Output Projection
        assert (
            nanotron_model.encoder[i].self_attn.o_proj.weight.shape == hf_model.encoder.layers[i].self_attn.out_proj.weight.shape
        )
        assert (
            nanotron_model.encoder[i].self_attn.o_proj.bias.shape == hf_model.encoder.layers[i].self_attn.out_proj.bias.shape
        )

        with torch.no_grad():
            hf_model.encoder.layers[i].self_attn.out_proj.weight.copy_(
                nanotron_model.encoder[i].self_attn.o_proj.weight
            )
            hf_model.encoder.layers[i].self_attn.out_proj.bias.copy_(
                nanotron_model.encoder[i].self_attn.o_proj.bias
            )

        # Layer Norm 2
        assert (
            nanotron_model.encoder[i].layer_norm2.weight.shape == hf_model.encoder.layers[i].layer_norm2.weight.shape
        )
        assert (
            nanotron_model.encoder[i].layer_norm2.bias.shape == hf_model.encoder.layers[i].layer_norm2.bias.shape
        )

        with torch.no_grad():
            hf_model.encoder.layers[i].layer_norm2.weight.copy_(
                nanotron_model.encoder[i].layer_norm2.weight
            )
            hf_model.encoder.layers[i].layer_norm2.bias.copy_(
                nanotron_model.encoder[i].layer_norm2.bias
            )

        # MLP Layers
        assert (
            nanotron_model.encoder[i].mlp.fc1.weight.shape == hf_model.encoder.layers[i].mlp.fc1.weight.shape
        )
        assert (
            nanotron_model.encoder[i].mlp.fc1.bias.shape == hf_model.encoder.layers[i].mlp.fc1.bias.shape
        )

        with torch.no_grad():
            hf_model.encoder.layers[i].mlp.fc1.weight.copy_(
                nanotron_model.encoder[i].mlp.fc1.weight
            )
            hf_model.encoder.layers[i].mlp.fc1.bias.copy_(
                nanotron_model.encoder[i].mlp.fc1.bias
            )

        assert (
            nanotron_model.encoder[i].mlp.fc2.weight.shape == hf_model.encoder.layers[i].mlp.fc2.weight.shape
        )
        assert (
            nanotron_model.encoder[i].mlp.fc2.bias.shape == hf_model.encoder.layers[i].mlp.fc2.bias.shape
        )

        with torch.no_grad():
            hf_model.encoder.layers[i].mlp.fc2.weight.copy_(
                nanotron_model.encoder[i].mlp.fc2.weight
            )
            hf_model.encoder.layers[i].mlp.fc2.bias.copy_(
                nanotron_model.encoder[i].mlp.fc2.bias
            )

    log_rank("Copied Vision Layers", logger=logger, level=logging.INFO, rank=0)

    # Post Layer Norm
    assert (
        nanotron_model.post_layernorm.weight.shape == hf_model.post_layernorm.weight.shape
    )
    assert (
        nanotron_model.post_layernorm.bias.shape == hf_model.post_layernorm.bias.shape
    )

    with torch.no_grad():
        hf_model.post_layernorm.weight.copy_(nanotron_model.post_layernorm.weight)
        hf_model.post_layernorm.bias.copy_(nanotron_model.post_layernorm.bias)

    log_rank("Copied Post Layer Norm", logger=logger, level=logging.INFO, rank=0)

def copy_weights_from_nanotron_to_hf_remaining(
    nanotron_model: Idefics3Model,
    hf_model: AutoModel,
    nanotron_config: Idefics3Config
):
    log_rank("Copying weights from Idefic3 Llama embeddings to Nanotron model...", logger=logger, level=logging.INFO, rank=0)

    hf_vocab_size = hf_model.model.text_model.config.vocab_size

    assert (
        nanotron_model.combined_embeddings.pp_block.text_embeddings.token_embedding.weight[:hf_vocab_size].shape
        == hf_model.model.text_model.embed_tokens.weight.shape
    )
    with torch.no_grad():
        hf_model.model.text_model.embed_tokens.weight.copy_(
            nanotron_model.combined_embeddings.pp_block.text_embeddings.token_embedding.weight[:hf_vocab_size]
        )

    log_rank("Copying weights from Idefic3 Connector to Nanotron model...", logger=logger, level=logging.INFO, rank=0)

    assert (
        nanotron_model.combined_embeddings.pp_block.connector.modality_projector.proj.weight.shape == hf_model.model.connector.modality_projection.proj.weight.shape
    )

    with torch.no_grad():
        hf_model.model.connector.modality_projection.proj.weight.copy_(nanotron_model.combined_embeddings.pp_block.connector.modality_projector.proj.weight)

    log_rank("Copied Connector", logger=logger, level=logging.INFO, rank=0)

    vocab_size = hf_model.vocab_size

    assert (
        nanotron_model.lm_head.pp_block.weight[:vocab_size].shape == hf_model.lm_head.weight.shape
    )

    with torch.no_grad():
        hf_model.lm_head.weight.copy_(nanotron_model.lm_head.pp_block.weight[:vocab_size])
        
    log_rank("Copied Head", logger=logger, level=logging.INFO, rank=0)


def main(args):
    # Init Nanotron Parallel Utilities
    additional_vocab_size = 1
    parallel_config = ParallelismArgs(dp=1, pp=1, tp=1)

    parallel_context = ParallelContext(
        data_parallel_size=parallel_config.dp,
        pipeline_parallel_size=parallel_config.pp,
        tensor_parallel_size=parallel_config.tp,
    )

    set_ranks_logging_level(parallel_context=parallel_context, logging_config=LoggingArgs())

    # Load Nanotron checkpoint config
    log_rank(
        f"Loading Nanotron checkpoint config file: {args.pretrained_model_name_or_path}",
        logger=logger,
        level=logging.INFO,
        rank=0,
    )

    nanotron_config = get_config_from_file(
        os.path.join(args.pretrained_model_name_or_path, "config.yaml"), config_class=Config, model_config_class=None
    )
    nanotron_idefics3_config = nanotron_config.model.model_config
    nanotron_llama_config = nanotron_idefics3_config.text_config
    nanotron_vision_config = nanotron_idefics3_config.vision_config


    # Init Idefics3 Nanotron model
    log_rank("Init empty Nanotron Idefics3 Model", logger=logger, level=logging.INFO, rank=0)

    nanotron_model = build_model(
        model_builder=lambda: Idefics3ForTraining(
            config=nanotron_idefics3_config,
            parallel_context=parallel_context,
            parallel_config=parallel_config,
        ),
        parallel_context=parallel_context,
        dtype=TORCH_DTYPE,
        device=DEVICE,
    )

    mark_tied_parameters(model=nanotron_model, parallel_context=parallel_context)
    sanity_check(root_module=nanotron_model)

    # # Load Nanotron Checkpoint
    log_rank("Loading Nanotron Idefics3 Model...", logger=logger, level=logging.INFO, rank=0)
    load_weights(
        model=nanotron_model, parallel_context=parallel_context, root_folder=Path(args.pretrained_model_name_or_path)
    )

    # Build empty HF Model
    # log_rank("Init empty HF Llama3 Model", logger=logger, level=logging.INFO, rank=0)

    # hf_llama_model = AutoModelForCausalLM.from_config(  # WARN This takes a long time
    #     config=LlamaConfigHF(**asdict(nanotron_llama_config)),
    #     torch_dtype=TORCH_DTYPE,
    #     attn_implementation="flash_attention_2",
    # ).to(DEVICE)

    # log_rank("Init empty HF SigLIP Model", logger=logger, level=logging.INFO, rank=0)
    # hf_siglip_model = AutoModel.from_config(
    #     config=SigLIPConfigHF(**asdict(nanotron_vision_config)),
    #     torch_dtype=TORCH_DTYPE,
    #     attn_implementation="flash_attention_2",
    # ).to(DEVICE)


    log_rank("Init empty HF Idefics3 Model", logger=logger, level=logging.INFO, rank=0)

    with init_empty_weights():
        hf_idefics3_model = AutoModelForVision2Seq.from_config(
            config=Idefics3ConfigHF(**asdict(nanotron_idefics3_config)),
            torch_dtype=TORCH_DTYPE,
            attn_implementation="flash_attention_2",
        ).to_empty(device=DEVICE)



    # hf_idefics3_model = AutoModelForVision2Seq.from_pretrained(
    #     args.hf_pretrained_model_name_or_path, torch_dtype=TORCH_DTYPE, attn_implementation="flash_attention_2"
    # ).to(DEVICE)

    # Copy weights from Nanotron to Hugging Face
    copy_weights_from_nanotron_to_hf_llama(
        nanotron_model=nanotron_model.model.llama,
        # hf_model=hf_llama_model,
        hf_model=hf_idefics3_model.model.text_model,
        nanotron_llama_config=nanotron_idefics3_config.text_config,
        additional_vocab_size=additional_vocab_size,
    )

    log_rank("Copied weights from Nanotron Llama model to HF model!", logger=logger, level=logging.INFO, rank=0)

    copy_weights_from_nanotron_to_hf_vision(
        nanotron_model=nanotron_model.model.combined_embeddings.pp_block.vision_model,
        # hf_model=hf_siglip_model,
        hf_model=hf_idefics3_model.model.vision_model,
        nanotron_vision_config=nanotron_vision_config,
    )

    log_rank("Copied weights from Nanotron SigLIP model to HF model!", logger=logger, level=logging.INFO, rank=0)

    copy_weights_from_nanotron_to_hf_remaining(
        nanotron_model=nanotron_model.model,
        hf_model=hf_idefics3_model,
        nanotron_config=nanotron_idefics3_config,
    )

    # log_rank("Copied weights from Nanotron Idefics3 model to HF model!", logger=logger, level=logging.INFO, rank=0)

    hf_checkpoint_path = Path(
        args.huggingface_checkpoint_path,
    )

    # save_weights(
    #     model=hf_idefics3_model,
    #     parallel_context=parallel_context,
    #     root_folder=hf_checkpoint_path,
    # )

    # Store weights
    log_rank("Saving HF model Checkpoint and Tokenizer!", logger=logger, level=logging.INFO, rank=0)
    # hf_llama_model.save_pretrained(args.hugging_face_checkpoint_path_llama, from_pt=True)
    # # Store tokenizer
    # tokenizer_llama = AutoTokenizer.from_pretrained(nanotron_llama_config.tokenizer.tokenizer_name_or_path)
    # tokenizer_llama.save_pretrained(args.hugging_face_checkpoint_path_llama)
    # log_rank(
    #     f"Checkpoint conversion finished, check {args.hugging_face_checkpoint_path_llama}",
    #     logger=logger,
    #     level=logging.INFO,
    #     rank=0,
    # )

    # # Store weights
    # hf_siglip_model.save_pretrained(args.hugging_face_checkpoint_path_siglip, from_pt=True)
    # # Store tokenizer
    # tokenizer_siglip = AutoTokenizer.from_pretrained(nanotron_vision_config.tokenizer.tokenizer_name_or_path)
    # tokenizer_siglip.save_pretrained(args.hugging_face_checkpoint_path_siglip)

    hf_idefics3_model.save_pretrained(args.huggingface_checkpoint_path, from_pt=True)
    hf_idefics3_model.config.save_pretrained(args.huggingface_checkpoint_path)
    # tokenizer_idefics3 = AutoTokenizer.from_pretrained(nanotron_config.tokenizer.tokenizer_name_or_path)
    # tokenizer_idefics3.save_pretrained(args.huggingface_checkpoint_path)

    log_rank(
        f"Checkpoint conversion finished, check {args.huggingface_checkpoint_path}",
        logger=logger,
        level=logging.INFO,
        rank=0,
    )

if __name__ == "__main__":
    _args = get_args()
    main(_args)

                                                                                                                                                                                                        
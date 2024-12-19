"""
HF_HUB_ENABLE_HF_TRANSFER=1 torchrun --nproc-per-node 1 tools/idefics3/build_nanotron_from_hf.py --nanotron-checkpoint-path nanotron_checkpoints/Nanotron-Idefics3-8B-Llama3 --pretrained-model-name-or-path-llama3 meta-llama/Meta-Llama-3-8B-Instruct --pretrained-model-name-or-path-siglip google/siglip-so400m-patch14-384
"""
import sys
sys.path.append('.venv/lib/python3.10/site-packages')

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import torch
from tqdm import tqdm
import yaml
from nanotron import logging
from nanotron.config.config import Config, GeneralArgs, LoggingArgs, ModelArgs, TokenizerArgs
from nanotron.config.models_config import ExistingCheckpointInit, Idefics3VisionConfig, Idefics3Config
from nanotron.config.parallelism_config import ParallelismArgs
from nanotron.logging import log_rank, set_ranks_logging_level
from nanotron.config.models_config import LlamaConfig as LlamaConfigNanotron
from nanotron.models.base import build_model
from nanotron.models.idefics import Idefics3ForTraining, Idefics3Model, VisionTransformer
from nanotron.parallel.context import ParallelContext
from nanotron.parallel.parameters import sanity_check
from nanotron.serialize.weights import save_weights
from nanotron.trainer import mark_tied_parameters

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

logger = logging.get_logger(__name__)

DEVICE = torch.device("cpu")
TORCH_DTYPE = torch.bfloat16


def copy_weights_from_hf_to_nanotron_llama(nanotron_model, hf_model, nanotron_config, 
    additional_vocab_size):
    nanotron_llama_config = nanotron_config.text_config
    # Copy params from HF to Nanotron
    log_rank("Copying weights from HF model to Nanotron model...", logger=logger, level=logging.INFO, rank=0)
    # Token embeddings
    log_rank("Copying Token Embeddings...", logger=logger, level=logging.INFO, rank=0)

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
            nanotron_model.decoder[i].pp_block.input_layernorm.weight.copy_(
                hf_model.layers[i].input_layernorm.weight
            )

        # Self attn
        ## QKV
        tmp_qkv_proj = torch.cat(
            [
                hf_model.layers[i].self_attn.q_proj.weight,
                hf_model.layers[i].self_attn.k_proj.weight,
                hf_model.layers[i].self_attn.v_proj.weight,
            ],
            dim=0,
        )
        assert tmp_qkv_proj.shape == nanotron_model.decoder[i].pp_block.attn.qkv_proj.weight.shape
        with torch.no_grad():
            nanotron_model.decoder[i].pp_block.attn.qkv_proj.weight.copy_(tmp_qkv_proj)

        ## O
        assert (
            hf_model.layers[i].self_attn.o_proj.weight.shape
            == nanotron_model.decoder[i].pp_block.attn.o_proj.weight.shape
        )
        with torch.no_grad():
            nanotron_model.decoder[i].pp_block.attn.o_proj.weight.copy_(
                hf_model.layers[i].self_attn.o_proj.weight
            )

        # MLP
        ## Gate Up Proj
        tmp_gate_up_proj = torch.cat(
            [
                hf_model.layers[i].mlp.gate_proj.weight,
                hf_model.layers[i].mlp.up_proj.weight,
            ],
            dim=0,
        )

        assert tmp_gate_up_proj.shape == nanotron_model.decoder[i].pp_block.mlp.gate_up_proj.weight.shape
        with torch.no_grad():
            nanotron_model.decoder[i].pp_block.mlp.gate_up_proj.weight.copy_(tmp_gate_up_proj)

        ## Down Proj
        assert (
            hf_model.layers[i].mlp.down_proj.weight.shape
            == nanotron_model.decoder[i].pp_block.mlp.down_proj.weight.shape
        )
        with torch.no_grad():
            nanotron_model.decoder[i].pp_block.mlp.down_proj.weight.copy_(
                hf_model.layers[i].mlp.down_proj.weight
            )

        # Post attn layer norm
        assert (
            hf_model.layers[i].post_attention_layernorm.weight.shape
            == nanotron_model.decoder[i].pp_block.post_attention_layernorm.weight.shape
        )
        with torch.no_grad():
            nanotron_model.decoder[i].pp_block.post_attention_layernorm.weight.copy_(
                hf_model.layers[i].post_attention_layernorm.weight
            )

        # Last layer norm
        log_rank("Copying Final Layer Norm...", logger=logger, level=logging.INFO, rank=0)
        assert nanotron_model.final_layer_norm.pp_block.weight.shape == hf_model.norm.weight.shape
        with torch.no_grad():
            nanotron_model.final_layer_norm.pp_block.weight.copy_(hf_model.norm.weight)

def nanotron_config_from_hf_config_llama(hf_config, additional_vocab_size=3):
    return LlamaConfigNanotron(
        bos_token_id=hf_config.bos_token_id,
        eos_token_id=hf_config.eos_token_id,
        hidden_act=hf_config.hidden_act,
        hidden_size=hf_config.hidden_size,
        initializer_range=hf_config.initializer_range,
        intermediate_size=hf_config.intermediate_size,
        is_llama_config=True,
        max_position_embeddings=hf_config.max_position_embeddings,
        num_attention_heads=hf_config.num_attention_heads,
        num_hidden_layers=hf_config.num_hidden_layers,
        num_key_value_heads=hf_config.num_key_value_heads,
        pad_token_id=None,
        pretraining_tp=hf_config.pretraining_tp,
        rms_norm_eps=hf_config.rms_norm_eps,
        rope_scaling=hf_config.rope_scaling,
        rope_theta=hf_config.rope_theta,
        rope_interleaved=False,
        tie_word_embeddings=hf_config.tie_word_embeddings,
        use_cache=hf_config.use_cache,
        vocab_size=hf_config.vocab_size + additional_vocab_size,
    )



def copy_weights_from_hf_to_nanotron_vision(
    nanotron_model: VisionTransformer,
    hf_model: AutoModel,
    nanotron_vision_config: Idefics3VisionConfig
):
    log_rank("Copying weights from Idefic3 ViT model to Nanotron model...", logger=logger, level=logging.INFO, rank=0)

    # Vision Embeddings
    log_rank("Copying Vision Embeddings...", logger=logger, level=logging.INFO, rank=0)

    assert (
        nanotron_model.embeddings.patch_embedding.weight.shape == hf_model.embeddings.patch_embedding.weight.shape
    )

    assert(
        nanotron_model.embeddings.patch_embedding.bias.shape == hf_model.embeddings.patch_embedding.bias.shape
    )

    assert (
        nanotron_model.embeddings.position_embedding.weight.shape
        == hf_model.embeddings.position_embedding.weight.shape
    )

    with torch.no_grad():
        nanotron_model.embeddings.patch_embedding.weight.copy_(hf_model.embeddings.patch_embedding.weight)

        nanotron_model.embeddings.patch_embedding.bias.copy_(hf_model.embeddings.patch_embedding.bias)

        nanotron_model.embeddings.position_embedding.weight.copy_(hf_model.embeddings.position_embedding.weight)
        

    log_rank("Copied Vision Embeddings", logger=logger, level=logging.INFO, rank=0)

    for i in tqdm(
        range(nanotron_vision_config.num_hidden_layers),
        desc="Copying Vision Layers",
        total=nanotron_vision_config.num_hidden_layers,
    ):
        assert (
            nanotron_model.encoder[i].layer_norm1.weight.shape == hf_model.encoder.layers[i].layer_norm1.weight.shape
        )

        with torch.no_grad():
            nanotron_model.encoder[i].layer_norm1.weight.copy_(hf_model.encoder.layers[i].layer_norm1.weight)

        assert (
            nanotron_model.encoder[i].layer_norm1.bias.shape == hf_model.encoder.layers[i].layer_norm1.bias.shape
        )

        with torch.no_grad():
            nanotron_model.encoder[i].layer_norm1.bias.copy_(hf_model.encoder.layers[i].layer_norm1.bias)

        tmp_qkv_proj = torch.cat(
            [
                hf_model.encoder.layers[i].self_attn.q_proj.weight,
                hf_model.encoder.layers[i].self_attn.k_proj.weight,
                hf_model.encoder.layers[i].self_attn.v_proj.weight,
            ],
            dim=0,
        )

        assert (
            tmp_qkv_proj.shape == nanotron_model.encoder[i].self_attn.qkv_proj.weight.shape
        )

        with torch.no_grad():
            nanotron_model.encoder[i].self_attn.qkv_proj.weight.copy_(tmp_qkv_proj)
        
        tmp_qkv_proj_bias = torch.cat(
            [
                hf_model.encoder.layers[i].self_attn.q_proj.bias,
                hf_model.encoder.layers[i].self_attn.k_proj.bias,
                hf_model.encoder.layers[i].self_attn.v_proj.bias,
            ],
            dim=0,
        )

        assert (
            tmp_qkv_proj_bias.shape == nanotron_model.encoder[i].self_attn.qkv_proj.bias.shape
        )

        with torch.no_grad():
            nanotron_model.encoder[i].self_attn.qkv_proj.bias.copy_(tmp_qkv_proj_bias)

        ## O

        assert (
            nanotron_model.encoder[i].self_attn.o_proj.weight.shape == hf_model.encoder.layers[i].self_attn.out_proj.weight.shape
        )

        with torch.no_grad():
            nanotron_model.encoder[i].self_attn.o_proj.weight.copy_(hf_model.encoder.layers[i].self_attn.out_proj.weight)

        assert (
            nanotron_model.encoder[i].self_attn.o_proj.bias.shape == hf_model.encoder.layers[i].self_attn.out_proj.bias.shape
        )

        with torch.no_grad():
            nanotron_model.encoder[i].self_attn.o_proj.bias.copy_(hf_model.encoder.layers[i].self_attn.out_proj.bias)

        # Layer Norm 2

        assert (
            nanotron_model.encoder[i].layer_norm2.weight.shape == hf_model.encoder.layers[i].layer_norm2.weight.shape
        )

        with torch.no_grad():
            nanotron_model.encoder[i].layer_norm2.weight.copy_(hf_model.encoder.layers[i].layer_norm2.weight)

        assert (
            nanotron_model.encoder[i].layer_norm2.bias.shape == hf_model.encoder.layers[i].layer_norm2.bias.shape
        )

        with torch.no_grad():
            nanotron_model.encoder[i].layer_norm2.bias.copy_(hf_model.encoder.layers[i].layer_norm2.bias)

        # MLP
        ## FC1

        assert (
            nanotron_model.encoder[i].mlp.fc1.weight.shape == hf_model.encoder.layers[i].mlp.fc1.weight.shape
        )

        with torch.no_grad():
            nanotron_model.encoder[i].mlp.fc1.weight.copy_(hf_model.encoder.layers[i].mlp.fc1.weight)

        assert (
            nanotron_model.encoder[i].mlp.fc1.bias.shape == hf_model.encoder.layers[i].mlp.fc1.bias.shape
        )

        with torch.no_grad():
            nanotron_model.encoder[i].mlp.fc1.bias.copy_(hf_model.encoder.layers[i].mlp.fc1.bias)

        ## FC2

        assert (
            nanotron_model.encoder[i].mlp.fc2.weight.shape == hf_model.encoder.layers[i].mlp.fc2.weight.shape
        )

        with torch.no_grad():
            nanotron_model.encoder[i].mlp.fc2.weight.copy_(hf_model.encoder.layers[i].mlp.fc2.weight)

        assert (
            nanotron_model.encoder[i].mlp.fc2.bias.shape == hf_model.encoder.layers[i].mlp.fc2.bias.shape
        )
        
        with torch.no_grad():
            nanotron_model.encoder[i].mlp.fc2.bias.copy_(hf_model.encoder.layers[i].mlp.fc2.bias)

    log_rank("Copied Vision Layers", logger=logger, level=logging.INFO, rank=0)

    # Post layer norm

    assert (
        nanotron_model.post_layernorm.weight.shape == hf_model.post_layernorm.weight.shape
    )

    with torch.no_grad():
        nanotron_model.post_layernorm.weight.copy_(hf_model.post_layernorm.weight)

    assert (
        nanotron_model.post_layernorm.bias.shape == hf_model.post_layernorm.bias.shape
    )

    with torch.no_grad():
        nanotron_model.post_layernorm.bias.copy_(hf_model.post_layernorm.bias)

    log_rank("Copied Post Layer Norm", logger=logger, level=logging.INFO, rank=0)

def copy_weights_from_hf_to_nanotron_remaining(
    nanotron_model: Idefics3Model,
    hf_model_llama: AutoModel,
    nanotron_config: Idefics3Config
):

    log_rank("Copying weights from Idefic3 Llama embeddings to Nanotron model...", logger=logger, level=logging.INFO, rank=0)

    hf_vocab_size = hf_model_llama.config.vocab_size

    assert (
        nanotron_model.combined_embeddings.pp_block.text_embeddings.token_embedding.weight[:hf_vocab_size].shape
        == hf_model_llama.embed_tokens.weight.shape
    )
    with torch.no_grad():
        nanotron_model.combined_embeddings.pp_block.text_embeddings.token_embedding.weight[:hf_vocab_size].copy_(
            hf_model_llama.embed_tokens.weight
        )


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="Nanotron Model")
    group.add_argument(
        "--nanotron-checkpoint-path",
        type=str,
        required=True,
        help="A path to a directory to store the converted Nanotron Checkpoint",
    )

    group = parser.add_argument_group(title="HuggingFace LLama3 Model")
    group.add_argument(
        "--pretrained-model-name-or-path-llama3",
        type=str,
        required=True,
        help="A path to a directory containing model weights saved using save_pretrained() or the model id of a pretrained model hosted inside a model repo on the Hugging Face Hub",
    )

    group = parser.add_argument_group(title="HuggingFace SigLIP Model")
    group.add_argument(
        "--pretrained-model-name-or-path-siglip",
        type=str,
        required=True,
        help="A path to a directory containing model weights saved using save_pretrained() or the model id of a pretrained model hosted inside a model repo on the Hugging Face Hub",
    )


    args = parser.parse_args()

    return args


def main(args):
    # Init Nanotron Parallel Utilities
    parallel_config = ParallelismArgs(dp=1, pp=1, tp=1)

    parallel_context = ParallelContext(
        data_parallel_size=parallel_config.dp,
        pipeline_parallel_size=parallel_config.pp,
        tensor_parallel_size=parallel_config.tp,
    )

    set_ranks_logging_level(parallel_context=parallel_context, logging_config=LoggingArgs())

    # Load Llama3-8B HF model
    log_rank(
        f"Loading pretrained Llama3 Model: {args.pretrained_model_name_or_path_llama3}",
        logger=logger,
        level=logging.INFO,
        rank=0,
    )

    hf_model_llama = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model_name_or_path_llama3, torch_dtype=TORCH_DTYPE, attn_implementation="flash_attention_2"
    ).to(DEVICE)
    hf_config_llama = hf_model_llama.config


    # Set Nanotron LlamaConfig
    vocab_size = hf_config_llama.vocab_size

    # Expand & ensure that it's divisible by 4

    additional_vocab_size = 4 - (vocab_size % 4)
    nanotron_llama_config = nanotron_config_from_hf_config_llama(hf_config_llama, additional_vocab_size)

    # Load SigLIP HF model
    log_rank(
        f"Loading pretrained SigLIP Model: {args.pretrained_model_name_or_path_siglip}",
        logger=logger,
        level=logging.INFO,
        rank=0,
    )

    hf_model_siglip = AutoModel.from_pretrained(
        args.pretrained_model_name_or_path_siglip, torch_dtype=TORCH_DTYPE, 
    attn_implementation="flash_attention_2",
    ).to(DEVICE)
    hf_config_siglip = hf_model_siglip.config.vision_config

    # Set Nanotron SigLIPConfig
    nanotron_vision_config = Idefics3VisionConfig(
        hidden_size=hf_config_siglip.hidden_size,
        image_size=hf_config_siglip.image_size,
        intermediate_size=hf_config_siglip.intermediate_size,
        num_hidden_layers= hf_config_siglip.num_hidden_layers,
        num_attention_heads=hf_config_siglip.num_attention_heads,
        num_key_value_heads=hf_config_siglip.num_attention_heads,
        num_channels=hf_config_siglip.num_channels,
        patch_size=hf_config_siglip.patch_size,
        hidden_act=hf_config_siglip.hidden_act,
        layer_norm_eps=hf_config_siglip.layer_norm_eps,
        attention_dropout=hf_config_siglip.attention_dropout,
        is_using_mup=False    
    )
    
    pad_token_id = hf_config_llama.pad_token_id
    if pad_token_id is None:
        pad_token_id = 128002

    nanotron_idefics3_config = Idefics3Config(
        text_config=nanotron_llama_config,
        vision_config=nanotron_vision_config,
        image_token_id=vocab_size + 1,
        pad_token_id=pad_token_id,
        scale_factor=2,
        vocab_size=vocab_size + additional_vocab_size,
    )

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

    copy_weights_from_hf_to_nanotron_vision(
        nanotron_model=nanotron_model.model.combined_embeddings.pp_block.vision_model,
        hf_model=hf_model_siglip.vision_model,
        nanotron_vision_config=nanotron_vision_config,
    )

    log_rank("Copied weights from HF SigLIP model to Nanotron model!", logger=logger, level=logging.INFO, rank=0)
    
    # Copy weights from HF to Nanotron
    copy_weights_from_hf_to_nanotron_llama(
        nanotron_model=nanotron_model.model.llama,
        hf_model=hf_model_llama.model,
        nanotron_config=nanotron_idefics3_config,
        additional_vocab_size=additional_vocab_size
    )

    log_rank("Copied weights from HF Llama model to Nanotron model!", logger=logger, level=logging.INFO, rank=0)

    
    copy_weights_from_hf_to_nanotron_remaining(
        nanotron_model=nanotron_model.model,
        hf_model_llama=hf_model_llama.model,
        nanotron_config=nanotron_idefics3_config,
    )

    nanotron_checkpoint_path = Path(
        args.nanotron_checkpoint_path
    )

    save_weights(
        model=nanotron_model,
        root_folder=nanotron_checkpoint_path,
        parallel_context=parallel_context,
    )

    # Store Config and Model Config files
    with open(nanotron_checkpoint_path / "config.yaml", "w") as f:
        config = Config(
            general=GeneralArgs(project="Nanotron", run="Idefics-Custom"),
            parallelism=parallel_config,
            model=ModelArgs(
                init_method=ExistingCheckpointInit(nanotron_checkpoint_path),
                model_config=nanotron_idefics3_config,
            ),
            tokenizer=TokenizerArgs(nanotron_checkpoint_path),
        )
        log_rank("Saving config ...", logger=logger, level=logging.INFO, rank=0)
        yaml.dump(config.as_dict(), f)

    with open(nanotron_checkpoint_path / "model_config.json", "w") as f:
        log_rank("Saving model config ...", logger=logger, level=logging.INFO, rank=0)
        json.dump(asdict(nanotron_idefics3_config), f)

    log_rank(
        f"Checkpoint conversion finished, check {args.nanotron_checkpoint_path}",
        logger=logger,
        level=logging.INFO,
        rank=0,
    )

    
if __name__ == "__main__":
    _args = get_args()
    main(_args)
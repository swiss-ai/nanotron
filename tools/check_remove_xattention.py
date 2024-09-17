"""
torchrun --nproc-per-node 1 tools/check_remove_xattention.py
"""
import numpy as np
import torch
from nanotron.config import ParallelismArgs
from nanotron.config.models_config import LlamaConfig as LlamaConfigNanotron
from nanotron.data.dataloader_builder import build_nanoset_dataloader
from nanotron.data.nanoset import Nanoset
from nanotron.models import build_model
from nanotron.models.llama_sft import LlamaForSFT
from nanotron.parallel import ParallelContext
from nanotron.parallel.pipeline_parallel.engine import AllForwardAllBackwardPipelineEngine
from nanotron.parallel.tensor_parallel.nn import TensorParallelLinearMode
from nanotron.trainer import mark_tied_parameters
from nanotron.utils import main_rank_first
from torch.nn import CrossEntropyLoss
from torch.testing import assert_close
from transformers import AutoModelForCausalLM, LlamaConfig

dtype = torch.bfloat16
device = torch.device("cuda")
PATH_TO_LLAMA = "/store/swissai/a06/models/Meta-Llama-3.1-8B"
PATH_TO_DATATROVE_DATASET = "/store/swissai/a06/datasets_raw/SlimPajama-6B-eos"

# NOTE(tj.solergibert) This script is for testing porpuses. ONLY use 1 GPU
DP = 1
PP = 1
TP = 1

# NOTE(tj.solergibert) How many K-first tokens must match
# NOTE(tj.solergibert) After running lot's of tests, MOST (If not 100%)  of the times the most probable token matches. Sometimes there are slightly differences in the next tokens,
# usually when the first token has a very high probability and the rest are left with < 1e-2.
TOPK_MATCH = 1

BATCHES = 15


def hf_build_labels_completions_only(input_ids, is_completitions):
    labels = np.where(
        is_completitions, input_ids, -100
    )  # Mask tokens that don't belong to the completitions by the Assistant
    return torch.tensor(np.array(labels, dtype=np.int64))


def main():
    hf_model = AutoModelForCausalLM.from_pretrained(
        PATH_TO_LLAMA, torch_dtype=dtype, attn_implementation="flash_attention_2"
    ).to(device)
    hf_config = LlamaConfig.from_pretrained(PATH_TO_LLAMA)

    parallel_config = ParallelismArgs(
        dp=DP,
        pp=PP,
        tp=TP,
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

    nanotron_config = LlamaConfigNanotron(
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
        vocab_size=hf_config.vocab_size,
    )

    nanotron_model = build_model(
        model_builder=lambda: LlamaForSFT(
            config=nanotron_config,
            parallel_context=parallel_context,
            parallel_config=parallel_config,
            random_states=None,
        ),
        parallel_context=parallel_context,
        dtype=dtype,
        device=device,
    )

    mark_tied_parameters(model=nanotron_model, parallel_context=parallel_context)

    # Copy Llama3-8B-Instruct parameters
    # Token embeddings
    assert (
        nanotron_model.model.token_position_embeddings.pp_block.token_embedding.weight.shape
        == hf_model.model.embed_tokens.weight.shape
    )

    with torch.no_grad():
        nanotron_model.model.token_position_embeddings.pp_block.token_embedding.weight.copy_(
            hf_model.model.embed_tokens.weight
        )  #  = hf_model.model.embed_tokens.weight.data

    # Decoder layers
    for i in range(nanotron_config.num_hidden_layers):
        # Input layer norm
        assert (
            hf_model.model.layers[i].input_layernorm.weight.shape
            == nanotron_model.model.decoder[i].pp_block.input_layernorm.weight.shape
        )
        with torch.no_grad():
            nanotron_model.model.decoder[i].pp_block.input_layernorm.weight.copy_(
                hf_model.model.layers[i].input_layernorm.weight
            )  #  = hf_model.model.layers[i].input_layernorm.weight
        # Self attn
        ## QKV
        tmp_qkv_proj = torch.cat(
            [
                hf_model.model.layers[i].self_attn.q_proj.weight,
                hf_model.model.layers[i].self_attn.k_proj.weight,
                hf_model.model.layers[i].self_attn.v_proj.weight,
            ],
            dim=0,
        )
        assert tmp_qkv_proj.shape == nanotron_model.model.decoder[i].pp_block.attn.qkv_proj.weight.shape
        with torch.no_grad():
            nanotron_model.model.decoder[i].pp_block.attn.qkv_proj.weight.copy_(
                tmp_qkv_proj
            )  #  = tmp_qkv_proj # torch.nn.Parameter(tmp_qkv_proj)

        ## O
        assert (
            hf_model.model.layers[i].self_attn.o_proj.weight.shape
            == nanotron_model.model.decoder[i].pp_block.attn.o_proj.weight.shape
        )
        with torch.no_grad():
            nanotron_model.model.decoder[i].pp_block.attn.o_proj.weight.copy_(
                hf_model.model.layers[i].self_attn.o_proj.weight
            )  #  = hf_model.model.layers[i].self_attn.o_proj.weight
        # MLP
        ## Gate Proj
        assert (
            hf_model.model.layers[i].mlp.gate_proj.weight.shape
            == nanotron_model.model.decoder[i].pp_block.mlp.gate_proj.weight.shape
        )
        with torch.no_grad():
            nanotron_model.model.decoder[i].pp_block.mlp.gate_proj.weight.copy_(
                hf_model.model.layers[i].mlp.gate_proj.weight
            )

        ## Up Proj
        assert (
            hf_model.model.layers[i].mlp.up_proj.weight.shape
            == nanotron_model.model.decoder[i].pp_block.mlp.up_proj.weight.shape
        )
        with torch.no_grad():
            nanotron_model.model.decoder[i].pp_block.mlp.up_proj.weight.copy_(
                hf_model.model.layers[i].mlp.up_proj.weight
            )

        ## Down Proj
        assert (
            hf_model.model.layers[i].mlp.down_proj.weight.shape
            == nanotron_model.model.decoder[i].pp_block.mlp.down_proj.weight.shape
        )
        with torch.no_grad():
            nanotron_model.model.decoder[i].pp_block.mlp.down_proj.weight.copy_(
                hf_model.model.layers[i].mlp.down_proj.weight
            )  #  = hf_model.model.layers[i].mlp.down_proj.weight

        # Post attn layer norm
        assert (
            hf_model.model.layers[i].post_attention_layernorm.weight.shape
            == nanotron_model.model.decoder[i].pp_block.post_attention_layernorm.weight.shape
        )
        with torch.no_grad():
            nanotron_model.model.decoder[i].pp_block.post_attention_layernorm.weight.copy_(
                hf_model.model.layers[i].post_attention_layernorm.weight
            )  #  = hf_model.model.layers[i].post_attention_layernorm.weight

    # Last layer norm
    assert nanotron_model.model.final_layer_norm.pp_block.weight.shape == hf_model.model.norm.weight.shape
    with torch.no_grad():
        nanotron_model.model.final_layer_norm.pp_block.weight.copy_(
            hf_model.model.norm.weight
        )  #  = hf_model.model.norm.weight
    # LM_Head
    assert nanotron_model.model.lm_head.pp_block.weight.shape == hf_model.lm_head.weight.shape
    with torch.no_grad():
        nanotron_model.model.lm_head.pp_block.weight.copy_(hf_model.lm_head.weight)  # = hf_model.lm_head.weight

    # Create Nanoset
    with main_rank_first(parallel_context.world_pg):
        train_dataset = Nanoset(
            dataset_folders=PATH_TO_DATATROVE_DATASET,
            sequence_length=4096,
            token_size=4,
            train_split_num_samples=10000000,
        )

    # Prepare dataloader
    train_dataloader = build_nanoset_dataloader(
        train_dataset,
        4096,
        remove_document_xattention=True,
        parallel_context=parallel_context,
        input_pp_rank=0,
        output_pp_rank=0,
        micro_batch_size=1,
        consumed_train_samples=0,
        dataloader_num_workers=0,
        dataloader_drop_last=True,
    )

    hf_model.eval()
    nanotron_model.eval()

    for i, batch in enumerate(train_dataloader):
        if i == BATCHES:
            break
        print(f"Checking sample {i}!")

        # Some DL Checks
        assert batch["input_ids"].shape == batch["label_ids"].shape
        assert batch["input_ids"].shape == batch["position_ids"].shape
        assert batch["input_ids"].shape == batch["label_mask"].shape

        with torch.inference_mode():
            output_nanotron = nanotron_model.model(
                input_ids=batch["input_ids"].cuda(), position_ids=batch["position_ids"].cuda()
            )
            output_hf = hf_model(input_ids=batch["input_ids"].cuda(), position_ids=batch["position_ids"].cuda())

        # Assertion of the logits
        # This will always fail! We aren't performing the SAME operations. Nanotron packs QKV matrices, MLP & LayerNorm is different. So we don't have to focus on MATCHING LOGITS BUT GENERATIONS
        # assert_close(output_hf.logits, output_nanotron.transpose(0, 1), rtol=1e-1, atol=1e-1)

        predicted_tokens = [33, 95, 132, 428, 744, 912, 1288]
        for predicted_token in predicted_tokens:
            print(predicted_token)
            next_tokens_hf = torch.softmax(output_hf.logits[0, predicted_token, :], -1)
            hf_topk_next_tokens = torch.topk(next_tokens_hf, 10)

            next_tokens_nanotron = torch.softmax(output_nanotron.transpose(0, 1)[0, predicted_token, :], -1)
            nanotron_topk_next_tokens = torch.topk(next_tokens_nanotron, 10)
            assert all(
                hf_topk_next_tokens[1][:TOPK_MATCH] == nanotron_topk_next_tokens[1][:TOPK_MATCH]
            ), f"HF: {hf_topk_next_tokens[1][:TOPK_MATCH]} \n\n{hf_topk_next_tokens[0][:TOPK_MATCH]}\n\n Nanotron: {nanotron_topk_next_tokens[1][:TOPK_MATCH]}\n\n{nanotron_topk_next_tokens[0][:TOPK_MATCH]}"

        print("All generations match!\nChecking Loss")

        # Loss check
        nanotron_loss = nanotron_model.loss(
            sharded_logits=output_nanotron,
            label_ids=batch["label_ids"].cuda(),
            label_mask=batch["label_mask"].cuda(),
        )["loss"]

        # Creating labels_ids for HF loss computation
        hf_labels = hf_build_labels_completions_only(
            batch["label_ids"].flatten().tolist(), batch["label_mask"].flatten().tolist()
        )
        shift_logits = output_hf.logits.contiguous()
        shift_labels = hf_labels.contiguous()
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, 128256)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to("cuda")
        hf_loss = loss_fct(shift_logits, shift_labels)

        assert_close(nanotron_loss, hf_loss, atol=1e-2, rtol=1e-2)  # -3 is fine for most cases too
        print("Loss match!")

    print("\n\n\nBoth generations and losses match!")


if __name__ == "__main__":
    main()
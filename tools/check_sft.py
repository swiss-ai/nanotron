import torch
from nanotron.config import ParallelismArgs
from nanotron.config.models_config import LlamaConfig as LlamaConfigNanotron
from nanotron.data.chat_dataset import ChatDataset
from nanotron.data.dataloader_builder import build_chat_dataloader
from nanotron.models import build_model
from nanotron.models.llama_sft import LlamaForSFT
from nanotron.parallel import ParallelContext
from nanotron.parallel.pipeline_parallel.engine import AllForwardAllBackwardPipelineEngine
from nanotron.parallel.tensor_parallel.nn import TensorParallelLinearMode
from nanotron.trainer import mark_tied_parameters
from torch.testing import assert_close
from transformers import AutoModelForCausalLM, LlamaConfig

dtype = torch.bfloat16
device = torch.device("cuda")
PATH_TO_LLAMA = "/mloscratch/homes/solergib/models/Meta-Llama-3-8B-Instruct"

# NOTE(tj.solergibert) This script is for testing porpuses. ONLY use 1 GPU
DP = 1
PP = 1
TP = 1

# NOTE(tj.solergibert) How many K-first tokens must match
TOPK_MATCH = 3


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
        ## Gate Up Proj
        tmp_gate_up_proj = torch.cat(
            [
                hf_model.model.layers[i].mlp.gate_proj.weight,
                hf_model.model.layers[i].mlp.up_proj.weight,
            ],
            dim=0,
        )

        assert tmp_gate_up_proj.shape == nanotron_model.model.decoder[i].pp_block.mlp.gate_up_proj.weight.shape
        with torch.no_grad():
            nanotron_model.model.decoder[i].pp_block.mlp.gate_up_proj.weight.copy_(
                tmp_gate_up_proj
            )  #  = tmp_gate_up_proj
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

    # Create ChatDataloaders
    train_dataset = ChatDataset(
        dataset_path="Open-Orca/SlimOrca",
        tokenizer_name_or_path=PATH_TO_LLAMA,
        sequence_length=2048,
        train_on_completions_only=True,
        remove_cross_attention=True,
        split="train",
        conversation_column_name="conversations",
        dp_rank=parallel_context.dp_pg.rank(),
        dp_ranks_size=parallel_context.dp_pg.size(),
    )

    # Prepare dataloader
    train_dataloader = build_chat_dataloader(
        dataset=train_dataset,
        sequence_length=2048,
        parallel_context=parallel_context,
        input_pp_rank=0,
        output_pp_rank=0,
    )

    batch = next(iter(train_dataloader))
    # Some DL Checks
    assert batch["input_ids"].shape == batch["label_ids"].shape
    assert batch["input_ids"].shape == batch["position_ids"].shape
    assert batch["input_ids"].shape == batch["label_mask"].shape

    hf_model.eval()
    nanotron_model.eval()

    with torch.inference_mode():
        output_nanotron = nanotron_model.model(
            input_ids=batch["input_ids"].cuda(), position_ids=batch["position_ids"].cuda()
        )
        output_hf = hf_model(input_ids=batch["input_ids"].cuda(), position_ids=batch["position_ids"].cuda())

    predicted_tokens = [37, 89, 125, 423, 698, 912, 1298, 1723]
    for predicted_token in predicted_tokens:
        next_tokens_hf = torch.softmax(output_hf.logits[0, predicted_token, :], -1)
        hf_topk_next_tokens = torch.topk(next_tokens_hf, 10)

        next_tokens_nanotron = torch.softmax(output_nanotron.transpose(0, 1)[0, predicted_token, :], -1)
        nanotron_topk_next_tokens = torch.topk(next_tokens_nanotron, 10)
        assert all(hf_topk_next_tokens[1][:TOPK_MATCH] == nanotron_topk_next_tokens[1][:TOPK_MATCH])

    print("All generations match!")
    # One last assertion of the logits
    assert_close(output_hf.logits, output_nanotron.transpose(0, 1), rtol=1e-1, atol=1e-1)


if __name__ == "__main__":
    main()

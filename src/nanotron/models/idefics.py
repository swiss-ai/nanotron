import torch
from typing import Dict, Optional, Union
from torch import nn

from nanotron import logging
from nanotron.config.config import Config
from nanotron.config.models_config import LlamaConfig, RandomInit, SpectralMupInit
from nanotron.config.parallelism_config import ParallelismArgs
from nanotron.logging import log_rank
from nanotron.models.base import NanotronModel
from nanotron.nn.layer_norm import TritonLayerNorm
from nanotron.parallel.context import ParallelContext
from nanotron.parallel.parameters import NanotronParameter
from nanotron.parallel.pipeline_parallel.block import PipelineBlock
from nanotron.parallel.pipeline_parallel.p2p import P2P
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from nanotron.parallel.tensor_parallel.distributed_differentiable_primitives import differentiable_all_gather, differentiable_identity, differentiable_scatter
from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
from nanotron.parallel.tensor_parallel.nn import TensorParallelColumnLinear, TensorParallelEmbedding, TensorParallelRowLinear
from nanotron.distributed import dist
from nanotron.config import Idefics3VisionConfig, Idefics3Config
from nanotron.generation.generate_store import AttachableStore
from nanotron.random import RandomStates, branch_random_state
from nanotron.scaling.parametrization import SpectralMupParametrizator, StandardParametrizator
from nanotron.utils import checkpoint_method
from nanotron.models.llama import GLUActivation, LlamaDecoderLayer, LlamaModel
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from nanotron.parallel.tensor_parallel.functional import sharded_cross_entropy


logger = logging.get_logger(__name__)

class LlamaEmbeddings(nn.Module, AttachableStore):
    def __init__(self, tp_pg: dist.ProcessGroup, config: LlamaConfig, parallel_config: Optional[ParallelismArgs]):
        super().__init__()
        self.token_embedding = TensorParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=config.pad_token_id,
            pg=tp_pg,
            mode=TensorParallelLinearMode.ALL_REDUCE,
        )
        self.pg = tp_pg

    def forward(self, input_ids: torch.Tensor, input_mask: torch.Tensor):  # [batch_size, seq_length]
        store = self.get_local_store()
        if store is not None:
            if "past_length" in store:
                past_length = store["past_length"]
            else:
                past_length = torch.zeros(1, dtype=torch.long, device=input_ids.device).expand(input_ids.shape[0])

            cumsum_mask = input_mask.cumsum(-1, dtype=torch.long)
            # Store new past_length in store
            store["past_length"] = past_length + cumsum_mask[:, -1]

        # Format input in `[seq_length, batch_size]` to support high TP with low batch_size
        input_embeds = self.token_embedding(input_ids)
        return {"input_embeds": input_embeds}


class VisionEmbedding(nn.Module, AttachableStore):
    """
    Sharded implementation of the Idefics3VisionEmbeddings from huggingface for nanotron. Uses CLIPVit for megatron as a reference.
    """
    def __init__(self, tp_pg: dist.ProcessGroup, config: Idefics3VisionConfig, parallel_config: Optional[ParallelismArgs]):
        super().__init__()
        self.tp_pg = tp_pg
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches_per_side = self.image_size // self.patch_size
        self.num_patches = self.num_patches_per_side ** 2
        self.num_positions = self.num_patches

        self.position_embedding = TensorParallelEmbedding(
            num_embeddings=self.num_positions,
            embedding_dim=self.embed_dim,
            pg=tp_pg,
            mode=TensorParallelLinearMode.ALL_REDUCE,
        )

    def forward(self, pixel_values: torch.FloatTensor, patch_attention_mask: torch.BoolTensor) -> Dict[str, torch.Tensor]:
        batch_size, _, max_im_h, max_im_w = pixel_values.shape

        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        max_nb_patches_h, max_nb_patches_w = max_im_h // self.patch_size, max_im_w // self.patch_size

        boundaries = torch.arange(1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side)
        position_ids = torch.full(size=(batch_size, max_nb_patches_h * max_nb_patches_w), fill_value=0)

        for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
            nb_patches_h = p_attn_mask[:, 0].sum()
            nb_patches_w = p_attn_mask[0].sum()

            fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
            fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)

            bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
            bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

            pos_ids = (bucket_coords_h[:, None] * self.num_patches_per_side + bucket_coords_w).flatten()
            position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids

        first_dim = position_ids.shape[0]
        group_size = self.tp_pg.size()

        if first_dim % group_size != 0:
            position_ids = nn.functional.pad(position_ids, (0, 0, 0, group_size - first_dim % group_size), mode="constant", value=0)

        position_ids = position_ids.to(self.position_embedding.weight.device)

        position_ids = self.position_embedding(position_ids)

        embeddings = embeddings + position_ids[:first_dim]

        return {
            "embeddings": embeddings,
        }
        

class VisionCoreAttention(nn.Module):
    def __init__(self, config: Idefics3VisionConfig, parallel_config: Optional[ParallelismArgs]):
        super().__init__()
        
        assert (
            config.hidden_size % config.num_attention_heads == 0
        ), "hidden_size must be divisible by num_attention_heads"

        self.d_qk = config.hidden_size // config.num_attention_heads
        self.d_v = config.hidden_size // config.num_attention_heads

        self.is_using_mup = config.is_using_mup
        self.checkpoint_attention = False

        self.dropout = config.attention_dropout

    @checkpoint_method(attr_name="checkpoint_attention")
    def forward(
        self,
        query_states: torch.Tensor,  # [batch_size, q_length, n_local_q_heads, inner_dim]
        key_states: torch.Tensor,  # [batch_size, kv_length, n_local_kv_heads, inner_dim]
        value_states: torch.Tensor,  # [batch_size, kv_length, n_local_kv_heads, inner_dim]
    ):
        from flash_attn.flash_attn_interface import flash_attn_func

        # NOTE: this scale is for µTransfer,
        # in SP, we use sqrt(1/d_h)
        softmax_scale = 1 / query_states.shape[-1] if self.is_using_mup else None
        causal = False
        dropout_rate = self.dropout if self.training else 0.0
        
        attn_output = flash_attn_func(
            q=query_states,
            k=key_states,
            v=value_states,
            dropout_p=dropout_rate,
            softmax_scale=softmax_scale,
            causal=causal,
            return_attn_probs=False,
        )

        return attn_output 
    
class VisionSelfAttention(nn.Module, AttachableStore):
    def __init__(self, config: Idefics3VisionConfig, parallel_config: Optional[ParallelismArgs],
    tp_pg: dist.ProcessGroup, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        assert (
            config.num_attention_heads % tp_pg.size() == 0
        ), f"Number of attention heads ({config.num_attention_heads}) must be divisible by TP size ({tp_pg.size()})."
        try:
            assert (
                config.num_key_value_heads % tp_pg.size() == 0
            ), f"Number of key/value heads ({config.num_key_value_heads}) must be divisible by TP size ({tp_pg.size()})."
        except AttributeError:
            log_rank(
                "WARNING: num_key_value_heads not defined, assuming it is equal to num_attention_heads",
                logger=logger,
                level=logging.WARNING,
                rank=0,
            )
            # If num_key_value_heads is not defined, we assume that it is equal to num_attention_heads
            config.num_key_value_heads = config.num_attention_heads
        assert (
            config.num_attention_heads % config.num_key_value_heads == 0
        ), f"Number of attention heads ({config.num_attention_heads}) must be divisible by number of key/value heads ({config.num_key_value_heads})."
        self.n_local_q_heads = config.num_attention_heads // tp_pg.size()
        self.n_local_kv_heads = config.num_key_value_heads // tp_pg.size()
        self.n_repeats = config.num_attention_heads // config.num_key_value_heads
        self.is_gqa = config.num_attention_heads != config.num_key_value_heads  # Whether we are using GQA or not
        self.d_qk = config.hidden_size // config.num_attention_heads
        self.d_v = config.hidden_size // config.num_attention_heads
        self.d_model = config.hidden_size
        self.is_using_mup = config.is_using_mup



        tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE
        tp_linear_async_communication = (
            parallel_config.tp_linear_async_communication if parallel_config is not None else False
        )

        qkv_contiguous_chunks = (
            config.num_attention_heads * self.d_qk,
            config.num_key_value_heads * self.d_qk,
            config.num_key_value_heads * self.d_qk
        )

        self.qkv_proj = TensorParallelColumnLinear(
            self.d_model,
            config.num_attention_heads * self.d_qk + 2 * config.num_key_value_heads * self.d_qk,
            pg=tp_pg,
            mode=tp_mode,
            bias=True,
            async_communication=tp_linear_async_communication,
            contiguous_chunks=qkv_contiguous_chunks,
            tp_recompute_allgather=parallel_config.tp_recompute_allgather
        )

        self.o_proj = TensorParallelRowLinear(
            config.num_attention_heads * self.d_qk,
            self.d_model,
            pg=tp_pg,
            mode=tp_mode,
            bias=True,
            async_communication=tp_linear_async_communication,
        )

        self.attention = VisionCoreAttention(
            config,
            parallel_config=parallel_config,
        )

    def forward(
        self,
        image_hidden_states: torch.Tensor,
        sequence_mask
    ):
        from flash_attn import bert_padding
        from flash_attn.flash_attn_interface import (
            flash_attn_varlen_func,
            flash_attn_with_kvcache,
        )

        hidden_states = image_hidden_states

        qkv_states = self.qkv_proj(
            hidden_states
        )  # [batch_size, seq_length, n_local_q_heads * d_qk + 2 * n_local_kv_heads * d_qk]
        batch_size, q_length, _ = qkv_states.shape


        if self.is_gqa:
            query_states, key_states, value_states = torch.split(
                qkv_states,
                [
                    self.n_local_q_heads * self.d_qk,
                    self.n_local_kv_heads * self.d_qk,
                    self.n_local_kv_heads * self.d_qk,
                ],
                dim=-1,
            )

            query_states = (
                query_states.contiguous().view(batch_size, q_length, self.n_local_q_heads, self.d_qk)
            )
            key_states = (
                key_states.contiguous().view(batch_size, q_length, self.n_local_kv_heads, self.d_qk)
            )
            value_states = (
                value_states.contiguous().view(batch_size, q_length, self.n_local_kv_heads, self.d_qk)
            )
        else:
            query_states, key_states, value_states = (
                qkv_states.view(batch_size, q_length, 3, self.n_local_q_heads, self.d_qk)
                .permute(2, 0, 1, 3, 4)
                .contiguous()
            )  # [3, batch_size, seq_length, n_local_q_heads, d_qk]


        # Apply rotary embeddings to query/key states
        # NOTE: The layout is different from models/llama.py which is [batch_size, num_heads, seq_length, d_qk]
        # Here it is, [batch_size, seq_length, num_heads, d_qk]
        # [2, batch_size, seq_length, num_heads, d_qk]
        key_value_states = torch.cat([key_states.unsqueeze(0), value_states.unsqueeze(0)], dim=0)
        # [batch_size, seq_length, 2, num_heads, d_qk]
        key_value_states = key_value_states.permute(1, 2, 0, 3, 4).contiguous()

        # [batch_size, seq_length, num_heads, d_qk]
        key_states, value_states = torch.split(key_value_states, 1, dim=2)

        kv_length = key_states.shape[1]
        key_states = key_states.view(batch_size, kv_length, self.n_local_kv_heads, self.d_qk)
        value_states = value_states.view(batch_size, kv_length, self.n_local_kv_heads, self.d_v)

        attention_output = self.attention(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
        )   

        attention_output = (
            attention_output.contiguous().view(batch_size, q_length, self.n_local_q_heads * self.d_v)
        )

        output = self.o_proj(attention_output)

        return {"image_hidden_states": output, "sequence_mask": sequence_mask}
    

class VisionMLP(nn.Module):
    def __init__(
            self,
            config: Idefics3VisionConfig,
            parallel_config: Optional[ParallelismArgs],
            tp_pg: dist.ProcessGroup,
    ):
        super().__init__()

        # TODO @thomasw21: refactor so that we store that default in a single place.
        tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE
        tp_linear_async_communication = (
            parallel_config.tp_linear_async_communication if parallel_config is not None else False
        )

        first_contiguous_chunks = (
            config.intermediate_size,  # shape of up_linear
        )
        self.fc1 = TensorParallelColumnLinear(
            config.hidden_size,
            config.intermediate_size,
            pg=tp_pg,
            mode=tp_mode,
            bias=True,
            async_communication=tp_linear_async_communication,
            contiguous_chunks=first_contiguous_chunks,
            tp_recompute_allgather=parallel_config.tp_recompute_allgather,
        )
        self.fc2 = TensorParallelRowLinear(
            config.intermediate_size,
            config.hidden_size,
            pg=tp_pg,
            mode=tp_mode,
            bias=True,
            async_communication=tp_linear_async_communication and tp_mode is TensorParallelLinearMode.REDUCE_SCATTER,
        )
        self.act = torch.compile(lambda x: nn.functional.gelu(x, approximate="tanh"))

    def forward(self, image_hidden_states):  # [seq_length, batch_size, hidden_dim]
        merged_states = self.fc1(image_hidden_states)
        image_hidden_states = self.fc2(self.act(merged_states))
        return {"image_hidden_states": image_hidden_states}
    


class VisionEncoderLayer(nn.Module):
    def __init__(
        self,
        config: Idefics3VisionConfig,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
        layer_id: int,
    ):
        super().__init__()

        self.self_attn = VisionSelfAttention(
            config,
            parallel_config=parallel_config,
            tp_pg=tp_pg,
            layer_idx=layer_id,
        )

        self.layer_norm1 = TritonLayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        
        self.layer_norm2 = TritonLayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )


        self.mlp = VisionMLP(
            config,
            parallel_config=parallel_config,
            tp_pg=tp_pg,
        )

        self.tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE
        self.layer_id = layer_id


    def forward(
        self,
        image_hidden_states: Union[torch.Tensor, TensorPointer],
        sequence_mask: Union[torch.Tensor, TensorPointer],
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        
        hidden_states = image_hidden_states

        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        output = self.self_attn(image_hidden_states=hidden_states, sequence_mask=sequence_mask)
        hidden_states = output["image_hidden_states"]

        hidden_states = hidden_states + residual

        residual = hidden_states

        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(image_hidden_states=hidden_states)["image_hidden_states"]
        hidden_states = hidden_states + residual

        return {
            "image_hidden_states": hidden_states,
            "sequence_mask": output["sequence_mask"],
        }
    

class VisionTransformer(nn.Module):
    def __init__(
        self, 
        config: Idefics3VisionConfig,
        parallel_context: ParallelContext,
        parallel_config: Optional[ParallelismArgs],
    ):
        super().__init__()
        self.config = config
        self.tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE

        self.embeddings = VisionEmbedding(
            tp_pg=parallel_context.tp_pg,
            config=config,
            parallel_config=parallel_config,
        )

        self.encoder = nn.ModuleList(
            [
                VisionEncoderLayer(
                    config=config,
                    parallel_config=parallel_config,
                    tp_pg=parallel_context.tp_pg,
                    layer_id=i,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        self.post_layernorm = TritonLayerNorm(
            normalized_shape=config.hidden_size,
            eps=config.layer_norm_eps,
        )

    def forward(
        self,
        pixel_values: Union[torch.Tensor, TensorPointer],
        pixel_attention_mask: Union[torch.Tensor, TensorPointer] = None,
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        batch_size = pixel_values.size(0)

        batch_size, num_images, num_channels, height, width = pixel_values.size()

        pixel_values = pixel_values.view(batch_size * num_images, num_channels, height, width)

        nb_values_per_image = pixel_values.shape[1:].numel()
        real_images_inds = (pixel_values == 0.0).sum(dim=(-1, -2, -3)) != nb_values_per_image
        pixel_values = pixel_values[real_images_inds].contiguous()

        if pixel_attention_mask is None:
            pixel_attention_mask = torch.ones(
                size=(pixel_values.size(0), pixel_values.size(2), pixel_values.size(3)),
                dtype=torch.bool,
                device=pixel_values.device,
                )
        else:
            # Remove padding images from the mask/pP p
            pixel_attention_mask = pixel_attention_mask.view(
                batch_size * num_images, *pixel_attention_mask.shape[2:]
            )
            pixel_attention_mask = pixel_attention_mask[real_images_inds].contiguous()

        patch_size = self.config.patch_size
        patches_subgrid = pixel_attention_mask.unfold(dimension=1, size=patch_size, step=patch_size)
        patches_subgrid = patches_subgrid.unfold(dimension=2, size=patch_size, step=patch_size)
        patch_attention_mask = (patches_subgrid.sum(dim=(-1, -2)) == patch_size * patch_size).bool()

        pixel_values = pixel_values.bfloat16()

        if patch_attention_mask is None:
            patch_size = self.config.patch_size
            patch_attention_mask = torch.ones(
                (
                    batch_size,
                    pixel_values.size(2) // patch_size,
                    pixel_values.size(3) // patch_size,
                )
            )

            patch_attention_mask = patch_attention_mask.to(pixel_values.device, dtype=torch.bool)

        image_hidden_states = self.embeddings(
            pixel_values=pixel_values,
            patch_attention_mask=patch_attention_mask,
        )["embeddings"]

        patch_attention_mask = patch_attention_mask.view(batch_size, -1)

        hidden_encoder_states = {
            "image_hidden_states": image_hidden_states,
            "sequence_mask": patch_attention_mask,
        }

        for i, encoder_layer in enumerate(self.encoder):
            hidden_encoder_states = encoder_layer(**hidden_encoder_states)

        image_hidden_states = hidden_encoder_states["image_hidden_states"]
        image_hidden_states = self.post_layernorm(input=image_hidden_states)

        return image_hidden_states


class Idefics3MLP(nn.Module):
    def __init__(
        self,
        config: Idefics3VisionConfig,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
    ):
        super().__init__()

        # TODO @thomasw21: refactor so that we store that default in a single place.
        tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE
        tp_linear_async_communication = (
            parallel_config.tp_linear_async_communication if parallel_config is not None else False
        )

        gate_up_contiguous_chunks = (
            config.intermediate_size,  # shape of gate_linear
            config.intermediate_size,  # shape of up_linear
        )
        self.gate_up_proj = TensorParallelColumnLinear(
            config.hidden_size,
            2 * config.intermediate_size,
            pg=tp_pg,
            mode=tp_mode,
            bias=False,
            async_communication=tp_linear_async_communication,
            contiguous_chunks=gate_up_contiguous_chunks,
            tp_recompute_allgather=parallel_config.tp_recompute_allgather,
        )
        self.down_proj = TensorParallelRowLinear(
            config.intermediate_size,
            config.hidden_size,
            pg=tp_pg,
            mode=tp_mode,
            bias=False,
            async_communication=tp_linear_async_communication and tp_mode is TensorParallelLinearMode.REDUCE_SCATTER,
        )
        self.split_silu_mul = torch.compile(GLUActivation(config.hidden_act))

    def forward(self, image_hidden_states):  # [seq_length, batch_size, hidden_dim]
        merged_states = self.gate_up_proj(image_hidden_states)
        image_hidden_states = self.down_proj(self.split_silu_mul(merged_states))
        return {"image_hidden_states": image_hidden_states}
    
class Idefics3SimpleMLP(nn.Module):
    def __init__(
        self,
        config: Idefics3Config,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
    ):
        super().__init__()

        # TODO @thomasw21: refactor so that we store that default in a single place.
        tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE
        tp_linear_async_communication = (
            parallel_config.tp_linear_async_communication if parallel_config is not None else False
        )

        hidden_size = config.vision_config.hidden_size

        self.input_size = hidden_size * (config.scale_factor ** 2)
        self.output_size = config.text_config.hidden_size
        self.proj = nn.Linear(
            self.input_size,
            self.output_size,
            bias = False
        )

    def forward(self, image_hidden_states):
        image_hidden_states = self.proj(image_hidden_states)
        return {"image_hidden_states": image_hidden_states}
    

class Idefics3Connector(nn.Module):
    def __init__(
        self,
        config: Idefics3Config,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup
    ):
        super().__init__()
        self.scale_factor = config.scale_factor
        self.modality_projector = Idefics3SimpleMLP(
            config=config,
            parallel_config=parallel_config,
            tp_pg=tp_pg,
        )

    def pixel_shuffle(self, x, scale_factor=2):
        bsz, seq, embed_dim = x.size()
        height = width = int(seq**0.5)

        x = x.view(bsz, height, width, embed_dim)
        x = x.view(bsz, height, int(width / scale_factor), embed_dim * scale_factor)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz, int(width / scale_factor), int(height / scale_factor), embed_dim * (scale_factor**2))
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz, int(seq / (scale_factor**2)), embed_dim * (scale_factor**2))
        return x
    
    def forward(self, image_hidden_states):
        image_hidden_states = self.pixel_shuffle(image_hidden_states, self.scale_factor)
        image_hidden_states = self.modality_projector(image_hidden_states=image_hidden_states)["image_hidden_states"]
        return {"image_hidden_states": image_hidden_states}

class InputsMerger(nn.Module):
    def __init__(
        self, 
        config: Idefics3Config,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup            
    ):
        super().__init__()
        self.tp_pg = tp_pg
        self.tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE
        
        self.image_token_id = config.image_token_id

    def forward(
        self,
        input_ids: Union[torch.Tensor, TensorPointer],
        inputs_embeds: Union[torch.Tensor, TensorPointer],
        image_hidden_states: Union[torch.Tensor, TensorPointer],
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        
        num_images, _, vision_hidden_size = image_hidden_states.shape
        special_image_token_mask = input_ids == self.image_token_id
        new_inputs_embeds = inputs_embeds.clone()
        reshaped_image_hidden_states = image_hidden_states.view(-1, vision_hidden_size)
        new_inputs_embeds[special_image_token_mask] = reshaped_image_hidden_states

        new_inputs_embeds = new_inputs_embeds.transpose(0, 1)

        if self.tp_mode is TensorParallelLinearMode.ALL_REDUCE:
            new_inputs_embeds = differentiable_identity(new_inputs_embeds, group=self.tp_pg)
        elif self.tp_mode is TensorParallelLinearMode.REDUCE_SCATTER:
            new_inputs_embeds = differentiable_scatter(new_inputs_embeds, group=self.tp_pg)
        else:
            raise ValueError(f"Got unexpected mode: {self.tp_mode}.") 
        return {"new_inputs_embeds": new_inputs_embeds}


class CombinedEmbeddings(nn.Module):
    def __init__(
        self,
        config: Idefics3Config,
        parallel_config: Optional[ParallelismArgs],
        parallel_context: ParallelContext,
        tp_pg: dist.ProcessGroup,
    ):
        super().__init__()
        self.text_embeddings = LlamaEmbeddings(
            tp_pg=tp_pg,
            config=config.text_config,
            parallel_config=parallel_config,
        )

        self.vision_model = VisionTransformer(
            config=config.vision_config,
            parallel_context=parallel_context,
            parallel_config=parallel_config,
        )

        self.connector = Idefics3Connector(
            config=config,
            parallel_config=parallel_config,
            tp_pg=tp_pg,
        )

        self.inputs_merger = InputsMerger(
            config=config,
            parallel_config=parallel_config,
            tp_pg=tp_pg,
        )

    def forward(
        self,
        input_ids: Union[torch.Tensor, TensorPointer],
        input_mask: Union[torch.Tensor, TensorPointer],
        pixel_values: Union[torch.Tensor, TensorPointer],
        pixel_attention_mask: Union[torch.Tensor, TensorPointer] = None,
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:

        llama_output = self.text_embeddings(
            input_ids=input_ids,
            input_mask=input_mask,
        )

        vision_output = self.vision_model(
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
        )

        connector_output = self.connector(
            image_hidden_states=vision_output,
        )

        inputs_merger_output = self.inputs_merger(
            input_ids=input_ids,
            inputs_embeds=llama_output["input_embeds"],
            image_hidden_states=connector_output["image_hidden_states"],
        )

        inputs_merger_output["input_mask"] = input_mask

        return inputs_merger_output

class Idefics3Model(nn.Module):
    def __init__(
        self,
        config: Idefics3Config,
        parallel_context: ParallelContext,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
    ):
        super().__init__()

        self.config = config
        self.image_token_id = config.image_token_id
        self.tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE
        self.tp_pg = tp_pg
        tp_linear_async_communication = (
            parallel_config.tp_linear_async_communication if parallel_config is not None else False
        )
        self.p2p = P2P(parallel_context.pp_pg, device=torch.device("cuda"))
        
        self.combined_embeddings = PipelineBlock(
            p2p=self.p2p,
            module_builder=CombinedEmbeddings,
            module_kwargs={
                "config": config,
                "parallel_config": parallel_config,
                "parallel_context": parallel_context,
                "tp_pg": tp_pg,
            },
            module_input_keys={
                "input_ids",
                "input_mask",
                "pixel_values",
            },
            module_output_keys={
                "new_inputs_embeds",
                "input_mask",
            }
        )

        self.llama = LlamaModel(
            config=config.text_config,
            parallel_context=parallel_context,
            parallel_config=parallel_config,
            p2p = self.p2p
        )
        
        del self.llama.lm_head
        del self.llama.token_position_embeddings

        self.lm_head = PipelineBlock(
            p2p=self.p2p,
            # Understand that this means that we return sharded logits that are going to need to be gathered
            module_builder=TensorParallelColumnLinear,
            module_kwargs={
                "in_features": config.text_config.hidden_size,
                "out_features": config.text_config.vocab_size,
                "pg": parallel_context.tp_pg,
                "bias": False,
                # TODO @thomasw21: refactor so that we store that default in a single place.
                "mode": self.tp_mode,
                "async_communication": tp_linear_async_communication,
                "tp_recompute_allgather": parallel_config.tp_recompute_allgather,
            },
            module_input_keys={"x"},
            module_output_keys={"logits"},
        )


        self.cast_to_fp32 = PipelineBlock(
            p2p=self.llama.p2p,
            module_builder=lambda: lambda x: x.float(),
            module_kwargs={},
            module_input_keys={"x"},
            module_output_keys={"output"},
        )
        
    def forward_with_hidden_states(
        self,
        input_ids: Union[torch.Tensor, TensorPointer],
        input_mask: Union[torch.Tensor, TensorPointer],
        pixel_values: Union[torch.Tensor, TensorPointer],
        pixel_attention_mask: Union[torch.Tensor, TensorPointer] = None,
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        # Calculated combined visual and textual embeddings
        embeds = self.combined_embeddings(
            input_ids=input_ids,
            input_mask=input_mask,
            pixel_values=pixel_values,
        )

        hidden_encoder_states = {
            "hidden_states": embeds["new_inputs_embeds"],
            "sequence_mask": embeds["input_mask"],
        }
        
        for encoder_block in self.llama.decoder:
            hidden_encoder_states = encoder_block(**hidden_encoder_states)

        hidden_states = self.llama.final_layer_norm(input=hidden_encoder_states["hidden_states"])["hidden_states"]

        sharded_logits = self.lm_head(x=hidden_states)["logits"]

        fp32_sharded_logits = self.cast_to_fp32(x=sharded_logits)["output"]

        return fp32_sharded_logits, hidden_states

    def forward(
        self,
        input_ids: Union[torch.Tensor, TensorPointer],
        input_mask: Union[torch.Tensor, TensorPointer],
        pixel_values: Union[torch.Tensor, TensorPointer] = None,
        pixel_attention_mask: Union[torch.Tensor, TensorPointer] = None,
    ):
        return self.forward_with_hidden_states(
            input_ids=input_ids, input_mask=input_mask, pixel_values=pixel_values, pixel_attention_mask=pixel_attention_mask
        )[0]
    
    def get_block_compute_costs_vision(self):
        config = self.config.vision_config
        d_ff = config.intermediate_size
        d_qkv = config.hidden_size // config.num_attention_heads

        return {
            CombinedEmbeddings: 4 * config.num_attention_heads * d_qkv * config.hidden_size
            + 3 * d_ff * config.hidden_size * config.num_hidden_layers
        }
    
    def get_block_compute_costs(self):
        llama_cost = self.llama.get_block_compute_costs()
        costs = self.get_block_compute_costs_vision()

        costs[LlamaDecoderLayer] = llama_cost[LlamaDecoderLayer]
        costs[TensorParallelColumnLinear] = self.config.text_config.hidden_size * self.config.text_config.vocab_size

        return costs
    

class Idefics3ForTraining(NanotronModel):
    def __init__(
        self,
        config: Idefics3Config,
        parallel_context: ParallelContext,
        parallel_config: Optional[ParallelismArgs],
        random_states: Optional[RandomStates] = None,
    ):
        super().__init__()

        self.model = Idefics3Model(
            config=config,
            parallel_context=parallel_context,
            parallel_config=parallel_config,
            tp_pg=parallel_context.tp_pg
        )

        self.loss = PipelineBlock(
            p2p=self.model.llama.p2p,
            module_builder=Loss,
            module_kwargs={"tp_pg": parallel_context.tp_pg},
            module_input_keys={
                "sharded_logits",
                "label_ids",
                "label_mask",
            },
            module_output_keys={"loss"},
        )
        self.parallel_context = parallel_context
        self.config = config
        self.parallel_config = parallel_config

    def forward(
        self,
        input_ids: Union[torch.Tensor, TensorPointer],
        input_mask: Union[torch.Tensor, TensorPointer],
        pixel_values: Union[torch.Tensor, TensorPointer],
        label_ids: Union[torch.Tensor, TensorPointer],
        label_mask: Union[torch.Tensor, TensorPointer],
        pixel_attention_mask: Union[torch.Tensor, TensorPointer] = None,
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        outputs = self.model(
            input_ids=input_ids,
            input_mask=input_mask,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
        )

        loss = self.loss(
            sharded_logits=outputs,
            label_ids=label_ids,
            label_mask=label_mask,
        )["loss"]

        return {"loss": loss}
    

    @torch.no_grad()
    def init_model_randomly(self, config: Config):
        """Initialize model parameters randomly.
        Note:
            Layernorm weight all 0 or 1 depending on `apply_layernorm_1p`
        """
        init_method = config.model.init_method
        if isinstance(init_method, RandomInit):
            parametrizator_cls = StandardParametrizator
        elif isinstance(init_method, SpectralMupInit):
            parametrizator_cls = SpectralMupParametrizator
        else:
            raise ValueError(f"Unknown init method {init_method}")

        parametrizator = parametrizator_cls(config=config.model)

        log_rank(
            f"Parametrizing model parameters using {parametrizator.__class__.__name__}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        model = self
        initialized_parameters = set()
        # Handle tensor parallelism
        module_id_to_prefix = {id(module): f"{module_name}." for module_name, module in model.named_modules()}
        # Fix the root_model
        module_id_to_prefix[id(model)] = ""

        for param_name, param in model.named_parameters():
            assert isinstance(param, NanotronParameter)

            module_name, param_name = param_name.rsplit(".", 1)

            if param.is_tied:
                tied_info = param.get_tied_info()
                full_param_name = tied_info.get_full_name_from_module_id_to_prefix(
                    module_id_to_prefix=module_id_to_prefix
                )
            else:
                full_param_name = f"{module_name}.{param_name}"

            if full_param_name in initialized_parameters:
                # Already initialized
                continue

            module = model.get_submodule(module_name)
            parametrizator.parametrize(param_name, module)

            assert full_param_name not in initialized_parameters
            initialized_parameters.add(full_param_name)

        assert initialized_parameters == {
            param.get_tied_info().get_full_name_from_module_id_to_prefix(module_id_to_prefix=module_id_to_prefix)
            if param.is_tied
            else name
            for name, param in model.named_parameters()
        }, f"Somehow the initialized set of parameters don't match:\n - Expected: { {name for name, _ in model.named_parameters()} }\n - Got: {initialized_parameters}"

    def get_block_compute_costs(self):
        return self.model.get_block_compute_costs()

    def get_embeddings_lm_head_tied_names(self):
        """Get the names of the tied embeddings and lm_head weights"""
        if self.config.text_config.tie_word_embeddings is True:
            return ["model.llama.token_position_embeddings.pp_block.token_embedding.weight", "model.llama.lm_head.pp_block.weight"]
        else:
            return []

    def get_flops_per_sec(self, iteration_time_in_sec, sequence_length, global_batch_size):
        return self.model.llama.get_flops_per_sec(iteration_time_in_sec, sequence_length, global_batch_size)


class VisionTransformerNanotron(NanotronModel):
    def __init__(
        self,
        config: Idefics3VisionConfig,
        parallel_context: ParallelContext,
        parallel_config: Optional[ParallelismArgs],
        random_states: Optional[RandomStates] = None,
    ):
        super().__init__()

        p2p = P2P(parallel_context.pp_pg, device=torch.device("cuda"))

        self.model = VisionTransformer(
            config=config,
            p2p=p2p,
            parallel_context=parallel_context,
            parallel_config=parallel_config,
        )

        self.parallel_context = parallel_context
        self.config = config
        self.parallel_config = parallel_config

    def forward(
        self,
        pixel_values: Union[torch.Tensor, TensorPointer],
        patch_attention_mask: Union[torch.Tensor, TensorPointer],
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        return self.model(pixel_values=pixel_values, patch_attention_mask=patch_attention_mask)

    @torch.no_grad()
    def init_model_randomly(self, config: Config):
        """Initialize model parameters randomly.
        Note:
            Layernorm weight all 0 or 1 depending on `apply_layernorm_1p`
        """
        init_method = config.model.init_method
        if isinstance(init_method, RandomInit):
            parametrizator_cls = StandardParametrizator
        elif isinstance(init_method, SpectralMupInit):
            parametrizator_cls = SpectralMupParametrizator
        else:
            raise ValueError(f"Unknown init method {init_method}")

        parametrizator = parametrizator_cls(config=config.model)

        log_rank(
            f"Parametrizing model parameters using {parametrizator.__class__.__name__}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        model = self
        initialized_parameters = set()
        # Handle tensor parallelism
        module_id_to_prefix = {id(module): f"{module_name}." for module_name, module in model.named_modules()}
        # Fix the root_model
        module_id_to_prefix[id(model)] = ""

        for param_name, param in model.named_parameters():
            assert isinstance(param, NanotronParameter)

            module_name, param_name = param_name.rsplit(".", 1)

            if param.is_tied:
                tied_info = param.get_tied_info()
                full_param_name = tied_info.get_full_name_from_module_id_to_prefix(
                    module_id_to_prefix=module_id_to_prefix
                )
            else:
                full_param_name = f"{module_name}.{param_name}"

            if full_param_name in initialized_parameters:
                # Already initialized
                continue

            module = model.get_submodule(module_name)
            parametrizator.parametrize(param_name, module)

            assert full_param_name not in initialized_parameters
            initialized_parameters.add(full_param_name)

        assert initialized_parameters == {
            param.get_tied_info().get_full_name_from_module_id_to_prefix(module_id_to_prefix=module_id_to_prefix)
            if param.is_tied
            else name
            for name, param in model.named_parameters()
        }, f"Somehow the initialized set of parameters don't match:\n - Expected: { {name for name, _ in model.named_parameters()} }\n - Got: {initialized_parameters}"

    def get_block_compute_costs(self):
        return self.model.get_block_compute_costs()

    def get_embeddings_lm_head_tied_names(self):
        """Get the names of the tied embeddings and lm_head weights"""
        return []
    
@torch.jit.script
def masked_mean(loss, label_mask, dtype):
    # type: (Tensor, Tensor, torch.dtype) -> Tensor
    return (loss * label_mask).sum(dtype=dtype) / label_mask.sum()

class Loss(nn.Module):
    def __init__(self, tp_pg: dist.ProcessGroup):
        super().__init__()
        self.tp_pg = tp_pg

    def forward(
        self,
        sharded_logits: torch.Tensor,  # [seq_length, batch_size, logits]
        label_ids: torch.Tensor,  # [batch_size, seq_length]
        label_mask: torch.Tensor,  # [batch_size, seq_length]
    ) -> Dict[str, torch.Tensor]:
        # Megatron by defaults cast everything in fp32. `--f16-lm-cross-entropy` is an option you can use to keep current precision.
        # https://github.com/NVIDIA/Megatron-LM/blob/f267e6186eae1d6e2055b412b00e2e545a8e896a/megatron/model/gpt_model.py#L38
        
        loss = sharded_cross_entropy(
            sharded_logits, label_ids.transpose(0, 1).contiguous(), group=self.tp_pg, dtype=torch.float
        ).transpose(0, 1)
        # TODO @thomasw21: It's unclear what kind of normalization we want to do.
        loss = masked_mean(loss, label_mask, dtype=torch.float)
        # I think indexing causes a sync we don't actually want
        # loss = loss[label_mask].sum()
        return {"loss": loss}
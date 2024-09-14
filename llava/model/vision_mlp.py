import torch
import torch.utils.checkpoint
from torch import nn
import math
import numpy as np
import torch.nn.functional as F

from transformers.models.llama.modeling_llama import LlamaSdpaAttention, LlamaDecoderLayer, LlamaRMSNorm, rotate_half, repeat_kv
class VisionMLP(nn.Module):
	def __init__(self, config):
		super().__init__()
		intermediate_size = config.hidden_size // 4
		self.context_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
		self.input_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
		self.proj = nn.Sequential(
			nn.Linear(intermediate_size*2, intermediate_size, bias=False),
			nn.SiLU(),
			nn.Linear(intermediate_size, config.hidden_size, bias=False)
		)
		self.layernorm_post = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

	def forward(self, image_full, image_concise, side_len_full, side_len_concise, attention_mask=None):
		bs = image_full.shape[0]
		reduce_factor = side_len_full//side_len_concise

		image_full = image_full.view(bs, side_len_concise, side_len_concise, -1)
		image_concise = image_concise.view(bs, side_len_concise, side_len_concise, -1)
		image_concise = image_concise.repeat_interleave(reduce_factor, 1).repeat_interleave(reduce_factor, 2)
		image_concise = self.context_proj(image_concise)
		residual = image_full
		image_full = self.input_proj(image_full)
		image_full = torch.cat([image_full, image_concise], -1)
		image_full = self.layernorm_post(self.proj(image_full) + residual) 

		image_full = image_full.flatten(1,2)

		return image_full
	
def apply_rotary_pos_emb(q, k, cos, sin, position_ids_q, position_ids_k, unsqueeze_dim=1):
	cos_q = cos[0][position_ids_q].unsqueeze(unsqueeze_dim)
	sin_q = sin[0][position_ids_q].unsqueeze(unsqueeze_dim)
	q_embed = (q * cos_q) + (rotate_half(q) * sin_q)
	cos_k = cos[0][position_ids_k].unsqueeze(unsqueeze_dim)
	sin_k = sin[0][position_ids_k].unsqueeze(unsqueeze_dim)
	k_embed = (k * cos_k) + (rotate_half(k) * sin_k)
	return q_embed, k_embed
	
# Adapted from LlamaAttention.forward
def LlamaSdpaAttention_forward(
	self,
	hidden_states,
	kv_states,
	attention_mask = None,
	position_ids_q = None,
	position_ids_kv = None,
	past_key_value = None,
	output_attentions = False,
	use_cache = False,
	cache_position = None,
	position_embeddings = None
):

	bsz, q_len, _ = hidden_states.size()
	kv_seq_len = kv_states.shape[1]

	query_states = self.q_proj(hidden_states)
	key_states = self.k_proj(kv_states)
	value_states = self.v_proj(kv_states)

	query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
	key_states = key_states.view(bsz, kv_seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
	value_states = value_states.view(bsz, kv_seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

	if position_embeddings is None:
		cos, sin = self.rotary_emb(value_states, torch.arange(position_ids_kv.shape[1], device=position_ids_kv.device, dtype=position_ids_kv.dtype).unsqueeze(0).expand(position_ids_kv.shape[0], -1))
	else:
		cos, sin = position_embeddings

	# In case static cache is used, it is an instance attribute.
	past_key_value = getattr(self, "past_key_value", past_key_value)

	if past_key_value is not None:
		# sin and cos are specific to RoPE models; cache_position needed for the static cache
		cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
		key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

	query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids_q, position_ids_kv)

	key_states = repeat_kv(key_states, self.num_key_value_groups)
	value_states = repeat_kv(value_states, self.num_key_value_groups)

	if attention_mask is not None:
		if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
			raise ValueError(
				f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
			)

	# SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
	# Reference: https://github.com/pytorch/pytorch/issues/112577.
	if query_states.device.type == "cuda" and attention_mask is not None:
		query_states = query_states.contiguous()
		key_states = key_states.contiguous()
		value_states = value_states.contiguous()

	attn_output = torch.nn.functional.scaled_dot_product_attention(
		query_states,
		key_states,
		value_states,
		attn_mask=attention_mask,
		dropout_p=self.attention_dropout if self.training else 0.0,
		# The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
		is_causal=self.is_causal and attention_mask is None and q_len > 1,
	)

	attn_output = attn_output.transpose(1, 2).contiguous()
	attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

	attn_output = self.o_proj(attn_output)

	return attn_output, None, past_key_value

LlamaSdpaAttention.forward = LlamaSdpaAttention_forward

def decoder_forward(
	self,
	hidden_states,
	kv_states=None,
	attention_mask = None,
	position_ids_q = None,
	position_ids_kv = None,
	past_key_value = None,
	output_attentions = False,
	use_cache = False,
	cache_position = None,
	position_embeddings = None,
	**kwargs,):
		residual = hidden_states

		hidden_states = self.input_layernorm(hidden_states)
		if kv_states is None:
			kv_states = hidden_states
			position_ids_kv = position_ids_q
		else:
			kv_states = self.input_layernorm(kv_states)

		# Cross Attention
		hidden_states, self_attn_weights, present_key_value = self.self_attn(
			hidden_states=hidden_states,
			kv_states = kv_states,
			attention_mask=attention_mask,
			position_ids_q=position_ids_q,
			position_ids_kv=position_ids_kv,
			past_key_value=past_key_value,
			output_attentions=output_attentions,
			use_cache=use_cache,
			cache_position=cache_position,
			position_embeddings=position_embeddings,
			**kwargs,
		)
		hidden_states = residual + hidden_states

		# Fully Connected
		residual = hidden_states
		hidden_states = self.post_attention_layernorm(hidden_states)
		hidden_states = self.mlp(hidden_states)
		hidden_states = residual + hidden_states

		outputs = (hidden_states,)

		if output_attentions:
			outputs += (self_attn_weights,)

		if use_cache:
			outputs += (present_key_value,)

		return outputs
LlamaDecoderLayer.forward = decoder_forward
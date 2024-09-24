#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import numpy as np

from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging
logger = logging.get_logger(__name__)

from torch.nn import CrossEntropyLoss


# , LlamaModel, LlamaForCausalLM, GenerationConfig
# from .modeling_llama import LlamaModel, LlamaForCausalLM
from transformers import LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

import os

class LlavaConfig(LlamaConfig):
	model_type = "llava_llama"
	temperature: float = 0.0  # reset to 0.0, previously 0.9 for Vicuna
	max_new_tokens: int = 1024
	do_sample: bool = False
	top_p: Optional[float] = None
	# rope_scaling: Optional[dict] = {}

def get_image_compress(hidden_states_image_full, compress_reduce_factor, single_crop_len=576):
	bs = hidden_states_image_full.shape[0]
	num_image_crops = hidden_states_image_full.shape[1]//single_crop_len
	h_full = w_full = int(single_crop_len**0.5)
	h_compress = w_compress = h_full//compress_reduce_factor

	hidden_states_image_full = hidden_states_image_full.view(bs*num_image_crops, h_full, w_full, -1)
	
	hidden_states_image_compress = nn.functional.interpolate(
	hidden_states_image_full.permute(0, 3, 1, 2).contiguous(),
		size=(h_compress, w_compress),
		mode='bilinear',
		align_corners=False
	)
	hidden_states_image_compress = hidden_states_image_compress.permute(0, 2, 3, 1).contiguous().view(bs, num_image_crops*h_compress*w_compress, -1)
	return hidden_states_image_compress

class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
	config_class = LlavaConfig

	def __init__(self, config: LlamaConfig):
		super(LlavaLlamaModel, self).__init__(config)
		
	def forward(
		self,
		input_ids: torch.LongTensor = None,
		attention_mask: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		past_key_values: Optional[List[torch.FloatTensor]] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
		cache_position: Optional[torch.LongTensor] = None,
		attention_mask_regular_4d: Optional[torch.Tensor] = None,
		attention_mask_compress_4d: Optional[torch.Tensor] = None,
		position_ids_image_compress: Optional[torch.LongTensor] = None,
		num_image_crops: Optional[int] = None,
	) -> Union[Tuple, BaseModelOutputWithPast]:
		output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
		output_hidden_states = (
			output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
		)
		use_cache = use_cache if use_cache is not None else self.config.use_cache

		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		if (input_ids is None) ^ (inputs_embeds is not None):
			raise ValueError(
				"You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
			)

		if self.gradient_checkpointing and self.training and use_cache:
			logger.warning_once(
				"`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
			)
			use_cache = False

		if inputs_embeds is None:
			inputs_embeds = self.embed_tokens(input_ids)

		past_seen_tokens = 0
		if use_cache:  # kept for BC (cache positions)
			if not isinstance(past_key_values, StaticCache):
				past_key_values = DynamicCache.from_legacy_cache(past_key_values)
				past_seen_tokens = past_key_values.get_seq_length()
		
		if cache_position is None:
			if isinstance(past_key_values, StaticCache):
				raise ValueError("cache_position is a required argument when using StaticCache.")
			cache_position = torch.arange(
				past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
			)
		first_forward = (past_seen_tokens==0)

		if past_seen_tokens > 0:
			position_ids = None
		if position_ids is None:
			position_ids = cache_position.unsqueeze(0)

		if past_seen_tokens == 0:
			mask_start_layer = os.getenv('MASK_START_LAYER')
			if mask_start_layer is not None:
				mask_start_layer = int(mask_start_layer)
				attention_mask_regular_4d_mask = attention_mask_regular_4d.clone()
				per_crop_token_len = self.config.per_crop_token_len
				image_full_len = num_image_crops * per_crop_token_len
				tensor = torch.full((image_full_len, image_full_len), torch.finfo(inputs_embeds.dtype).min, dtype = attention_mask_regular_4d.dtype, device=attention_mask_regular_4d.device)
				tensor.fill_diagonal_(0)
				tensor = tensor.view(1, 1, image_full_len, image_full_len).repeat(attention_mask_regular_4d_mask.shape[0], 1, 1, 1)
				attention_mask_regular_4d_mask[:, :, :image_full_len, :image_full_len] = tensor
			causal_mask = attention_mask_regular_4d
		else:
			causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position)
		hidden_states = inputs_embeds

		# decoder layers
		all_hidden_states = () if output_hidden_states else None
		all_self_attns = () if output_attentions else None
		next_decoder_cache = None

		per_crop_token_len = self.config.per_crop_token_len
		compress_reduce_factor = self.config.compress_reduce_factor
		compress_v = self.config.compress_v
		compress_v_start_layer = self.config.compress_v_start_layer

		hidden_states = inputs_embeds

		if first_forward:
			image_full_len = num_image_crops * per_crop_token_len
			newline_len = num_image_crops
			image_compress_len = num_image_crops * per_crop_token_len // compress_reduce_factor**2
			text_len = inputs_embeds.shape[1] - image_full_len - newline_len
			hidden_states_image_full = hidden_states[:, :image_full_len]
			hidden_states_newline_full = hidden_states[:, image_full_len:image_full_len+newline_len]
			hidden_states_text = hidden_states[:, image_full_len+newline_len:]

		for layer_i, decoder_layer in enumerate(self.layers):
			if output_hidden_states:
				all_hidden_states += (hidden_states,)

			if not compress_v or layer_i < compress_v_start_layer or not first_forward:
				if first_forward and mask_start_layer is not None:
					if layer_i >= mask_start_layer:
						causal_mask = attention_mask_regular_4d_mask
					else:
						causal_mask = attention_mask_regular_4d
				if self.gradient_checkpointing and self.training:
					layer_outputs = self._gradient_checkpointing_func(
						decoder_layer.__call__,
						hidden_states,
						causal_mask,
						position_ids,
						position_ids,
						past_key_values,
						output_attentions,
						use_cache,
						cache_position,
						None,
						False,

					)
				else:
					layer_outputs = decoder_layer(
						hidden_states,
						causal_mask,
						position_ids,
						position_ids,
						past_key_value=past_key_values,
						output_attentions=output_attentions,
						use_cache=use_cache,
						cache_position = cache_position,
						position_embeddings = None,
						compress_v=False,

					)

				hidden_states = layer_outputs[0]

			else:
				if layer_i == compress_v_start_layer:
					hidden_states_image_full = hidden_states[:, :image_full_len]
					hidden_states_newline_full = hidden_states[:, image_full_len:image_full_len+newline_len]
					hidden_states_text = hidden_states[:, image_full_len+newline_len:]

					position_ids_image_full = position_ids[:, :image_full_len]
					position_ids_newline_full = position_ids[:, image_full_len:image_full_len+newline_len]
					position_ids_text = position_ids[:, image_full_len+newline_len:]

					hidden_states_image_compress = get_image_compress(hidden_states_image_full, compress_reduce_factor, per_crop_token_len)

					position_ids_compress_q = torch.cat([position_ids_image_compress, position_ids_newline_full, position_ids_text], 1)
					position_ids_compress_kv = torch.cat([position_ids_image_compress, position_ids_image_full,  position_ids_newline_full, position_ids_text], 1)


				if self.gradient_checkpointing and self.training:
					layer_outputs = self._gradient_checkpointing_func(
						decoder_layer.__call__,
						torch.cat([hidden_states_image_compress, hidden_states_image_full, hidden_states_newline_full, hidden_states_text], 1),
						attention_mask_compress_4d,
						position_ids_compress_q,
						position_ids_compress_kv,
						past_key_values,
						output_attentions,
						use_cache,
						cache_position,
						None,
						True,
						image_compress_len,
						image_full_len
					)
				else:
					layer_outputs = decoder_layer(
						torch.cat([hidden_states_image_compress, hidden_states_image_full, hidden_states_newline_full, hidden_states_text], 1),
						attention_mask_compress_4d,
						position_ids_compress_q,
						position_ids_compress_kv,
						past_key_value=past_key_values,
						output_attentions=output_attentions,
						use_cache=use_cache,
						cache_position = cache_position,
						position_embeddings = None,
						compress_v=True,
						image_compress_len=image_compress_len,
						image_full_len=image_full_len,
					)

				hidden_states_image_compress = layer_outputs[0][:, :image_compress_len]
				hidden_states_newline = layer_outputs[0][:, image_compress_len:image_compress_len+newline_len]
				hidden_states_text = layer_outputs[0][:, image_compress_len+newline_len:]

				hidden_states_image_full = self.vision_mlp_layers[layer_i-compress_v_start_layer](hidden_states_image_full, hidden_states_image_compress, compress_reduce_factor, per_crop_token_len)

				if layer_i == len(self.layers) - 1:
					hidden_states = torch.cat([hidden_states_image_full, hidden_states_newline, hidden_states_text], 1)

			if use_cache:
				next_decoder_cache = layer_outputs[2 if output_attentions else 1]

			if output_attentions:
				all_self_attns += (layer_outputs[1],)

		hidden_states = self.norm(hidden_states)

		# add hidden states from the last decoder layer
		if output_hidden_states:
			all_hidden_states += (hidden_states,)
		
		if first_forward and compress_v and use_cache:
			for layer_i in range(len(next_decoder_cache.key_cache)):
				if layer_i >= compress_v_start_layer:
					key_cache = next_decoder_cache.key_cache[layer_i]
					key_cache = key_cache[:, :, image_compress_len:]
					next_decoder_cache.key_cache[layer_i] = key_cache

					value_cache = next_decoder_cache.value_cache[layer_i]
					value_cache = value_cache[:, :, image_compress_len:]
					next_decoder_cache.value_cache[layer_i] = value_cache

					if layer_i == 0:
						next_decoder_cache.seen_tokens -= image_compress_len

		next_cache = None
		if use_cache:
			next_cache = (
				next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
			)

		if not return_dict:
			return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
		return BaseModelOutputWithPast(
			last_hidden_state=hidden_states,
			past_key_values=next_cache,
			hidden_states=all_hidden_states,
			attentions=all_self_attns,
		)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
	config_class = LlavaConfig

	def __init__(self, config):
		LlamaForCausalLM.__init__(self, config)

		# configure default generation settings
		config.model_type = "llava_llama"
		# config.rope_scaling = None

		self.model = LlavaLlamaModel(config)
		self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
		# Initialize weights and apply final processing
		self.post_init()

	def get_model(self):
		return self.model

	def forward(
		self,
		input_ids: torch.LongTensor = None,
		attention_mask: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		past_key_values: Optional[List[torch.FloatTensor]] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		labels: Optional[torch.LongTensor] = None,
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		images: Optional[torch.FloatTensor] = None,
		image_sizes: Optional[List[List[int]]] = None,
		return_dict: Optional[bool] = None,
		modalities: Optional[List[str]] = ["image"],
		dpo_forward: Optional[bool] = None,
		cache_position=None,
	) -> Union[Tuple, CausalLMOutputWithPast]:

		prepare_inputs_labels = inputs_embeds is None
		if inputs_embeds is None:
			(input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, attention_mask_regular_4d, attention_mask_compress_4d, position_ids_image_compress, num_image_crops) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes)

		# decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
		outputs = self.model(
			input_ids=input_ids,
			attention_mask=attention_mask,
			position_ids=position_ids,
			past_key_values=past_key_values,
			inputs_embeds=inputs_embeds,
			use_cache=use_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
			cache_position=cache_position,
			attention_mask_regular_4d=attention_mask_regular_4d if prepare_inputs_labels else self.attention_mask_regular_4d,
			attention_mask_compress_4d=attention_mask_compress_4d if prepare_inputs_labels else self.attention_mask_compress_4d,
			position_ids_image_compress=position_ids_image_compress if prepare_inputs_labels else self.position_ids_image_compress,
			num_image_crops=num_image_crops if prepare_inputs_labels else self.num_image_crops,
		)

		hidden_states = outputs[0]
		if self.config.pretraining_tp > 1:
			lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
			logits = [nn.functional.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
			logits = torch.cat(logits, dim=-1)
		else:
			logits = self.lm_head(hidden_states)
		logits = logits.float()

		loss = None
		if labels is not None:
			# Shift so that tokens < n predict n
			# shift_logits = logits[..., :-1, :].contiguous()
			# shift_labels = labels[..., 1:].contiguous()
			shift_logits = logits
			shift_labels = labels
			# Flatten the tokens
			loss_fct = CrossEntropyLoss()
			shift_logits = shift_logits.view(-1, self.config.vocab_size)
			shift_labels = shift_labels.view(-1)
			# Enable model parallelism
			shift_labels = shift_labels.to(shift_logits.device)
			loss = loss_fct(shift_logits, shift_labels)

		if not return_dict:
			output = (logits,) + outputs[1:]
			return (loss,) + output if loss is not None else output

		return CausalLMOutputWithPast(
			loss=loss,
			logits=logits,
			past_key_values=outputs.past_key_values,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)

	@torch.no_grad()
	def generate(
		self,
		inputs: Optional[torch.Tensor] = None,
		images: Optional[torch.Tensor] = None,
		image_sizes: Optional[torch.Tensor] = None,
		modalities: Optional[List[str]] = ["image"],
		**kwargs,
	) -> Union[GenerateOutput, torch.LongTensor]:
		position_ids = kwargs.pop("position_ids", None)
		attention_mask = kwargs.pop("attention_mask", None)
		if "inputs_embeds" in kwargs:
			raise NotImplementedError("`inputs_embeds` is not supported")

		if images is not None:
			(inputs, position_ids, attention_mask, _, inputs_embeds, _, attention_mask_regular_4d, attention_mask_compress_4d, position_ids_image_compress, num_image_crops) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
			self.attention_mask_regular_4d = attention_mask_regular_4d
			self.attention_mask_compress_4d = attention_mask_compress_4d
			self.position_ids_image_compress = position_ids_image_compress
			self.num_image_crops = num_image_crops
		else:
			inputs_embeds = self.get_model().embed_tokens(inputs)

		return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

	def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
		images = kwargs.pop("images", None)
		image_sizes = kwargs.pop("image_sizes", None)
		inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
		if images is not None:
			inputs["images"] = images
		if image_sizes is not None:
			inputs["image_sizes"] = image_sizes
		return inputs

from transformers.models.llama.modeling_llama import LlamaSdpaAttention, LlamaDecoderLayer, LlamaRMSNorm, rotate_half, repeat_kv

def decoder_forward(
	self,
	hidden_states,
	attention_mask = None,
	position_ids_q = None,
	position_ids_kv = None,
	past_key_value = None,
	output_attentions = False,
	use_cache = False,
	cache_position = None,
	position_embeddings = None,
	compress_v=False,
	image_compress_len=36,
	image_full_len=576,
	**kwargs,):
		if compress_v:
			residual = torch.cat([hidden_states[:, :image_compress_len], hidden_states[:, image_compress_len+image_full_len:]], 1)
			hidden_states = self.input_layernorm(hidden_states)
			kv_states = hidden_states
			hidden_states = torch.cat([hidden_states[:, :image_compress_len], hidden_states[:, image_compress_len+image_full_len:]], 1)
		else:
			residual = hidden_states
			hidden_states = self.input_layernorm(hidden_states)
			kv_states = hidden_states
			position_ids_kv = position_ids_q

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

def apply_rotary_pos_emb(q, k, cos_q, sin_q, cos_k, sin_k, position_ids_q, position_ids_k, unsqueeze_dim=1):
	cos_q = cos_q.unsqueeze(unsqueeze_dim)
	sin_q = sin_q.unsqueeze(unsqueeze_dim)
	q_embed = (q * cos_q) + (rotate_half(q) * sin_q)
	cos_k = cos_k.unsqueeze(unsqueeze_dim)
	sin_k = sin_k.unsqueeze(unsqueeze_dim)
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
		cos_q, sin_q = self.rotary_emb(value_states, position_ids_q)
		cos_k, sin_k = self.rotary_emb(value_states, position_ids_kv)
	else:
		cos, sin = position_embeddings
		cos_q, sin_q = cos, sin
		cos_k, sin_k = cos, sin

	# In case static cache is used, it is an instance attribute.
	past_key_value = getattr(self, "past_key_value", past_key_value)

	query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos_q, sin_q, cos_k, sin_k, position_ids_q, position_ids_kv)

	if past_key_value is not None:
		# sin and cos are specific to RoPE models; cache_position needed for the static cache
		# cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
		cache_kwargs = {}
		key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

	key_states = repeat_kv(key_states, self.num_key_value_groups)
	value_states = repeat_kv(value_states, self.num_key_value_groups)

	causal_mask = attention_mask
	# if attention_mask is not None and cache_position is not None:
	if attention_mask is not None:
		causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

	# SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
	# Reference: https://github.com/pytorch/pytorch/issues/112577.
	if query_states.device.type == "cuda" and causal_mask is not None:
		query_states = query_states.contiguous()
		key_states = key_states.contiguous()
		value_states = value_states.contiguous()

	attn_output = torch.nn.functional.scaled_dot_product_attention(
		query_states,
		key_states,
		value_states,
		attn_mask=causal_mask,
		dropout_p=self.attention_dropout if self.training else 0.0,
	)

	attn_output = attn_output.transpose(1, 2).contiguous()
	attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

	attn_output = self.o_proj(attn_output)

	return attn_output, None, past_key_value

LlamaSdpaAttention.forward = LlamaSdpaAttention_forward



AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)


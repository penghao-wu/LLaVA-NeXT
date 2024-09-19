#    Copyright 2024 Hao Zhang
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


from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from transformers.cache_utils import Cache, DynamicCache, StaticCache

from transformers.utils import logging
logger = logging.get_logger(__name__)

from torch.nn import CrossEntropyLoss

# from ...constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM

# from .qwen.modeling_qwen import QWenLMHeadModel, QWenModel
# from .qwen.configuration_qwen import QWenConfig

def get_image_concise(image_full_hidden_states, concise_reduce_factor, single_crop_len=576):
	bs = image_full_hidden_states.shape[0]
	num_image_crops = image_full_hidden_states.shape[1]//single_crop_len
	h_full = w_full = int(single_crop_len**0.5)
	h_concise = w_concise = h_full//concise_reduce_factor

	image_full_hidden_states = image_full_hidden_states.view(bs*num_image_crops, h_full, w_full, -1)
	
	image_concise_hidden_states = nn.functional.interpolate(
	image_full_hidden_states.permute(0, 3, 1, 2).contiguous(),
		size=(h_concise, w_concise),
		mode='bilinear',
		align_corners=False
	)
	image_concise_hidden_states = image_concise_hidden_states.permute(0, 2, 3, 1).contiguous().view(bs, num_image_crops*h_concise*w_concise, -1)
	return image_concise_hidden_states

class LlavaQwenConfig(Qwen2Config):
	model_type = "llava_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
	config_class = LlavaQwenConfig

	def __init__(self, config: Qwen2Config):
		super(LlavaQwenModel, self).__init__(config)


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
		length_info: Optional[dict] = None,
		position_ids_fast_q: Optional[torch.LongTensor] = None,
		position_ids_fast_kv: Optional[torch.LongTensor] = None,
		attention_masks_fast_4d: Optional[torch.Tensor] = None,
	) -> Union[Tuple, BaseModelOutputWithPast]:
		output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
		output_hidden_states = (
			output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
		)
		use_cache = use_cache if use_cache is not None else self.config.use_cache

		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		# retrieve input_ids and inputs_embeds
		if input_ids is not None and inputs_embeds is not None:
			raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
		elif input_ids is not None:
			batch_size, seq_length = input_ids.shape
		elif inputs_embeds is not None:
			batch_size, seq_length, _ = inputs_embeds.shape
		else:
			raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

		if self.gradient_checkpointing and self.training:
			if use_cache:
				logger.warning_once(
					"`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
				)
				use_cache = False

		past_key_values_length = 0

		if use_cache:
			use_legacy_cache = not isinstance(past_key_values, Cache)
			if use_legacy_cache:
				past_key_values = DynamicCache.from_legacy_cache(past_key_values)
			past_key_values_length = past_key_values.get_usable_length(seq_length)

		if position_ids is None:
			device = input_ids.device if input_ids is not None else inputs_embeds.device
			position_ids = torch.arange(
				past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
			)
			position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
		else:
			position_ids = position_ids.view(-1, seq_length).long()

		if inputs_embeds is None:
			inputs_embeds = self.embed_tokens(input_ids)

		# if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
		#     is_padding_right = attention_mask[:, -1].sum().item() != batch_size
		#     if is_padding_right:
		#         raise ValueError(
		#             "You are attempting to perform batched generation with padding_side='right'"
		#             " this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to "
		#             " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
		#         )

		# if self._attn_implementation == "flash_attention_2":
		#     # 2d mask is passed through the layers
		#     attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
		# elif self._attn_implementation == "sdpa" and not output_attentions:
		#     # output_attentions=True can not be supported when using SDPA, and we fall back on
		#     # the manual implementation that requires a 4D causal mask in all cases.
		#     attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
		#         attention_mask,
		#         (batch_size, seq_length),
		#         inputs_embeds,
		#         past_key_values_length,
		#     )
		# else:
		#     # 4d mask is passed through the layers
		#     attention_mask = _prepare_4d_causal_attention_mask(
		#         attention_mask,
		#         (batch_size, seq_length),
		#         inputs_embeds,
		#         past_key_values_length,
		#         sliding_window=self.config.sliding_window,
		#     )

		hidden_states = inputs_embeds

		# decoder layers
		all_hidden_states = () if output_hidden_states else None
		all_self_attns = () if output_attentions else None
		next_decoder_cache = None

		fast_vision_start_layer = self.config.fast_vision_start_layer

		for layer_i, decoder_layer in enumerate(self.layers):
			if output_hidden_states:
				all_hidden_states += (hidden_states,)

			if not self.config.fast_vision or layer_i < fast_vision_start_layer:
				if self.gradient_checkpointing and self.training:
					layer_outputs = self._gradient_checkpointing_func(
						decoder_layer.__call__,
						hidden_states,
						None,
						attention_mask,
						position_ids,
						None,
						past_key_values,
						output_attentions,
						use_cache,
					)
				else:
					layer_outputs = decoder_layer(
						hidden_states,
						None,
						attention_mask,
						position_ids,
						None,
						past_key_value=past_key_values,
						output_attentions=output_attentions,
						use_cache=use_cache,
					)

				hidden_states = layer_outputs[0]

			else:
				if layer_i == fast_vision_start_layer:
					image_full_len = length_info['image_full_len']
					image_concise_len = length_info['image_concise_len']
					text_len = length_info['text_len']
					newline_len = length_info['newline_len']

					image_full_hidden_states = []
					text_hidden_states = []
					newline_hidden_states = []

					image_full_hidden_states = hidden_states[:, :image_full_len[0]].contiguous()
					newline_hidden_states = hidden_states[:, image_full_len[0]:image_full_len[0]+newline_len[0]]
					text_hidden_states = hidden_states[:, image_full_len[0]+newline_len[0]:]
						
					concise_reduce_factor = self.config.concise_reduce_factor
					image_concise_hidden_states = get_image_concise(image_full_hidden_states, concise_reduce_factor, self.config.image_token_len_per_side**2)
					
					hidden_states_fast_q = []
					hidden_states_fast_kv = []
					# q [image_concise, newline, text]
					# key&value [image_full, image_concise, newline, text]

					hidden_states_fast_q = torch.cat([image_concise_hidden_states, newline_hidden_states, text_hidden_states], 1)
					hidden_states_fast_kv = torch.cat([image_full_hidden_states, image_concise_hidden_states, newline_hidden_states, text_hidden_states], 1)


				if self.gradient_checkpointing and self.training:
					layer_outputs = self._gradient_checkpointing_func(
						decoder_layer.__call__,
						hidden_states_fast_q,
						hidden_states_fast_kv,
						attention_masks_fast_4d,
						position_ids_fast_q,
						position_ids_fast_kv,
						past_key_values,
						output_attentions,
						use_cache,
					)
				else:
					layer_outputs = decoder_layer(
						hidden_states_fast_q,
						hidden_states_fast_kv,
						attention_masks_fast_4d,
						position_ids_fast_q,
						position_ids_fast_kv,
						past_key_values,
						output_attentions,
						use_cache,
					)

				hidden_states_fast_q = layer_outputs[0]

				h = w = self.config.image_token_len_per_side
				h_concise = h//concise_reduce_factor
				w_concise = w//concise_reduce_factor

				image_concise_hidden_states = hidden_states_fast_q[:, :image_concise_len[0]].contiguous()
				newline_hidden_states = hidden_states_fast_q[:, image_concise_len[0]:image_concise_len[0]+newline_len[0]]
				text_hidden_states = hidden_states_fast_q[:, image_concise_len[0]+newline_len[0]:]

				image_full_hidden_states = self.vision_mlp_layers[layer_i-fast_vision_start_layer](image_full_hidden_states, image_concise_hidden_states, h, h_concise)

				hidden_states_fast_q = torch.cat([image_concise_hidden_states, newline_hidden_states, text_hidden_states], 1)
				hidden_states_fast_kv = torch.cat([image_full_hidden_states, image_concise_hidden_states, newline_hidden_states, text_hidden_states], 1)

				hidden_states = hidden_states_fast_q

			if use_cache:
				next_decoder_cache = layer_outputs[2 if output_attentions else 1]

			if output_attentions:
				all_self_attns += (layer_outputs[1],)

		hidden_states = self.norm(hidden_states)

		# add hidden states from the last decoder layer
		if output_hidden_states:
			all_hidden_states += (hidden_states,)

		next_cache = None
		if use_cache:
			next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

		if not return_dict:
			return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
		return BaseModelOutputWithPast(
			last_hidden_state=hidden_states,
			past_key_values=next_cache,
			hidden_states=all_hidden_states,
			attentions=all_self_attns,
		)


class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
	config_class = LlavaQwenConfig

	def __init__(self, config):
		# super(Qwen2ForCausalLM, self).__init__(config)
		Qwen2ForCausalLM.__init__(self, config)
		config.model_type = "llava_qwen"
		config.rope_scaling = None

		self.model = LlavaQwenModel(config)
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
		dpo_forward: Optional[bool] = False,
		cache_position=None,
	) -> Union[Tuple, CausalLMOutputWithPast]:

		if inputs_embeds is None:
			(input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, labels_fast, length_info, position_ids_fast_q, position_ids_fast_kv, attention_masks_fast_4d) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes)
			if self.get_model().config.fast_vision:
				labels = labels_fast

		output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
		output_hidden_states = (
			output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
		)
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
			length_info=length_info,
			position_ids_fast_q=position_ids_fast_q,
			position_ids_fast_kv=position_ids_fast_kv,
			attention_masks_fast_4d=attention_masks_fast_4d
		)

		hidden_states = outputs[0]
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
			(inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
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






from transformers.models.qwen2.modeling_qwen2 import Qwen2SdpaAttention, Qwen2DecoderLayer, Qwen2RMSNorm, rotate_half, repeat_kv

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


def apply_rotary_pos_emb(q, k, cos, sin, position_ids_q, position_ids_k, unsqueeze_dim=1):
	cos_q = cos[position_ids_q].unsqueeze(unsqueeze_dim)
	sin_q = sin[position_ids_q].unsqueeze(unsqueeze_dim)
	q_embed = (q * cos_q) + (rotate_half(q) * sin_q)
	cos_k = cos[position_ids_k].unsqueeze(unsqueeze_dim)
	sin_k = sin[position_ids_k].unsqueeze(unsqueeze_dim)
	k_embed = (k * cos_k) + (rotate_half(k) * sin_k)
	return q_embed, k_embed

Qwen2DecoderLayer.forward = decoder_forward

# Copied from transformers.models.mistral.modeling_mistral.MistralSdpaAttention with Mistral->Qwen2
def Qwen2SdpaAttention_forward(
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

	cos, sin = self.rotary_emb(value_states, seq_len=max(q_len, kv_seq_len))
	query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids_q, position_ids_kv)

	# In case static cache is used, it is an instance attribute.
	past_key_value = getattr(self, "past_key_value", past_key_value)

	if past_key_value is not None:
		cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
		key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

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

Qwen2SdpaAttention.forward = Qwen2SdpaAttention_forward

AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)

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


class LlavaConfig(LlamaConfig):
	model_type = "llava_llama"
	temperature: float = 0.0  # reset to 0.0, previously 0.9 for Vicuna
	max_new_tokens: int = 1024
	do_sample: bool = False
	top_p: Optional[float] = None
	# rope_scaling: Optional[dict] = {}


def get_image_concise(image_full_hidden_states, concise_reduce_factor, single_crop_len=576):
	split_sizes = []
	all_image_full_crops = []
	for cur_image_full_hidden_states in image_full_hidden_states:
		cur_image_num = cur_image_full_hidden_states.shape[0]//single_crop_len
		split_sizes.append(cur_image_num)
		all_image_full_crops.append(cur_image_full_hidden_states.view(cur_image_num, single_crop_len, -1))
	all_image_full_crops = torch.cat(all_image_full_crops, 0)

	h_full = w_full = int(single_crop_len**0.5)
	h_concise = w_concise = h_full//concise_reduce_factor

	all_image_full_crops = all_image_full_crops.view(sum(split_sizes), h_full, w_full, -1)
	
	all_image_concise_crops = nn.functional.interpolate(
	all_image_full_crops.permute(0, 3, 1, 2).contiguous(),
		size=(h_concise, w_concise),
		mode='bilinear',
		align_corners=False
	)
	all_image_concise_crops = all_image_concise_crops.permute(0, 2, 3, 1).contiguous().flatten(1,2)
	image_concise_hidden_states = torch.split(all_image_concise_crops, split_sizes)
	return image_concise_hidden_states

class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
	config_class = LlavaConfig

	def __init__(self, config: LlamaConfig):
		super(LlavaLlamaModel, self).__init__(config)
		
	def forward(
		self,
		input_ids: torch.LongTensor = None,
		attention_mask: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
		cache_position: Optional[torch.LongTensor] = None,
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

		return_legacy_cache = False
		if (
			use_cache and not isinstance(past_key_values, Cache) and not self.training
		):  # kept for BC (non `Cache` `past_key_values` inputs)
			return_legacy_cache = True
			past_key_values = DynamicCache.from_legacy_cache(past_key_values)
			logger.warning_once(
				"We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
				"Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
			)

		if cache_position is None:
			past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
			cache_position = torch.arange(
				past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
			)
		if position_ids is None:
			position_ids = cache_position.unsqueeze(0)

		# causal_mask = self._update_causal_mask(
		# 	attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
		# )
		causal_mask = attention_mask
		hidden_states = inputs_embeds

		# max_len = max((position_ids.shape[1], position_ids_fast_kv.shape[1]))
		# # create position embeddings to be shared across the decoder layers
		# position_embeddings = self.rotary_emb(hidden_states, torch.arange(max_len, device=position_ids.device, dtype=position_ids.dtype).unsqueeze(0).expand(position_ids.shape[0], -1))
		position_embeddings = None

		# decoder layers
		all_hidden_states = () if output_hidden_states else None
		all_self_attns = () if output_attentions else None
		next_decoder_cache = None

		fast_vision_start_layer = self.config.fast_vision_start_layer
		bs = hidden_states.shape[0]

		for layer_i, decoder_layer in enumerate(self.layers):
			if output_hidden_states:
				all_hidden_states += (hidden_states,)

			if not self.config.fast_vision or layer_i < fast_vision_start_layer:
				if self.gradient_checkpointing and self.training:
					layer_outputs = self._gradient_checkpointing_func(
						decoder_layer.__call__,
						hidden_states,
						None,
						causal_mask,
						position_ids,
						None,
						past_key_values,
						output_attentions,
						use_cache,
						cache_position,
						position_embeddings,
					)
				else:
					layer_outputs = decoder_layer(
						hidden_states,
						attention_mask=causal_mask,
						position_ids=position_ids,
						past_key_value=past_key_values,
						output_attentions=output_attentions,
						use_cache=use_cache,
						cache_position=cache_position,
						position_embeddings=position_embeddings,
					)

				hidden_states = layer_outputs[0]

			else:
				if layer_i == fast_vision_start_layer:
					image_full_len = length_info['image_full_len']
					image_concise_len = length_info['image_concise_len']
					text_len = length_info['text_len']
					newline_len = length_info['text_len']

					image_full_hidden_states = []
					text_hidden_states = []
					newline_hidden_states = []

					
					for batch_i in range(bs):
						image_full_hidden_states.append(hidden_states[batch_i][:image_full_len[batch_i]])
						newline_hidden_states.append(hidden_states[batch_i][image_full_len[batch_i]:image_full_len[batch_i]+newline_len[batch_i]])
						text_hidden_states.append(hidden_states[batch_i][image_full_len[batch_i]+newline_len[batch_i]:image_full_len[batch_i]+newline_len[batch_i]+text_len[batch_i]])
						
					concise_reduce_factor = self.config.concise_reduce_factor
					image_concise_hidden_states = get_image_concise(image_full_hidden_states, concise_reduce_factor, self.config.image_token_len_per_side**2)
					
					hidden_states_fast_q = []
					hidden_states_fast_kv = []
					# q [image_concise, newline, text]
					# key&value [image_full, image_concise, newline, text]
					for batch_i in range(bs):
						cur_hidden_states_fast_q = torch.cat([image_concise_hidden_states[batch_i].flatten(0, 1), newline_hidden_states[batch_i], text_hidden_states[batch_i]])
						cur_hidden_states_fast_kv = torch.cat([image_full_hidden_states[batch_i],image_concise_hidden_states[batch_i].flatten(0, 1), newline_hidden_states[batch_i], text_hidden_states[batch_i]])
						if len(cur_hidden_states_fast_q) < position_ids_fast_q.shape[1]:
							padding_len = position_ids_fast_q.shape[1] - len(cur_hidden_states_fast_q)
							cur_hidden_states_fast_q = torch.cat([cur_hidden_states_fast_q, torch.zeros((padding_len, cur_hidden_states_fast_q.shape[-1]), device=cur_hidden_states_fast_q.device, dtype=cur_hidden_states_fast_q.dtype)])
						if len(cur_hidden_states_fast_kv) < position_ids_fast_kv.shape[1]:
							padding_len = cur_hidden_states_fast_kv.shape[1] - len(cur_hidden_states_fast_kv)
							cur_hidden_states_fast_kv = torch.cat([cur_hidden_states_fast_kv, torch.zeros((padding_len, cur_hidden_states_fast_kv.shape[-1]), device=cur_hidden_states_fast_kv.device, dtype=cur_hidden_states_fast_kv.dtype)])
						hidden_states_fast_q.append(cur_hidden_states_fast_q)
						hidden_states_fast_kv.append(cur_hidden_states_fast_kv)
					hidden_states_fast_q = torch.stack(hidden_states_fast_q)
					hidden_states_fast_kv = torch.stack(hidden_states_fast_kv)


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
						cache_position,
						position_embeddings
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
						cache_position,
						position_embeddings
					)

				hidden_states_fast_q = layer_outputs[0]

				image_concise_hidden_states = []
				image_full_hidden_states = []
				split_sizes = []
				h = w = self.config.image_token_len_per_side
				h_concise = h//concise_reduce_factor
				w_concise = w//concise_reduce_factor
				for batch_i in range(bs):
					cur_image_concise_hidden_states = layer_outputs[0][batch_i][:image_concise_len[batch_i]]
					image_num =  cur_image_concise_hidden_states.shape[0]//(h_concise*w_concise)
					split_sizes.append(image_num)

					image_concise_hidden_states.append(cur_image_concise_hidden_states.view(image_num, h_concise , w_concise, -1))
					image_full_hidden_states[batch_i] = image_full_hidden_states[batch_i].view(image_num, h , w, -1)

				image_concise_hidden_states = torch.cat(image_concise_hidden_states)
				image_full_hidden_states = torch.cat(image_full_hidden_states)

				image_full_hidden_states = self.vision_mlp_layers[i](image_concise_hidden_states, image_concise_hidden_states, h, h_concise)
				image_full_hidden_states = torch.split(image_full_hidden_states, split_sizes)

				for batch_i in range(bs):
					hidden_states_fast_kv[batch_i][:image_full_len[batch_i]] = image_full_hidden_states[batch_i].flatten(0, 1)
					hidden_states_fast_kv[batch_i][image_full_len[batch_i]:image_full_len[batch_i]+image_concise_len[batch_i]+newline_len[batch_i]+text_len[batch_i]] = hidden_states_fast_q[batch_i][image_concise_len[batch_i]:image_concise_len[batch_i]+newline_len[batch_i]+text_len[batch_i]]

				hidden_states = hidden_states_fast_q

			if use_cache:
				next_decoder_cache = layer_outputs[2 if output_attentions else 1]

			if output_attentions:
				all_self_attns += (layer_outputs[1],)

		hidden_states = self.norm(hidden_states)

		# add hidden states from the last decoder layer
		if output_hidden_states:
			all_hidden_states += (hidden_states,)

		next_cache = next_decoder_cache if use_cache else None
		if return_legacy_cache:
			next_cache = next_cache.to_legacy_cache()

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

		if inputs_embeds is None:
			(input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, labels_fast, length_info, position_ids_fast_q, position_ids_fast_kv, attention_masks_fast_4d) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes)
			if self.get_model().config.fast_vision:
				labels = labels_fast

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
			length_info=length_info,
			position_ids_fast_q=position_ids_fast_q,
			position_ids_fast_kv=position_ids_fast_kv,
			attention_masks_fast_4d=attention_masks_fast_4d
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
			shift_logits = logits[..., :-1, :].contiguous()
			shift_labels = labels[..., 1:].contiguous()
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
		modalities = kwargs.pop("modalities", None) if "modalities" in kwargs and modalities is None else modalities
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


AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)

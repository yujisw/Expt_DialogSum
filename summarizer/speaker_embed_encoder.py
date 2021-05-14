import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.configuration_bart import BartConfig
from transformers.modeling_bart import (
    EncoderLayer,
    SinusoidalPositionalEmbedding,
    LearnedPositionalEmbedding,
    LayerNorm,
    invert_mask,
)
from transformers.modeling_outputs import BaseModelOutput

class TurnConverter():
    def __init__(self, eot_id, pad_id):
        self.eot_id=eot_id
        self.pad_id=pad_id
        self.current_speaker_id=1
    
    def init_speaker_id(self):
        self.current_speaker_id=1
    
    def change_speaker_id(self):
        if self.current_speaker_id==1:
            self.current_speaker_id = 2
        elif self.current_speaker_id==2:
            self.current_speaker_id = 1
    
    def convert_id_to_speaker_id(self, w_id):
        if w_id==self.pad_id:
            return 0
        elif w_id==self.eot_id:
            self.change_speaker_id()
        return self.current_speaker_id

    def convert_batch(self, input_ids):
        batch_speaker_ids = []
        for text_ids in input_ids:
            speaker_ids = []
            self.init_speaker_id()
            for w_id in text_ids:
                speaker_ids.append(self.convert_id_to_speaker_id(w_id))
            batch_speaker_ids.append(speaker_ids)
        return torch.tensor(batch_speaker_ids).to('cuda')

# class SpeakerConverter():
#     def __init__(self, says_id, eot_id, eod_id=1, pad_id=0):
#         self.says_id=says_id
#         self.eot_id=eot_id
#         self.eod_id=eod_id
#         self.pad_id=pad_id
#         self.current_speaker_id=1
#         self.speaker_list = []
    
#     def init_attr(self):
#         self.current_speaker_id=1
#         self.speaker_list = []

#     def get_speaker_id(self, speaker_name):
#         if speaker_name in self.speaker_list:
#             return self.speaker_list.index(speaker_name)+1
#         else:
#             self.speaker_list.append(speaker_name)
#             return len(self.speaker_list)
        
#     def change_speaker_id(self, speaker_ids):
#         if [self.eod_id] == speaker_ids:
#             self.current_speaker_id = 0
#         else:
#             self.current_speaker_id = self.get_speaker_id('_'.join([str(w_id) for w_id in speaker_ids]))
#         return self.current_speaker_id
    
#     def convert_batch(self, input_ids):
#         batch_speaker_ids = []
#         for text_idx, text_seq in enumerate(input_ids):
#             speaker_ids = []
#             self.init_attr()
#             text_len = len(text_seq)
#             for w_idx in range(text_len):
#                 if w_idx==0:
#                     for i in range(w_idx, text_len):
#                         if self.says_id == text_seq[i]:
#                             speaker_ids.append(self.current_speaker_id)
#                             self.change_speaker_id(text_seq[w_idx:i])
#                             break
#                 elif self.eot_id == text_seq[w_idx]:
#                     for i in range(w_idx+1, text_len):
#                         if self.eod_id == text_seq[i]:
#                             speaker_ids.append(self.current_speaker_id)
#                             self.change_speaker_id([self.eod_id])
#                             break
#                         elif self.says_id == text_seq[i]:
#                             speaker_ids.append(self.current_speaker_id)
#                             self.change_speaker_id(text_seq[w_idx+1:i])
#                             break
#                 else:
#                     speaker_ids.append(self.current_speaker_id)
#             batch_speaker_ids.append(speaker_ids)
#         return torch.tensor(batch_speaker_ids).to('cuda')

class SpeakerConverter():
    def __init__(self, sep_id, says_id, turn_embed=False, eod_id=1, pad_id=0):
        self.sep_id=sep_id
        self.says_id=says_id # id for "[SAYS]" or ":"
        self.turn_embed=turn_embed
        self.eod_id=eod_id
        self.pad_id=pad_id
    
    def _input_ids_to_speaker_ids(self, id_seq):
        """
        Requirements:
            id_seq: 1データの入力トークンidベクトル
        """
        sep_indices = (id_seq == self.sep_id).nonzero(as_tuple=True)[0]
        start_indices = torch.cat([torch.LongTensor([0]).to('cuda'),sep_indices+1]).tolist()
        name_list = []
        for start_idx in start_indices:
            for idx, word_id in enumerate(id_seq[start_idx:]):
                if word_id.item()==self.says_id or word_id.item()==self.eod_id:
                    name_list.append(str(id_seq[start_idx:start_idx+idx]))
                    break
        assert len(start_indices) == len(name_list)

        name_dict= {name:id+1 for id, name in enumerate(sorted(set(name_list), key=name_list.index))}
        mask = np.sign(abs(np.diff([0]+[name_dict[name] for name in name_list])))>0
        start_indices = np.array(start_indices)[mask]
        name_list = np.array(name_list)[mask]

        length = len(id_seq)
        end_idx = (id_seq == self.eod_id).nonzero()[0][0].item()
        speaker_ids = torch.zeros_like(id_seq).to('cuda')
        for start_idx, speaker_id in zip(start_indices, [name_dict[name] for name in name_list]):
            speaker_ids[start_idx:] = torch.ones(length-start_idx, dtype=torch.int32).to('cuda') * speaker_id
        speaker_ids[end_idx:] = torch.ones(length-end_idx, dtype=torch.int32).to('cuda') * self.pad_id # which means torch.zeros, because self.pad_id is basically 0
        return speaker_ids

    def convert_batch(self, input_ids):
        speaker_ids = [self._input_ids_to_speaker_ids(id_seq) for id_seq in input_ids]
        return torch.stack(speaker_ids).to('cuda')

class BartEncoderWithSpeakerEmbedding(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    :class:`EncoderLayer`.
    Args:
        config: BartConfig
    """

    def __init__(self, config: BartConfig, embed_tokens, says_id, eot_id, ratio_to_token_embedding=0, speaker_embed_scale=0, use_turn_embeds=False, partial_embed=False, eod_id=1, pad_id=0):
        super().__init__()

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        self.embed_dim = embed_tokens.embedding_dim
        self.embed_scale = math.sqrt(self.embed_dim) if config.scale_embedding else 1.0
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = config.max_position_embeddings

        self.embed_tokens = embed_tokens

        # speaker embedding setup
        # self.says_id = embed_tokens.num_embeddings - 3
        # self.eot_id = embed_tokens.num_embeddings - 1
        self.says_id = says_id
        self.eot_id = eot_id

        if use_turn_embeds:
            max_speaker_num = 2
            self.speaker_converter = TurnConverter(eot_id = self.eot_id, pad_id=pad_id)
        else:
            max_speaker_num = 14
            # self.speaker_converter = SpeakerConverter(says_id = self.says_id, eot_id = self.eot_id, eod_id=1, pad_id=0)
            self.speaker_converter = SpeakerConverter(sep_id=self.eot_id, says_id=self.says_id, turn_embed=False, eod_id=eod_id, pad_id=pad_id)

        # setting for speaker_embed_scale
        assert ratio_to_token_embedding!=0 or speaker_embed_scale!=0, "Please set ratio_to_token_embedding or speaker_embed_scale with a positive value."
        assert ratio_to_token_embedding==0 or speaker_embed_scale==0, "Do not set both ratio_to_token_embedding and speaker_embed_scale."
        if ratio_to_token_embedding!=0:
            self.speaker_embed_scale = self.embed_scale * ratio_to_token_embedding
        elif speaker_embed_scale!=0:
            self.speaker_embed_scale = speaker_embed_scale
        else:
            assert False, "Could not set speaker_embed_scale."

        self.partial_embed = partial_embed
        if self.partial_embed:
            # valid_idx = torch.LongTensor(list(range(0, self.embed_dim//4)) + list(range(self.embed_dim//2,self.embed_dim*3//4))) # first half
            # valid_idx = torch.LongTensor(list(range(self.embed_dim//4, self.embed_dim//2)) + list(range(self.embed_dim*3//4,self.embed_dim))) # second half
            # valid_idx = torch.LongTensor(list(range(0, self.embed_dim*1//8)) + list(range(self.embed_dim*4//8,self.embed_dim*5//8))) # first quarter
            # valid_idx = torch.LongTensor(list(range(self.embed_dim*1//8, self.embed_dim*2//8)) + list(range(self.embed_dim*5//8,self.embed_dim*6//8))) # second quarter
            # valid_idx = torch.LongTensor(list(range(self.embed_dim*2//8, self.embed_dim*3//8)) + list(range(self.embed_dim*6//8,self.embed_dim*7//8))) # third quarter
            valid_idx = torch.LongTensor(list(range(self.embed_dim*3//8, self.embed_dim*4//8)) + list(range(self.embed_dim*7//8,self.embed_dim))) # forth quarter
            # valid_idx = torch.LongTensor(list(range(self.embed_dim*3//8, self.embed_dim//2)) + list(range(self.embed_dim*7//8,self.embed_dim)))
            self.mask = torch.eye(self.embed_dim)[valid_idx].sum(axis=0).to('cuda')
            print("mask requires_grad:",self.mask.requires_grad)
            print("mask[0]:",self.mask)
            self.embed_speaker = nn.Embedding(max_speaker_num+1, self.embed_dim, padding_idx=pad_id)
        else:
            self.embed_speaker = nn.Embedding(max_speaker_num+1, self.embed_dim, padding_idx=pad_id)
        
        if config.static_position_embeddings:
            self.embed_positions = SinusoidalPositionalEmbedding(
                config.max_position_embeddings, self.embed_dim, self.padding_idx
            )
        else:
            self.embed_positions = LearnedPositionalEmbedding(
                config.max_position_embeddings,
                self.embed_dim,
                pad_id,
                config.extra_pos_embeddings,
            )
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = LayerNorm(self.embed_dim) if config.normalize_embedding else nn.Identity()
        # mbart has one extra layer_norm
        self.layer_norm = LayerNorm(config.d_model) if config.add_final_layer_norm else None

    def forward(
        self, input_ids, attention_mask=None, output_attentions=False, output_hidden_states=False, return_dict=True
    ):
        """
        Args:
            input_ids (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            attention_mask (torch.LongTensor): indicating which indices are padding tokens
        Returns:
            BaseModelOutput or Tuple comprised of:
                - **x** (Tensor): the last encoder layer's output of shape `(src_len, batch, embed_dim)`
                - **encoder_states** (tuple(torch.FloatTensor)): all intermediate hidden states of shape `(src_len,
                  batch, embed_dim)`. Only populated if *output_hidden_states:* is True.
                - **all_attentions** (tuple(torch.FloatTensor)): Attention weights for each layer.
                During training might not be of length n_layers because of layer dropout.
        """
        # check attention mask and invert
        if attention_mask is not None:
            attention_mask = invert_mask(attention_mask)

        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        embed_pos = self.embed_positions(input_ids)
        
        batch_speaker_ids = self.speaker_converter.convert_batch(input_ids)
        embed_spk = self.embed_speaker(batch_speaker_ids) * self.speaker_embed_scale
        
        # x = inputs_embeds + embed_pos
        x = inputs_embeds + embed_pos
        if self.partial_embed:
            embed_spk = embed_spk * self.mask
            # x[:,:,self.embed_dim*3//8:self.embed_dim//2] = x[:,:,self.embed_dim*3//8:self.embed_dim//2] + embed_spk[:,:,:self.embed_dim//8]
            # x[:,:,self.embed_dim*7//8:] = x[:,:,self.embed_dim*7//8:] + embed_spk[:,:,self.embed_dim//8:]
        # else:
        #     x = x + embed_spk
        x = x + embed_spk

        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = [] if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states.append(x)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                attn = None
            else:
                # print("before encoder_layer", x.shape)
                x, attn = encoder_layer(x, attention_mask, output_attentions=output_attentions)

            if output_attentions:
                all_attentions = all_attentions + (attn,)

        if self.layer_norm:
            x = self.layer_norm(x)
        if output_hidden_states:
            encoder_states.append(x)
            # T x B x C -> B x T x C
            encoder_states = tuple(hidden_state.transpose(0, 1) for hidden_state in encoder_states)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if not return_dict:
            return tuple(v for v in [x, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=x, hidden_states=encoder_states, attentions=all_attentions)

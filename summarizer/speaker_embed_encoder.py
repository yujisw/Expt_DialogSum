import math
import torch
import torch.nn as nn
from transformers.configuration_bart import BartConfig
from transformers.modeling_bart import EncoderLayer, SinusoidalPositionalEmbedding, LayerNorm

class SpeakerConverter():
    def __init__(self, speaker_num, eot_idx):
        self.speaker_num = speaker_num
        self.eot_idx=eot_idx
        self.current_speaker_id=1
    
    def init_speaker_id(self):
        self.current_speaker_id=1
    
    def change_speaker_id(self):
        if self.current_speaker_id==1:
            self.current_speaker_id = 2
        elif self.current_speaker_id==2:
            self.current_speaker_id = 1
    
    def convert_id_to_speaker_id(self, w_id):
        if w_id==0:
            return 0
        elif w_id==self.eot_idx:
            self.change_speaker_id()
        return self.current_speaker_id

    def convert_batch(self, input_ids):
        batch_speaker_ids = []
        for text_ids in input_ids:
            speaker_ids = []
            sc.init_speaker_id()
            for w_id in text_ids:
                speaker_ids.append(sc.convert_id_to_speaker_id(w_id.item()))
            batch_speaker_ids.append(speaker_ids)
        return torch.tensor(batch_speaker_ids)

class BartEncoderWithSpeakerEmbedding(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    :class:`EncoderLayer`.
    Args:
        config: BartConfig
    """

    def __init__(self, config: BartConfig, embed_tokens):
        super().__init__()

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = embed_tokens.embedding_dim
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = config.max_position_embeddings

        self.embed_tokens = embed_tokens

        # speaker embedding setup
        self.eot_idx = embed_tokens.num_embeddings - 1
        speaker_num = 2 #TODO
        self.speaker_converter = SpeakerConverter(speaker_num = speaker_num, eot_idx = self.eot_idx)
        self.speaker_embed_scale = 0.1
        self.embed_speaker = nn.Embedding(speaker_num+1, embed_dim, padding_idx=0)
        
        if config.static_position_embeddings:
            self.embed_positions = SinusoidalPositionalEmbedding(
                config.max_position_embeddings, embed_dim, self.padding_idx
            )
        else:
            self.embed_positions = LearnedPositionalEmbedding(
                config.max_position_embeddings,
                embed_dim,
                self.padding_idx,
                config.extra_pos_embeddings,
            )
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = LayerNorm(embed_dim) if config.normalize_embedding else nn.Identity()
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
        x = inputs_embeds + embed_pos + embed_spk
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

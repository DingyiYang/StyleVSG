import math
import random
from collections import namedtuple, OrderedDict
import copy

import numpy as np
import torch
import torch.nn as nn
import torchtext
import torch.nn.functional as F
from mass.lstm import FactoredLSTM
from mass.lstm import DLNLSTM
from torch.autograd import Variable

import fairseq
from fairseq import options, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    MultiheadAttention,
    LayerNorm,
    AdaptiveInput,
    CharacterTokenEmbedder,
    PositionalEmbedding,
    AdaptiveSoftmax,
)

from mass.sinusoidal_positional_embedding import SinusoidalPositionalEmbedding
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from .learned_positional_embedding import LearnedPositionalEmbedding
from .multihead_attention import MultiheadAttention as localMultiheadAttention

import clip

DEFAULT_MAX_SOURCE_POSITIONS = 512
DEFAULT_MAX_TARGET_POSITIONS = 512


@register_model('transformer_mass')
class TransformerMASSModel(FairseqEncoderDecoderModel):
    """
    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')

        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')

        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--no-decoder-final-norm', action='store_true',
                            help='don\'t add an extra layernorm after the last decoder block')

        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion')
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--adaptive-softmax-factor', type=float, metavar='N',
                            help='adaptive input factor')
        parser.add_argument('--no-token-positional-embeddings', action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--character-embeddings', action='store_true',
                            help='if set, uses character embedding convolutions to produce token embeddings')
        parser.add_argument('--character-filters', type=str, metavar='LIST',
                            default='[(1, 64), (2, 128), (3, 192), (4, 256), (5, 256), (6, 256), (7, 256)]',
                            help='size of character embeddings')
        parser.add_argument('--character-embedding-dim', default=4, type=int, metavar='N',
                            help='size of character embeddings')
        parser.add_argument('--adaptive-input', action='store_true',
                            help='if set, uses adaptive input')
        parser.add_argument('--adaptive-input-factor', type=float, metavar='N',
                            help='adaptive input factor')
        parser.add_argument('--adaptive-input-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive input cutoff points.')
        parser.add_argument('--tie-adaptive-weights', action='store_true',
                            help='if set, ties the weights of adaptive softmax and adaptive input')
        parser.add_argument('--tie-adaptive-proj', action='store_true',
                            help='if set, ties the projection weights of adaptive softmax and adaptive input')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for decoder')
        parser.add_argument('--decoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--layernorm-embedding', action='store_true',
                            help='add layernorm to embedding')
        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')

        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--load-from-pretrained-model', type=str, default=None,
                            help='Load from pretrained model')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim
            )

        encoder = TransformerEncoder(args, src_dict, encoder_embed_tokens)
        decoder = TransformerDecoder(args, tgt_dict, decoder_embed_tokens)

        model = TransformerMASSModel(encoder, decoder)

        if args.load_from_pretrained_model is not None:
            states = torch.load(args.load_from_pretrained_model, map_location='cpu')
            model.load_state_dict(states)
            args.load_from_pretrained_model = None  # Clear this param

        return TransformerMASSModel(encoder, decoder)

    def max_positions(self):
        return (self.encoder.max_positions(), self.decoder.max_positions())

    def forward(self, src_tokens=None, src_sen_pos=None, src_lengths=None, prev_output_tokens=None,
                **kwargs):
        """
        Run the forward pass for an encoder-decoder model.

        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., teacher forcing) to
        the decoder to produce the next outputs::

            encoder_out = self.encoder(src_tokens, src_lengths)
            return self.decoder(prev_output_tokens, encoder_out)

        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            # ADD: src_sen_pos: 输入的图片对应的是哪张图片/句子

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        encoder_out = self.encoder(src_tokens, src_sen_pos, src_lengths=src_lengths, **kwargs)
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        return decoder_out


@register_model('transformer_mix')
class TransformerMixModel(FairseqEncoderDecoderModel):
    """
    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, encoder, decoder, model_lang_pairs, whether_concat, whether_add_memory):
        self.model_lang_pairs = model_lang_pairs
        self.whether_concat = whether_concat
        self.whether_add_memory = whether_add_memory
        self.padding_idx = 0
        # self.memory_initializer = memory_initializer
        print("whether_concat", self.whether_concat)
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')

        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')

        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--no-decoder-final-norm', action='store_true',
                            help='don\'t add an extra layernorm after the last decoder block')

        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion')
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--adaptive-softmax-factor', type=float, metavar='N',
                            help='adaptive input factor')
        parser.add_argument('--no-token-positional-embeddings', action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--character-embeddings', action='store_true',
                            help='if set, uses character embedding convolutions to produce token embeddings')
        parser.add_argument('--character-filters', type=str, metavar='LIST',
                            default='[(1, 64), (2, 128), (3, 192), (4, 256), (5, 256), (6, 256), (7, 256)]',
                            help='size of character embeddings')
        parser.add_argument('--character-embedding-dim', default=4, type=int, metavar='N',
                            help='size of character embeddings')
        parser.add_argument('--adaptive-input', action='store_true',
                            help='if set, uses adaptive input')
        parser.add_argument('--adaptive-input-factor', type=float, metavar='N',
                            help='adaptive input factor')
        parser.add_argument('--adaptive-input-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive input cutoff points.')
        parser.add_argument('--tie-adaptive-weights', action='store_true',
                            help='if set, ties the weights of adaptive softmax and adaptive input')
        parser.add_argument('--tie-adaptive-proj', action='store_true',
                            help='if set, ties the projection weights of adaptive softmax and adaptive input')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for decoder')
        parser.add_argument('--decoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--layernorm-embedding', action='store_true',
                            help='add layernorm to embedding')
        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')

        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--load-from-pretrained-model', type=str, default=None,
                            help='Load from pretrained model')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        model_lang_pairs = task.model_lang_pairs
        # make sure all arguments are present in older models
        base_architecture(args)
        whether_concat = args.whether_concat

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim
            )

        whether_mix=False
        if(args.whether_use_text_clip_fea=="True"):whether_mix=True
        if(args.decoder_type=="DLNLSTM"):whether_mix=True

        encoder = TransformerEncoder(args, src_dict, encoder_embed_tokens, whether_mix=whether_mix)
        decoder = TransformerMixDecoder(args, tgt_dict, decoder_embed_tokens)
        if(args.decoder_type=="FactoredLSTM"):
            decoder = FactoredLSTM(args, tgt_dict, decoder_embed_tokens)
        elif(args.decoder_type=="DLNLSTM"):
            decoder = DLNLSTM(args, tgt_dict, decoder_embed_tokens)

        if (args.whether_add_memory == "True"):
            decoder = TransformerMixDecoderWithMemory(args, tgt_dict, decoder_embed_tokens)

        model = TransformerMixModel(encoder, decoder, model_lang_pairs, whether_concat=whether_concat,
                                    whether_add_memory=args.whether_add_memory)

        if args.load_from_pretrained_model is not None:
            states = torch.load(args.load_from_pretrained_model, map_location='cpu')
            if 'model' in states:
                states = states['model']

            # now we need to manipulate the states
            num_decoder_output_types = len(args.model_lang_pairs)

            new_states = OrderedDict()
            layers_to_copy = []
            if options.eval_bool(args.divide_decoder_self_attn_norm):
                layers_to_copy.append('self_attn_layer_norm')
            if options.eval_bool(args.divide_decoder_embed_norm):
                layers_to_copy.append('emb_layer_norm')
            if options.eval_bool(args.divide_decoder_final_norm):
                layers_to_copy.append('final_layer_norm')
            if options.eval_bool(args.divide_decoder_encoder_attn_norm):
                layers_to_copy.append('encoder_attn_layer_norm')
            if options.eval_bool(args.divide_decoder_self_attn_query):
                layers_to_copy.append('self_attn.in_proj')
            if options.eval_bool(args.divide_decoder_encoder_attn_query):
                layers_to_copy.append('encoder_attn.in_proj')

            for k, v in states.items():
                if 'decoder' in k and any(item in k for item in layers_to_copy):
                    layer_idx = k.split('.')[2]
                    suffix = k.split('.')[-1] if 'in_proj' not in k else k.split('_')[-1]
                    layer_name = k.split('.')[-2]
                    for type_idx in range(num_decoder_output_types):
                        if 'emb_layer_norm' in k:
                            new_states['decoder.{}.{}.{}'.format(layer_name, type_idx, suffix)] \
                                = copy.deepcopy(v)
                        else:
                            if 'in_proj' in k:
                                dim = int(v.shape[0] / 3)
                                new_states[
                                    'decoder.layers.{}.{}.q_proj.{}.{}'.format(layer_idx, layer_name, type_idx, suffix)] \
                                    = copy.deepcopy(v[:dim])
                                new_states[
                                    'decoder.layers.{}.{}.k_proj.{}'.format(layer_idx, layer_name, suffix)] \
                                    = copy.deepcopy(v[dim:2 * dim])
                                new_states[
                                    'decoder.layers.{}.{}.v_proj.{}'.format(layer_idx, layer_name, suffix)] \
                                    = copy.deepcopy(v[2 * dim:])
                            else:
                                new_states['decoder.layers.{}.{}.{}.{}'.format(layer_idx, layer_name, type_idx, suffix)] \
                                    = copy.deepcopy(v)
                else:
                    new_states[k] = v

            del states
            print('loading pretrained model from: {}'.format(args.load_from_pretrained_model))
            model.load_state_dict(new_states)
            args.load_from_pretrained_model = None  # Clear this param

        return model

    def max_positions(self):
        """Maximum length supported by the model."""
        return {
            key: (self.encoder.max_positions(), self.decoder.max_positions())
            for key in self.model_lang_pairs
        }

    def forward(self, src_tokens=None, src_sen_pos=None, src_lengths=None, img_features=None, text_features=None,
                prev_output_tokens=None, prev_sen_pos=None, multi_target=None, **kwargs):
        """
        Run the forward pass for an encoder-decoder model.

        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., teacher forcing) to
        the decoder to produce the next outputs::

            encoder_out = self.encoder(src_tokens, src_lengths)
            return self.decoder(prev_output_tokens, encoder_out)

        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """

        src_tokens = src_tokens[:, :-1]
        pic_num = 5
        token_num = int(src_tokens.size(1) / pic_num)
        sen_pos_num = int(src_sen_pos.size(1) / pic_num)
        src_sen_pos = torch.cat([src_sen_pos[:, sen_pos_num * i:sen_pos_num * (i) + token_num] for i in range(5)],
                                dim=1)
        temp_device = src_tokens.device
        if (self.whether_concat == True or self.whether_concat == "True"):
            # 判断是否有返回memory_unit
            # print(encoder_out.encoder_padding_mask)
            decoder_out = []
            # print("token_num",token_num)
            batch_size = src_tokens.size(0)

            encoder_out = self.encoder(src_tokens, src_sen_pos, src_lengths=src_lengths,
                                       img_features=img_features, text_features=text_features, **kwargs)
            encoder_out_last = encoder_out.encoder_out
            visual_hidden = encoder_out.visual_hidden
            # print("encoder_out_last",encoder_out_last.shape)
            # print("visual_hidden",visual_hidden.shape)
            concat_padding_mask = encoder_out.encoder_padding_mask
            prev_ms = [encoder_out.encoder_out, encoder_out.encoder_padding_mask]

            token_num = int(encoder_out_last.size(0) / 5)
            for i in range(pic_num):
                if (self.whether_add_memory == "True"):
                    t_padding = concat_padding_mask
                    if (img_features is not None): t_padding = None
                    if (t_padding != None): t_padding = t_padding[:, i * token_num:(i + 1) * token_num]
                    t_encoder_out = EncoderOut(
                        encoder_out=encoder_out_last[i * token_num:(i + 1) * token_num, :, :],  # T x B x C
                        encoder_padding_mask=t_padding,  # B x T
                        encoder_states=None,  # List[T x B x C]
                        visual_hidden=visual_hidden[i * token_num:(i + 1) * token_num, :, :],
                    )
                else:
                    t_padding = encoder_out.encoder_padding_mask
                    if (img_features is not None): t_padding = None
                    if (t_padding != None): t_padding = t_padding[:, i * token_num:(i + 1) * token_num]
                    t_encoder_out = EncoderOut(
                        encoder_out=encoder_out_last[i * token_num:(i + 1) * token_num, :, :],  # T x B x C
                        encoder_padding_mask=t_padding,  # B x T
                        encoder_states=None,  # List[T x B x C]
                        visual_hidden=visual_hidden[i * token_num:(i + 1) * token_num, :, :],
                    )

                prev_sen = multi_target[i]

                if (self.whether_add_memory == "True"):
                    whether_update = True
                    t_decoder_out, extra, prev_ms = \
                        self.decoder(prev_output_tokens[i],
                                     visual_hidden=visual_hidden[i * token_num:(i + 1) * token_num, :, :],
                                     temp_sen_id=i + 1, prev_sen_pos=None, encoder_out=t_encoder_out, last_sen=prev_sen,
                                     prev_ms=prev_ms, whether_update=whether_update, text_features = text_features[:, i, :] if text_features is not None else None, **kwargs)
                    decoder_out_i = (t_decoder_out, extra)
                    decoder_out.append(decoder_out_i)
                else:
                    decoder_out_i = self.decoder(prev_output_tokens[i], prev_sen_pos=i+1, encoder_out=t_encoder_out,
                                                 **kwargs)
                    decoder_out.append(decoder_out_i)

            return decoder_out

        encoder_out = self.encoder(src_tokens, src_sen_pos, src_lengths=src_lengths,
                                   img_features=img_features, text_features=text_features, **kwargs)
        decoder_out = self.decoder(prev_output_tokens, prev_sen_pos=None, encoder_out=encoder_out, **kwargs)
        return decoder_out


class ResidualOutput(nn.Module):
    def __init__(self, config):
        super(ResidualOutput, self).__init__()
        self.dense = nn.Linear(config.encoder_embed_dim, config.encoder_embed_dim)
        self.LayerNorm = BertLayerNorm(config.encoder_embed_dim, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.decoder_embed_dim % config.decoder_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.decoder_embed_dim, config.decoder_attention_heads))
        self.num_attention_heads = config.decoder_attention_heads
        self.attention_head_size = int(config.decoder_embed_dim / config.decoder_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.decoder_embed_dim, self.all_head_size)
        self.key = nn.Linear(config.decoder_embed_dim, self.all_head_size)
        self.value = nn.Linear(config.decoder_embed_dim, self.all_head_size)

        self.dropout = nn.Dropout(config.dropout)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # (N, L, nh, dh)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (N, nh, L, dh)

    def forward(self, query_states, key_states, value_states, attention_mask):
        """
        Args:
            query_states: (N, Lq, D)
            key_states: (N, L, D)
            value_states: (N, L, D)
            attention_mask: (N, Lq, L)

        Returns:

        """
        # only need to mask the dimension where the softmax (last dim) is applied, as another dim (second last)
        # will be ignored in future computation anyway
        # print("attention_mask",attention_mask)
        attention_mask = (1 - attention_mask.unsqueeze(1)) * -10000.  # (N, 1, Lq, L)
        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(key_states)
        mixed_value_layer = self.value(value_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # (N, nh, Lq, dh)
        key_layer = self.transpose_for_scores(mixed_key_layer)  # (N, nh, L, dh)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # (N, nh, L, dh)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # (N, nh, Lq, L)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.decoder_embed_dim, config.decoder_embed_dim)
        self.LayerNorm = BertLayerNorm(config.decoder_embed_dim, eps=1e-5)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        """
        Args:
            input_tensor: (N, L, D)
            attention_mask: (N, Lq, L)

        Returns:

        """
        self_output = self.self(input_tensor, input_tensor, input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class MemoryInitializer(nn.Module):
    def __init__(self, args):
        super(MemoryInitializer, self).__init__()
        # init memory
        self.embed_dim = args.encoder_embed_dim
        self.n_memory_cells = args.n_memory_cells
        # print("n_memory_cells",self.n_memory_cells)
        self.init_memory_bias = nn.Parameter(
            torch.randn(1, args.n_memory_cells, 1)
        )  # (1, M, D)
        self.init_memory_fc = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            # BertLayerNorm(self.embed_dim),
            LayerNorm(self.embed_dim, export=False),
            nn.Dropout(args.dropout)
        )

    def forward(self, input_states, attention_mask=None):
        """ initialize the model with the first input states
            input_states: (N, L, D)
            attention_mask: (N, L)
        """
        pooled_input_states = None
        temp_device = input_states.device
        if (attention_mask is None):
            pooled_input_states = torch.sum(input_states, dim=1)
            pooled_input_states = pooled_input_states / input_states.size(1)
        else:
            pooled_input_states = torch.sum(input_states * attention_mask.unsqueeze(-1), dim=1)  # (N, D)
            pooled_input_states = pooled_input_states / attention_mask.sum(1, keepdim=True)  # (N, D) no zero here
        pooled_input_states = pooled_input_states.unsqueeze(1).repeat(1, self.n_memory_cells, 1)  # (N, M, D)
        pooled_input_states = pooled_input_states + self.init_memory_bias  # (N, M, D)
        # print(pooled_input_states)
        init_memory = self.init_memory_fc(pooled_input_states)  # (N, M, D)
        return init_memory


class MemoryUpdater(nn.Module):
    def __init__(self, args):
        super(MemoryUpdater, self).__init__()
        self.embed_dim = args.encoder_embed_dim
        num_attention_heads = args.encoder_attention_heads
        attention_dropout = 0.1
        add_bias_kv = False
        add_zero_attn = False
        # self.memory_update_attention = BertSelfAttention(args)
        self.memory_update_attention = MultiheadAttention(
            self.embed_dim,
            num_attention_heads,
            dropout=args.dropout,
            add_bias_kv=False,
            add_zero_attn=False,
            self_attention=True
        )

        self.mc = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.sc = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        self.mz = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.sz = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(self, prev_m, input_states, attention_mask=None):
        """ This module should have access to all the text at this step,
        since its state will not be used for generation at current step
        Args:
            prev_m: (N, M, D), M is memory size
            input_states: (N, L, D)
            attention_mask: (N, L)
        Returns:

        """

        n_memory_cells = prev_m.shape[1]
        update_mask = None
        temp_device = input_states.device
        # attention中要mask的位置为1

        y = input_states
        s_t, _ = self.memory_update_attention(
            query=prev_m,
            key=y,
            value=y,
            key_padding_mask=attention_mask,
            incremental_state=None,
            need_weights=False,
            attn_mask=None,
        )

        c_t = torch.tanh(self.mc(prev_m) + self.sc(s_t))  # (N, M, D)

        z_t = torch.sigmoid(self.mz(prev_m) + self.sz(s_t))  # (N, M, D)

        updated_memory = (1 - z_t) * c_t + z_t * prev_m  # (N, M, D)
        return updated_memory


class TransformerEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
            self,
            embedding_dim: float = 768,
            ffn_embedding_dim: float = 3072,
            num_attention_heads: float = 8,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.1,
            activation_fn: str = 'relu',
            add_bias_kv: bool = False,
            add_zero_attn: bool = False,
            export: bool = False,
    ) -> None:
        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=True
        )

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim, export=export)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)

    def forward(
            self,
            x: torch.Tensor,
            self_attn_mask: torch.Tensor = None,
            self_attn_padding_mask: torch.Tensor = None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)
        return x, attn


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block.
    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False,
                 no_train_encoder_attn=False):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.cross_self_attention = getattr(args, 'cross_self_attention', False)
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not self.cross_self_attention,
        )
        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu')
        )
        self.activation_dropout = getattr(args, 'activation_dropout', 0)
        if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = getattr(args, 'relu_dropout', 0)
        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, 'char_inputs', False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = MultiheadAttention(
                self.embed_dim,
                args.decoder_attention_heads,
                kdim=getattr(args, 'encoder_embed_dim', None),
                vdim=getattr(args, 'encoder_embed_dim', None),
                dropout=args.attention_dropout,
                encoder_decoder_attention=True,
            )
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.no_train_encoder_attn = no_train_encoder_attn

        self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(
            self,
            x,
            encoder_out=None,
            encoder_padding_mask=None,
            incremental_state=None,
            prev_self_attn_state=None,
            prev_attn_state=None,
            self_attn_mask=None,
            self_attn_padding_mask=None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """

        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        if prev_self_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state = {"prev_key": prev_key, "prev_value": prev_value}
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            self.self_attn._set_input_buffer(incremental_state, saved_state)

        if self.cross_self_attention and not (
                incremental_state is not None and "prev_key" in self.self_attn._get_input_buffer(incremental_state)):
            if self_attn_mask is not None:
                self_attn_mask = torch.cat((x.new(x.size(0), encoder_out.size(0)).zero_(), self_attn_mask), dim=1)
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    encoder_padding_mask = self_attn_padding_mask.new(encoder_out.size(1), encoder_out.size(0)).zero_()
                self_attn_padding_mask = torch.cat((encoder_padding_mask, self_attn_padding_mask), dim=1)
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        if self.encoder_attn is not None and not self.no_train_encoder_attn:
            residual = x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)
            if prev_attn_state is not None:
                if incremental_state is None:
                    incremental_state = {}
                prev_key, prev_value = prev_attn_state[:2]
                saved_state = {"prev_key": prev_key, "prev_value": prev_value}
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            # ADD:
            encoder_attention_mask = None
            encoder_attention_mask = encoder_out.eq(0)
            for t_i in range(encoder_attention_mask.size(0)):
                for t_j in range(encoder_attention_mask.size(1)):
                    if (t_j == 0):
                        encoder_attention_mask[t_i][t_j] = True
                    else:
                        encoder_attention_mask[t_i][t_j] = False

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                # attn_mask = encoder_attention_mask,
                static_kv=True,
            )
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            if self_attn_padding_mask is not None:
                self_attn_state = saved_state["prev_key"], saved_state["prev_value"], saved_state[
                    "prev_key_padding_mask"]
            else:
                self_attn_state = saved_state["prev_key"], saved_state["prev_value"]
            return x, attn, self_attn_state
        return x, attn

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn


class TransformerMixDecoderLayer(nn.Module):
    """Decoder layer block.
    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False,
                 no_train_encoder_attn=False):
        super().__init__()
        # check how many norms we need
        decoder_output_types, decoder_input_types = [], []
        decoder_output_types += [item.split('-')[-1] for item in args.model_lang_pairs]
        decoder_input_types += [item.split('-')[0] for item in args.model_lang_pairs]
        self.decoder_output_types_dict = {type: idx for idx, type in enumerate(decoder_output_types)}
        self.decoder_input_types_dict = {type: idx for idx, type in enumerate(decoder_input_types)}

        self.divide_decoder_self_attn_norm = options.eval_bool(args.divide_decoder_self_attn_norm)
        self.divide_decoder_final_norm = options.eval_bool(args.divide_decoder_final_norm)
        self.divide_decoder_encoder_attn_norm = options.eval_bool(args.divide_decoder_encoder_attn_norm)

        # check whether we divide the self-attn and encoder-attn query
        self.divide_decoder_self_attn_query = options.eval_bool(args.divide_decoder_self_attn_query)
        self.divide_decoder_encoder_attn_query = options.eval_bool(args.divide_decoder_encoder_attn_query)

        self.embed_dim = args.decoder_embed_dim
        self.cross_self_attention = getattr(args, 'cross_self_attention', False)
        if self.divide_decoder_self_attn_query:
            self.self_attn = localMultiheadAttention(
                embed_dim=self.embed_dim,
                num_heads=args.decoder_attention_heads,
                dropout=args.attention_dropout,
                add_bias_kv=add_bias_kv,
                add_zero_attn=add_zero_attn,
                self_attention=not self.cross_self_attention,
                tgt_types=self.decoder_output_types_dict,
                enable_torch_version=False,
            )
        else:
            self.self_attn = MultiheadAttention(
                embed_dim=self.embed_dim,
                num_heads=args.decoder_attention_heads,
                dropout=args.attention_dropout,
                add_bias_kv=add_bias_kv,
                add_zero_attn=add_zero_attn,
                self_attention=not self.cross_self_attention,
            )
        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu')
        )
        self.activation_dropout = getattr(args, 'activation_dropout', 0)
        if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = getattr(args, 'relu_dropout', 0)
        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, 'char_inputs', False)
        if self.divide_decoder_self_attn_norm:
            self.self_attn_layer_norm = nn.ModuleList([LayerNorm(self.embed_dim, export=export)
                                                       for _ in range(len(decoder_output_types))])
        else:
            self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            if self.divide_decoder_encoder_attn_query:
                self.encoder_attn = localMultiheadAttention(
                    self.embed_dim,
                    args.decoder_attention_heads,
                    kdim=getattr(args, 'encoder_embed_dim', None),
                    vdim=getattr(args, 'encoder_embed_dim', None),
                    dropout=args.attention_dropout,
                    encoder_decoder_attention=True,
                    tgt_types=self.decoder_input_types_dict,
                    enable_torch_version=False,
                )
            else:
                self.encoder_attn = MultiheadAttention(
                    self.embed_dim,
                    args.decoder_attention_heads,
                    kdim=getattr(args, 'encoder_embed_dim', None),
                    vdim=getattr(args, 'encoder_embed_dim', None),
                    dropout=args.attention_dropout,
                    encoder_decoder_attention=True,
                )
            if self.divide_decoder_encoder_attn_norm:
                self.encoder_attn_layer_norm = nn.ModuleList([LayerNorm(self.embed_dim, export=export)
                                                              for _ in range(len(decoder_input_types))])
            else:
                self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.no_train_encoder_attn = no_train_encoder_attn

        self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)

        if self.divide_decoder_final_norm:
            self.final_layer_norm = nn.ModuleList([LayerNorm(self.embed_dim, export=export)
                                                   for _ in range(len(decoder_output_types))])
        else:
            self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(
            self,
            x,
            encoder_out=None,
            encoder_padding_mask=None,
            incremental_state=None,
            prev_self_attn_state=None,
            prev_attn_state=None,
            self_attn_mask=None,
            self_attn_padding_mask=None,
            lang_pair=None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """

        # print(encoder_padding_mask)

        src, tgt = lang_pair.split('-')

        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        if prev_self_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state = {"prev_key": prev_key, "prev_value": prev_value}
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            self.self_attn._set_input_buffer(incremental_state, saved_state)

        if self.cross_self_attention and not (
                incremental_state is not None and "prev_key" in self.self_attn._get_input_buffer(incremental_state)):
            if self_attn_mask is not None:
                self_attn_mask = torch.cat((x.new(x.size(0), encoder_out.size(0)).zero_(), self_attn_mask), dim=1)
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    encoder_padding_mask = self_attn_padding_mask.new(encoder_out.size(1), encoder_out.size(0)).zero_()
                self_attn_padding_mask = torch.cat((encoder_padding_mask, self_attn_padding_mask), dim=1)
            y = torch.cat((encoder_out, x), dim=0)
        else:
            # print("decoder not concat")
            y = x


        if self.divide_decoder_self_attn_query:
            x, attn = self.self_attn(
                query=x,
                key=y,
                value=y,
                key_padding_mask=self_attn_padding_mask,
                incremental_state=incremental_state,
                need_weights=False,
                attn_mask=self_attn_mask,
                tgt_type=tgt,
            )
        else:
            x, attn = self.self_attn(
                query=x,
                key=y,
                value=y,
                key_padding_mask=self_attn_padding_mask,
                incremental_state=incremental_state,
                need_weights=False,
                attn_mask=self_attn_mask,
            )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if self.divide_decoder_self_attn_norm:
            x = self.maybe_layer_norm(self.self_attn_layer_norm[self.decoder_output_types_dict[tgt]], x, after=True)
        else:
            x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        if self.encoder_attn is not None and not self.no_train_encoder_attn:
            residual = x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)
            if prev_attn_state is not None:
                if incremental_state is None:
                    incremental_state = {}
                prev_key, prev_value = prev_attn_state[:2]
                saved_state = {"prev_key": prev_key, "prev_value": prev_value}
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)
            # print("before_attn",encoder_out.shape,encoder_out)

            if self.divide_decoder_encoder_attn_query:
                x, attn = self.encoder_attn(
                    query=x,
                    key=encoder_out,
                    value=encoder_out,
                    key_padding_mask=encoder_padding_mask,
                    incremental_state=incremental_state,
                    # attn_mask=encoder_attention_mask,
                    static_kv=True,
                    tgt_type=src,
                )
            else:
                x, attn = self.encoder_attn(
                    query=x,
                    key=encoder_out,
                    value=encoder_out,
                    key_padding_mask=encoder_padding_mask,
                    incremental_state=incremental_state,
                    # attn_mask=encoder_attention_mask,
                    static_kv=True,
                )
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            if self.divide_decoder_encoder_attn_norm:
                x = self.maybe_layer_norm(self.encoder_attn_layer_norm[self.decoder_input_types_dict[src]], x,
                                          after=True)
            else:
                x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if self.divide_decoder_final_norm:
            x = self.maybe_layer_norm(self.final_layer_norm[self.decoder_output_types_dict[tgt]], x, after=True)
        else:
            x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)

        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            if self_attn_padding_mask is not None:
                self_attn_state = saved_state["prev_key"], saved_state["prev_value"], saved_state[
                    "prev_key_padding_mask"]
            else:
                self_attn_state = saved_state["prev_key"], saved_state["prev_value"]
            return x, attn, self_attn_state
        return x, attn

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m

def ids_to_words(ids, temp_dict, final_ids, padding_idx):
    eos_id = temp_dict.eos()
    if (final_ids is not None):
        final_id = final_ids[1]  # torch.nonzero(ids==eos_id).squeeze()
        ids = ids[:final_id]
    ids = ids[:65]
    final_str = (" ".join([temp_dict[t_id] for t_id in ids])).replace(" [SEP] ", " ").replace(" [MASK] ", " ").replace(
        " [PAD] ", " ").replace("[", "").replace("]", "")
    #print(final_str)
    return final_str

def ids_to_clip_token(ids, clip_dict):
    t_device = ids.device
    clip_ids = torch.zeros(ids.size(0), 77).to(t_device).type_as(clip_dict[0])
    start_id = torch.tensor([49406]).to(t_device)
    end_id = torch.tensor([49407]).to(t_device)
    for i in range(ids.size(0)):
        t_ids = torch.cat([clip_dict[int(t_id)] for t_id in ids[i]]).to(t_device)
        if (len(t_ids) > 75):
            t_ids = torch.cat([t_ids[:75], end_id])
        t_ids = torch.cat([start_id, t_ids])[:77]
        clip_ids[i][:len(t_ids)] = t_ids
    return clip_ids


def build_clip_dict(temp_dict):
    clip_dict = {}
    clip_eos = 49407
    for i in range(len(temp_dict.count)):
        t_w = temp_dict[i]
        t_w = t_w.replace("[", "").replace("]", "")
        t_clip_tok = clip.tokenize(t_w)
        end_loc = torch.nonzero(t_clip_tok == clip_eos).squeeze()
        clip_dict[i] = t_clip_tok[0][1:end_loc[1]]
        # print(t_w,clip_dict[i])
    clip_dict[temp_dict.eos()] = torch.tensor([clip_eos])
    clip_dict[temp_dict.pad()] = torch.tensor([0])
    clip_dict[temp_dict.unk()] = torch.tensor([])
    #print(clip_dict)
    return clip_dict


def get_sinusoid_encoding_table_ldpe(n_position, d_hid, padding_idx=None, length=6):
    ''' Sinusoid position encoding table '''

    # print("positions: {}".format(n_position))

    def cal_angle(position, hid_idx):
        return (length - position) / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.from_numpy(sinusoid_table)


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    # print("positions: {}".format(n_position))

    def cal_angle(position, hid_idx):
        return (position) / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.from_numpy(sinusoid_table)


EncoderOut = namedtuple('TransformerEncoderOut', [
    'encoder_out',  # T x B x C
    'encoder_padding_mask',  # B x T
    'encoder_states',  # List[T x B x C]
    'visual_hidden',
])

# ADD: 包含有memory的encoder output
EncoderOutWithMemory = namedtuple('TransformerEncoderOut', [
    'memory_unit',  #
    'encoder_out',  # T x B x C
    'encoder_padding_mask',  # B x T
    'encoder_states',  # List[T x B x C]
])


class TransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """  # 如果添加memory_unit 为true，则返回值包含记忆状态prev_ms

    def __init__(self, args, dictionary, embed_tokens, whether_mix=False):
        # temp_dict = dictionary
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([3]))

        self.dropout = args.dropout
        self.layer_wise_attention = getattr(args, 'layer_wise_attention', False)

        embed_dim = embed_tokens.embedding_dim
        self.embed_dim = embed_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions
        self.whether_use_glove = False
        if (args.whether_use_img_fea == "True"):
            self.embed_tokens = None
        else:
            self.embed_tokens = embed_tokens

        self.embed_scale = math.sqrt(embed_dim)
        if(args.decoder_type=="FactoredLSTM" or args.decoder_type=="DLNLSTM"):
            self.embed_scale = 1
        self.embed_positions = None


        self.embed_positions_sen = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(10, embed_dim, padding_idx=self.padding_idx),
            freeze=True
        )

        self.layers = nn.ModuleList([])

        self.layers.extend([
            TransformerEncoderLayer(
                args.encoder_embed_dim,
                args.encoder_ffn_embed_dim,
                args.encoder_attention_heads,
                args.dropout,
                args.attention_dropout,
                args.activation_dropout,
                args.activation_fn,
            )
            for i in range(args.encoder_layers)
        ])


        self.emb_layer_norm = LayerNorm(embed_dim)

        self.style_layers = None
        if(whether_mix):
            self.style_layers = nn.ModuleList([])

            self.style_layers.extend([
                TransformerEncoderLayer(
                    args.encoder_embed_dim,
                    args.encoder_ffn_embed_dim,
                    args.encoder_attention_heads,
                    args.dropout,
                    args.attention_dropout,
                    args.activation_dropout,
                    args.activation_fn,
                )
                for i in range(args.encoder_layers)
            ])

            self.style_emb_layer_norm = LayerNorm(embed_dim)
        self.apply(init_bert_params)

    def forward(self, src_tokens, src_sen_pos, src_lengths, img_features, text_features, return_all_hiddens=False, lang_pair=None, **unused):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # if applying text embedding, the embed_layer will be different  from these for images
        if(lang_pair is not None and lang_pair.split("-")[0]!="src" and self.style_layers is not None):
            text_src = True
        else:
            text_src =False
        if self.layer_wise_attention:
            return_all_hiddens = True

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # embed tokens and positions

        if (img_features is not None):
            #print(img_features)
            encoder_padding_mask = None

            x = self.embed_scale * img_features

            token_num = int(x.size(1) / 5)
            sen_pos_num = int(src_sen_pos.size(1) / 5)
            src_sen_pos = torch.cat(
                [src_sen_pos[:, sen_pos_num * i:sen_pos_num * (i) + token_num] for i in range(5)], dim=1)
            #print("src_sen_pos",src_sen_pos)
        else:
            x = self.embed_scale * self.embed_tokens(src_tokens)


        sen_pos_embed = self.embed_positions_sen(src_sen_pos)
        x = x.type_as(sen_pos_embed[0][0])

        frozen = False

        x += sen_pos_embed
        x = x.to(torch.float32)

        visual_hidden = x.transpose(0, 1).clone()

        if self.emb_layer_norm:
            if(frozen):
                with torch.no_grad():
                    if (text_src):
                        x = self.style_emb_layer_norm(x)
                    else:
                        x = self.emb_layer_norm(x)
            else:
                if (text_src):
                    x = self.style_emb_layer_norm(x)
                else:
                    x = self.emb_layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        if encoder_padding_mask is not None:
            x *= 1 - encoder_padding_mask.unsqueeze(-1).type_as(x)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = [] if return_all_hiddens else None

        # encoder layers

        if(text_src):
            for layer in self.style_layers:
                #print(lang_pair)
                x, _ = layer(x, self_attn_padding_mask=encoder_padding_mask)
                if return_all_hiddens:
                    encoder_states.append(x)
        else:
            if(frozen):
                with torch.no_grad():
                    for layer in self.layers:
                        x, _ = layer(x, self_attn_padding_mask=encoder_padding_mask)
                        if return_all_hiddens:
                            encoder_states.append(x)
            else:
                for layer in self.layers:
                    x, _ = layer(x, self_attn_padding_mask=encoder_padding_mask)
                    if return_all_hiddens:
                        encoder_states.append(x)

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_states=encoder_states,  # List[T x B x C]
            visual_hidden=visual_hidden,
        )

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out.encoder_out is not None:
            encoder_out = encoder_out._replace(
                encoder_out=encoder_out.encoder_out.index_select(1, new_order)
            )
        if encoder_out.visual_hidden is not None:
            encoder_out = encoder_out._replace(
                visual_hidden=encoder_out.visual_hidden.index_select(1, new_order)
            )
        if encoder_out.encoder_padding_mask is not None:
            encoder_out = encoder_out._replace(
                encoder_padding_mask=encoder_out.encoder_padding_mask.index_select(0, new_order)
            )
        if encoder_out.encoder_states is not None:
            for idx, state in enumerate(encoder_out.encoder_states):
                encoder_out.encoder_states[idx] = state.index_select(1, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())


class TransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False, no_train_encoder_attn=False):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([3]))

        self.dropout = args.dropout
        self.decoder_layerdrop = args.decoder_layerdrop
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.output_embed_dim = args.decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.project_in_dim = Linear(input_embed_dim, embed_dim, bias=False) if embed_dim != input_embed_dim else None

        max_target_positions = 100
        self.embed_positions = LearnedPositionalEmbedding(
            max_target_positions + 1 + self.padding_idx, embed_dim, self.padding_idx,
        )

        self.cross_self_attention = getattr(args, 'cross_self_attention', False)
        self.layer_wise_attention = getattr(args, 'layer_wise_attention', False)

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(args, no_encoder_attn, no_train_encoder_attn=no_train_encoder_attn)
            for _ in range(args.decoder_layers)
        ])

        self.adaptive_softmax = None

        self.project_out_dim = Linear(embed_dim, self.output_embed_dim, bias=False) \
            if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights else None

        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), self.output_embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=self.output_embed_dim ** -0.5)

        if args.decoder_normalize_before and not getattr(args, 'no_decoder_final_norm', False):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None
        if getattr(args, 'layernorm_embedding', False):
            self.emb_layer_norm = LayerNorm(embed_dim)
        else:
            self.emb_layer_norm = None

    def forward(
            self,
            prev_output_tokens,
            encoder_out=None,
            incremental_state=None,
            features_only=False,
            **extra_args
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            **extra_args
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
            self,
            prev_output_tokens,
            prev_sen_pos,
            encoder_out=None,
            incremental_state=None,
            full_context_alignment=False,
            alignment_layer=None,
            alignment_heads=None,
            **unused,
    ):
        """
        Similar to *forward* but only return features.
        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).
        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).
        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        if alignment_layer is None:
            alignment_layer = len(self.layers) - 1

        # embed positions
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.glove_embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.emb_layer_norm:
            x = self.emb_layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn = None
        inner_states = [x]
        for idx, layer in enumerate(self.layers):
            encoder_state = None
            if encoder_out is not None:
                if self.layer_wise_attention:
                    encoder_state = encoder_out.encoder_states[idx]
                else:
                    encoder_state = encoder_out.encoder_out

            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if not self.training or (dropout_probability > self.decoder_layerdrop):
                x, layer_attn = layer(
                    x,
                    encoder_state,
                    encoder_out.encoder_padding_mask if encoder_out is not None else None,
                    incremental_state,
                    self_attn_mask=self_attn_mask,
                    self_attn_padding_mask=self_attn_padding_mask,
                )
                inner_states.append(x)
                if layer_attn is not None and idx == alignment_layer:
                    attn = layer_attn.float()

        # if attn is not None:
        #     if alignment_heads is not None:
        #         attn = attn[:alignment_heads]
        #
        #     average probabilities over heads
        # attn = attn.mean(dim=0)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {'attn': attn, 'inner_states': inner_states}

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                return F.linear(features, self.embed_tokens.weight)
            else:
                return F.linear(features, self.embed_out)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions())

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
                not hasattr(self, '_future_mask')
                or self._future_mask is None
                or self._future_mask.device != tensor.device
                or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)

        for i in range(len(self.layers)):
            # update layer norms
            layer_norm_map = {
                '0': 'self_attn_layer_norm',
                '1': 'encoder_attn_layer_norm',
                '2': 'final_layer_norm'
            }
            for old, new in layer_norm_map.items():
                for m in ('weight', 'bias'):
                    k = '{}.layers.{}.layer_norms.{}.{}'.format(name, i, old, m)
                    if k in state_dict:
                        state_dict['{}.layers.{}.{}.{}'.format(name, i, new, m)] = state_dict[k]
                        del state_dict[k]

        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict


class TransformerMixDecoder(TransformerDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False, no_train_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn, no_train_encoder_attn)
        self.register_buffer('version', torch.Tensor([3]))

        self.dropout = args.dropout
        self.dictionary = dictionary
        self.decoder_layerdrop = args.decoder_layerdrop
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.output_embed_dim = args.decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.project_in_dim = Linear(input_embed_dim, embed_dim, bias=False) if embed_dim != input_embed_dim else None


        self.embed_positions = LearnedPositionalEmbedding(
            args.max_target_positions + 1 + self.padding_idx, embed_dim, self.padding_idx,
        )
        # for position embedding
        self.embed_positions_sen = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(10, embed_dim, padding_idx=self.padding_idx),
            freeze=True
        )

        self.cross_self_attention = getattr(args, 'cross_self_attention', False)
        self.layer_wise_attention = getattr(args, 'layer_wise_attention', False)

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerMixDecoderLayer(args, no_encoder_attn, no_train_encoder_attn=no_train_encoder_attn)
            for _ in range(args.decoder_layers)
        ])

        self.adaptive_softmax = None

        self.project_out_dim = Linear(embed_dim, self.output_embed_dim, bias=False) \
            if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights else None

        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), self.output_embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=self.output_embed_dim ** -0.5)

        if args.decoder_normalize_before and not getattr(args, 'no_decoder_final_norm', False):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        # check how many norms we need
        decoder_output_types, decoder_input_types = [], []
        decoder_output_types += [item.split('-')[-1] for item in args.model_lang_pairs]
        decoder_input_types += [item.split('-')[0] for item in args.model_lang_pairs]
        self.decoder_output_types_dict = {type: idx for idx, type in enumerate(decoder_output_types)}
        self.decoder_input_types_dict = {type: idx for idx, type in enumerate(decoder_input_types)}

        self.divide_decoder_embed_norm = options.eval_bool(args.divide_decoder_embed_norm)
        if getattr(args, 'layernorm_embedding', False):
            if self.divide_decoder_embed_norm:
                self.emb_layer_norm = nn.ModuleList([LayerNorm(embed_dim)
                                                     for _ in range(len(decoder_output_types))])
            else:
                self.emb_layer_norm = LayerNorm(embed_dim)
        else:
            self.emb_layer_norm = None

        if getattr(args, 'style_cls', 0):
            self.style_cls = nn.Linear(embed_dim, 2)
        else:
            self.style_cls = None

    def forward(
            self,
            prev_output_tokens,
            prev_sen_pos,
            encoder_out=None,
            incremental_state=None,
            features_only=False,
            lang_pair=None,
            **extra_args
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        # print(prev_output_tokens)
        # print(encoder_out)
        x, extra = self.extract_features(
            prev_output_tokens,
            prev_sen_pos,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            lang_pair=lang_pair,
            **extra_args
        )
        if not features_only:
            x = self.output_layer(x)
        #print("x",x.shape)
        return x, extra

    # ADD: 将bi-gru的输出进行拼接

    def extract_features(
            self,
            prev_output_tokens,
            prev_sen_pos,
            encoder_out=None,
            incremental_state=None,
            full_context_alignment=False,
            alignment_layer=None,
            alignment_heads=None,
            lang_pair=None,
            **unused,
    ):
        """
        Similar to *forward* but only return features.
        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).
        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).
        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        if alignment_layer is None:
            alignment_layer = len(self.layers) - 1

        # ADD: 根据之前的输出构建prev_sen_pos数组
        positions = None

        positions_orig = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        )

        if incremental_state is not None:
            # print(positions_orig.shape)
            positions_tmp = positions_orig[:, -1:]
            # print(positions_tmp.shape)
        else:
            positions_tmp = positions_orig
        positions = positions + positions_tmp if positions is not None else positions_tmp


        # ADD:

        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions


        src, tgt = lang_pair.split('-')
        if self.emb_layer_norm:
            if self.divide_decoder_embed_norm:
                x = self.emb_layer_norm[self.decoder_output_types_dict[tgt]](x)
            else:
                x = self.emb_layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask = None
        # print(prev_output_tokens)
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn = None
        inner_states = [x]
        for idx, layer in enumerate(self.layers):
            encoder_state = None
            if encoder_out is not None:
                if self.layer_wise_attention:
                    encoder_state = encoder_out.encoder_states[idx]
                else:
                    encoder_state = encoder_out.encoder_out

            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if not self.training or (dropout_probability > self.decoder_layerdrop):
                x, layer_attn = layer(
                    x,
                    encoder_state,
                    encoder_out.encoder_padding_mask if encoder_out is not None else None,
                    incremental_state,
                    self_attn_mask=self_attn_mask,
                    self_attn_padding_mask=self_attn_padding_mask,
                    lang_pair=lang_pair,
                )
                inner_states.append(x)
                if layer_attn is not None and idx == alignment_layer:
                    attn = layer_attn.float()

        # if attn is not None:
        #     if alignment_heads is not None:
        #         attn = attn[:alignment_heads]
        #
        #     average probabilities over heads
        # attn = attn.mean(dim=0)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        # get the style classification prob
        if self.style_cls is not None:
            style_prob = self.style_cls(x.max(dim=1)[0])
        else:
            style_prob = None

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {'attn': attn, 'inner_states': inner_states, 'style_prob': style_prob}


# memory-augmented decoder
class TransformerMixDecoderWithMemory(TransformerDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False, no_train_encoder_attn=False):
        # temp_dict = dictionary
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn, no_train_encoder_attn)
        self.register_buffer('version', torch.Tensor([3]))

        self.dropout = args.dropout
        self.dictionary = dictionary
        self.decoder_layerdrop = args.decoder_layerdrop
        self.share_input_output_embed = args.share_decoder_input_output_embed
        self.clip_id_dict = None

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.output_embed_dim = args.decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens
        # 若使用glove embed
        self.whether_use_glove = False
        self.clip_encode_text = None
        if(args.whether_read_text_clip_fea!="True"):
            clip_model, _ = clip.load("ViT-B/32")
            self.clip_encode_text = clip_model.encode_text
            print("load clip model")


        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.project_in_dim = Linear(input_embed_dim, embed_dim, bias=False) if embed_dim != input_embed_dim else None

        self.embed_positions = LearnedPositionalEmbedding(
            args.max_target_positions + 1 + self.padding_idx, embed_dim, self.padding_idx,
        )

        # ADD: memory_state的初始化和更新
        self.memory_update = MemoryUpdater(args)
        self.memory_initialize = MemoryInitializer(args)
        # self.output = ResidualOutput(args)

        self.memory_augmented_attention = MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=False,
            add_zero_attn=False,
            self_attention=True,
        )

        self.memory_projection = Linear(embed_dim, self.output_embed_dim, bias=True)
        self.hidden_dense = Linear(embed_dim, self.output_embed_dim, bias=True)
        self.embed_positions_sen = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(10, embed_dim, padding_idx=self.padding_idx),
            freeze=True
        )

        self.cross_self_attention = getattr(args, 'cross_self_attention', False)
        self.layer_wise_attention = getattr(args, 'layer_wise_attention', False)

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerMixDecoderLayer(args, no_encoder_attn, no_train_encoder_attn=no_train_encoder_attn)
            for _ in range(args.decoder_layers)
        ])

        self.adaptive_softmax = None

        self.project_out_dim = Linear(embed_dim, self.output_embed_dim, bias=False) \
            if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights else None

        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), self.output_embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=self.output_embed_dim ** -0.5)

        if args.decoder_normalize_before and not getattr(args, 'no_decoder_final_norm', False):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None
        self.memory_layer_norm = LayerNorm(embed_dim)
        self.sentence_layer_norm = LayerNorm(embed_dim)

        # check how many norms we need
        decoder_output_types, decoder_input_types = [], []
        decoder_output_types += [item.split('-')[-1] for item in args.model_lang_pairs]
        decoder_input_types += [item.split('-')[0] for item in args.model_lang_pairs]
        self.decoder_output_types_dict = {type: idx for idx, type in enumerate(decoder_output_types)}
        self.decoder_input_types_dict = {type: idx for idx, type in enumerate(decoder_input_types)}

        self.divide_decoder_embed_norm = options.eval_bool(args.divide_decoder_embed_norm)
        self.model_transfer_proj = Linear(embed_dim, embed_dim, bias=True)

        self.activation_fn_style = nn.ModuleList([LayerNorm(embed_dim, export=False)
                                                   for _ in range(len(decoder_output_types))])
        self.activation_fn = LayerNorm(embed_dim)
        self.whether_clip_encode = True
        self.token_type_embeddings = nn.Embedding(2, embed_dim)
        if(args.whether_clip_encode=="False"):
            self.whether_clip_encode = False
            print("use token embeding to update memory")
        if getattr(args, 'layernorm_embedding', False):
            if self.divide_decoder_embed_norm:
                self.emb_layer_norm = nn.ModuleList([LayerNorm(embed_dim)
                                                     for _ in range(len(decoder_output_types))])
            else:
                self.emb_layer_norm = LayerNorm(embed_dim)
        else:
            self.emb_layer_norm = None

        if getattr(args, 'style_cls', 0):
            self.style_cls = nn.Linear(embed_dim, 2)
        else:
            self.style_cls = None

    def forward(
            self,
            prev_output_tokens,
            prev_sen_pos,
            visual_hidden=None,
            temp_sen_id=0,
            encoder_out=None,
            incremental_state=None,
            features_only=False,
            lang_pair=None,
            prev_ms=None,
            last_sen=None,
            whether_update=True,
            text_features=None,
            **extra_args
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        # ADD:
        # 如果考虑memory_unit，则根据之前的memory和之前的图片句子信息进行memory更新
        encoder_padding_mask = encoder_out.encoder_padding_mask
        visual_padding_mask = encoder_out.encoder_padding_mask

        #改为encoder_out
        t_encoder_out = encoder_out.visual_hidden
        visual_encoder_out = encoder_out.encoder_out
        t_device = prev_output_tokens.device

        no_memory = False

        if no_memory==True:
            if (len(prev_ms) == 2):
                prev_ms = self.memory_initialize(t_encoder_out[:visual_encoder_out.size(0), :, :].transpose(0, 1),
                                                 None)
                # print("prev_ms",prev_ms.shape)
                prev_ms.to(t_device)
            x, extra = self.extract_features(
                prev_output_tokens,
                prev_sen_pos,
                encoder_out=encoder_out,
                incremental_state=incremental_state,
                lang_pair=lang_pair,
                prev_ms=prev_ms,
                temp_sen_id=temp_sen_id,
                **extra_args
            )
            if not features_only:
                x = self.output_layer(x)
            return x, extra, prev_ms

        if (last_sen is not None and whether_update == True):
            # print("temp_sen_id",temp_sen_id)
            whether_clip_encode = self.whether_clip_encode
            #whether_clip_encode = False
            if (whether_clip_encode == False):
                last_sen_encode = self.embed_scale * self.embed_tokens(last_sen)
            else:
                if(text_features is None):
                    #print("use clip model")
                    eos_id = self.dictionary.eos()
                    final_ids = torch.nonzero(last_sen == eos_id).squeeze()
                    # print(last_sen)
                    # print(final_ids)\
                    eos_num = int(len(final_ids) / last_sen.size(0))
                    with torch.no_grad():
                        try:
                            text = clip.tokenize([ids_to_words(last_sen[t_sen_i], self.dictionary,
                                                               final_ids[t_sen_i * eos_num], self.padding_idx) for t_sen_i
                                                  in range(last_sen.size(0))]).to(t_device)
                        except:
                            text = clip.tokenize([ids_to_words(last_sen[t_sen_i], self.dictionary,
                                                               None, self.padding_idx) for t_sen_i
                                                  in range(last_sen.size(0))]).to(t_device)
                        if(self.clip_encode_text is None):
                            clip_model, _ = clip.load("ViT-B/32")
                            self.clip_encode_text = clip_model.encode_text
                            print("load clip model")
                        clip_text_features = self.clip_encode_text(text)
                        text_features = clip_text_features
                last_sen_encode = self.embed_scale * text_features.unsqueeze(1)

            last_sen_padding = last_sen.eq(self.padding_idx)

            if (whether_clip_encode):
                encoder_padding_mask = None
            else:
                if (encoder_padding_mask is None):
                    encoder_padding_mask = torch.BoolTensor(t_encoder_out.size(1), t_encoder_out.size(0)).fill_(0).to(
                        t_device)
                encoder_padding_mask = torch.cat((encoder_padding_mask, last_sen_padding), dim=1)

            last_sen_pos = torch.full((last_sen_encode.size(0), last_sen_encode.size(1)), temp_sen_id).to(t_device)
            if(whether_clip_encode==False):
                last_sen_pos = last_sen_pos.masked_fill(
                    last_sen_padding,
                    float('0'),
                )
            last_sen_pos = last_sen_pos.type(torch.LongTensor).to(t_device)

            last_sen_encode = last_sen_encode.to(torch.float32)

            last_sen_encode += self.embed_positions_sen(last_sen_pos)

            last_sen_encode = last_sen_encode.transpose(0, 1)

            t_encoder_out = torch.cat((t_encoder_out, last_sen_encode), dim=0)

        t_encoder_out = self.model_transfer_proj(t_encoder_out)
        t_encoder_out = self.activation_fn(t_encoder_out)
        if (prev_ms == None):
            prev_ms = self.memory_initialize(t_encoder_out.transpose(0, 1), encoder_padding_mask)
            prev_ms.to(t_device)
            print("use t_encoder to initial")
        if (len(prev_ms) == 2):
            #初始化时只看到了visual信息
            initial_padding_mask=None
            if(visual_padding_mask is not None):
                bsz = visual_encoder_out.size(1)
                initial_padding_mask = torch.ones(bsz, visual_encoder_out.size(0)).to(t_device)
                initial_padding_mask = initial_padding_mask.masked_fill(
                    visual_padding_mask,
                    0,
                )

            prev_ms = self.memory_initialize(t_encoder_out[:visual_encoder_out.size(0), :, :].transpose(0, 1), initial_padding_mask)
            prev_ms.to(t_device)

        ms = prev_ms
        if (whether_update == True):
            ms = self.memory_update(prev_ms.transpose(0, 1), t_encoder_out,
                                    attention_mask=encoder_padding_mask).transpose(0, 1)

        ms.to(t_device)

        inter_encoder_out = t_encoder_out[:visual_encoder_out.size(0), :, :]

        concat_mh = torch.cat([prev_ms.transpose(0, 1), inter_encoder_out],
                              dim=0)  # [M+L, N,  Di]
        memory_attention_mask = None
        if (visual_padding_mask is not None):
            memory_attention_mask = torch.cat((visual_padding_mask[:, 0:1], visual_padding_mask), dim=1)


        memory_attention_output, _ = self.memory_augmented_attention(
            query=inter_encoder_out,
            key=concat_mh,
            value=concat_mh,
            key_padding_mask=memory_attention_mask,
            incremental_state=None,
            need_weights=False,
            attn_mask=None,
        )

        memory_attention_output = self.memory_projection(memory_attention_output)  # (N, L, Di) -> (N, L, D)'''
        # print("memory_attention_output",memory_attention_output.shape)
        #memory_attention_output = self.hidden_dense(memory_attention_output)
        memory_attention_output = F.dropout(memory_attention_output, p=self.dropout, training=self.training)
        #memory_attention_output =
        layer_output = self.memory_layer_norm(memory_attention_output + visual_encoder_out)

        augmented_encoder_out = EncoderOut(
            encoder_out=layer_output,  # T x B x C
            encoder_padding_mask=None,  # B x T
            encoder_states=None,  # List[T x B x C]
            visual_hidden=encoder_out.visual_hidden,
        )

        x, extra = self.extract_features(
            prev_output_tokens,
            prev_sen_pos,
            encoder_out=augmented_encoder_out,
            incremental_state=incremental_state,
            lang_pair=lang_pair,
            prev_ms=prev_ms,
            temp_sen_id=temp_sen_id,
            **extra_args
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra, ms

    # ADD: 将bi-gru的输出进行拼接

    def extract_features(
            self,
            prev_output_tokens,
            prev_sen_pos,
            encoder_out=None,
            incremental_state=None,
            full_context_alignment=False,
            alignment_layer=None,
            alignment_heads=None,
            lang_pair=None,
            prev_ms=None,
            temp_sen_id=0,
            **unused,
    ):
        """
        Similar to *forward* but only return features.
        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).
        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).
        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        t_device = prev_output_tokens.device

        if alignment_layer is None:
            alignment_layer = len(self.layers) - 1

        prev_sen_padding = prev_output_tokens.eq(self.padding_idx)

        prev_sen_pos = torch.full((prev_output_tokens.size(0), prev_output_tokens.size(1)), temp_sen_id).to(t_device)
        prev_sen_pos = prev_sen_pos.masked_fill(
            prev_sen_padding,
            float('0'),
        )
        prev_sen_pos = prev_sen_pos.type(torch.LongTensor).to(t_device)

        # print(prev_sen_pos)
        positions = self.embed_positions_sen(
            prev_sen_pos
        )
        if incremental_state is not None:
            print("incremental")
            positions = positions[:, -1:]
        else:
            positions = positions


        positions_orig = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        )

        if incremental_state is not None:
            # print(positions_orig.shape)
            positions_tmp = positions_orig[:, -1:]
            # print(positions_tmp.shape)
        else:
            positions_tmp = positions_orig
        positions = positions + positions_tmp if positions is not None else positions_tmp
        # print(len(prev_sen_pos[0]))
        # print(len(prev_sen_pos))

        # ADD:

        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)


        if positions is not None:
            x += positions


        src, tgt = lang_pair.split('-')
        if self.emb_layer_norm:
            if self.divide_decoder_embed_norm:
                x = self.emb_layer_norm[self.decoder_output_types_dict[tgt]](x)
            else:
                x = self.emb_layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask = None
        # print(prev_output_tokens)
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn = None
        inner_states = [x]
        for idx, layer in enumerate(self.layers):
            encoder_state = None
            encoder_mask = None
            if encoder_out is not None:

                encoder_state = encoder_out.encoder_out
                encoder_mask = encoder_out.encoder_padding_mask

            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if not self.training or (dropout_probability > self.decoder_layerdrop):
                x, layer_attn = layer(
                    x,
                    encoder_state,
                    encoder_mask,
                    incremental_state,
                    self_attn_mask=self_attn_mask,
                    self_attn_padding_mask=self_attn_padding_mask,
                    lang_pair=lang_pair,
                )
                # ADD：计算memory_augmented x

                inner_states.append(x)
                if layer_attn is not None and idx == alignment_layer:
                    attn = layer_attn.float()


        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        # get the style classification prob
        if self.style_cls is not None:
            style_prob = self.style_cls(x.max(dim=1)[0])
        else:
            style_prob = None

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {'attn': attn, 'inner_states': inner_states, 'style_prob': style_prob}


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


@register_model_architecture('transformer_mass', 'transformer_mass')
def base_architecture(args):
    args.activation_fn = getattr(args, 'activation_fn', 'relu')
    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.)

    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)

    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 2048)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)  ## added for EXPR

    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)

    args.no_scale_embedding = getattr(args, 'no_scale_embedding', False)  ## added for EXPR
    args.layernorm_embedding = getattr(args, 'layernorm_embedding', True)  ## added for EXPR
    args.decoder_normalize_before = False  ## added for EXPR
    args.no_decoder_final_norm = getattr(args, 'no_decoder_final_norm', True)

    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.adaptive_softmax_factor = getattr(args, 'adaptive_softmax_factor', 4)
    args.adaptive_input = getattr(args, 'adaptive_input', False)
    args.adaptive_input_factor = getattr(args, 'adaptive_input_factor', 4)
    args.adaptive_input_cutoff = getattr(args, 'adaptive_input_cutoff', None)
    args.tie_adaptive_weights = getattr(args, 'tie_adaptive_weights', False)
    args.tie_adaptive_proj = getattr(args, 'tie_adaptive_proj', False)

@register_model_architecture('transformer_mass', 'transformer_mass_base')
@register_model_architecture('transformer_mix', 'transformer_mix_base')
def transformer_base(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 3072)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 12)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)

    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.1)
    args.activation_fn = getattr(args, 'activation_fn', 'gelu')

    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 768)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 3072)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 12)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)

    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    base_architecture(args)


@register_model_architecture('transformer_mass', 'transformer_mass_middle')
def transformer_middle(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)

    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    transformer_base(args)


@register_model_architecture('transformer_mass', 'transformer_mass_big')
def transformer_big(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 12)
    args.decoder_layers = getattr(args, 'decoder_layers', 12)
    transformer_middle(args)

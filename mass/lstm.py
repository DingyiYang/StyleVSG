import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from fairseq.models import (
    FairseqIncrementalDecoder,
)

from fairseq.modules import (
    LayerNorm,
)


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


class FactoredLSTM(FairseqIncrementalDecoder):
    def __init__(self, args, dictionary, embed_tokens):
        #super(FactoredLSTM, self).__init__()
        super().__init__(dictionary)
        hidden_dim = embed_tokens.embedding_dim
        emb_dim = embed_tokens.embedding_dim
        factored_dim = hidden_dim
        vocab_size = len(dictionary)
        self.padding_idx = embed_tokens.padding_idx
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.embed_scale = math.sqrt(emb_dim)

        # embedding\
        self.B = nn.Linear(factored_dim, hidden_dim)
        self.embed_tokens = embed_tokens #nn.Embedding(vocab_size, emb_dim)
        self.mode_transfer = nn.Linear(factored_dim, hidden_dim)

        self.embed_positions_sen = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(10, hidden_dim, padding_idx=self.padding_idx),
            freeze=True
        )

        # factored lstm weights
        self.U_i = nn.Linear(factored_dim, hidden_dim)
        self.S_fi = nn.Linear(factored_dim, factored_dim)
        self.V_i = nn.Linear(emb_dim, factored_dim)
        self.W_i = nn.Linear(hidden_dim, hidden_dim)

        self.U_f = nn.Linear(factored_dim, hidden_dim)
        self.S_ff = nn.Linear(factored_dim, factored_dim)
        self.V_f = nn.Linear(emb_dim, factored_dim)
        self.W_f = nn.Linear(hidden_dim, hidden_dim)

        self.U_o = nn.Linear(factored_dim, hidden_dim)
        self.S_fo = nn.Linear(factored_dim, factored_dim)
        self.V_o = nn.Linear(emb_dim, factored_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)

        self.U_c = nn.Linear(factored_dim, hidden_dim)
        self.S_fc = nn.Linear(factored_dim, factored_dim)
        self.V_c = nn.Linear(emb_dim, factored_dim)
        self.W_c = nn.Linear(hidden_dim, hidden_dim)

        self.S_hi = nn.Linear(factored_dim, factored_dim)
        self.S_hf = nn.Linear(factored_dim, factored_dim)
        self.S_ho = nn.Linear(factored_dim, factored_dim)
        self.S_hc = nn.Linear(factored_dim, factored_dim)

        # self.S_ri = nn.Linear(factored_dim, factored_dim)
        # self.S_rf = nn.Linear(factored_dim, factored_dim)
        # self.S_ro = nn.Linear(factored_dim, factored_dim)
        # self.S_rc = nn.Linear(factored_dim, factored_dim)

        # weight for output
        self.C = nn.Linear(hidden_dim, vocab_size)

    def forward_step(self, embedded, h_0, c_0, mode):
        i = self.V_i(embedded)
        f = self.V_f(embedded)
        o = self.V_o(embedded)
        c = self.V_c(embedded)

        if mode == "factual":
            i = self.S_fi(i)
            f = self.S_ff(f)
            o = self.S_fo(o)
            c = self.S_fc(c)
        elif mode == "stylized":
            i = self.S_hi(i)
            f = self.S_hf(f)
            o = self.S_ho(o)
            c = self.S_hc(c)
        # elif mode == "romantic":
        #     i = self.S_ri(i)
        #     f = self.S_rf(f)
        #     o = self.S_ro(o)
        #     c = self.S_rc(c)
        else:
            sys.stderr.write("mode name wrong!")

        i_t = F.sigmoid(self.U_i(i) + self.W_i(h_0))
        f_t = F.sigmoid(self.U_f(f) + self.W_f(h_0))
        o_t = F.sigmoid(self.U_o(o) + self.W_o(h_0))
        c_tilda = F.tanh(self.U_c(c) + self.W_c(h_0))

        c_t = f_t * c_0 + i_t * c_tilda
        h_t = o_t * c_t

        outputs = self.C(h_t)

        return outputs, h_t, c_t

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
        #   def forward(self, captions, features=None, mode="factual"):
        '''
        Args:
            features: fixed vectors from images, [batch, emb_dim]
            captions: [batch, max_len]
            mode: type of caption to generate
        '''
        t_device = prev_output_tokens.device
        temp_sen_id = prev_sen_pos
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

        captions = prev_output_tokens
        batch_size = captions.size(0)
        embedded = self.embed_tokens(captions)#self.B(self.embed_tokens(captions))  # [batch, max_len, emb_dim]
        #embedded += positions
        features = encoder_out.encoder_out.transpose(0,1)#self.mode_transfer(encoder_out.encoder_out.transpose(0,1))
        src, tgt = lang_pair.split("-")
        mode = "factual"
        if(tgt!="tgt"):mode = "stylized"
        # concat features and captions
        if src == "src":
            #print(mode)
            #print(lang_pair)
            if features is None:
                sys.stderr.write("features is None!")
            embedded = torch.cat((features, embedded), 1)


        # initialize hidden state
        h_t = Variable(torch.Tensor(batch_size, self.hidden_dim))
        c_t = Variable(torch.Tensor(batch_size, self.hidden_dim))
        nn.init.uniform(h_t)
        nn.init.uniform(c_t)

        if torch.cuda.is_available():
            h_t = h_t.cuda()
            c_t = c_t.cuda()

        all_outputs = []
        # iterate
        for ix in range(embedded.size(1)):
            emb = embedded[:, ix, :]
            outputs, h_t, c_t = self.forward_step(emb, h_t, c_t, mode=mode)
            all_outputs.append(outputs)

        all_outputs = torch.stack(all_outputs, 1)
        if(src == "src"):
            all_outputs = all_outputs[:,1:,:]
        #print("all_outputs", all_outputs.shape)
        extra = {'attn': None, 'inner_states': None, 'style_prob': None}

        return all_outputs, extra

class DLNLSTM(FairseqIncrementalDecoder):
    def __init__(self, args, dictionary, embed_tokens):
        #super(FactoredLSTM, self).__init__()
        super().__init__(dictionary)
        hidden_dim = embed_tokens.embedding_dim
        emb_dim = embed_tokens.embedding_dim
        factored_dim = hidden_dim
        vocab_size = len(dictionary)
        self.padding_idx = embed_tokens.padding_idx
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.embed_scale = math.sqrt(emb_dim)

        # embedding
        self.B = embed_tokens #nn.Embedding(vocab_size, emb_dim)

        self.embed_positions_sen = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(10, hidden_dim, padding_idx=self.padding_idx),
            freeze=True
        )

        # factored lstm weights
        self.f_transfer = nn.Linear(factored_dim, hidden_dim)
        self.s_transfer = nn.Linear(factored_dim, hidden_dim)
        self.U_i = nn.Linear(factored_dim, hidden_dim)
        self.S_fi = nn.Linear(factored_dim, factored_dim)
        self.V_i = nn.Linear(emb_dim, factored_dim)
        self.W_i = nn.Linear(hidden_dim, hidden_dim)

        self.U_f = nn.Linear(factored_dim, hidden_dim)
        self.S_ff = nn.Linear(factored_dim, factored_dim)
        self.V_f = nn.Linear(emb_dim, factored_dim)
        self.W_f = nn.Linear(hidden_dim, hidden_dim)

        self.U_o = nn.Linear(factored_dim, hidden_dim)
        self.S_fo = nn.Linear(factored_dim, factored_dim)
        self.V_o = nn.Linear(emb_dim, factored_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)

        self.U_c = nn.Linear(factored_dim, hidden_dim)
        self.S_fc = nn.Linear(factored_dim, factored_dim)
        self.V_c = nn.Linear(emb_dim, factored_dim)
        self.W_c = nn.Linear(hidden_dim, hidden_dim)

        export = False
        #factual的layer_norm
        self.ln_f_i = LayerNorm(emb_dim, export=False)
        self.ln_f_j = LayerNorm(emb_dim, export=False)
        self.ln_f_f = LayerNorm(emb_dim, export=False)
        self.ln_f_o = LayerNorm(emb_dim, export=False)
        self.ln_f_c = LayerNorm(emb_dim, export=False)

        # style的layer_norm
        self.ln_s_i = LayerNorm(emb_dim, export=False)
        self.ln_s_j = LayerNorm(emb_dim, export=False)
        self.ln_s_f = LayerNorm(emb_dim, export=False)
        self.ln_s_o = LayerNorm(emb_dim, export=False)
        self.ln_s_c = LayerNorm(emb_dim, export=False)

        # self.S_hi = nn.Linear(factored_dim, factored_dim)
        # self.S_hf = nn.Linear(factored_dim, factored_dim)
        # self.S_ho = nn.Linear(factored_dim, factored_dim)
        # self.S_hc = nn.Linear(factored_dim, factored_dim)

        # self.S_ri = nn.Linear(factored_dim, factored_dim)
        # self.S_rf = nn.Linear(factored_dim, factored_dim)
        # self.S_ro = nn.Linear(factored_dim, factored_dim)
        # self.S_rc = nn.Linear(factored_dim, factored_dim)

        # weight for output
        self.C = nn.Linear(hidden_dim, vocab_size)

    def forward_step(self, embedded, h_0, c_0, mode):
        '''
        i = self.V_i(embedded)
        f = self.V_f(embedded)
        o = self.V_o(embedded)
        c = self.V_c(embedded)

        i = self.S_fi(i)
        f = self.S_ff(f)
        o = self.S_fo(o)
        c = self.S_fc(c)


        if mode == "factual":
            i = self.S_fi(i)
            f = self.S_ff(f)
            o = self.S_fo(o)
            c = self.S_fc(c)
        elif mode == "stylized":
            i = self.S_hi(i)
            f = self.S_hf(f)
            o = self.S_ho(o)
            c = self.S_hc(c)
        # elif mode == "romantic":
        #     i = self.S_ri(i)
        #     f = self.S_rf(f)
        #     o = self.S_ro(o)
        #     c = self.S_rc(c)
        else:
            sys.stderr.write("mode name wrong!")
        '''

        i = self.U_i(embedded) + self.W_i(h_0)
        f = self.U_f(embedded) + self.W_f(h_0)
        o = self.U_o(embedded) + self.W_o(h_0)
        c = self.U_c(embedded) + self.W_c(h_0)


        if(mode=="factual"):
            i = self.ln_f_i(i)
            f = self.ln_f_f(f)
            o = self.ln_f_o(o)
            c = self.ln_f_j(c)
        elif(mode=="stylized"):
            i = self.ln_s_i(i)
            f = self.ln_s_f(f)
            o = self.ln_s_o(o)
            c = self.ln_s_j(c)
        else:
            print("mode wrong")

        i_t = F.sigmoid(i)
        f_t = F.sigmoid(f)
        o_t = F.sigmoid(o)
        c_tilda = F.tanh(c)

        c_t = f_t * c_0 + i_t * c_tilda
        if (mode == "factual"):
            c_t = self.ln_f_c(c_t)
        elif (mode == "stylized"):
            c_t = self.ln_s_c(c_t)
        else:
            print("mode wrong")
        h_t = o_t * c_t

        outputs = self.C(h_t)

        return outputs, h_t, c_t

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
        #   def forward(self, captions, features=None, mode="factual"):
        '''
        Args:
            features: fixed vectors from images, [batch, emb_dim]
            captions: [batch, max_len]
            mode: type of caption to generate
        '''
        t_device = prev_output_tokens.device
        temp_sen_id = prev_sen_pos
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

        captions = prev_output_tokens
        batch_size = captions.size(0)
        embedded = self.B(captions)  # [batch, max_len, emb_dim]
        #embedded += positions
        features = encoder_out.encoder_out.transpose(0,1)
        src, tgt = lang_pair.split("-")
        mode = "factual"
        if(tgt!="tgt"):mode = "stylized"
        # concat features and captions

        #print(mode)

        if features is None:
            sys.stderr.write("features is None!")

        '''

        if (mode == "factual"):
            features = self.f_transfer(features)
        elif (mode == "stylized"):
            features = self.s_transfer(features)'''
        embedded = torch.cat((features, embedded), 1)

        # initialize hidden state
        h_t = Variable(torch.Tensor(batch_size, self.hidden_dim))
        c_t = Variable(torch.Tensor(batch_size, self.hidden_dim))
        nn.init.uniform(h_t)
        nn.init.uniform(c_t)

        if torch.cuda.is_available():
            h_t = h_t.cuda()
            c_t = c_t.cuda()

        all_outputs = []
        # iterate
        for ix in range(embedded.size(1)):
            emb = embedded[:, ix, :]
            outputs, h_t, c_t = self.forward_step(emb, h_t, c_t, mode=mode)
            all_outputs.append(outputs)

        all_outputs = torch.stack(all_outputs, 1)
        #if(src == "src"):
        all_outputs = all_outputs[:,1:,:]
        #print("all_outputs", all_outputs.shape)
        extra = {'attn': None, 'inner_states': None, 'style_prob': None}

        return all_outputs, extra

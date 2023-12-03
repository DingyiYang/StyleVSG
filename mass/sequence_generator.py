import math
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np

from fairseq import search, utils
from fairseq.data import data_utils
from fairseq.models import FairseqIncrementalDecoder
from collections import namedtuple
import nltk


class SequenceGenerator(object):
    def __init__(
        self,
        tgt_dict,
        beam_size=1,
        max_len_a=0,
        max_len_b=512,
        min_len=1,
        normalize_scores=True,
        len_penalty=1.,
        unk_penalty=0.,
        retain_dropout=False,
        sampling=False,
        sampling_topk=-1,
        sampling_topp=-1.0,
        temperature=1.,
        diverse_beam_groups=-1,
        diverse_beam_strength=0.5,
        match_source_len=False,
        no_repeat_ngram_size=0,
        lambda_repeat_penalty_intra=20,
        lambda_repeat_penalty_inter=10,
        sen_len_constraint=5,
        whether_add_memory=False,
        whether_concat = False,
        whether_use_img_fea = False,
        mode_type = "test",
        whether_clip_encode="True",
    ):
        """Generates translations of a given source sentence.
        Args:
            tgt_dict (~fairseq.data.Dictionary): target dictionary
            beam_size (int, optional): beam width (default: 1)
            max_len_a/b (int, optional): generate sequences of maximum length
                ax + b, where x is the source length
            min_len (int, optional): the minimum length of the generated output
                (not including end-of-sentence)
            normalize_scores (bool, optional): normalize scores by the length
                of the output (default: True)
            len_penalty (float, optional): length penalty, where <1.0 favors
                shorter, >1.0 favors longer sentences (default: 1.0)
            unk_penalty (float, optional): unknown word penalty, where <0
                produces more unks, >0 produces fewer (default: 0.0)
            retain_dropout (bool, optional): use dropout when generating
                (default: False)
            sampling (bool, optional): sample outputs instead of beam search
                (default: False)
            sampling_topk (int, optional): only sample among the top-k choices
                at each step (default: -1)
            sampling_topp (float, optional): only sample among the smallest set
                of words whose cumulative probability mass exceeds p
                at each step (default: -1.0)
            temperature (float, optional): temperature, where values
                >1.0 produce more uniform samples and values <1.0 produce
                sharper samples (default: 1.0)
            diverse_beam_groups/strength (float, optional): parameters for
                Diverse Beam Search sampling
            match_source_len (bool, optional): outputs should match the source
                length (default: False)
        """
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        #ADD
        #print("tgt_dict")
        #print(tgt_dict)
        self.tgt_dict=tgt_dict
        self.sent_split_tokens=[]
        for i in range(len(tgt_dict)):
            if str(tgt_dict[i]) in ["?", ".", "!","''"]:
                self.sent_split_tokens.append(i)


        self.split_w = []
        stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
        for i in range(len(tgt_dict)):
            if str(tgt_dict[i]) in stop_words:
                self.split_w.append([",",".","!","?"])

        self.vocab_size = len(tgt_dict)
        self.beam_size = beam_size
        # the max beam size is the dictionary size - 1, since we never select pad
        self.beam_size = min(beam_size, self.vocab_size - 1)
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.min_len = min_len
        self.normalize_scores = normalize_scores
        self.len_penalty = len_penalty
        self.unk_penalty = unk_penalty
        self.retain_dropout = retain_dropout
        self.temperature = temperature
        self.match_source_len = match_source_len
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.lambda_repeat_penalty_intra = lambda_repeat_penalty_intra
        self.lambda_repeat_penalty_inter = lambda_repeat_penalty_inter
        self.sen_len_constraint = sen_len_constraint
        self.whether_add_memory = whether_add_memory
        self.whether_concat = whether_concat
        self.whether_use_img_fea = whether_use_img_fea
        self.mode_type =mode_type
        self.whether_clip_encode = whether_clip_encode
        print("sen_len_constr: {}".format(self.sen_len_constraint))
        print("lambda_repeat_penalty_inter: {}".format(self.lambda_repeat_penalty_inter))
        assert sampling_topk < 0 or sampling, '--sampling-topk requires --sampling'
        assert sampling_topp < 0 or sampling, '--sampling-topp requires --sampling'
        assert temperature > 0, '--temperature must be greater than 0'
        self.noun_dict=[]

        # ADD
        all_tokens = [tgt_dict[i] for i in range(len(tgt_dict))]
        tags = nltk.pos_tag(all_tokens)
        for (word, pos) in tags:
            if("NN" in pos):
                self.noun_dict.append(word)
        print(self.noun_dict)

        if sampling:
            self.search = search.Sampling(tgt_dict, sampling_topk, sampling_topp)
        elif diverse_beam_groups > 0:
            self.search = search.DiverseBeamSearch(tgt_dict, diverse_beam_groups, diverse_beam_strength)
        elif match_source_len:
            self.search = search.LengthConstrainedBeamSearch(
                tgt_dict, min_len_a=1, min_len_b=0, max_len_a=1, max_len_b=0,
            )
        else:
            self.search = search.BeamSearch(tgt_dict)
    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        """Generate a batch of translations.
        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        #model = EnsembleModel(models)
        #print(sample['net_input']['src_tokens'])
        out = None
        pre_generate_tokens = None
        prev_ms = None
        model = EnsembleModel(models)
        all_encoders = []
        concat_encoder_out = None
        concat_padding_mask = None
        model.eval()
        encoder_input = {
            k: v for k, v in sample['net_input'].items()
            if k != 'prev_output_tokens'
        }
        #print('img_features',encoder_input['img_features'])
        src_tokens = encoder_input['src_tokens']
        if(int(len(src_tokens[0]))%5!=0):
            src_tokens = src_tokens[:,:-1]
        encoder_input["src_tokens"]=src_tokens
        input_size = src_tokens.size()
        # batch dimension goes first followed by source lengths
        bsz = input_size[0]
        src_len = input_size[1]
        beam_size = self.beam_size
        token_num = int(encoder_input['src_tokens'].size(1) / 5)
        #更改src_sen_pos
        src_sen_pos = encoder_input['src_sen_pos']
        sen_pos_num = int(src_sen_pos.size(1) / 5)
        src_sen_pos = torch.cat([src_sen_pos[:, sen_pos_num * i:sen_pos_num * (i) + token_num] for i in range(5)],
                                dim=1)
        encoder_input['src_sen_pos']=src_sen_pos
        # print(token_num)
        EncoderOut = namedtuple('TransformerEncoderOut', [
            'encoder_out',  # T x B x C
            'encoder_padding_mask',  # B x T
            'encoder_states',  # List[T x B x C]
            'visual_hidden',
        ])
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        whether_split_encoder = False
        visual_hidden = None
        if(whether_split_encoder is False):
            all_encoder_out = model.forward_encoder(encoder_input)
            #_, visual_hidden = all_encoder_out[0]
            #visual_hidden = visual_hidden.index_select(1, new_order)
            all_encoder_out = model.reorder_encoder_out(all_encoder_out, new_order)

            if(self.whether_concat==False or self.whether_concat=="False"):
                t_out = self._generate(model, sample, pic_id=None, pre_generate_tokens=None, **kwargs, )
                return t_out
            #print(all_encoder_out[0])
            all_encoder_padding_mask = all_encoder_out[0].encoder_padding_mask
            if(all_encoder_padding_mask is None):
                all_encoder_padding_mask = src_tokens.eq(self.pad).index_select(0, new_order)

            prev_ms = [all_encoder_out[0].encoder_out, all_encoder_padding_mask]
            for t_id in range(5):
                i = t_id
                encoder_out= all_encoder_out[0]
                encoder_out_last = encoder_out.encoder_out
                # print(src_tokens)
                # print(prefix_tokens)
                t_padding = encoder_out.encoder_padding_mask
                visual_hidden=encoder_out.visual_hidden
                token_num = int(encoder_out_last.size(0)/5)
                if (t_padding != None): t_padding = t_padding[:, i * token_num:(i + 1) * token_num]
                t_encoder_outs = EncoderOut(
                    encoder_out=encoder_out_last[i * token_num:(i + 1) * token_num, :, :],  # T x B x C
                    encoder_padding_mask=t_padding,  # B x T
                    encoder_states=encoder_out.encoder_states,  # List[T x B x C]
                    visual_hidden=visual_hidden[i * token_num:(i + 1) * token_num, :, :]
                )
                all_encoders.append([t_encoder_outs])
        else:
            for t_id in range(5):
                model.eval()
                encoder_input = {
                    k: v for k, v in sample['net_input'].items()
                    if k != 'prev_output_tokens'
                }
                src_tokens = encoder_input['src_tokens']
                input_size = src_tokens.size()
                # batch dimension goes first followed by source lengths
                bsz = input_size[0]
                src_len = input_size[1]
                beam_size = self.beam_size
                i = t_id
                token_num = int(encoder_input['src_tokens'].size(1) / 5)
                # print(token_num)
                encoder_input['src_sen_pos'] = encoder_input['src_sen_pos'][:, i * token_num:(i + 1) * token_num]
                encoder_input['src_tokens'] = encoder_input['src_tokens'][:, i * token_num:(i + 1) * token_num]
                t_encoder_out = model.forward_encoder(encoder_input)
                new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
                new_order = new_order.to(src_tokens.device).long()
                t_encoder_out = model.reorder_encoder_out(t_encoder_out, new_order)

                all_encoders.append(t_encoder_out)
                t_padding_mask = src_tokens[:, i * token_num:(i + 1) * token_num].eq(self.pad)
                t_padding_mask = t_padding_mask.index_select(0, new_order)
                if (concat_encoder_out == None):
                    concat_encoder_out = t_encoder_out[0].encoder_out
                    concat_padding_mask = t_padding_mask
                else:
                    concat_encoder_out = torch.cat((concat_encoder_out, t_encoder_out[0].encoder_out), dim=0)
                    concat_padding_mask = torch.cat((concat_padding_mask, t_padding_mask), dim=1)
            if (self.whether_add_memory == "True"):
                prev_ms = [concat_encoder_out, concat_padding_mask]
        multi_target = sample["multi_target"]

        last_out = None
        for t_id in range(5):
            #print(t_id)
            if(self.whether_add_memory=="True"):
                whether_update = False
                if(t_id>0):
                    lang_pair = None
                    if ('lang_pair' in encoder_input.keys()):
                        lang_pair = encoder_input['lang_pair']
                    _, _, prev_ms = model.forward_decoder(
                        last_out, encoder_outs=all_encoders[t_id-1], temperature=self.temperature,
                        lang_pair=lang_pair,
                        whether_add_memory=True, prev_ms=prev_ms, last_sen=last_out, whether_update=True, temp_sen_id = t_id, visual_hidden = visual_hidden[i*token_num:(i+1)*token_num,:,:]
                    )
                    #print("do_update",prev_ms)
                t_out, prev_ms = self._generate_with_memory(model, sample, pic_id = t_id, pre_generate_tokens = pre_generate_tokens, prev_ms=prev_ms, last_sen = last_out, encoder_outs = all_encoders[t_id], whether_update=whether_update, temp_sen_id = t_id+1, visual_hidden = visual_hidden[i*token_num:(i+1)*token_num,:,:], **kwargs, )
                #在得到了t_out之后，对prev_ms进行更新
            else:
                t_out = self._generate(model, sample, pic_id = t_id, pre_generate_tokens = pre_generate_tokens, encoder_outs = all_encoders[t_id], temp_sen_id= t_id+1, **kwargs, )

            t_device=src_tokens.device
            if(out == None):out =t_out
            else:
                #print(len(out))
                for i in range(len(out)):
                    for j in range(len(out[i])):
                        #out[i][j]['tokens'] = torch.cat([out[i][j]['tokens'],torch.tensor([4]).to(t_device),t_out[i][j]['tokens']])
                        out[i][j]['tokens'] = torch.cat(
                            [out[i][j]['tokens'], t_out[i][j]['tokens']])
                        out[i][j]['score'] = out[i][j]['score']*t_out[i][j]['score']
                        if(t_out[i][j]['attention'] is not None):
                            out[i][j]['attention'] = torch.cat((out[i][j]['attention'],t_out[i][j]['attention']),dim=1)
                        else:
                            out[i][j]['attention']=None

            if(pre_generate_tokens==None): pre_generate_tokens=[]

            for i in range(len(out)):
                #t_generate_tokens.append(t_out[i][0]['tokens'][:-1])
                if(t_id == 0):
                    pre_generate_tokens.append(t_out[i][0]['tokens'][:-1])
                else:
                    pre_generate_tokens[i] = torch.cat([pre_generate_tokens[i], t_out[i][0]['tokens'][:-1]])
            last_out = data_utils.collate_tokens(
                [s[0]['tokens'][:-1] for s in t_out],
                self.pad, self.eos, left_pad=False, move_eos_to_beginning=False,
            )
            last_out = last_out.index_select(0, new_order)

        return out

    @torch.no_grad()
    def _generate(
        self,
        model,
        sample,
        prefix_tokens=None,
        bos_token=None,
        pic_id=None,
        pre_generate_tokens = None,
        encoder_outs = None,
        temp_sen_id = 1,
        **kwargs,
    ):
        '''
        pre_output_tokens: 前面生成的句子，用于降低重复性
        '''
        if not self.retain_dropout:
            model.eval()

        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        encoder_input = {
            k: v for k, v in sample['net_input'].items()
            if k != 'prev_output_tokens'
        }
        lang_pair = None
        if('lang_pair' in encoder_input.keys()):
            lang_pair = encoder_input['lang_pair']

        src_tokens = encoder_input['src_tokens']
        src_sen_pos = encoder_input['src_sen_pos']
        src_lengths = (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
        input_size = src_tokens.size()
        # batch dimension goes first followed by source lengths
        bsz = input_size[0]
        src_len = input_size[1]
        beam_size = self.beam_size

        i = pic_id
        token_num = int(encoder_input['src_tokens'].size(1) / 5)
        # print(token_num)
        encoder_input['src_sen_pos'] = encoder_input['src_sen_pos'][:, i * token_num:(i + 1) * token_num]
        encoder_input['src_tokens'] = encoder_input['src_tokens'][:, i * token_num:(i + 1) * token_num]
        lang_pair = None
        if ('lang_pair' in encoder_input.keys()):
            lang_pair = encoder_input['lang_pair']

        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                # exclude the EOS marker
                model.max_decoder_positions() - 1,
            )


        EncoderOut = namedtuple('TransformerEncoderOut', [
            'encoder_out',  # T x B x C
            'encoder_padding_mask',  # B x T
            'encoder_states',  # List[T x B x C]
            'visual_hidden',
        ])
        encoder_out = encoder_outs[0]

        i = pic_id

        finalized = self.get_single_generated(encoder_outs, model, src_tokens[:,i * token_num:(i + 1) * token_num], bsz, beam_size, max_len, bos_token, lang_pair, prefix_tokens, pre_generate_tokens, temp_sen_id=temp_sen_id)
        #print(i, t_finalized)`

        return finalized

    @torch.no_grad()
    def _generate_with_memory(
            self,
            model,
            sample,
            prefix_tokens=None,
            bos_token=None,
            pic_id=None,
            pre_generate_tokens=None,
            prev_ms = None,
            last_sen = None,
            encoder_outs = None,
            whether_update = True,
            temp_sen_id = 0,
            visual_hidden=None,
            **kwargs,
    ):
        '''
        pre_output_tokens: 前面生成的句子，用于降低重复性
        '''
        if not self.retain_dropout:
            model.eval()

        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        encoder_input = {
            k: v for k, v in sample['net_input'].items()
            if k != 'prev_output_tokens'
        }
        #print("target",sample["target"][0])
        src_tokens = encoder_input['src_tokens']
        src_sen_pos = encoder_input['src_sen_pos']
        src_lengths = (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
        input_size = src_tokens.size()
        # batch dimension goes first followed by source lengths
        bsz = input_size[0]
        src_len = input_size[1]
        beam_size = self.beam_size
        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                # exclude the EOS marker
                model.max_decoder_positions() - 1,
            )

        i = pic_id
        token_num = int(encoder_input['src_tokens'].size(1)/5)
        #print(token_num)
        encoder_input['src_sen_pos'] = encoder_input['src_sen_pos'][:, i * token_num:(i + 1) * token_num]
        encoder_input['src_tokens'] = encoder_input['src_tokens'][:,i * token_num:(i + 1) * token_num]
        lang_pair = None
        if ('lang_pair' in encoder_input.keys()):
            lang_pair = encoder_input['lang_pair']


        EncoderOut = namedtuple('TransformerEncoderOut', [
            'encoder_out',  # T x B x C
            'encoder_padding_mask',  # B x T
            'encoder_states',  # List[T x B x C]
            'visual_hidden',
        ])
        encoder_out = encoder_outs[0]


        finalized, prev_ms = self.get_single_generated(encoder_outs, model, src_tokens[:,i * token_num:(i + 1) * token_num], bsz, beam_size, max_len, bos_token,
                                              lang_pair, prefix_tokens, pre_generate_tokens, prev_ms=prev_ms, last_sen = last_sen, whether_add_memory = True, pic_id=pic_id, whether_update=whether_update,temp_sen_id=temp_sen_id,visual_hidden=visual_hidden)
        # print(i, t_finalized)`

        return finalized, prev_ms

    @torch.no_grad()
    def get_single_generated(
            self,
            encoder_outs,
            model,
            src_tokens,
            bsz,
            beam_size,
            max_len,
            bos_token,
            lang_pair,
            prefix_tokens,
            pre_generate_tokens,
            prev_ms = None,
            visual_hidden = None,
            last_sen = None,
            whether_add_memory = False,
            whether_update = True,
            pic_id = None,
            temp_sen_id =0,
    ):
        # initialize buffers
        #prev_ms = prev_ms
        # ADD:
        prev_token_bag =[]
        prev_token_bag_loc = []
        prev_token_bag_num = []
        if(pre_generate_tokens is not None):
            for bdidx in range(len(pre_generate_tokens)):
                token_bag = []
                token_bag_loc = {}
                token_bag_num = {}
                for t_loc,pre_tok in enumerate(pre_generate_tokens[bdidx]):
                    #pre_tok=int(pre_tok)
                    if (pre_tok not in token_bag and pre_tok not in self.split_w):
                    #if (pre_tok not in self.split_w):
                        token_bag.append(pre_tok)
                        #token_bag_num[pre_tok]=0
                    token_bag_loc[pre_tok]=t_loc
                prev_token_bag.append(token_bag)
                prev_token_bag_loc.append(token_bag_loc)
                #prev_token_bag_num.append(token_bag_num)
        return_ms = prev_ms
        scores = src_tokens.new(bsz * beam_size, max_len + 1).float().fill_(0)
        scores_buf = scores.clone()
        tokens = src_tokens.new(bsz * beam_size, max_len + 2).long().fill_(self.pad)
        tokens_buf = tokens.clone()
        tokens[:, 0] = self.eos if bos_token is None else bos_token
        attn, attn_buf = None, None
        src_lengths = (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)

        # The blacklist indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then the blacklist would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        blacklist = src_tokens.new_zeros(bsz, beam_size).eq(-1)  # forward and backward-compatible False mask

        # list of completed sentences
        finalized = [[] for i in range(bsz)]
        finished = [False for i in range(bsz)]
        num_remaining_sent = bsz

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(tokens)
        cand_offsets = torch.arange(0, cand_size).type_as(tokens)

        # helper function for allocating buffers on the fly
        buffers = {}

        def buffer(name, type_of=tokens):  # noqa
            if name not in buffers:
                buffers[name] = type_of.new()
            return buffers[name]

        def is_finished(sent, step, unfin_idx):
            """
            Check whether we've finished generation for a given sentence, by
            comparing the worst score among finalized hypotheses to the best
            possible score among unfinalized hypotheses.
            """
            assert len(finalized[sent]) <= beam_size
            if len(finalized[sent]) == beam_size or step == max_len:
                return True
            return False

        def finalize_hypos(step, bbsz_idx, eos_scores):
            """
            Finalize the given hypotheses at this step, while keeping the total
            number of finalized hypotheses per sentence <= beam_size.
            Note: the input must be in the desired finalization order, so that
            hypotheses that appear earlier in the input are preferred to those
            that appear later.
            Args:
                step: current time step
                bbsz_idx: A vector of indices in the range [0, bsz*beam_size),
                    indicating which hypotheses to finalize
                eos_scores: A vector of the same size as bbsz_idx containing
                    scores for each hypothesis
            """
            assert bbsz_idx.numel() == eos_scores.numel()

            # clone relevant token and attention tensors
            tokens_clone = tokens.index_select(0, bbsz_idx)
            tokens_clone = tokens_clone[:, 1:step + 2]  # skip the first index, which is EOS
            assert not tokens_clone.eq(self.eos).any()
            tokens_clone[:, step] = self.eos
            attn_clone = attn.index_select(0, bbsz_idx)[:, :, 1:step+2] if attn is not None else None

            # compute scores per token position
            pos_scores = scores.index_select(0, bbsz_idx)[:, :step+1]
            pos_scores[:, step] = eos_scores
            # convert from cumulative to per-position scores
            pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

            # normalize sentence-level scores
            if self.normalize_scores:
                eos_scores /= (step + 1) ** self.len_penalty

            cum_unfin = []
            prev = 0
            for f in finished:
                if f:
                    prev += 1
                else:
                    cum_unfin.append(prev)

            sents_seen = set()
            for i, (idx, score) in enumerate(zip(bbsz_idx.tolist(), eos_scores.tolist())):
                unfin_idx = idx // beam_size
                sent = unfin_idx + cum_unfin[unfin_idx]

                sents_seen.add((sent, unfin_idx))

                if self.match_source_len and step > src_lengths[unfin_idx]:
                    score = -math.inf

                def get_hypo():

                    if attn_clone is not None:
                        # remove padding tokens from attn scores
                        hypo_attn = attn_clone[i]
                    else:
                        hypo_attn = None

                    return {
                        'tokens': tokens_clone[i],
                        'score': score,
                        'attention': hypo_attn,  # src_len x tgt_len
                        'alignment': None,
                        'positional_scores': pos_scores[i],
                    }

                if len(finalized[sent]) < beam_size:
                    finalized[sent].append(get_hypo())

            newly_finished = []
            for sent, unfin_idx in sents_seen:
                # check termination conditions for this sentence
                if not finished[sent] and is_finished(sent, step, unfin_idx):
                    finished[sent] = True
                    newly_finished.append(unfin_idx)
            return newly_finished

        reorder_state = None
        batch_idxs = None
        #ADD:
        #print("tokens: "+str(tokens[1, 0:max_len].tolist()))

        # ADD: 对于每个图片生成句子，组合后返回

        for step in range(max_len + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(batch_idxs)
                    reorder_state.view(-1, beam_size).add_(corr.unsqueeze(-1) * beam_size)
                model.reorder_incremental_state(reorder_state)
                encoder_outs = model.reorder_encoder_out(encoder_outs, reorder_state)
                #print(reorder_state.shape)
                if(visual_hidden is not None): visual_hidden = visual_hidden.index_select(1, reorder_state)
                if(prev_ms is not None):prev_ms = prev_ms.index_select(0, reorder_state)
                if(last_sen is not None):last_sen = last_sen.index_select(0, reorder_state)
            #print(tokens[:, :step + 1])
            if(whether_add_memory==False):
                lprobs, avg_attn_scores = model.forward_decoder(
                    tokens[:, :step + 1], encoder_outs=encoder_outs, temperature=self.temperature, lang_pair=lang_pair,temp_sen_id=temp_sen_id,
                )
            else:
                #whether_update = True
                if(step>0):whether_update = False

                lprobs, avg_attn_scores, prev_ms = model.forward_decoder(
                    tokens[:, :step + 1], encoder_outs=encoder_outs, temperature=self.temperature, lang_pair=lang_pair, whether_add_memory=True, prev_ms=prev_ms, last_sen = last_sen, whether_update=whether_update, temp_sen_id=temp_sen_id, visual_hidden = visual_hidden,
                )
                #print(prev_ms.shape)
                if(step==0):return_ms = prev_ms

            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            # handle max length constraint
            if step >= max_len-1:
                lprobs[:, :self.eos] = -math.inf
                lprobs[:, self.eos + 1:] = -math.inf

            # handle prefix tokens (possibly with different lengths)
            if prefix_tokens is not None and step < prefix_tokens.size(1) and step < max_len:
                prefix_toks = prefix_tokens[:, step].unsqueeze(-1).repeat(1, beam_size).view(-1)
                prefix_lprobs = lprobs.gather(-1, prefix_toks.unsqueeze(-1))
                prefix_mask = prefix_toks.ne(self.pad)
                lprobs[prefix_mask] = -math.inf
                lprobs[prefix_mask] = lprobs[prefix_mask].scatter_(
                    -1, prefix_toks[prefix_mask].unsqueeze(-1), prefix_lprobs[prefix_mask]
                )
                # if prefix includes eos, then we should make sure tokens and
                # scores are the same across all beams
                eos_mask = prefix_toks.eq(self.eos)
                if eos_mask.any():
                    # validate that the first beam matches the prefix
                    first_beam = tokens[eos_mask].view(-1, beam_size, tokens.size(-1))[:, 0, 1:step + 1]
                    eos_mask_batch_dim = eos_mask.view(-1, beam_size)[:, 0]
                    target_prefix = prefix_tokens[eos_mask_batch_dim][:, :step]
                    assert (first_beam == target_prefix).all()

                    def replicate_first_beam(tensor, mask):
                        tensor = tensor.view(-1, beam_size, tensor.size(-1))
                        tensor[mask] = tensor[mask][:, :1, :]
                        return tensor.view(-1, tensor.size(-1))

                    # copy tokens, scores and lprobs from the first beam to all beams
                    tokens = replicate_first_beam(tokens, eos_mask_batch_dim)
                    scores = replicate_first_beam(scores, eos_mask_batch_dim)
                    lprobs = replicate_first_beam(lprobs, eos_mask_batch_dim)
            elif step < self.min_len:
                # minimum length constraint (does not apply if using prefix_tokens)
                lprobs[:, self.eos] = -math.inf

            # ADD: give a repetition penalty score
            # ADD: handle the max sentence length constraint
            #print("lambda_repeat_penalty_intra: {}, lambda_repeat_penalty_inter: {}".format(self.lambda_repeat_penalty_intra, self.lambda_repeat_penalty_inter))
            for bbsz_idx in range(bsz * beam_size):
                gen_tokens = tokens[bbsz_idx, : step + 1].tolist()
                t_sent = 1
                sent_len = 0
                for i in range(len(gen_tokens)):
                    if gen_tokens[-(i+1)] in self.sent_split_tokens:
                        t_sent = 0
                        sent_len += 1
                        continue
                    if t_sent == 1:
                        lprobs[bbsz_idx, gen_tokens[-(i+1)]] -= self.lambda_repeat_penalty_intra
                    else:
                        lprobs[bbsz_idx, gen_tokens[-(i+1)]] -= (self.lambda_repeat_penalty_inter / (step))
                    #对于pre_generate_tokens中出现的token，降低其概览

                if(pre_generate_tokens!=None):
                    temp_batch = int(bbsz_idx/(beam_size))
                    story_length = len(pre_generate_tokens[temp_batch])
                    token_bag = prev_token_bag[temp_batch]
                    token_bag_loc = prev_token_bag_loc[temp_batch]
                    #token_bag_num = prev_token_bag_num[temp_batch]
                    for pre_tok in token_bag:
                        #if(self.tgt_dict[pre_tok] in self.noun_dict):
                        lprobs[bbsz_idx, pre_tok] -= (self.lambda_repeat_penalty_inter / (story_length))
                        #story_length -= 1
                if(sent_len>=self.sen_len_constraint):
                    lprobs[bbsz_idx, :self.eos] = -math.inf
                    lprobs[bbsz_idx, self.eos + 1:] = -math.inf


            if step >= max_len-1:
                lprobs[:, :self.eos] = -math.inf
                lprobs[:, self.eos + 1:] = -math.inf

            if self.no_repeat_ngram_size > 0:
                # for each beam and batch sentence, generate a list of previous ngrams
                gen_ngrams = [{} for bbsz_idx in range(bsz * beam_size)]
                for bbsz_idx in range(bsz * beam_size):
                    gen_tokens = tokens[bbsz_idx].tolist()
                    for ngram in zip(*[gen_tokens[i:] for i in range(self.no_repeat_ngram_size)]):
                        gen_ngrams[bbsz_idx][tuple(ngram[:-1])] = \
                                gen_ngrams[bbsz_idx].get(tuple(ngram[:-1]), []) + [ngram[-1]]

            # Record attention scores
            if avg_attn_scores is not None:
                if attn is None:
                    if(whether_add_memory==False):
                        attn = scores.new(bsz * beam_size, int(avg_attn_scores.size(1)), max_len + 2)
                    else:
                        attn = scores.new(bsz * beam_size, int(avg_attn_scores.size(1)), max_len + 2)
                    attn_buf = attn.clone()
                attn[:, :, step + 1].copy_(avg_attn_scores)

            scores = scores.type_as(lprobs)
            scores_buf = scores_buf.type_as(lprobs)
            eos_bbsz_idx = buffer('eos_bbsz_idx')
            eos_scores = buffer('eos_scores', type_of=scores)

            self.search.set_src_lengths(src_lengths)

            if self.no_repeat_ngram_size > 0:
                def calculate_banned_tokens(bbsz_idx):
                    # before decoding the next token, prevent decoding of ngrams that have already appeared
                    ngram_index = tuple(tokens[bbsz_idx, step + 2 - self.no_repeat_ngram_size:step + 1].tolist())
                    return gen_ngrams[bbsz_idx].get(ngram_index, [])

                def calculate_banned_tokens_in_pre_generated(bbsz_idx):
                    temp_batch = int(bbsz_idx / (beam_size))
                    temp_pre_tokens = pre_generate_tokens[temp_batch].tolist()
                    t_grams = {}
                    for ngram in zip(*[temp_pre_tokens[i:] for i in range(self.no_repeat_ngram_size)]):
                        t_grams[tuple(ngram[:-1])] = \
                            t_grams.get(tuple(ngram[:-1]), []) + [ngram[-1]]
                    ngram_index = tuple(tokens[bbsz_idx, step + 2 - self.no_repeat_ngram_size:step + 1].tolist())
                    return t_grams.get(ngram_index, [])



                if step + 2 - self.no_repeat_ngram_size >= 0:
                    # no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
                    banned_tokens = [calculate_banned_tokens(bbsz_idx) for bbsz_idx in range(bsz * beam_size)]

                    # ADD: 如果前面有已经生成的句子，需统计使得不出现repeat_ngram
                    if(pre_generate_tokens!=None):
                        other_banned = [calculate_banned_tokens_in_pre_generated(bbsz_idx) for bbsz_idx in range(bsz * beam_size)]
                        banned_tokens = [banned_tokens[bbsz_idx]+other_banned[bbsz_idx] for bbsz_idx in range(bsz * beam_size)]
                else:
                    banned_tokens = [[] for bbsz_idx in range(bsz * beam_size)]

                for bbsz_idx in range(bsz * beam_size):
                    lprobs[bbsz_idx, banned_tokens[bbsz_idx]] = -math.inf


            cand_scores, cand_indices, cand_beams = self.search.step(
                step,
                lprobs.view(bsz, -1, self.vocab_size),
                scores.view(bsz, self.beam_size, -1)[:, :, :step],
            )

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # finalize hypotheses that end in eos, except for blacklisted ones
            # or candidates with a score of -inf
            eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-math.inf)
            eos_mask[:, :beam_size][blacklist] = 0

            # only consider eos when it's among the top beam_size indices
            torch.masked_select(
                cand_bbsz_idx[:, :beam_size],
                mask=eos_mask[:, :beam_size],
                out=eos_bbsz_idx,
            )

            finalized_sents = set()
            if eos_bbsz_idx.numel() > 0:
                torch.masked_select(
                    cand_scores[:, :beam_size],
                    mask=eos_mask[:, :beam_size],
                    out=eos_scores,
                )
                finalized_sents = finalize_hypos(step, eos_bbsz_idx, eos_scores)
                num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            assert step < max_len

            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = cand_indices.new_ones(bsz)
                batch_mask[cand_indices.new(finalized_sents)] = 0
                batch_idxs = batch_mask.nonzero().squeeze(-1)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]
                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]
                src_lengths = src_lengths[batch_idxs]
                blacklist = blacklist[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                scores_buf.resize_as_(scores)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens_buf.resize_as_(tokens)
                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, attn.size(1), -1)
                    attn_buf.resize_as_(attn)
                bsz = new_bsz
            else:
                batch_idxs = None

            # Set active_mask so that values > cand_size indicate eos or
            # blacklisted hypos and values < cand_size indicate candidate
            # active hypos. After this, the min values per row are the top
            # candidate active hypos.
            active_mask = buffer('active_mask')
            eos_mask[:, :beam_size] |= blacklist
            torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[:eos_mask.size(1)],
                out=active_mask,
            )

            # get the top beam_size active hypotheses, which are just the hypos
            # with the smallest values in active_mask
            active_hypos, new_blacklist = buffer('active_hypos'), buffer('new_blacklist')
            torch.topk(
                active_mask, k=beam_size, dim=1, largest=False,
                out=(new_blacklist, active_hypos)
            )

            # update blacklist to ignore any finalized hypos
            blacklist = new_blacklist.ge(cand_size)[:, :beam_size]
            assert (~blacklist).any(dim=1).all()

            active_bbsz_idx = buffer('active_bbsz_idx')
            torch.gather(
                cand_bbsz_idx, dim=1, index=active_hypos,
                out=active_bbsz_idx,
            )
            active_scores = torch.gather(
                cand_scores, dim=1, index=active_hypos,
                out=scores[:, step].view(bsz, beam_size),
            )

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses
            torch.index_select(
                tokens[:, :step + 1], dim=0, index=active_bbsz_idx,
                out=tokens_buf[:, :step + 1],
            )
            torch.gather(
                cand_indices, dim=1, index=active_hypos,
                out=tokens_buf.view(bsz, beam_size, -1)[:, :, step + 1],
            )
            if step > 0:
                torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx,
                    out=scores_buf[:, :step],
                )
            torch.gather(
                cand_scores, dim=1, index=active_hypos,
                out=scores_buf.view(bsz, beam_size, -1)[:, :, step],
            )

            # copy attention for active hypotheses
            if attn is not None:
                torch.index_select(
                    attn[:, :, :step + 2], dim=0, index=active_bbsz_idx,
                    out=attn_buf[:, :, :step + 2],
                )

            # swap buffers
            tokens, tokens_buf = tokens_buf, tokens
            scores, scores_buf = scores_buf, scores
            if attn is not None:
                attn, attn_buf = attn_buf, attn

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(len(finalized)):
            finalized[sent] = sorted(finalized[sent], key=lambda r: r['score'], reverse=True)
        if(whether_add_memory==True):return finalized, return_ms
        return finalized


class EnsembleModel(torch.nn.Module):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        self.incremental_states = None
        '''
        if all(isinstance(m.decoder, FairseqIncrementalDecoder) for m in models):
            self.incremental_states = {m: {} for m in models}'''

    def has_encoder(self):
        return hasattr(self.models[0], 'encoder')

    def max_decoder_positions(self):
        return min(m.max_decoder_positions() for m in self.models)

    @torch.no_grad()
    def forward_encoder(self, encoder_input):
        if not self.has_encoder():
            return None
        return [model.encoder(**encoder_input) for model in self.models]

    @torch.no_grad()
    def forward_decoder(self, tokens, encoder_outs, temperature=1., lang_pair=None, whether_add_memory = False, prev_ms = None, last_sen = None, whether_update = True, temp_sen_id=0, visual_hidden=None):
        if len(self.models) == 1:
            if(whether_add_memory==False):
                return self._decode_one(
                    tokens,
                    self.models[0],
                    encoder_outs[0] if self.has_encoder() else None,
                    self.incremental_states,
                    log_probs=True,
                    temperature=temperature,
                    lang_pair=lang_pair,
                    temp_sen_id=temp_sen_id,
                )
            else:
                return self._decode_one_with_memory(
                    tokens,
                    self.models[0],
                    encoder_outs[0] if self.has_encoder() else None,
                    self.incremental_states,
                    log_probs=True,
                    temperature=temperature,
                    lang_pair=lang_pair,
                    prev_ms=prev_ms,
                    last_sen = last_sen,
                    whether_update=whether_update,
                    temp_sen_id=temp_sen_id,
                    visual_hidden=visual_hidden,
                )

        log_probs = []
        avg_attn = None
        for model, encoder_out in zip(self.models, encoder_outs):
            probs, attn = self._decode_one(
                tokens,
                model,
                encoder_out,
                self.incremental_states,
                log_probs=True,
                temperature=temperature,
                lang_pair=lang_pair,
            )
            log_probs.append(probs)
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(len(self.models))
        if avg_attn is not None:
            avg_attn.div_(len(self.models))
        return avg_probs, avg_attn

    def _decode_one(
        self, tokens, model, encoder_out, incremental_states, log_probs,
        temperature=1., lang_pair=None,temp_sen_id = 1,
    ):
        if self.incremental_states is not None:
            decoder_out = list(model.forward_decoder(
                tokens, prev_sen_pos=temp_sen_id, encoder_out=encoder_out,incremental_state=self.incremental_states[model], lang_pair=lang_pair,
            ))
        else:
            decoder_out = list(model.forward_decoder(tokens, prev_sen_pos=temp_sen_id, encoder_out=encoder_out, lang_pair=lang_pair,))
        decoder_out[0] = decoder_out[0][:, -1:, :]
        if temperature != 1.:
            decoder_out[0].div_(temperature)
        attn = decoder_out[1]
        if type(attn) is dict:
            attn = attn.get('attn', None)
        if attn is not None:
            attn = attn[:, -1, :]
        probs = model.get_normalized_probs(decoder_out, log_probs=log_probs)
        probs = probs[:, -1, :]
        return probs, attn

    def _decode_one_with_memory(
        self, tokens, model, encoder_out, incremental_states, log_probs,
        temperature=1., lang_pair=None, prev_ms = None, last_sen = None, whether_update = True, temp_sen_id=0,visual_hidden=None,
    ):
        if self.incremental_states is not None:
            decoder_out = list(model.forward_decoder(
                tokens, temp_sen_id=temp_sen_id, visual_hidden=visual_hidden, prev_sen_pos=None, encoder_out=encoder_out,incremental_state=self.incremental_states[model], lang_pair=lang_pair, prev_ms = prev_ms, last_sen = last_sen, whether_update=whether_update
            ))
        else:
            decoder_out = list(model.forward_decoder(tokens, temp_sen_id=temp_sen_id, visual_hidden=visual_hidden, prev_sen_pos=None, encoder_out=encoder_out, lang_pair=lang_pair, prev_ms = prev_ms, last_sen = last_sen, whether_update=whether_update) )
        decoder_out[0] = decoder_out[0][:, -1:, :]
        if temperature != 1.:
            decoder_out[0].div_(temperature)
        attn = decoder_out[1]
        if type(attn) is dict:
            attn = attn.get('attn', None)
        if attn is not None:
            attn = attn[:, -1, :]
        ms = decoder_out[2]
        probs = model.get_normalized_probs(decoder_out, log_probs=log_probs)
        probs = probs[:, -1, :]
        return probs, attn, ms

    def reorder_encoder_out(self, encoder_outs, new_order):
        if not self.has_encoder():
            return
        return [
            model.encoder.reorder_encoder_out(encoder_out, new_order)
            for model, encoder_out in zip(self.models, encoder_outs)
        ]

    def reorder_incremental_state(self, new_order):
        if self.incremental_states is None:
            return
        for model in self.models:
            model.decoder.reorder_incremental_state(self.incremental_states[model], new_order)

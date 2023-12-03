import os
from collections import OrderedDict
import itertools
import json
import torch
import numpy as np
import fairseq
from fairseq.tasks import register_task
from fairseq.tasks.fairseq_task import FairseqTask
from fairseq.tasks.multilingual_translation import MultilingualTranslationTask
from fairseq.tasks.translation import TranslationTask
from fairseq.tasks.language_modeling import LanguageModelingTask
from fairseq import utils, options
import time
from tqdm import tqdm
import mass.Constants
from .model_loss import ModelLoss

import fairseq.criterions.label_smoothed_cross_entropy

from .bert_dictionary import BertDictionary
from .language_pair_dataset import LanguagePairDataset

from fairseq.data import (
    data_utils,
    Dictionary,
    MonolingualDataset,
    TokenBlockDataset,
    TransformEosDataset,
    TruncatedDictionary,
)

from fairseq.data import (
    BacktranslationDataset,
    IndexedCachedDataset,
    IndexedDataset,
    IndexedRawTextDataset,
    NoisingDataset,
    RoundRobinZipDatasets,
    AppendTokenDataset,
    ConcatDataset,
    indexed_dataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    TransformEosLangPairDataset,
)


@register_task('translation_mass')
class TranslationMASSTask(TranslationTask):
    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)

    @classmethod
    def load_dictionary(cls, filename):
        return BertDictionary.load_from_file(filename)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)



@register_task('translation_mass_lm')
class TranslationMASSLMTask(LanguageModelingTask):
    def __init__(self, args, dictionary, output_dictionary=None, targets=None):
        super().__init__(args, dictionary, output_dictionary=output_dictionary, targets=targets)

    # @classmethod
    # def setup_task(cls, args, **kwargs):
    #     """Setup the task (e.g., load dictionaries).
    #     Args:
    #         args (argparse.Namespace): parsed command-line arguments
    #     """
    #     if getattr(args, "raw_text", False):
    #         utils.deprecation_warning(
    #             "--raw-text is deprecated, please use --dataset-impl=raw"
    #         )
    #         args.dataset_impl = "raw"
    #     elif getattr(args, "lazy_load", False):
    #         utils.deprecation_warning(
    #             "--lazy-load is deprecated, please use --dataset-impl=lazy"
    #         )
    #         args.dataset_impl = "lazy"
    #
    #     dictionary = None
    #     output_dictionary = None
    #     if args.data:
    #         paths = args.data.split(":")
    #         assert len(paths) > 0
    #         # dictionary = Dictionary.load(os.path.join(paths[0], "dict.txt"))
    #         dictionary = cls.load_dictionary(os.path.join(paths[0], "dict.txt"))
    #         print("| dictionary: {} types".format(len(dictionary)))
    #         output_dictionary = dictionary
    #         if args.output_dictionary_size >= 0:
    #             output_dictionary = TruncatedDictionary(
    #                 dictionary, args.output_dictionary_size
    #             )
    #
    #     # upgrade old checkpoints
    #     if hasattr(args, "exclude_self_target"):
    #         args.self_target = not args.exclude_self_target
    #
    #     targets = []
    #     if getattr(args, "self_target", False):
    #         targets.append("self")
    #     if getattr(args, "future_target", False):
    #         targets.append("future")
    #     if getattr(args, "past_target", False):
    #         targets.append("past")
    #     if len(targets) == 0:
    #         # standard language modeling
    #         targets = ["future"]
    #
    #     return cls(args, dictionary, output_dictionary, targets=targets)

    @classmethod
    def load_dictionary(cls, filename):
        print(filename)
        return BertDictionary.load_from_file(filename)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)




def _get_bt_dataset_key(lang_pair):
    return "bt:" + lang_pair


#style数据集的key
def _get_denoising_dataset_key(lang_pair):
    return "" + lang_pair


def _lang_token(lang: str):
    return '__{}__'.format(lang)


def _lang_token_index(dic: Dictionary, lang: str):
    """Return language token index."""
    idx = dic.index(_lang_token(lang))
    assert idx != dic.unk_index, \
        'cannot find language token for lang {}'.format(lang)
    return idx


# ported from UnsupervisedMT
def parse_lambda_config(x):
    """
    Parse the configuration of lambda coefficient (for scheduling).
    x = "3"                  # lambda will be a constant equal to x
    x = "0:1,1000:0"         # lambda will start from 1 and linearly decrease
                             # to 0 during the first 1000 iterations
    x = "0:0,1000:0,2000:1"  # lambda will be equal to 0 for the first 1000
                             # iterations, then will linearly increase to 1 until iteration 2000
    """
    split = x.split(',')
    if len(split) == 1:
        return float(x), None
    else:
        split = [s.split(':') for s in split]
        assert all(len(s) == 2 for s in split)
        assert all(k.isdigit() for k, _ in split)
        assert all(int(split[i][0]) < int(split[i + 1][0]) for i in range(len(split) - 1))
        return float(split[0][1]), [(int(k), float(v)) for k, v in split]


def load_langpair_dataset(
    data_path, split,
    src, src_dict,
    tgt, tgt_dict,
    combine, dataset_impl, upsample_primary,
    left_pad_source, left_pad_target, max_source_positions,
    max_target_positions, prepend_bos=False, load_alignments=False,
    truncate_source=False, lang_pair=None,
    whether_concat=False,
    whether_use_img_fea=False,
    all_img_features=None, img_id_dict=None,
    whether_read_text_clip_fea = False, whether_use_text_clip_fea=False,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
        #print(filename)
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []
    # ADD: 读取src dataset的term/object对应的图片位置
    word2idx = {}
    for i in range(len(tgt_dict)):
        word2idx[tgt_dict[i]] = i
    src_sen_pos_dataset = []
    tgt_sen_pos_dataset = []
    src_graph_objs = []
    src_sizes = None
    top_obj_num = 10
    img_features = None

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else '')

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

        src_dataset = data_utils.load_indexed_dataset(prefix + src, src_dict, dataset_impl)
        if truncate_source:
            print('truncated!')
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    #StripTokenDataset(src_dataset ),
                    max_source_positions - 1,
                ),
                #src_dict.eos(),
            )
        src_datasets.append(src_dataset)
        tgt_datasets.append(
            AppendTokenDataset(
                data_utils.load_indexed_dataset(prefix + tgt, tgt_dict, dataset_impl),
                tgt_dict.eos(),
            )
        )

        # ADD: src_sen_pos
        src_sen_pos_file = os.path.join(data_path, split+'.src.sen.pos')
        tgt_sen_pos_file = os.path.join(data_path, split+'.tgt.sen.pos')
        #if indexed_dataset.dataset_exists(src_sen_pos_file, impl="None"):

        with open(src_sen_pos_file) as src_sen_pos_f:
            for line in src_sen_pos_f:
                line=line.split("\n")[0]
                line = line.strip()
                if(line==""):t_arr=[]
                else:
                    t_arr = [int(i/top_obj_num)+1 for i in range(top_obj_num*5)]
                a = np.array(t_arr)
                src_sen_pos_dataset.append(torch.from_numpy(a))
        with open(tgt_sen_pos_file) as tgt_sen_pos_f:
            for line in tgt_sen_pos_f:
                line=line.split("\n")[0]
                line = line.strip()
                if(line==""):t_arr=[]
                else:
                    t_arr = [int(i) for i in line.split(" ")]
                t_arr.append(tgt_dict.eos())
                t_arr.append(tgt_dict.eos())
                a = np.array(t_arr)
                tgt_sen_pos_dataset.append(torch.from_numpy(a))

        text_clip_fea = None
        temp_lang = "vist"
        if(whether_read_text_clip_fea == "True"):
            if("fairy" in lang_pair):
                temp_lang = "fairy"
                print("load fair clip text")
            elif("romance" in lang_pair):
                temp_lang = "romance"
                print("load fair clip text")
            elif("humor" in lang_pair):
                temp_lang = "humor"
                print("load fair clip text")

            text_clip_f = "/features/clip/text_clip/{}/clip_text_features_{}.npy".format(temp_lang,split)
            text_clip_fea = torch.from_numpy(np.load(text_clip_f))
            print("temp_lang",text_clip_fea.shape)

        if (whether_use_img_fea == "True"):
            if(whether_use_text_clip_fea=="True" and temp_lang!="vist"):
                img_features = text_clip_fea
                print(temp_lang,"use text as input")
                break

            print("use img features")
            img_features=[]

            img_seq_file = os.path.join(data_path, 'img_seq.{}'.format(split))
            with open(img_seq_file) as img_seqs:
                for line in img_seqs:
                    line = line.split("\n")[0]
                    line = line.strip()
                    if (line == ""):
                        t_arr = []
                    else:
                        t_arr = [str(img_id) for img_id in line.split(" ")]
                    # t_arr.append(src_dict.eos())
                    t_story_fea = []
                    pre_loc = 0
                    for img_id in t_arr:
                        try:
                            t_loc = img_id_dict[img_id]
                            pre_loc = t_loc
                        except:
                            #print(img_id)
                            t_loc = pre_loc
                            #print(img_id,t_loc)
                        t_story_fea.append(all_img_features[t_loc])

                    img_features.append(t_story_fea)
            #print(img_features)
            img_features = torch.from_numpy(np.array(img_features))







        print('| {} {} {}-{} {} examples'.format(data_path, split_k, src, tgt, len(src_datasets[-1])))

        if not combine:
            break

    #print(src_sen_pos_dataset[0])
    #print(src_datasets[0][0])

    assert len(src_datasets) == len(tgt_datasets)
    #assert len(src_datasets[0]) == len(src_sen_pos_dataset)
    #assert len(tgt_datasets[0]) == len(tgt_sen_pos_dataset)

    if len(src_datasets) == 1:
        src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
        #src_sen_pos_dataset = src_sen_pos_datasets[0]
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        #src_sen_pos_dataset = ConcatDataset(src_sen_pos_datasets, sample_ratios)

    #print(src_sen_pos_dataset[0])

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())
        #src_sen_pos_dataset = PrependTokenDataset(src_sen_pos_dataset, src_dict.bos())

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, '{}.align.{}-{}'.format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(align_path, None, dataset_impl)
    src_size = src_dataset.sizes
    return LanguagePairDataset(
        src_dataset,src_size , src_dict,
        tgt_dataset, tgt_dataset.sizes, tgt_dict,
        src_sen_pos_dataset, len(src_sen_pos_dataset),
        tgt_sen_pos_dataset, len(tgt_sen_pos_dataset),
        img_features,
        text_clip_fea,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        max_source_positions=max_source_positions,
        max_target_positions=max_target_positions,
        align_dataset=align_dataset,
        lang_pair=lang_pair,
        multi_tgt=whether_concat,
    )


@register_task('translation_mix')
class SummDAETranslationTask(FairseqTask):
    """A task for training multiple translation models simultaneously.
    We iterate round-robin over batches from multiple language pairs, ordered
    according to the `--lang-pairs` argument.
    The training loop is roughly:
        for i in range(len(epoch)):
            for lang_pair in args.lang_pairs:
                batch = next_batch_for_lang_pair(lang_pair)
                loss = criterion(model_for_lang_pair(lang_pair), batch)
                loss.backward()
            optimizer.step()
    In practice, `next_batch_for_lang_pair` is abstracted in a FairseqDataset
    (e.g., `RoundRobinZipDatasets`) and `model_for_lang_pair` is a model that
    implements the `FairseqMultiModel` interface.
    During inference it is required to specify a single `--source-lang` and
    `--target-lang`, instead of `--lang-pairs`.
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--dae-styles', default=None, metavar='STYLE',
                            help='comma-separated list of styles for DAE(in training order): humor, romance')
        parser.add_argument('--lambda-parallel-config', default="1.0", type=str, metavar='CONFIG',
                            help='cross-entropy reconstruction coefficient (parallel data). '
                                 'use fixed weight during training if set to floating point number. '
                                 'use piecewise linear function over number of updates to schedule the '
                                 'weight with the format: w0:step0,w1:step1,...')
        parser.add_argument('--lambda-denoising-config', default="0.0", type=str, metavar='CONFIG',
                            help='Cross-entropy reconstruction coefficient (denoising autoencoding)'
                                 'use fixed weight during training if set to floating point number. '
                                 'use piecewise linear function over number of updates to schedule the '
                                 'weight with the format: w0:step0,w1:step1,...')
        parser.add_argument('--max-word-shuffle-distance', default=3.0, type=float, metavar='N',
                            help='maximum word shuffle distance for denoising autoencoding data generation')
        parser.add_argument('--word-dropout-prob', default=0.1, type=float, metavar='N',
                            help='word dropout probability for denoising autoencoding data generation')
        parser.add_argument('--word-blanking-prob', default=0.2, type=float, metavar='N',
                            help='word blanking probability for denoising autoencoding data generation')

        parser.add_argument('--load-alignments', action='store_true', help='load the binarized alignments')
        parser.add_argument('--truncate-source', default=False, action='store_true',
                            help='boolean to truncate source to max-source-positions')

        parser.add_argument('data', metavar='DIR', help='path to data directory')
        parser.add_argument('--lang-pairs', default=None, metavar='PAIRS',
                            help='comma-separated list of language pairs (in training order): en-de,en-fr,de-fr')
        parser.add_argument('--model_lang_pairs', nargs="+", metavar='PAIRS',
                            help='Full list of lang pairs')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language (only needed for inference)')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language (only needed for inference)')
        parser.add_argument('--lazy-load', action='store_true',
                            help='load the dataset lazily')
        parser.add_argument('--raw-text', default=False, action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left (default: True)')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left (default: False)')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        parser.add_argument('--encoder-langtok', default=None, type=str, choices=['src', 'tgt'],
                            metavar='SRCTGT',
                            help='replace beginning-of-sentence in source sentence with source or target '
                                 'language token. (src/tgt)')

        parser.add_argument('--decoder-langtok', action='store_true',
                            help='replace beginning-of-sentence in target sentence with target language token')

        parser.add_argument('--divide-decoder-self-attn-norm', default='False', type=str, metavar='BOOL',
                            help='Decoder self-attn norm dependent on styles')
        parser.add_argument('--divide-decoder-embed-norm', default='False', type=str, metavar='BOOL',
                            help='Decoder embed norm dependent on styles')
        parser.add_argument('--divide-decoder-final-norm', default='False', type=str, metavar='BOOL',
                            help='Decoder final norm dependent on styles')
        parser.add_argument('--divide-decoder-encoder-attn-norm', default='False', type=str, metavar='BOOL',
                            help='Decoder encoder_attn norm dependent on styles')
        parser.add_argument('--divide-decoder-self-attn-query', default='False', type=str, metavar='BOOL',
                            help='Decoder self-attn query dependent on styles')
        parser.add_argument('--divide-decoder-encoder-attn-query', default='False', type=str, metavar='BOOL',
                            help='Decoder encoder-attn query dependent on styles')
        parser.add_argument('--lambda_repeat_penalty_intra', default=20, type=float,
                            help='repetition penalty parameter')
        parser.add_argument('--lambda_repeat_penalty_inter', default=5, type=float,
                            help='repetition penalty parameter')
        parser.add_argument('--sen_len_constraint', default=5, type=float,
                            help='sentence length constraint')
        parser.add_argument('--whether_concat', default='False', type=str, metavar='BOOL',
                            help='whether to generate one sentence for one pic')
        parser.add_argument('--encoder_type', default='transformer', type=str,
                            help='encoder type')
        parser.add_argument('--decoder_type', default='transformer', type=str,
                            help='decoder type')
        parser.add_argument('--pic_token_num', default=5, type=int,
                            help='token numbers for one picture')
        # 是否添加memory unit
        parser.add_argument('--whether_add_memory', default='False', type=str, metavar='BOOL',
                            help='whether to add memory unit')
        parser.add_argument('--n_memory_cells', default=1, type=int,
                            help='the length of memory unit')
        parser.add_argument('--whether_use_img_fea', default='False', type=str, metavar='BOOL',
                            help='whether to use image features')
        parser.add_argument('--whether_use_resnet_fea', default='False', type=str, metavar='BOOL',
                            help='whether to use resnet image features')
        parser.add_argument('--whether_use_text_clip_fea', default='False', type=str, metavar='BOOL',
                            help='whether to use text features')
        parser.add_argument('--whether_read_text_clip_fea', default='False', type=str, metavar='BOOL',
                            help='whether to use object features')
        parser.add_argument('--mode_type', default='train', type=str, metavar='BOOL',
                            help='train/test')
        parser.add_argument('--whether_clip_encode', default='True', type=str, metavar='BOOL',
                            help='whether use clip feature to update memory')

        # fmt: on

    def __init__(self, args, dicts, training):
        super().__init__(args)
        self.args = args
        self.dicts = dicts
        self.training = training
        if training:
            if args.lang_pairs is not None:
                self.summ_pair = args.lang_pairs[0]
                args.source_lang, args.target_lang = args.lang_pairs[0].split('-')
            else:
                self.summ_pair = None
        # else:
            # self.lang_pairs = ['{}-{}'.format(args.source_lang, args.target_lang)]
        self.langs = list(dicts.keys())

        #style数据集的pairs赋值
        self.dae_styles = args.dae_styles
        if self.dae_styles is not None:
            self.dae_styles = self.dae_styles.split(',')
            self.dae_pairs = ['{}_src-{}_tgt'.format(style, style) for style in self.dae_styles]
        else:
            self.dae_pairs = None
        self.model_lang_pairs = args.model_lang_pairs
        self.eval_lang_pairs = self.model_lang_pairs
        assert len(self.model_lang_pairs)

        self.lambda_parallel, self.lambda_parallel_steps = parse_lambda_config(args.lambda_parallel_config)
        self.lambda_denoising, self.lambda_denoising_steps = parse_lambda_config(args.lambda_denoising_config)


        self.generate_pairs = None

        self.all_img_features = None
        self.img_id_dict = None

        if(self.args.whether_use_img_fea=="True"):
            self.all_img_features = np.load("/features/clip/images/clip_features.npy")
            self.img_id_dict = json.load(open("/features/clip/images/img_id_dict.json","r"))

            print("load all features")
            #print(len(self.all_img_features),self.all_img_features[0])
            print(len(list(self.img_id_dict.keys())))

        # if (self.lambda_denoising > 0.0 or self.lambda_denoising_steps is not None):
        #     denoising_lang_pairs = [
        #         "%s-%s" % (tgt, tgt)
        #         for tgt in {lang_pair.split('-')[1] for lang_pair in args.lang_pairs}
        #     ]
        #

    @classmethod
    def setup_task(cls, args, **kwargs):
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        if getattr(args, 'raw_text', False):
            utils.deprecation_warning('--raw-text is deprecated, please use --dataset-impl=raw')
            args.dataset_impl = 'raw'
        elif getattr(args, 'lazy_load', False):
            utils.deprecation_warning('--lazy-load is deprecated, please use --dataset-impl=lazy')
            args.dataset_impl = 'lazy'

        if args.source_lang is not None or args.target_lang is not None:
            training = False
        else:
            training = True

        # load dictionaries
        dicts = OrderedDict()
        #非style数据集的dictionary
        if args.lang_pairs is not None:
            args.lang_pairs = args.lang_pairs.split(',')
            sorted_langs = sorted(list({x for lang_pair in args.lang_pairs for x in lang_pair.split('-')}))
            for lang in sorted_langs:
                paths = args.data.split(':')
                assert len(paths) > 0
                # dicts[lang] = Dictionary.load(os.path.join(paths[0], 'dict.{}.txt'.format(lang)))
                dicts[lang] = BertDictionary.load_from_file(os.path.join(paths[0], 'dict.{}.txt'.format(lang)))
                #print(dicts[lang].eos())
                #print(dicts[sorted_langs[0]].eos())
                if len(dicts) > 0:
                    assert dicts[lang].pad() == dicts[sorted_langs[0]].pad()
                    assert dicts[lang].eos() == dicts[sorted_langs[0]].eos()
                    assert dicts[lang].unk() == dicts[sorted_langs[0]].unk()
                if args.encoder_langtok is not None or args.decoder_langtok:
                    for lang_to_add in sorted_langs:
                        dicts[lang].add_symbol(_lang_token(lang_to_add))
                print('| [{}] dictionary: {} types'.format(lang, len(dicts[lang])))
        else:
            assert args.dae_styles

        #style数据集的dict
        if len(args.data.split(':')) > 2:
            #有多种style数据
            sorted_langs = sorted(list(args.dae_styles.split(',')))
            for idx, lang_style in enumerate(args.dae_styles.split(',')):
                paths = args.data.split(':')[1:]
                # dicts[lang] = Dictionary.load(os.path.join(paths[0], 'dict.{}.txt'.format(lang)))
                for suffix in ["_src","_tgt"]:
                    lang = lang_style+suffix
                    dicts[lang] = BertDictionary.load_from_file(os.path.join(paths[idx], 'dict.{}.txt'.format(lang)))
                    '''
                    if len(dicts) > 0:
                        assert dicts[lang].pad() == dicts[sorted_langs[0]].pad()
                        assert dicts[lang].eos() == dicts[sorted_langs[0]].eos()
                        assert dicts[lang].unk() == dicts[sorted_langs[0]].unk()'''
                    if args.encoder_langtok is not None or args.decoder_langtok:
                        for lang_to_add in sorted_langs:
                            dicts[lang].add_symbol(_lang_token(lang_to_add))
                    print('| [{}] dictionary: {} types'.format(lang, len(dicts[lang])))
        elif len(args.data.split(':')) > 1:
            style_pair=args.model_lang_pairs[1]
            print("syle_pairs_dict:"+style_pair)
            sorted_langs = sorted(list({x for x in style_pair.split('-')}))
            for lang in sorted_langs:
                paths = args.data.split(':')[1:]
                # dicts[lang] = Dictionary.load(os.path.join(paths[0], 'dict.{}.txt'.format(lang)))
                dicts[lang] = BertDictionary.load_from_file(os.path.join(paths[0], 'dict.{}.txt'.format(lang)))
                if len(dicts) > 0:
                    assert dicts[lang].pad() == dicts[sorted_langs[0]].pad()
                    assert dicts[lang].eos() == dicts[sorted_langs[0]].eos()
                    assert dicts[lang].unk() == dicts[sorted_langs[0]].unk()
                if args.encoder_langtok is not None or args.decoder_langtok:
                    for lang_to_add in sorted_langs:
                        dicts[lang].add_symbol(_lang_token(lang_to_add))
                print('| [{}] dictionary: {} types'.format(lang, len(dicts[lang])))
        '''
        if len(args.data.split(':')) > 1:
            sorted_langs = sorted(list(args.dae_styles.split(',')))
            for idx, lang in enumerate(args.dae_styles.split(',')):
                paths = args.data.split(':')[1:]
                # dicts[lang] = Dictionary.load(os.path.join(paths[0], 'dict.{}.txt'.format(lang)))
                dicts[lang] = BertDictionary.load_from_file(os.path.join(paths[idx], 'dict.txt'))
                if len(dicts) > 0:
                    assert dicts[lang].pad() == dicts[sorted_langs[0]].pad()
                    assert dicts[lang].eos() == dicts[sorted_langs[0]].eos()
                    assert dicts[lang].unk() == dicts[sorted_langs[0]].unk()
                if args.encoder_langtok is not None or args.decoder_langtok:
                    for lang_to_add in sorted_langs:
                        dicts[lang].add_symbol(_lang_token(lang_to_add))
                print('| [{}] dictionary: {} types'.format(lang, len(dicts[lang])))'''
        return cls(args, dicts, training)

    def build_model(self, args):
        from fairseq import models
        model = models.build_model(args, self)

        return model

    def load_dataset(self, split, epoch=0, combine=True, **kwargs):
        """Load a dataset split."""
        paths = self.args.data.split(':')
        assert len(paths) > 0
        # data_path = paths[epoch % len(paths)]
        para_data_path = paths[0]
        dae_data_paths = paths[1:]

        def split_exists(split, src, tgt, lang, data_path):
            if src is not None:
                filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
            else:
                filename = os.path.join(data_path, '{}.{}-None.{}'.format(split, src, tgt))
            if self.args.raw_text and IndexedRawTextDataset.exists(filename):
                return True
            elif not self.args.raw_text and IndexedDataset.exists(filename):
                return True
            return False

        def indexed_dataset(path, dictionary):
            if self.args.raw_text:
                return IndexedRawTextDataset(path, dictionary)
            elif IndexedDataset.exists(path):
                if self.args.lazy_load:
                    return IndexedDataset(path, fix_lua_indexing=True)
                else:
                    return IndexedCachedDataset(path, fix_lua_indexing=True)
            return None

        # load parallel datasets
        para_datasets = {}
        if (self.lambda_parallel > 0.0 or self.lambda_parallel_steps is not None):
            assert self.summ_pair is not None
            src, tgt = self.summ_pair.split('-')
            para_datasets[self.summ_pair] = load_langpair_dataset(
                para_data_path, split, src, self.dicts[src], tgt, self.dicts[tgt],
                combine=combine, dataset_impl=self.args.dataset_impl,
                upsample_primary=self.args.upsample_primary,
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
                max_source_positions=self.args.max_source_positions,
                max_target_positions=self.args.max_target_positions,
                load_alignments=self.args.load_alignments,
                truncate_source=self.args.truncate_source,
                lang_pair=self.summ_pair,
                whether_concat=self.args.whether_concat,
                whether_use_img_fea=self.args.whether_use_img_fea,
                all_img_features=self.all_img_features, img_id_dict=self.img_id_dict,
                whether_read_text_clip_fea=self.args.whether_read_text_clip_fea, whether_use_text_clip_fea=self.args.whether_use_text_clip_fea,
            )
        
        #style数据的loading, 可以参考parallel数据的
        noising_datasets = {}
        if (split != 'test' and self.lambda_denoising > 0.0 or self.lambda_denoising_steps is not None):
            for lang_pair, dae_data_path in zip(self.dae_pairs, dae_data_paths):
                self.style_pair = lang_pair
                src, tgt = self.style_pair.split('-')
                print(src)
                print(tgt)
                noising_datasets[self.style_pair] = load_langpair_dataset(
                    dae_data_path, split, src, self.dicts[src], tgt, self.dicts[tgt],
                    combine=combine, dataset_impl=self.args.dataset_impl,
                    upsample_primary=self.args.upsample_primary,
                    left_pad_source=self.args.left_pad_source,
                    left_pad_target=self.args.left_pad_target,
                    max_source_positions=self.args.max_source_positions,
                    max_target_positions=self.args.max_target_positions,
                    load_alignments=self.args.load_alignments,
                    truncate_source=self.args.truncate_source,
                    lang_pair=self.style_pair,
                    whether_concat=self.args.whether_concat,
                    whether_use_img_fea=self.args.whether_use_img_fea,
                    all_img_features=self.all_img_features, img_id_dict=self.img_id_dict,
                    whether_read_text_clip_fea=self.args.whether_read_text_clip_fea, whether_use_text_clip_fea=self.args.whether_use_text_clip_fea
                )


        # ADD：

        self.datasets[split] = RoundRobinZipDatasets(
            OrderedDict([
                (lang_pair, dataset)
                for lang_pair, dataset in para_datasets.items()
            ] + [
                (_get_denoising_dataset_key(lang_pair), dataset)
                for lang_pair, dataset in noising_datasets.items()
            ]),
            eval_key=None if split != 'test' else (list(para_datasets.keys()) + list(noising_datasets.keys()))[0]
        )

    def build_dataset_for_inference(self, src_tokens, src_sen_pos, src_lengths):
        # lang_pair = "%s-%s" % (self.args.source_lang, self.args.target_lang)
        lang_pair = self.args.lang_pairs
        return RoundRobinZipDatasets(
            OrderedDict([(
                lang_pair,
                self.alter_dataset_langtok(
                    LanguagePairDataset(
                        src_tokens, src_sen_pos, src_lengths,
                        self.source_dictionary,
                        lang_pair=lang_pair,
                    ),
                    src_eos=self.source_dictionary.eos(),
                    src_lang=self.args.source_lang,
                    tgt_eos=self.target_dictionary.eos(),
                    tgt_lang=self.args.target_lang,
                ),
            )]),
            eval_key=lang_pair,
        )

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        model.train()

        agg_loss, agg_sample_size, agg_logging_output = 0., 0., {}

        def forward_backward(model, samples, logging_output_key, weight):
            nonlocal agg_loss, agg_sample_size, agg_logging_output
            if samples is None or len(samples) == 0:
                return
            #print(discriminator)

            loss, sample_size, logging_output = criterion(model, samples)
            # loss, sample_size, logging_output = criterion(model, samples)
            if ignore_grad:
                loss *= 0
            else:
                loss *= weight

            #print(loss)
            optimizer.backward(loss)

            agg_loss += loss.detach().item()
            # TODO make summing of the sample sizes configurable
            agg_sample_size += sample_size
            #print(logging_output_key)
            agg_logging_output[logging_output_key] = logging_output

        if self.lambda_parallel > 0.0:
            forward_backward(model, sample[self.summ_pair], self.summ_pair, self.lambda_parallel)

        #训练步：style数据的forward_back_ward
        if self.lambda_denoising > 0.0:
            for lang_pair in self.dae_pairs:
                # _, tgt = lang_pair.split('-')
                sample_key = _get_denoising_dataset_key(lang_pair)
                forward_backward(model, sample[sample_key], sample_key, self.lambda_denoising)

        return agg_loss, agg_sample_size, agg_logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            agg_loss, agg_sample_size, agg_logging_output = 0., 0., {}
            #print(self.model_lang_pairs)
            #if(len(self.model_lang_pairs)!=2):print(self.model_lang_pairs)

            for lang_pair in self.model_lang_pairs:
                if sample is None or lang_pair not in sample or sample[lang_pair] is None or len(sample[lang_pair]) == 0:

                    continue

                loss, sample_size, logging_output = criterion(model, sample[lang_pair])
                agg_loss += loss.data.item()
                # TODO make summing of the sample sizes configurable
                agg_sample_size += sample_size
                agg_logging_output[lang_pair] = logging_output

        return agg_loss, agg_sample_size, agg_logging_output

    def inference_step(self, generator, models, sample, prefix_tokens=None):
        with torch.no_grad():
            return generator.generate(
                models,
                sample,
                prefix_tokens=prefix_tokens,
                bos_token=_lang_token_index(self.target_dictionary, self.args.target_lang)
                if self.args.decoder_langtok else self.target_dictionary.eos(),
            )

    def update_step(self, num_updates):
        def lambda_step_func(config, n_iter):
            """
            Update a lambda value according to its schedule configuration.
            """
            ranges = [i for i in range(len(config) - 1) if config[i][0] <= n_iter < config[i + 1][0]]
            if len(ranges) == 0:
                assert n_iter >= config[-1][0]
                return config[-1][1]
            assert len(ranges) == 1
            i = ranges[0]
            x_a, y_a = config[i]
            x_b, y_b = config[i + 1]
            return y_a + (n_iter - x_a) * float(y_b - y_a) / float(x_b - x_a)

        if self.lambda_parallel_steps is not None:
            self.lambda_parallel = lambda_step_func(self.lambda_parallel_steps, num_updates)
        if self.lambda_denoising_steps is not None:
            self.lambda_denoising = lambda_step_func(self.lambda_denoising_steps, num_updates)

    def aggregate_logging_outputs(self, logging_outputs, criterion):
        # aggregate logging outputs for each language pair
        # logging_output_keys = {
        #     key
        #     for logging_output in logging_outputs
        #     for key in logging_output
        # }
        # lang_pair_keys = set([self.summ_pair] if self.summ_pair is not None else [] + [
        #     _get_denoising_dataset_key(lang_pair)
        #     for lang_pair in self.dae_pairs
        # ] if self.dae_pairs is not None else [])
        # logging_output_keys = logging_output_keys.intersection(lang_pair_keys)
        logging_output_keys = self.model_lang_pairs + self.generate_pairs \
                if self.generate_pairs is not None else self.model_lang_pairs
        return self.upper_aggregate_logging_outputs(logging_outputs, criterion, logging_output_keys)

    def upper_aggregate_logging_outputs(self, logging_outputs, criterion, logging_output_keys=None):
        logging_output_keys = logging_output_keys or self.eval_lang_pairs
        # aggregate logging outputs for each language pair
        agg_logging_outputs = {
            key: criterion.__class__.aggregate_logging_outputs([
                logging_output.get(key, {}) for logging_output in logging_outputs
            ])
            for key in logging_output_keys
        }

        def sum_over_languages(key):
            return sum(logging_output[key] for logging_output in agg_logging_outputs.values())

        # flatten logging outputs
        flat_logging_output = {
            '{}:{}'.format(lang_pair, k): v
            for lang_pair, agg_logging_output in agg_logging_outputs.items()
            for k, v in agg_logging_output.items()
        }
        flat_logging_output['loss'] = sum_over_languages('loss')
        flat_logging_output['reward_loss'] = sum_over_languages('reward_loss')
        if any('nll_loss' in logging_output for logging_output in agg_logging_outputs.values()):
            flat_logging_output['nll_loss'] = sum_over_languages('nll_loss')
        flat_logging_output['sample_size'] = sum_over_languages('sample_size')
        flat_logging_output['nsentences'] = sum_over_languages('nsentences')
        flat_logging_output['ntokens'] = sum_over_languages('ntokens')
        return flat_logging_output

    @property
    def source_dictionary(self):
        return list(self.dicts.values())[0]

    @property
    def target_dictionary(self):
        return list(self.dicts.values())[1]

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        if len(self.datasets.values()) == 0:
            return {'%s-%s' % (self.args.source_lang, self.args.target_lang):
                        (self.args.max_source_positions, self.args.max_target_positions)}
        return OrderedDict([
            (key, (self.args.max_source_positions, self.args.max_target_positions))
            for split in self.datasets.keys()
            for key in self.datasets[split].datasets.keys()
        ])

    def grad_denom(self, sample_sizes, criterion):
        return criterion.__class__.grad_denom(sample_sizes)

    def init_logging_output(self, sample):
        return {
            'ntokens': sum(
                sample_lang.get('ntokens', 0)
                for sample_lang in sample.values()
            ) if sample is not None else 0,
            'nsentences': sum(
                sample_lang['target'].size(0) if 'target' in sample_lang else 0
                for sample_lang in sample.values()
            ) if sample is not None else 0,
        }

    def get_encoder_langtok(self, src_lang, tgt_lang):
        if self.args.encoder_langtok is None:
            return self.dicts[src_lang].eos()
        if self.args.encoder_langtok == 'src':
            return _lang_token_index(self.dicts[src_lang], src_lang)
        else:
            return _lang_token_index(self.dicts[src_lang], tgt_lang)

    def get_decoder_langtok(self, tgt_lang):
        if not self.args.decoder_langtok:
            return self.dicts[tgt_lang].eos()
        return _lang_token_index(self.dicts[tgt_lang], tgt_lang)

    def alter_dataset_langtok(self, lang_pair_dataset,
                              src_eos=None, src_lang=None, tgt_eos=None, tgt_lang=None):
        if self.args.encoder_langtok is None and not self.args.decoder_langtok:
            return lang_pair_dataset

        new_src_eos = None
        if self.args.encoder_langtok is not None and src_eos is not None \
                and src_lang is not None and tgt_lang is not None:
            new_src_eos = self.get_encoder_langtok(src_lang, tgt_lang)
        else:
            src_eos = None

        new_tgt_bos = None
        if self.args.decoder_langtok and tgt_eos is not None and tgt_lang is not None:
            new_tgt_bos = self.get_decoder_langtok(tgt_lang)
        else:
            tgt_eos = None

        return TransformEosLangPairDataset(
            lang_pair_dataset,
            src_eos=src_eos,
            new_src_eos=new_src_eos,
            tgt_bos=tgt_eos,
            new_tgt_bos=new_tgt_bos,
        )

    def build_generator(self, args):
        if getattr(args, 'score_reference', False):
            from fairseq.sequence_scorer import SequenceScorer
            return SequenceScorer(self.target_dictionary)
        else:
            from fairseq.sequence_generator import SequenceGeneratorWithAlignment
            from mass.sequence_generator import SequenceGenerator
            if getattr(args, 'print_alignment', False):
                seq_gen_cls = SequenceGeneratorWithAlignment
            else:
                seq_gen_cls = SequenceGenerator
            return seq_gen_cls(
                self.target_dictionary,
                beam_size=getattr(args, 'beam', 5),
                max_len_a=getattr(args, 'max_len_a', 0),
                max_len_b=getattr(args, 'max_len_b', 200),
                min_len=getattr(args, 'min_len', 1),
                normalize_scores=(not getattr(args, 'unnormalized', False)),
                len_penalty=getattr(args, 'lenpen', 1),
                unk_penalty=getattr(args, 'unkpen', 0),
                sampling=getattr(args, 'sampling', False),

                sampling_topk=getattr(args, 'sampling_topk', -1),
                sampling_topp=getattr(args, 'sampling_topp', -1.0),
                temperature=getattr(args, 'temperature', 1.),
                diverse_beam_groups=getattr(args, 'diverse_beam_groups', -1),
                diverse_beam_strength=getattr(args, 'diverse_beam_strength', 0.5),
                match_source_len=getattr(args, 'match_source_len', False),
                no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
                lambda_repeat_penalty_intra=getattr(args, 'lambda_repeat_penalty_intra', 20),
                lambda_repeat_penalty_inter=getattr(args, 'lambda_repeat_penalty_inter', 10),

                sen_len_constraint=getattr(args, 'sen_len_constraint', 5),
                whether_add_memory=getattr(args, 'whether_add_memory', False),
                whether_concat=getattr(args, 'whether_concat', False),
                whether_use_img_fea=getattr(args, 'whether_use_img_fea', False),
                mode_type=getattr(args, 'mode_type', "test"),
                whether_clip_encode=getattr(args, 'whether_clip_encode', "True"),
            )

    @classmethod
    def load_dictionary(cls, filename):
        return BertDictionary.load_from_file(filename)

    def build_generator_cls(self, args):
        from mass.sequence_generator import SequenceGenerator
        return SequenceGenerator(
            self.target_dictionary,
            beam_size=getattr(args, 'beam', 1),
            max_len_a=getattr(args, 'max_len_a', 0),
            max_len_b=getattr(args, 'max_len_b', 200),
            min_len=getattr(args, 'min_len', 1),
            normalize_scores=(not getattr(args, 'unnormalized', False)),
            len_penalty=getattr(args, 'lenpen', 1),
            unk_penalty=getattr(args, 'unkpen', 0),
            sampling=getattr(args, 'sampling', True),
            sampling_topk=getattr(args, 'sampling_topk', 5),
            sampling_topp=getattr(args, 'sampling_topp', -1.0),
            temperature=getattr(args, 'temperature', 1.),
            diverse_beam_groups=getattr(args, 'diverse_beam_groups', -1),
            diverse_beam_strength=getattr(args, 'diverse_beam_strength', 0.5),
            match_source_len=getattr(args, 'match_source_len', False),
            no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
            lambda_repeat_penalty_intra=getattr(args, 'lambda_repeat_penalty_intra', 20),
            lambda_repeat_penalty_inter=getattr(args, 'lambda_repeat_penalty_inter', 10),
            sen_len_constraint=getattr(args, 'sen_len_constraint', 5),
            whether_add_memory=getattr(args, 'whether_add_memory', False),
            whether_concat=getattr(args, 'whether_concat', False),
            whether_use_img_fea=getattr(args, 'whether_use_img_fea', False),
            mode_type=getattr(args, 'mode_type', "test"),
            whether_clip_encode=getattr(args, 'whether_clip_encode', "True"),
        )

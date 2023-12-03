#opyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import warnings
warnings.filterwarnings("ignore")

from fairseq import utils

import torch.nn as nn

from fairseq.criterions import FairseqCriterion, register_criterion
softmax = nn.Softmax(dim=1)


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)

    if ignore_index is not None:
        non_pad_mask = target.ne(ignore_index)
        nll_loss = nll_loss[non_pad_mask]
        smooth_loss = smooth_loss[non_pad_mask]
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        sum_nll_loss = nll_loss.sum()
        sum_smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * sum_nll_loss + eps_i * sum_smooth_loss
    return loss, sum_nll_loss

def format_label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        non_pad_mask = target.ne(ignore_index)
        nll_loss = nll_loss[non_pad_mask]
        smooth_loss = smooth_loss[non_pad_mask]
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        sum_nll_loss = nll_loss.sum()
        sum_smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * sum_nll_loss + eps_i * sum_smooth_loss
    return loss, sum_nll_loss


@register_criterion('model_loss')
class ModelLoss(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.decoder_type = args.decoder_type

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')

        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        #print(net_output)
        loss = None
        nll_loss = None
        sen_num=5
        sen_num = len(net_output)
        whether_concat = True
        if(sen_num!=5):
            loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
            with_reward_loss = loss.clone()
        else:
            for i in range(sen_num):
                t_sample = {'target':sample["multi_target"][i]}

                t_loss, t_nll_loss= self.compute_loss(model, net_output[i], t_sample, reduce=reduce)

                if(loss==None):loss = t_loss
                else: loss += t_loss
                if(nll_loss==None):nll_loss = t_nll_loss
                else: nll_loss += t_nll_loss
            with_reward_loss = loss.clone()

        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']

        #logging_output["loss"]=utils.item(with_reward_loss.data) if reduce else with_reward_loss.data
        #return with_reward_loss, sample_size, logging_output
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'reward_loss': utils.item(with_reward_loss.data) if reduce else with_reward_loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }

        #print(logging_output)
        return loss, sample_size, logging_output


    def compute_loss(self, model, net_output, sample, reduce=True):

        lprobs_batch = model.get_normalized_probs(net_output, log_probs=True)
        target_bacth = model.get_targets(sample, net_output)
        #print("lprobs_batch",lprobs_batch.shape)
        #print("target_bacth",target_bacth.shape)
        if(target_bacth.size(1)>lprobs_batch.size(1)):
            target_bacth = target_bacth[:,:lprobs_batch.size(1)].contiguous()
        elif(target_bacth.size(1)<lprobs_batch.size(1)):
            lprobs_batch = lprobs_batch[:,:target_bacth.size(1),:].contiguous()
        lprobs = lprobs_batch.view(-1, lprobs_batch.size(-1))
        target =target_bacth.view(-1)
        loss, nll_loss= label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )

        # print(reward_loss)

        return loss, nll_loss



    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'reward_loss': sum(log.get('reward_loss', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2) if ntokens > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }

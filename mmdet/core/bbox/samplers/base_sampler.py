# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

import torch

from .sampling_result import SamplingResult


class BaseSampler(metaclass=ABCMeta):
    """Base class of samplers."""

    def __init__(self,
                 num,
                 pos_fraction,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True,
                 **kwargs):
        self.num = num
        self.pos_fraction = pos_fraction
        self.neg_pos_ub = neg_pos_ub
        self.add_gt_as_proposals = add_gt_as_proposals
        self.pos_sampler = self
        self.neg_sampler = self

    """
        num (int)： 总样本采样数 = 正样本采样数 + 负样本采样数
        pos_fraction (float): 期望的正样本采样比例，那么期望的正样本采样数 = num * pos_fraction
        neg_pos_ub (int, optional): 基于正样本数的负样本数上限基准 => 每采样1个正样本，就要采样neg_pos_ub个负样本
        add_gt_as_proposal： 是否把gt当作proposal，两阶段检测器的roi_head会用到该参数
    """
    # 抽象方法：由子类实现正负样本采样规则 => self.pos_sampler._sample_pos / self.neg_sampler._sample_neg
    @abstractmethod
    def _sample_pos(self, assign_result, num_expected, **kwargs):
        """Sample positive samples."""
        pass

    @abstractmethod
    def _sample_neg(self, assign_result, num_expected, **kwargs):
        """Sample negative samples."""
        pass

    def sample(self,
               assign_result,
               bboxes,
               gt_bboxes,
               gt_labels=None,
               **kwargs):
        """Sample positive and negative bboxes.

        This is a simple implementation of bbox sampling given candidates,
        assigning results and ground truth bboxes.

        Args:
            assign_result (:obj:`AssignResult`): Bbox assigning results.
            bboxes (Tensor): Boxes to be sampled from.
            gt_bboxes (Tensor): Ground truth bboxes.
            gt_labels (Tensor, optional): Class labels of ground truth bboxes.

        Returns:
            :obj:`SamplingResult`: Sampling result.

        Example:
            >>> from mmdet.core.bbox import RandomSampler
            >>> from mmdet.core.bbox import AssignResult
            >>> from mmdet.core.bbox.demodata import ensure_rng, random_boxes
            >>> rng = ensure_rng(None)
            >>> assign_result = AssignResult.random(rng=rng)
            >>> bboxes = random_boxes(assign_result.num_preds, rng=rng)
            >>> gt_bboxes = random_boxes(assign_result.num_gts, rng=rng)
            >>> gt_labels = None
            >>> self = RandomSampler(num=32, pos_fraction=0.5, neg_pos_ub=-1,
            >>>                      add_gt_as_proposals=False)
            >>> self = self.sample(assign_result, bboxes, gt_bboxes, gt_labels)
        """
        if len(bboxes.shape) < 2:
            bboxes = bboxes[None, :]

        # 切片操作，保留第一个维度:bboxes_num; 第二个维度的前4个元素: 左上右下两点坐标
        bboxes = bboxes[:, :4]

        # gt_flag初始化为0
        gt_flags = bboxes.new_zeros((bboxes.shape[0], ), dtype=torch.uint8)
        # 是否也要把gt当作proposal
        if self.add_gt_as_proposals and len(gt_bboxes) > 0:
            if gt_labels is None:
                raise ValueError(
                    'gt_labels must be given when add_gt_as_proposals is True')
            # 把gt_bboxes也算在bboxes内，用cat连接
            bboxes = torch.cat([gt_bboxes, bboxes], dim=0)
            # 使用add_gt_添加label属性(如果传入的gt_labels还是None则没作用)
            assign_result.add_gt_(gt_labels)
            # cat连接后也修改gt_flags，gt_bboxes对应的gt_flags为1
            gt_ones = bboxes.new_ones(gt_bboxes.shape[0], dtype=torch.uint8)
            gt_flags = torch.cat([gt_ones, gt_flags])

        # 期望的正样本采样个数 = 总样本采样数 * 期望的正样本采样比例
        num_expected_pos = int(self.num * self.pos_fraction)
        # 调用子类实现的抽象方法，进行正样本采样
        pos_inds = self.pos_sampler._sample_pos(
            assign_result, num_expected_pos, bboxes=bboxes, **kwargs)
        # We found that sampled indices have duplicated items occasionally.
        # (may be a bug of PyTorch)
        # 采样后，unique()去重复
        pos_inds = pos_inds.unique()
        # 正样本采样后的个数
        num_sampled_pos = pos_inds.numel()
        # 期望负样本采样个数 = 总样本采样数- 正样本采样个数
        num_expected_neg = self.num - num_sampled_pos
        if self.neg_pos_ub >= 0:
            # 每采样1个正样本，就要采样neg_pos_ub个负样本
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        # 调用子类实现的抽象方法，进行负样本采样
        neg_inds = self.neg_sampler._sample_neg(
            assign_result, num_expected_neg, bboxes=bboxes, **kwargs)
        # 采样后，unique()去重复
        neg_inds = neg_inds.unique()

        # 构造sampling_result封装数据进行返回
        # gt_flags这个参数，几乎只为add_gt_as_proposals这个参数服务
        sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                                         assign_result, gt_flags)
        return sampling_result

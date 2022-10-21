# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
from mmcv.runner import force_fp32

from mmdet.core import (anchor_inside_flags, build_assigner, build_bbox_coder,
                        build_prior_generator, build_sampler, images_to_levels,
                        multi_apply, unmap)
from ..builder import HEADS, build_loss
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin


@HEADS.register_module()
class AnchorHead(BaseDenseHead, BBoxTestMixin):
    """Anchor-based head (RPN, RetinaNet, SSD, etc.).

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels. Used in child classes.
        anchor_generator (dict): Config dict for anchor generator
        bbox_coder (dict): Config of bounding box coder.
        reg_decoded_bbox (bool): If true, the regression loss would be
            applied directly on decoded bounding boxes, converting both
            the predicted boxes and regression targets to absolute
            coordinates format. Default False. It should be `True` when
            using `IoULoss`, `GIoULoss`, or `DIoULoss` in the bbox head.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
        
        num_classes（int）：不包括background的类别数
        in_channels（int）：输入特征图中的通道数
        feat_channels（int）：隐藏层（或称为中间层）的通道数，用于子类[为啥不直接在子类中定义？因为所有子类都要用]
        anchor_generator（dict）：该anchor_head采用的anchor_generator
        bbox_coder (dict）：该anchor_head采用的bbox_coder
        reg_decoded_bbox（bool）：如果为True，则直接用bbox的坐标值进行回归计算Loss，coder无效 => 适用IoU系列Loss
                                 默认为False，使用诸如delta_xywh_bbox_coder的编码器归一化成delta回归计算Loss => 适用L1Loss
        loss_cls（dict）： 该anchor_head采用的分类Loss
        loss_bbox（dict）：该anchor_head采用的回归Loss
        train_cfg（dict）：训练配置 => 主要设置assigner(iou_calculator)和sampler
        test_cfg（dict）： 测试配置 => 主要设置nms
        init_cfg（dict或list[dict]，可选）：初始化配置(比较少见)
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     scales=[8, 16, 32],
                     ratios=[0.5, 1.0, 2.0],
                     strides=[4, 8, 16, 32, 64]),
                 bbox_coder=dict(
                     type='DeltaXYWHBBoxCoder',
                     clip_border=True,
                     target_means=(.0, .0, .0, .0),
                     target_stds=(1.0, 1.0, 1.0, 1.0)),
                 reg_decoded_bbox=False,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=dict(type='Normal', layer='Conv2d', std=0.01)):
        super(AnchorHead, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        # 一般都是num_classes + 1（背景类）
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        # 检查一下num_class参数
        if self.cls_out_channels <= 0:
            raise ValueError(f'num_classes={num_classes} is too small')
        self.reg_decoded_bbox = reg_decoded_bbox

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            if hasattr(self.train_cfg,
                       'sampler') and self.train_cfg.sampler.type.split(
                           '.')[-1] != 'PseudoSampler':
                self.sampling = True
                sampler_cfg = self.train_cfg.sampler
                # avoid BC-breaking
                # sampler 用于正负样本均衡 => 如果使用的是FocalLoss则不用进行正负样本采样，采用PseudoSampler保持接口一致
                if loss_cls['type'] in [
                        'FocalLoss', 'GHMC', 'QualityFocalLoss'
                ]:
                    warnings.warn(
                        'DeprecationWarning: Determining whether to sampling'
                        'by loss type is deprecated, please delete sampler in'
                        'your config when using `FocalLoss`, `GHMC`, '
                        '`QualityFocalLoss` or other FocalLoss variant.')
                    self.sampling = False
                    sampler_cfg = dict(type='PseudoSampler')
            else:
                # 如果train_cfg中没有sampler配置，采用PseudoSampler保持接口一致
                self.sampling = False
                sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.fp16_enabled = False

        self.prior_generator = build_prior_generator(anchor_generator)

        # Usually the numbers of anchors for each level are the same
        # except SSD detectors. So it is an int in the most dense
        # heads but a list of int in SSDHead
        self.num_base_priors = self.prior_generator.num_base_priors[0]
        self._init_layers()

    @property
    def num_anchors(self):
        warnings.warn('DeprecationWarning: `num_anchors` is deprecated, '
                      'for consistency or also use '
                      '`num_base_priors` instead')
        return self.prior_generator.num_base_priors[0]

    @property
    def anchor_generator(self):
        warnings.warn('DeprecationWarning: anchor_generator is deprecated, '
                      'please use "prior_generator" instead')
        return self.prior_generator

    def _init_layers(self):
        """Initialize layers of the head."""
        self.conv_cls = nn.Conv2d(self.in_channels,
                                  self.num_base_priors * self.cls_out_channels,
                                  1)
        self.conv_reg = nn.Conv2d(self.in_channels, self.num_base_priors * 4,
                                  1)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level \
                    the channels number is num_base_priors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale \
                    level, the channels number is num_base_priors * 4.

        单层level上的forward=>只需要计算该层level上的cls_score和bbox_pred
        该函数主要被父类调用，完成巻积运算后用于计算loss
        """
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        return cls_score, bbox_pred

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_scores (list[Tensor]): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * num_classes.
                - bbox_preds (list[Tensor]): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * 4.
        multi_apply函数，将feat[neck输出的多尺度特征层]拆分成多个输入(每个输入一个level)，然后调用forward_single得到分层输出
        再使用tuple(map(list, zip(return from forward_single)))的数据形式保存并返回
        """
        return multi_apply(self.forward_single, feats)

    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
            device (torch.device | str): Device for returned tensors

        Returns:
            tuple:
                anchor_list (list[Tensor]): Anchors of each image.
                valid_flag_list (list[Tensor]): Valid flags of each image.
        获得多层级level的grid anchor坐标和其对应的valid flag
        """
        num_imgs = len(img_metas)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        # 得到所有level层级feature map上的grid anchor，每个层级一个list
        multi_level_anchors = self.prior_generator.grid_priors(
            featmap_sizes, device=device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        # 得到所有level层级feature map上的grid anchor的valid_flag,映射在padding上的flag为false，后续不计算loss，每个层级一个list
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.prior_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'], device)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list

    def _get_targets_single(self,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            label_channels=1,
                            unmap_outputs=True):
        """Compute regression and classification targets for anchors in a
        single image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            img_meta (dict): Meta info of the image.
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:  <=todo: 感觉这一段注释是错的，这写的应该是get_target的return注释
            tuple:
                labels_list (list[Tensor]): Labels of each level
                label_weights_list (list[Tensor]): Label weights of each level
                bbox_targets_list (list[Tensor]): BBox targets of each level
                bbox_weights_list (list[Tensor]): BBox weights of each level
                num_total_pos (int): Number of positive samples in all images
                num_total_neg (int): Number of negative samples in all images

        计算单个图像中，其gt_bboxes和grid anchor确定的target，的classification 和 regression

        输入参数:
            flat_anchors (Tensor): 单个图像的multi-level的grid anchor，但是被flatten了
            unmap_outputs: 是否将输出(inside_flat_anchor)映射回原始的anchor集合中

        返回:
            翻译返回一个tuple: 错的就不翻译了

            实际返回一个tuple:
            return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result)
                labels(Tensor): 单张图片multi-level上的label经过flatten， flat_label <=赋值的是num_class
                label_weights (Tensor): 单张图片multi-level上的flat_label weight
                bbox_targets (Tensor): 单张图片multi-level上的flat_bbox_targets
                bbox_weights (Tensor): 单张图片multi-level上的flat_bbox_weights
                pos_inds (Tensor): 单张图片multi-level上采样后的正样本数量
                neg_inds (Tensor): 单张图片multi-level上采样后的负样本数量
                sampling_result (SamplingResult): 采样结果（由self.sampler.sample()返回）

        """
        # 之前使用pad_shape检查得到valid_flags，现在又使用img_shape检查inside_flags
        # 一般train_cfg的allowed_border=-1，所以会进一步筛选
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        # inside_flag为True的flat_anchor才会进行后续的assign、sampler => 这之后都是对inside_flat_anchor操作了
        anchors = flat_anchors[inside_flags, :]

        # 输入anchors、gt_bboxes、gt_bboxes_ignore进行正负样本分配
        # 通过assinger.assign得到单张图片的assign_result
        assign_result = self.assigner.assign(
            anchors, gt_bboxes, gt_bboxes_ignore,
            None if self.sampling else gt_labels)
        # 输入assign_result、anchors、gt_bboxes进行正负样本采样
        # 通过sampler.sample得到单张图片的sampling_result
        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        # todo: 为什么label要用num_classes初始化？
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        # 如果该图像中含有正样本
        if len(pos_inds) > 0:
            # reg_decoded_bbox默认为False，进入该分支，
            # 表明要先经过coder.encode将bbox转为delta之类的编码，再回归计算Loss
            if not self.reg_decoded_bbox:
                # 只有正样本才encode
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            # reg_decoded_bbox为True，直接使用bbox回归
            else:
                # 使用sampling_result中的pos_gt_bboxes属性，直接返回正样本对应的gt_bboxes
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            # 重新建立bbox_target，被分配为负样本/忽略样本的anchor统统赋0，只赋值正样本的bbox(encode后的)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            # 负样本/忽略样本的bbox权重为0，正样本的bbox权重为1
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            # 正样本的权重，实际在label_weight[pos_inds]上
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        # 负样本的权重
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        # 把现在inside_flat_anchor拓展回原来的flat_anchor的尺度，fill参数代表那些在边界外的anchor要填充的数字
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result)

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True,
                    return_sampling_results=False):
        """Compute regression and classification targets for anchors in
        multiple images.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 4).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: Usually returns a tuple containing learning targets.

                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each
                  level.
                - bbox_targets_list (list[Tensor]): BBox targets of each level.
                - bbox_weights_list (list[Tensor]): BBox weights of each level.
                - num_total_pos (int): Number of positive samples in all
                  images.
                - num_total_neg (int): Number of negative samples in all
                  images.

            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end

        计算多张图像(数量是1个batch_size)的targets的classification和regression
        参数：
            anchor_list (list[list[Tensor]]): 外层的list代指图片列表，内层的list代指图片的feature levels，每个anchor四个参数(左上右下)
            valid_flag_list (list[list[Tensor]])： 同上(每个anchor一个参数，True or False)
            img_metas (list[dict])： 图像元数据，list代指图像列表
            gt_bboxes_list (list[Tensor])： GT的Bbox list
            gt_labels_list (list[Tensor])： GT的Label list
            gt_bboxes_ignore_list (list[Tensor])： 需要被忽视的GT的Bbox list
            label_channels (int): label的通道数
            unmap_outputs (bool): 是否将输出(inside_flat_anchor)映射回原始的anchor集合中

        返回：
                label_list(list[Tensor]): 一个batch_size的图片上按multi-level分组的label，一层level一个list
                label_weights_list (list[Tensor]): 一个batch_size的图片上按multi-level分组的label weight
                bbox_targets_list (list[Tensor]): 一个batch_size的图片上按multi-level分组的bbox_targets
                bbox_weights_list (list[Tensor]): 一个batch_size的图片上按multi-level分组的bbox_weights
                num_total_pos (int): 一个batch_size的图片上按multi-level分组的所有的正样本数量
                num_total_neg (int): 一个batch_size的图片上按multi-level分组上所有的负样本数量
        """
        # 根据输入参数，收集get_targets_single()需要的参数
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors to a single tensor
        # 这一步操作就是 flatten multi-level的grid anchors
        # 将单张图片的multi-level的grid anchors list及其对应valid flag list通过cat连接，每张图片用list.append连接形成图片列表
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

        # compute targets for each image
        # 相当于c++中的int* a = new int[10]，声明一个没有初始化(初始化为None)，但有固定size的变量
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]

        # 调用multi_apply(get_targets_single())，调用batch_size次
        results = multi_apply(
            self._get_targets_single,
            concat_anchor_list,
            concat_valid_flag_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)
        # 把get_targets_single()的返回值一一取出
        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         pos_inds_list, neg_inds_list, sampling_results_list) = results[:7]
        rest_results = list(results[7:])  # user-added return values
        # no valid anchors
        # 如果没有有效的anchor直接返回None
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        # 统计一组batch_size的img_metas (list[dict])中的所有采样后的正负样本数量
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        # 因为res return到loss()中, loss是按单层level调用loss_single()计算loss，所以要转为按multi-level分组
        # [target_img0, target_img1, ... target_img(num_img)] -> [target_level0, target_level1, ... target_level4]
        # num_img对应1个batch_size， level0 ~ level4 对应 FPN层的5层输出
        # target_img (list : num_img)，其中list(Tensor): Tensor(level0 + level1 + level2 + level3 + level4,)
        # target_level (list : num_level)，其中list(Tensor): Tensor(num_img, level0)...Tensor(num_img, level4)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        # result 拆分成tuple(img)， tuple(img) 转为 tuple(multi-level), 再封装成 res (除了sampling_result_list)
        res = (labels_list, label_weights_list, bbox_targets_list,
               bbox_weights_list, num_total_pos, num_total_neg)

        if return_sampling_results:
            res = res + (sampling_results_list, )
        # rest_results = list(results[7:])
        # 把除前7个tuple外的额外属性(一般是用户自定义的)也转为multi-level分组，然后加到res末尾
        for i, r in enumerate(rest_results):  # user-added return values
            rest_results[i] = images_to_levels(r, num_level_anchors)

        return res + tuple(rest_results)

    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (N, num_total_anchors, 4).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        # 计算单层level上的loss
        """

        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, 4)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_cls, loss_bbox

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.

        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device

        # 调用get_anchors获取multi-level上的所有grid anchor和其对应的valid_flag
        # list(num_img) 其中list: list(num_level)，再其中list: Tensor(level0) ~ Tensor(level4)
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        # label_channels
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        # 调用get_targets()，通过grid_anchor和gt_bboxes检索target，获取cls_reg_targets
        # get_targets()返回按multi-level分组表示的res(labels_list, label_weights_list, bbox_targets_list,
        #                bbox_weights_list, num_total_pos, num_total_neg, [sampling_results], [rest_results])
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)

        if cls_reg_targets is None:
            return None
        # 又拆出来,把num_total_pos和num_total_neg合并为num_total_samples
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        # list(num_img)，其中list:Tensor(level0 + level1 + ... level4)
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        # list(num_level)，其中list: Tensor(num_img, level0) ~ Tensor(num_img，level4)
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        # 调用multi_apply(loss_single)需要以下准备:
        # 已定义好的loss_single
        # 本身的传入参数cls_scores， bbox_preds
        # get_anchor返回处理的all_anchor_list
        # get_target返回处理的(labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_samples)
        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    def aug_test(self, feats, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            feats (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains features for all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5), where
                5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,), The length of list should always be 1.
        """
        return self.aug_test_bboxes(feats, img_metas, rescale=rescale)

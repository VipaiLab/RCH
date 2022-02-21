from abc import ABCMeta, abstractmethod
import math
import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16, force_fp32
import torch.distributed as dist
from mmseg.core import build_pixel_sampler
from mmseg.ops import resize
from ..builder import build_loss
from ..losses import accuracy


class BaseDecodeHead(BaseModule, metaclass=ABCMeta):
    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 input_transform=None,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 ignore_index=255,
                 sampler=None,
                 use_cl = False,
                 cl_type = 'CrossEntropyLoss',
                 mom = 0.9,
                 tmp = 0.5,
                 threshold = 0.0,
                 alp = 2,
                 align_corners=False,
                 init_cfg=dict(
                     type='Normal', std=0.01, override=dict(name='conv_seg'))):
        super(BaseDecodeHead, self).__init__(init_cfg)
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index
        self.loss_decode = build_loss(loss_decode)
        if use_cl:
            loss_cl_fn = dict(type=cl_type,use_sigmoid=False,loss_weight=0.1)
            self.loss_cl_fn = build_loss(loss_cl_fn)
            loss_cl_fp = dict(type=cl_type,use_sigmoid=False,loss_weight=0.1)
            self.loss_cl_fp = build_loss(loss_cl_fp)
        self.ignore_index = ignore_index
        self.align_corners = align_corners
        self.use_cl = use_cl
        self.mom = mom
        self.tmp = tmp
        self.threshold = threshold
        self.alp = alp
        if self.use_cl:
            avg_f = torch.randn(self.channels, self.num_classes)
            self.register_buffer('avg_f', avg_f)
            self.cl_conv = ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=1,
                    padding=0,
                    conv_cfg=None,
                    norm_cfg=self.norm_cfg,
                    act_cfg=None
                    )
        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None

        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.fp16_enabled = False
    def onehot(self, label):
        '''
        input: label: b h w
        output: b h w c
        '''
        lbl = label.clone()
        N = self.num_classes + 1
        lbl[lbl==self.ignore_index] = N - 1
        size = list(lbl.size())
        lbl = lbl.view(-1)
        ones = torch.sparse.torch.eye(N).cuda()
        ones = ones.index_select(0, lbl.long())
        size.append(N)
        return ones.view(*size)[...,:(N-1)].float()
    def get_mask_fn_fp(self, lbl_one, pred_one, logit):
        '''input: b h w c        output: b h w c'''
        fg = lbl_one.sum(3,keepdim=True) != 0
        fg = fg.float()
        pred_one = pred_one*fg
        tp = lbl_one*pred_one
        fn = lbl_one - tp
        fp = pred_one - tp
        num_tp = tp.sum(1).sum(1)
        num_pred = pred_one.sum(1).sum(1)
        num_gt = lbl_one.sum(1).sum(1)
        t_fn = num_tp/(num_gt + 1e-8) + 1e-6
        t_fp = num_tp/(num_pred + 1e-8) + 1e-6 
        self.t_fn = t_fn.unsqueeze(2)
        self.t_fp = t_fp.unsqueeze(2)
        return lbl_one, fn, fp
    def local_avg_tp_fn_fp(self, f, mask, fn, fp):
        f = f.detach()
        b,h,w,c = mask.size()
        d = f.size(1)
        f = f.reshape(b,d,h*w)

        mask = mask.reshape(b,h*w,c)
        mask_sum = mask.sum(1,keepdim=True)
        index = mask_sum > 0
        index = index.float()
        mask = mask/(torch.norm(mask,p=1,dim=1,keepdim=True) + 1e-8)
        f_mask = torch.bmm(f,mask)

        avg_f = self.avg_f.repeat(b,1,1)

        fn = fn.reshape(b,h*w,c)
        fn = fn/(torch.norm(fn,p=1,dim=1,keepdim=True) + 1e-8)
        f_fn = torch.bmm(f,fn)

        fp = fp.reshape(b,h*w,c)
        fp = fp/(torch.norm(fp,p=1,dim=1,keepdim=True) + 1e-8)
        f_fp = torch.bmm(f,fp)

        with torch.no_grad():
            f_mask_g = f_mask.sum(0)
            index = index.sum(0) + 1e-12
            dist.all_reduce(f_mask_g, op=dist.reduce_op.SUM)
            dist.all_reduce(index, op=dist.reduce_op.SUM)
            f_mask_g = f_mask_g/index
            index[index > 0.5] = self.mom
            index[index <= 0.5] = 1.0
            self.avg_f = self.avg_f*index
            self.avg_f += (1-index)*f_mask_g
        return avg_f, f_fn, f_fp
    def get_score(self, feature, lbl_one, logit, avg_f_tp, f_fn, f_fp):
        b,d,h,w = feature.size()
        c = self.num_classes
        lbl_one = lbl_one.permute(0,3,1,2).reshape(b,c,h*w)
        logit = logit.reshape(b,c,h*w)
        feature = feature.reshape(b,d,h*w)
        feature = feature/(torch.norm(feature, p=2, dim=1, keepdim=True)+1e-6)

        avg_f_tp = avg_f_tp.permute(0,2,1) # b,c,d
        avg_f_tp = avg_f_tp/(torch.norm(avg_f_tp, p=2, dim=2, keepdim=True)+1e-6)

        f_fn = f_fn.permute(0,2,1) # b,c,d
        f_fn = f_fn/(torch.norm(f_fn, p=2, dim=2, keepdim=True)+1e-6)

        f_fp = f_fp.permute(0,2,1) # b,c,d
        f_fp = f_fp/(torch.norm(f_fp, p=2, dim=2, keepdim=True)+1e-6)
        
       # b c n
        score_fn = torch.relu( torch.bmm(f_fn, feature) - self.t_fn )
        score_fp = torch.relu( torch.bmm(f_fp, feature) - self.t_fp )

        p_map = - ((1-logit)**self.alp)*lbl_one
        fn_map = p_map*score_fn
        fp_map = p_map*score_fp

        score_avg_tp = torch.bmm(avg_f_tp, feature)

        score_cl_fn = score_avg_tp + fn_map
        score_cl_fn = score_cl_fn.view(b,c,h,w)

        score_cl_fp = score_avg_tp + fp_map
        score_cl_fp = score_cl_fp.view(b,c,h,w)
        return score_cl_fn/self.tmp, score_cl_fp/self.tmp
    def cl_forward(self, feature, gt, logit):
        feature = self.cl_conv(feature)
        b,_,h,w = logit.size()
        gt = interpolate(gt, (h,w)).squeeze(1)
        pred = logit.max(1)[1]
        pred_one = self.onehot(pred)
        lbl_one = self.onehot(gt)

        logit = torch.softmax(logit, 1)
        logit_detach = logit.permute(0,2,3,1)
        mask, fn, fp = self.get_mask_fn_fp(lbl_one, pred_one, logit_detach)
        avg_f_tp, f_fn, f_fp = self.local_avg_tp_fn_fp(feature, mask, fn, fp)
        score_cl_fn, score_cl_fp = self.get_score(feature, lbl_one, logit, avg_f_tp, f_fn, f_fp)
        #score_batch = F.upsample(score_batch, size=(H,W), mode='bilinear', align_corners=True)
        #score_ds = F.upsample(score_ds, size=(H,W), mode='bilinear', align_corners=True)
        return {"cl_fn": score_cl_fn, "cl_fp": score_cl_fp}

    def extra_repr(self):
        """Extra repr."""
        s = f'input_transform={self.input_transform}, ' \
            f'ignore_index={self.ignore_index}, ' \
            f'align_corners={self.align_corners}'
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    @auto_fp16()
    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        seg_logits, feature = self.forward(inputs)
        if self.use_cl:
            logit = seg_logits.detach()
            gt = gt_semantic_seg.float().detach()
            cl_logits = self.cl_forward(feature, gt, logit)
            losses = self.losses(seg_logits, gt_semantic_seg, cl_logits)
            return losses
        else:
            losses = self.losses(seg_logits, gt_semantic_seg, False)
            return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        logit, f = self.forward(inputs)
        return logit

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label, cl_logits):
        """Compute segmentation loss."""
        loss = dict()
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
            if self.use_cl:
                cl_weight = self.sampler.sample(cl_logit, seg_label)
        else:
            seg_weight = None
            if self.use_cl:
                cl_weight = None
                cl_logit_fn = resize(
                                    input=cl_logits["cl_fn"],
                                    size=seg_label.shape[2:],
                                    mode='bilinear',
                                    align_corners=self.align_corners)
                cl_logit_fp = resize(
                                    input=cl_logits["cl_fp"],
                                    size=seg_label.shape[2:],
                                    mode='bilinear',
                                    align_corners=self.align_corners)
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        seg_label = seg_label.squeeze(1)
        loss['loss_seg'] = self.loss_decode(
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index)
        loss['acc_seg'] = accuracy(seg_logit, seg_label)
        if self.use_cl:
            loss['loss_fn'] = self.loss_cl_fn(
                cl_logit_fn,
                seg_label,
                weight=cl_weight,
                ignore_index=self.ignore_index)
            loss['acc_fn'] = accuracy(cl_logit_fn, seg_label)
            loss['loss_fp'] = self.loss_cl_fp(
                cl_logit_fp,
                seg_label,
                weight=cl_weight,
                ignore_index=self.ignore_index)
            loss['acc_fp'] = accuracy(cl_logit_fp, seg_label)
        return loss

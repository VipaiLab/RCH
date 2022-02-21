import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from ..builder import HEADS
from .decode_head_new import BaseDecodeHead


@HEADS.register_module()
class FCNHead(BaseDecodeHead):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 **kwargs):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        super(FCNHead, self).__init__(**kwargs)
        if num_convs == 0:
            assert self.in_channels == self.channels
        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        if self.use_cl:
            print("redefine cl module ",self.alp)
            num_channel = self.in_channel_lis[1]
            print('num channels is ', num_channel)
            self.begin_channel = self.in_channel_lis[0]
            self.end_channel = self.begin_channel + num_channel
            avg_f = torch.randn(num_channel, self.num_classes)
            self.register_buffer('avg_f', avg_f)
            self.cl_conv = ConvModule(
                    num_channel,
                    num_channel,
                    kernel_size=1,
                    padding=0,
                    conv_cfg=None,
                    norm_cfg=self.norm_cfg,
                    act_cfg=None
                    )

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        if self.use_cl:
            feat_cl = x[:,self.begin_channel:self.end_channel,:,:]
        feature = self.convs(x)
        if self.concat_input:
            feature = self.conv_cat(torch.cat([x, feature], dim=1))
        output = self.cls_seg(feature)
        if self.use_cl:
            return output, feat_cl
        else:
            return output, feature

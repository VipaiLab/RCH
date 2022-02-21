_base_ = [
    '../_base_/models/fcn_r50-d8_cl_region.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
model = dict(
    backbone=dict(dilations=(1, 1, 1, 2), strides=(1, 2, 2, 1)),
    decode_head=dict(dilation=6,scale_up=16,temp=0.25),
    auxiliary_head=dict(dilation=6))

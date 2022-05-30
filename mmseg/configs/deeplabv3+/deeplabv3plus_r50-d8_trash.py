_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py', '../_base_/datasets/_my_dataset.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
model = dict(
    pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101),
    decode_head=dict(num_classes=11), auxiliary_head=dict(num_classes=11))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

checkpoint_config = dict(interval=1)
runner = dict(type='EpochBasedRunner', max_epochs=50)
evaluation = dict(interval=1, save_best='mIoU', metric='mIoU', pre_eval=True)
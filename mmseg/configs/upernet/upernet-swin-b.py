_base_ = [
    '../_base_/models/upernet_swin.py', '../_base_/datasets/_my_dataset.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_base_patch4_window7_224_20220317-e9b98025.pth'
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32]),
    decode_head=dict(
        in_channels=[128, 256, 512, 1024], 
        num_classes=11,
        loss_decode=dict(type='FocalLoss', use_sigmoid=True)  #crossEntropyLoss 대신
    ),
    auxiliary_head=dict(
        in_channels=512, 
        num_classes=11,
        loss_decode=dict(type='FocalLoss', use_sigmoid=True)
        )
    )

optimizer = dict(
    _delete_=True,
    type='AdamW',   
    lr=0.00005,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

checkpoint_config = dict(interval=1)
runner = dict(type='EpochBasedRunner', max_epochs=70)
evaluation = dict(interval=1, save_best='mIoU', metric='mIoU', pre_eval=True)
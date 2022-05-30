_base_ = [
    '../_base_/models/upernet_swin.py', '../_base_/datasets/_my_dataset_aug.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
checkpoint_file = '/opt/ml/input/code/mmsegmentation/pretrain/swin_large_patch4_window12_384_22k.pth'
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        pretrain_img_size=384,
        embed_dims=192,
        patch_size=4,
        mlp_ratio=4,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU')),
    decode_head=dict(
        in_channels=[192,384,768,1536], 
        num_classes=11,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=True)
        ),
    auxiliary_head=dict(
        in_channels=768, 
        num_classes=11,
        loss_decode=dict(type='DiceLoss', use_sigmoid=True)
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
runner = dict(type='EpochBasedRunner', max_epochs=100)
evaluation = dict(interval=1, save_best='mIoU', metric='mIoU', pre_eval=True)
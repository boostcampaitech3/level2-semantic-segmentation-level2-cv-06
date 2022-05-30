_base_ = ['segformer_mit-b0_trash.py']

# model settings
model = dict(
    pretrained='/opt/ml/input/code/mmsegmentation/pretrain/mit_b5.pth',
    backbone=dict(
        embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[3, 6, 40, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512]))

checkpoint_config = dict(interval=1)
runner = dict(type='EpochBasedRunner', max_epochs=140)
evaluation = dict(interval=1, save_best='mIoU', metric='mIoU', pre_eval=True)
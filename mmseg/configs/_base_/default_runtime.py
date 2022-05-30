# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
        dict(type='WandbLoggerHook',interval=10,
            init_kwargs=dict(
                project='JHwan',
                entity = 'omakase',
                name = 'swin-l focal AdamW epo100 lr0.00005 f2'
            ),
        )  
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True

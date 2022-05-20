# HRNetV2 OCR
## Model
- Model에 대한 코드는 [HRNet-Semantic-Segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation) github을 참고 하였습니다.
- Model에 대한 ImageNet Pretrained 파일은 [HRNet-Image-Classification](https://github.com/HRNet/HRNet-Image-Classification) github에서 가져왔습니다.

## Structure
```
├─ configs                                 
│  └─ config.yaml                                 
├─ dataset.py
├─ inference.py
├─ loss.py
├─ optimizer.py
├─ model.py
├─ scheduler.py
├─ test.py
├─ train.py
├─ trainer.py
├─ utils.py
└─ config.yaml
```
- `configs/config.yaml` : HRNetv2 OCR의 모델 config
- `dataset.py` : Dataset과 Transforms 관련 파일
- `inference.py` : 학습 모델 추론 파일
- `loss.py` : Loss 관련 파일
- `optimizer.py` : Optimizer 관련 파일
- `scheduler.py` : Scheduler 관련 파일
- `test.py` : Inference시 불러오는 파일로, csv 파일 생성
- `train.py` : 모델 학습 config 준비 파일
- `trainer.py` : 모델 학습을 위한 파일
- `utils.py` : Metric과 Custom Scheduler 관련 파일
- `config.yaml` : 모델과 학습 파라미터와 관련 파일

## Training
```bash
$ python train.py \
    --seed {seed 설정} \
    --epoch {max epoch 설정} \
    --batch_size {batch size 설정}
    --learning_rate {learning rate 설정} \
    --fold {fold 사용 여부} \
    --config_dir {config file이 존재하는 디렉토리 설정}
    --viz_interval {wandb media 출력 간격 설정}
    --log_interval {wandb log 출력 간격 설정}
```

## Inference
```bash
$ python inference.py \
    --seed {seed 설정} \
    --batch_size {batch_size 설정} \
    --config_dir {config file이 존재하는 디렉토리 설정}
    --config {config 설정} \
    --config_model {model config 설정} \
    --checkpoint {pth 파일 설정}
```

## Reference
- [HRNet-Semantic-Segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation)
- [HRNet-Image-Classification](https://github.com/HRNet/HRNet-Image-Classification)
- [CustomCosineAnnealingWarmUpRestarts](https://gaussian37.github.io/dl-pytorch-lr_scheduler/)

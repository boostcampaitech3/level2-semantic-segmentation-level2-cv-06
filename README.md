# 🍣 오마카세 
![image](https://user-images.githubusercontent.com/91659448/170925131-0b8114f0-d00f-49d9-93a7-61e361c68aca.png)
- 대회 기간 : 2022.04.25 ~ 2022.05.12
- 목적 : 재활용 품목 분류를 위한 Semantic Segmentation

## ♻️ 재활용 품목 분류를 위한 Object Detection
### 🔎 배경
![image](https://user-images.githubusercontent.com/91659448/164387063-c84ae185-257c-4b90-8015-366cbe22a05d.png)

- 코로나19가 확산됨에 따라 언택트 시대가 도래하였습니다.
- 이에 발맞춰 배달 산업의 성장과 e커머스 시장의 확대되며 일회용품과 플라스틱의 사용 비율이 높아졌습니다.
- 이러한 문화는 해당 산업의 성장을 불러왔지만, "쓰레기 대란"과 "매립지 부족"과 같은 사회 문제를 낳고 있습니다.
- 분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나입니다. 
- 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정 받아 재활용 되지만 그렇지 않은 경우는 폐기물로 분류되어 매립 또는 소각되기 때문입니다.
- 따라서 우리는 사진에서 쓰레기를 Segmentation 하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다. 

### 💾 데이터 셋
- `전체 이미지 개수` : 4092장 (train 3468 장, test 624 장)
- `11개 클래스` : Background, General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
- `이미지 크기` : (512, 512)

### 📂 데이터 구조
```
level2-semantic-segmentation-level2-cv-06/
│
├─ 📂 models                                 
|  ├─ 📂 HRNetV2_OCR
│  │  ├─ 📂 configs
│  │  │  └── 📄 config.yaml.yaml
│  │  └─ 📄 *.py
│  │
│  └─ 📂 TransUNet
│     ├─ 📂 networks
│     │  ├─ 📄 vit_seg_configs.py
│     │  ├─ 📄 vit_seg_modeling.py
│     │  └─ 📄 vit_seg_modeling_resnet_skip.py
│     ├─ 📄 config.yaml
│     └─ 📄 *.py
│
├─ 📂 mmseg
│  └─ 📂 configs
│     ├─ 📂 _base_
│     │  ├─ 📂 datasets
│     │  ├─ 📂 models
│     │  ├─ 📂 schedules
│     │  └─ 📄 default_runtime.py
│     ├─ 📂 deeplabv3+
│     │  ├─ 📄 deeplabv3plus_r50-d8_trash.py
│     │  └─ 📄 deeplabv3plus_s101-d8_trash.py
│     ├─ 📂 segformer
│     │  ├─ 📄 segformer_mit-b0_trash.py
│     │  └─ 📄 segformer_mit-b5_trash.py
│     ├─ 📂 upernet
│     │  ├─ 📄 upernet-swin-b.py
│     │  ├─ 📄 upernet-swin-l.py
│     │  └─ 📄 upernet-swin-l-aug.py
│     ├─ 📄 inference.ipynb
│     └─ 📄 train.py
│
└─ 📂 utils
   ├─ 📄 concat_json.ipynb
   ├─ 📄 data_viz.ipynb
   ├─ 📄 inference_pseudo.ipynb
   ├─ 📄 json_to_segmap.ipynb
   ├─ 📄 MakeObjectAugDict.ipynb
   ├─ 📄 MakeObjectAugJson.ipynb
   └─ 📄 StratifiedGroupKFold.ipynb
```

## 🙂 멤버
| 박동훈 | 박성호 | 송민기 | 이무현 | 이주환 |  
| :-: | :-: | :-: | :-: | :-: |  
|[Github](https://github.com/BTOCC25) | [Github](https://github.com/pyhonic) | [Github](https://github.com/alsrl8) | [Github](https://github.com/PeterLEEEEEE) | [Github](https://github.com/JHwan96)


## 📋 역할
| 멤버 | 역할 |
| :-: | :-: |
|박동훈(T3086)| EDA, Dataset mislabeling 수정, mmsegmentation 적용 |
|박성호(T3090)| EDA, Dataset mislabeling 수정, SMP 및 Model 적용, param 실험 |
|송민기(T3112)| EDA, Dataset mislabeling 수정 |
|이무현(T3144)| EDA, Dataset mislabeling 수정, mmsegmentation 적용 |
|이주환(T3241)| EDA, Dataset mislabeling 수정, mmsegmentation 적용, Augmentation 실험 |


## 🧪 실험
|Property|Model|Backbone| Fold | mIoU@public|mIoU@private|
| :-: | :-: | :-: | :-: | :-: | :-: |
| 2-Stage | UperNet | Swin-L | Fold 2 | 0.7435 | 0.7513 | 
| 2-Stage | UperNet | Swin-L | Fold 5 |0.7412 | 0.7530 | 
| Pseudo Labeling | UperNet | Swin-L | Fold 5 |0.7109 | 0.7496 |
| Ensemble | UperNet |Swin-L | Fold 1~5 |0.7230 | 0.7549 |      
| Ensemble | UperNet, HRNetV2_OCR, deepLabV3+ | | | 0.7403 | 0.7611 |

## Reference
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
- [SMP](https://github.com/qubvel/segmentation_models.pytorch)
- [HRNet](https://github.com/HRNet/HRNet-Semantic-Segmentation)
- [TransUNet](https://github.com/Beckschen/TransUNet)
- [unlim](https://github.com/microsoft/unilm)
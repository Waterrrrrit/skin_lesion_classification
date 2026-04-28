# Lesion Classification (Skin Lesion 13-Class)

이 프로젝트는 일반 피부 병변 이미지 데이터를 이용해 13개 병변을 분류하는 baseline 학습 파이프라인입니다. "데이터 읽기 → 전처리/증강 → 모델 학습 → 검증 → 결과 저장"의 전체 흐름을 한 번에 실행할 수 있도록 구성되어 있습니다.

## 목표

* 일반 피부 병변 13개 클래스 분류 baseline 구축
* 전이학습 기반 EfficientNet-B3 모델을 사용하여 미세한 피부 병변에 대한 고해상도 특징 추출 성능 확보
* Validation 분리 기반 정량 평가(accuracy/precision/recall/f1) 수행 및 Focal Loss를 통한 데이터 불균형 문제 완화

## 데이터셋 설명

* **데이터 루트**: `dataset/`
* **split**: `Training`, `Validation`
* **이미지 폴더**: `images`
* **라벨 폴더**: `jsons`
* 이미지 파일과 JSON 라벨 파일은 동일한 stem을 공유
* JSON 내부 `diagnosis_name`을 정답 라벨로 사용
* **클래스 수**: 13
* **출처**: AI Hub, Kaggle 통합 데이터

## 클래스 목록

* 화이트헤드 (Whitehead)
* 블랙헤드 (Blackhead)
* 구진 (Papule)
* 농포 (Pustule)
* 결절 (Nodule)
* 피지선결석 (Sebaceous Calculus)
* 비립종 (Milium)
* 한관종 (Syringoma)
* 모공확장 (Enlarged Pores)
* 기미 (Melasma)
* 색소침착 (Pigmentation)
* 주사(딸기코) (Rosacea)
* 지루성 피부염 (Seborrheic Dermatitis)

## 클래스별 샘플 수

* **Train**: 각 클래스 1,000장
* **Val**: 각 클래스 1,000장

## 학습 파이프라인

1. `lesion_dataset.py` 가 이미지 + JSON 라벨을 읽어 학습용 데이터로 변환 (에러 데이터 자동 스킵)
2. `train.py` 가 전처리/증강 및 Focal Loss를 적용하여 배치 단위로 학습 수행
3. Validation 데이터로 성능 평가 후 F1-score 기준 best 모델 저장

## 기본 전처리/증강

* **Reinhard Normalization**: 데이터 도메인 간의 색상 편차를 줄이기 위해 LAB 색 공간에서 색감 교정
* **Resize(300, 300)**: B3 모델 권장 해상도를 고려하여 모든 입력 크기 통일
* **RandomHorizontalFlip(p=0.5)**: 데이터 증강
* **Normalize(ImageNet 기준)**: 평균 `[0.485, 0.456, 0.406]`, 표준편차 `[0.229, 0.224, 0.225]` 적용

## 모델 구성

* **Backbone**: `EfficientNet-B3` (B0 대비 고해상도 특징 추출에 유리)
* **Pretrained weights**: `EfficientNet_B3_Weights.DEFAULT`
* **Classifier 교체**: 13클래스로 노드 변경 및 과적합 방지를 위한 `Dropout(p=0.3)` 적용
* **Loss Function**: `FocalLoss(alpha=1.0, gamma=2.0)` 적용으로 데이터 불균형 완화

## 평가 지표

* accuracy
* precision_macro
* recall_macro
* f1_macro
* confusion matrix

## 결과 파일

* `best_model.pth` : 최고 성능(F1-score)을 기록한 모델 가중치
* `class_names.json` : 추론 시 사용할 13개 클래스 이름 목록
* `metrics_history.json` : epoch별 성능 로그
* `confusion_matrix.csv` : 최종 validation 기준 클래스 혼동행렬 (13x13 고정)

## 파일 역할

* `lesion_dataset.py`: 이미지와 JSON 라벨을 읽어 학습용 데이터로 만드는 파일
* `model.py`: EfficientNet-B3 기반 모델 정의 파일
* `preprocess.py`: Reinhard Normalization 전처리 클래스 정의 파일
* `loss.py`: Focal Loss 계산 함수 정의 파일
* `train.py`: 데이터 로드, 학습/검증 루프 및 평가 파일 저장 실행 파일
* `infer.py`: 학습된 가중치와 단일 이미지를 입력받아 추론 결과를 JSON으로 반환하는 파일

## 실행 예시

데이터셋이 기본 경로(`dataset/`)에 구성되어 있다면, 옵션 없이 아래 명령어로 즉시 학습이 가능합니다.

```bash
python train.py
```

배치 사이즈나 에폭 수 등 설정값을 직접 변경하여 실행할 수 있습니다.

```bash
python train.py --save-dir ./outputs --batch-size 8 --epochs 20 --lr 0.0001
```

## 추론 실행 예시

학습이 끝난 뒤에는 `infer.py` 로 이미지 1장 추론을 수행할 수 있습니다.

```bash
python infer.py --image ./sample.jpg --checkpoint ./outputs/best_model.pth --class-names ./outputs/class_names.json
```

## 추론 출력 형식

* `pred_class` : 최고 확률 클래스 index
* `pred_label` : 최고 확률 클래스 이름
* `probabilities` : 13개 클래스 전체 softmax 확률
* `confidence` : 최고 확률값

예시:

```json
{
    "pred_class": 6,
    "pred_label": "비립종",
    "probabilities": [0.01, 0.03, 0.02, 0.01, 0.00, 0.05, 0.72, 0.04, 0.03, 0.01, 0.01, 0.01, 0.06],
    "confidence": 0.72
}
```

## 모델과 백엔드 역할 분리

* 모델은 클래스 예측 결과와 softmax 확률, confidence를 반환
* 백엔드는 confidence threshold 등 운영 정책을 적용
* `판단 불가` 여부는 모델이 아니라 서비스 정책에서 결정

즉, 현재 추론 계층은 판단 재료를 반환하는 역할만 담당합니다.

## Model Source

* EfficientNet paper: `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks`
* torchvision docs: [https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b3.html](https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b3.html)
* torchvision source: [https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py](https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py)

## License

* `torchvision` repository license: BSD 3-Clause

## Package Install

```bash
pip install -r requirements.txt
```

PyTorch와 torchvision은 서버 CUDA 버전에 맞게 별도 설치를 권장합니다.

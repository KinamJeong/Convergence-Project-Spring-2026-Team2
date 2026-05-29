# HIPTrack + MVTD Online Augmentation Fine-Tuning

> MVTD(Maritime Visual Tracking Dataset)를 활용한 HIPTrack Fine-Tuning 브랜치.
> 온라인 데이터 증강(Online Augmentation)을 적용하여 해양 도메인 특화 학습을 수행합니다.

---

## 기존 코드 대비 변경/추가 사항

| 파일 | 작업 |
|---|---|
| `lib/train/data/mvtd_augmentation.py` | 신규 추가 — 온라인 증강 기법 모음 |
| `lib/train/dataset/mvtd.py` | 신규 추가 — MVTD 전용 데이터셋 클래스 |
| `lib/train/dataset/__init__.py` | 수정 — MVTD import 추가 |
| `lib/train/base_functions.py` | 수정 — MVTD 데이터셋 등록 |
| `experiments/hiptrack/hiptrack.yaml` | 수정 — MVTD 학습 설정 |

---

## 사전 준비

### 1. 환경 세팅

```bash
conda env create -f HIPTrack_env_cuda113.yaml
conda activate hiptrack
```

### 2. 프로젝트 경로 초기화

```bash
python tracking/create_default_local_file.py \
    --workspace_dir . \
    --data_dir ./data \
    --save_dir ./output
```

### 3. 로컬 경로 설정 

`lib/train/admin/local.py`를 열어서 **본인 환경에 맞게** 아래 경로를 수정해야 합니다.

```python
# 본인 HIPTrack 폴더 경로로 수정
self.mvtd_dir = '/본인경로/HIPTrack/data/mvtd/train'
```

(제 경로는 아래처럼 세팅해서 사용했습니다.)
```python
self.mvtd_dir = '/home/ivl5/hiptrack/HIPTrack/data/mvtd/train'
```

### 4. MVTD 데이터셋 준비 

GitHub에는 데이터가 포함되어 있지 않습니다. 아래 구조로 직접 배치해주세요.

```
HIPTrack/data/mvtd/
├── train/
│   ├── list.txt           ← 시퀀스 이름 목록 (없으면 자동 생성)
│   ├── 1-Ship/
│   │   ├── 00000001.jpg
│   │   ├── 00000002.jpg
│   │   ├── groundtruth.txt
│   │   ├── absence.label
│   │   └── cover.label
│   ├── 2-Boat/
│   └── ...
└── test/
    ├── video1/
    └── ...
```

### 5. 사전학습 모델 준비

아래 경로에 사전학습 가중치 파일을 배치해주세요.

```
HIPTrack/pretrained_models/
└── DropTrack_k700_800E_alldata.pth.tar
```

> MVTD 레포 공유 링크에서 다운로드:
> https://kuacae-my.sharepoint.com/:f:/g/personal/ahsan_bakht_ku_ac_ae/IgD3c4aIu83XQazIPFglANf8AVv_h1J8etNnaGo5PaCvfis?e=UHUr2R

---

## 학습 실행

### GPU 메모리 설정 확인

`experiments/hiptrack/hiptrack.yaml`에서 GPU 메모리에 맞게 `BATCH_SIZE`를 조정하세요.

```yaml
TRAIN:
  BATCH_SIZE: 8   # GPU 메모리 부족 시: 32 → 16 → 8 순으로 줄이기
```

### 학습 명령어

**멀티 GPU (권장)**
```bash
python3 tracking/train.py \
    --script hiptrack \
    --config hiptrack \
    --save_dir ./output_mvtd \
    --mode multiple \
    --nproc_per_node 4   # 사용 가능한 GPU 수로 변경
```

**싱글 GPU**
```bash
python3 tracking/train.py \
    --script hiptrack \
    --config hiptrack \
    --save_dir ./output_mvtd \
    --mode single
```

### 학습 재개 (중단 후 이어서)

체크포인트가 `output_mvtd/checkpoints/`에 저장되므로, 같은 명령어로 재실행하면 자동으로 이어서 학습합니다.

---

## 온라인 증강 동작 방식

기존 HIPTrack과 달리, 이미지를 디스크에 저장하지 않고 **학습 중 매 배치마다 실시간으로 증강**을 적용합니다.

```
[MVTD Train set 디스크]
        ↓
  DataLoader 호출
        ↓
  mvtd.py의 get_frames() 실행
        ↓
  ★ MVTDAugmentor.apply() — 온라인 증강 적용
    ├── 시퀀스 단위로 증강 전략 한 번 선택 (시간적 일관성 유지)
    ├── aug_prob=0.5 (50% 확률로 증강 적용)
    └── 적용 기법: hflip, zoom_in, zoom_out, synthetic_reflection,
                  specular_highlight, wave_blur, maritime_haze,
                  wake_overlay, foreground_occlusion,
                  background_jitter, color_jitter
        ↓
  모델 입력 (증강된 이미지)
```

> **테스트 시 증강 비적용**: 증강은 학습(train)에만 적용되며, 테스트/추론 시에는 원본 이미지 그대로 사용됩니다.

---

## 학습 설정 요약

| 항목 | 값 |
|---|---|
| 학습 데이터 | MVTD Train set |
| 검증 데이터 | MVTD Val set |
| Epoch | 30 |
| Learning Rate | 0.00001 |
| LR Drop Epoch | 20 |
| Batch Size | 8 (GPU 메모리에 따라 조정) |
| 증강 적용 확률 | 50% |

---

## 주의사항

- `data/`, `pretrained_models/`, `output/`, `output_mvtd/` 폴더는 `.gitignore`에 포함되어 있어 GitHub에 올라가지 않습니다.
- `data/mvtd/train`폴더에 MVTD 학습 데이터를 위치하면 됩니다.
- `test/` 데이터는 절대 학습에 사용하지 마세요.
- `local.py`의 경로 설정을 반드시 본인 환경에 맞게 수정해야 합니다.

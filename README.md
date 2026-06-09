# Convergence-Project-Spring-2026-Team2

---

## Installation

### Recommended Environment
- **OS:** Ubuntu 22.04.5 LTS
- **CUDA:** 11.3
- **Python:** 3.8.0
- **PyTorch:** 1.10.1 + cu113
- **GPU:** NVIDIA GeForce GTX 1660 Ti

### 📁 Data Preparation

> **MVTD Dataset:** [MVTD](https://huggingface.co/datasets/AhsanBB/Maritime_Visual_Tracking_Dataset_MVTD/tree/main)
>
> **Official HIPTrack's Pretrained Weights:** [Pretrained Weights](https://drive.google.com/drive/folders/1W-fJCnfxwz2IIC6O8R9ADOAK_ebkmC-d)
>
> **Our Work's Finetuned Weights:** [Finetuned Weights](https://drive.google.com/file/d/1Efe2X_CcCMZ3f_Z3ZzLdgXdH4FIhGUb-/view)

Please download Dataset & Checkpoints and place them as follows:

```
|- HIPTrack
   |- output
   |  └─ checkpoints
   |      └─ train
   |          └─ hiptrack
   |              └─ hiptrack
   |                  └─ HIPTrack_ep0099.pth.tar    # Official pretrained weights (rename from HIPTrack_all_data.pth.tar)
   |                  └─ HIPTrack_ep0034.pth.tar    # Our finetuned weights
   |- data
   |  └─ mvtd
   |      |- train                    # 129 sequences (Ship / Boat / USV / SailBoat)
   |      |   |- 1-Ship
   |      |   |- 3-Boat
   |      |   |- ...
   |      |   |- 159-Boat
   |      |   └─ list.txt
   |      |- validation               # 25 sequences (Ship / Boat / USV / SailBoat)
   |      |   |- 2-USV
   |      |   |- 9-Boat
   |      |   |- ...
   |      |   |- 156-Boat
   |      |   └─ list.txt
   |      └─ test                     # 23 sequences (MVTD official test set)
   |          |- 1-Boat
   |          |- 2-Ship
   |          |- ...
   |          |- 23-Boat
   |          └─ list.txt
   |- experiments
   |  └─ hiptrack
   |      └─ hiptrack.yaml            # Set TEST.EPOCH to match the checkpoint you want to use
   |- Run_train.txt
   └─ Run_test.txt
|- Run_Command.sh
```

> **Note:** The official pretrained weights (`HIPTrack_all_data.pth.tar`) must be renamed to `HIPTrack_ep0099.pth.tar` and placed in the path above. Make sure `TEST.EPOCH: 99` is set in `experiments/hiptrack/hiptrack.yaml`.

### Environment Setup

```bash
git clone https://github.com/Sejong-VLI-Courses/Convergence-Project-Spring-2026-Team2
cd Convergence-Project-Spring-2026-Team2
pip install -r requirements.txt
```

---

## Training

**Fine-tuning on MVTD:**

To change training parameters, edit `HIPTrack/experiments/hiptrack/hiptrack.yaml`.

```bash
bash Run_Command.sh HIPTrack train
```

---

## Evaluation

```bash
bash Run_Command.sh HIPTrack test
```

To run evaluation on the results:

```bash
python HIPTrack/tracking/analysis_result_solo.py
```

---

## Our Results

### Ablation Study  (Test Set Results)

| Configuration | AUC | OP75 |
|---|---|---|
| Baseline (Official Pretrained, ep0099) | 75.30 | 85.16 |
| + SwitchRecoveryModule + HRatioCorrectionModule | TODO | TODO |
| + Retraining (ep0034) | 81.19 | 86.69 |

| +  | TODO | TODO |
| **Full Pipeline (ep0034 + Both Modules)** | **80.61** | **86.09** |


---

## Method Overview

This project extends **HIPTrack** (CVPR 2024) for maritime vessel tracking on the **MVTD dataset**, introducing two plug-and-play inference modules on top of domain-specific fine-tuning.

### Modules

| Module | Description |
|---|---|
| **Retraining** | Re-trained HIPTrack on MVTD train set with online augmentation |
| **SwitchRecoveryModule** | Detects identity switches via response drop + cosine similarity, recovers bbox with constant-velocity extrapolation |
| **HRatioCorrectionModule** | Corrects systematic H-ratio underestimation caused by vessel scale decrease, using sliding window h-drop detection |

All modules operate as **post-processing** steps with no modification to HIPTrack internals, enabling clean ablation.

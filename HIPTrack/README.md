# Switch Recovery Module 적용 가이드

HIPTrack 추론 시 identity switch(타겟을 다른 객체로 오인해 계속 추적하는 현상)를 감지하고, motion 예측으로 원래 객체 위치로 복구하는 외부 모듈입니다.

검증 결과: 62-USV AUC 62.35 → 88.03 (+25.68), 10-USV AUC 41.50 → 43.21 (+1.71)

***

### 파일 준비

`switch_recovery.py`를 아래 경로에 복사합니다.

```
lib/test/tracker/switch_recovery.py
```

***

### hiptrack.py 수정 

#### 1. initialize() - 모듈 인스턴스 생성

`self.frame_id = 0`바로 다음에 추가합니다.

```python
# save states
        self.state = info['init_bbox']
        self.frame_id = 0

        # ===== switch 복구 모듈 초기화 =====
        from switch_recovery import SwitchRecoveryModule
        self.recovery_module = SwitchRecoveryModule()
```

#### 2. track() - 모듈 호출 및 state 보정

아래 줄을 찾습니다.

```python
self.state = clip_box(self.map_box_back(pred_box, resize_factor, gt_crop=gt_crop), H, W, margin=10)
```

그 다음에 오는 'score_max','response_max'계산 줄과 함께 아래처럼 수정합니다.

```python
self.state = clip_box(self.map_box_back(pred_box, resize_factor, gt_crop=gt_crop), H, W, margin=10)

        topk_states = []

        score_max = pred_score_map.max().item()
        response_max = response.max().item()

        # ===== switch 복구 모듈 =====
        corrected, info = self.recovery_module.update(response_max, list(self.state))
        if info.get("recovered"):
            self.state = clip_box(list(corrected), H, W, margin=10)
```

***

### 모듈 끄는 법 (baseline 비교 시)

개입 두 줄만 주석 처리하면 됩니다. 모듈 호출은 유지하되 state를 안 바꾸는 상태로, 트래커는 원래대로 동작합니다.

```python
corrected, info = self.recovery_module.update(response_max, list(self.state))
        # if info.get("recovered"):
        #     self.state = clip_box(list(corrected), H, W, margin=10)
```

***

### import 오류 시

'from switch_recovery import SwitchRecoveryModule'이 안 되면 상대 import로 바꿉니다.

```python
from .switch_recovery import SwitchRecoveryModule
```

***

### 모듈 파라미터 (SwitchRecoveryModule)

필요 시 인스턴스 생성 시 조정 가능합니다. 기본값은 두 USV 시퀀스 데이터로 튜닝한 값입니다.

| 파라미터  | 기본값 | 설명 | 
|----------|--------|-----| 
| ratio | 0.65 | baseline 대비 이 비율 미만이면 confidence 낮음으로 판정 | 
| persist | 3 | 낮음이 몇 프레임 연속이면 SUSPECT 트리거 |
| warmup | 30 | 초반 baseline 구축 프레임 수 |
| max_motion_speed | 4.0 | 이 이상 빠른 motion은 신뢰 불가로 보고 개입 보류 |
| recover_ratio | 0.75 | confidence가 이 비율 이상 회복되면 복구 성공 판정 |

***

### 적용 조건과 한계

- 잘 작동하는 경우: switch 이후 원래 객체가 motion 예측 위치 근처에 외따로 남아있는 경우. 62-USV가 대표 사례입니다.

- 제한적인 경우: switch 이후에도 강한 distractor(다른 움직이는 큰 객체)가 같은 영역에 있는 경우. 10-USV가 대표 사례입니다. 위치 이동만으로는 트래커의 선택을 바꾸기 어렵습니다.

- 미감지 경우: confidence가 얕게 떨어지는 switch(118-Boat, 부표로의 switch). confidence 기반 감지기가 SUSPECT를 트리거하기 전에 회복돼버려 개입 타이밍을 놓칩니다.




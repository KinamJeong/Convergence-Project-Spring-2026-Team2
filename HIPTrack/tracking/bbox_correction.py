print(">>> bbox_correction loaded from:", __file__)
from collections import deque


class HRatioCorrectionModule:
    """
    HIPTrack의 H_ratio 과소추정 보정 모듈 (Type A).

    문제: response_max가 낮아질수록 bbox의 h가 GT 대비 체계적으로 과소추정됨.
         (154-Boat: H_ratio≈0.848, 78-USV: H_ratio≈0.867 @ low response 구간)

    해결: warmup 구간(초반 N프레임)에서 response_max baseline을 구축하고,
         이후 response가 threshold 아래로 떨어질 때 response 구간별 scale factor로
         h를 보정함. center_y는 유지.

    개입 위치: hiptrack.py의 track() 내부,
               clip_box() → self.state 확정 직후,
               recovery_module.update() 호출 전.
    """

    def __init__(self,
                 init_h=None,
                 h_window=30,
                 h_drop_th=0.025,
                 warmup=30,
                 baseline_window=50,
                 thresh_ratio=0.96,
                 thresh_abs_cap=0.97, #0.9
                 scale_mild=1.05,
                 scale_moderate=1.10,
                 scale_severe=1.15,
                 resp_min=0.30):

        self.init_h = init_h
        self.h_window = h_window
        self.h_drop_th = h_drop_th
        self.warmup = warmup
        self.baseline_window = baseline_window
        self.thresh_ratio = thresh_ratio
        self.thresh_abs_cap = thresh_abs_cap
        self.scale_mild = scale_mild
        self.scale_moderate = scale_moderate
        self.scale_severe = scale_severe
        self.resp_min = resp_min

        self.frame_id = 0
        self.baseline = None
        self.threshold = None
        self.enabled = False
        self.resp_history = deque(maxlen=baseline_window)
        self.h_ratio_history = deque(maxlen=h_window * 2)

    def update(self, response_max: float, bbox: list) -> tuple:
        """
        Parameters
        ----------
        response_max : float
            현재 프레임의 response_max (hiptrack.py의 response.max().item())
        bbox : list [x, y, w, h]
            clip_box() 직후의 self.state

        Returns
        -------
        (corrected_bbox, info)
            corrected_bbox : list [x, y, w, h]  보정된 bbox (보정 없으면 입력 그대로)
            info           : dict  디버그 정보
        """
        self.frame_id += 1
        x, y, w, h = bbox

        # warmup: baseline 구축
        if self.frame_id <= self.warmup:
            self.resp_history.append(response_max)
            self.baseline = sum(self.resp_history) / len(self.resp_history)
            self.threshold = min(self.baseline * self.thresh_ratio,
                                 self.thresh_abs_cap)
            if self.init_h and self.init_h > 0:
                self.h_ratio_history.append(h / self.init_h)
            return list(bbox), self._info(response_max, scale=1.0, corrected=False)

        # baseline 갱신 (안정 프레임만)
        if response_max >= self.threshold:
            self.resp_history.append(response_max)
            self.baseline = sum(self.resp_history) / len(self.resp_history)
            self.threshold = min(self.baseline * self.thresh_ratio,
                                 self.thresh_abs_cap)
        
         # h/init_h 이력 갱신
        if self.init_h and self.init_h > 0:
            self.h_ratio_history.append(h / self.init_h)

         # 실시간 감지: h 지속 감소 + response 저하 동시 체크
        cond_resp = response_max < self.threshold
        cond_h_drop = False
        if len(self.h_ratio_history) >= self.h_window * 2:
            recent = list(self.h_ratio_history)[-self.h_window:]
            prev   = list(self.h_ratio_history)[-self.h_window * 2:-self.h_window]
            h_drop = (sum(prev) / len(prev)) - (sum(recent) / len(recent))
            cond_h_drop = h_drop > self.h_drop_th

        cond_size = (h < self.init_h * 0.80)
        cond_resp_min = (response_max >= self.resp_min)
        self.enabled = cond_h_drop and cond_size and cond_resp_min 

        # 디버그 출력 (50프레임마다)
        #if self.frame_id % 50 == 0:
        #    if len(self.h_ratio_history) >= self.h_window * 2:
        #        recent = list(self.h_ratio_history)[-self.h_window:]
        #        prev   = list(self.h_ratio_history)[-self.h_window * 2:-self.h_window]
        #        h_drop = (sum(prev)/len(prev)) - (sum(recent)/len(recent))
        #        print(f"f{self.frame_id:4d} | resp={response_max:.4f} | "
        #            f"threshold={self.threshold:.4f} | cond_resp={cond_resp} | "
        #            f"h_drop={h_drop:.4f} | cond_h_drop={cond_h_drop} | "
        #            f"enabled={self.enabled}")
        #####################################


        if not self.enabled:
            return list(bbox), self._info(response_max, scale=1.0, corrected=False)
        
        # response 구간별 scale 결정
        mid_boundary = self.threshold - (self.threshold - 0.62) * 0.5  # ≈0.68~0.70
        if response_max >= mid_boundary:
            scale = self.scale_mild        # [mid_boundary, threshold)
        elif response_max >= 0.62:
            scale = self.scale_moderate    # [0.62, mid_boundary)
        else:
            scale = self.scale_severe      # [0, 0.62)

        # h 보정 (center_y 유지)
        h_new = h * scale
        y_new = y - (h_new - h) / 2  # center_y = y + h/2 → 유지

        corrected = [x, y_new, w, h_new]
        return corrected, self._info(response_max, scale=scale, corrected=True)

    def _info(self, response_max: float, scale: float, corrected: bool) -> dict:
        return {
            "frame":        self.frame_id,
            "response_max": round(response_max, 4),
            "baseline":     round(self.baseline, 4) if self.baseline else None,
            "threshold":    round(self.threshold, 4) if self.threshold else None,
            "scale":        round(scale, 4),
            "corrected":    corrected,
            "enabled":      self.enabled,
        }

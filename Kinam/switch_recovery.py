print(">>> switch_recovery loaded from:", __file__)
from collections import deque
import math


class SwitchRecoveryModule:
    """
    화면 내 identity switch 감지 + motion 기반 복구.
    HIPTrack track() 안에서 매 프레임 호출하며, self.state(bbox)를 보정한다.

    상태: NORMAL -> SUSPECT -> LOST -> (복구) -> NORMAL
    """

    def __init__(self,
                 # --- 조건1: switch 감지 ---
                 ratio=0.65, persist=3, warmup=30,
                 baseline_window=50, update_thresh=0.85,
                 # --- LOST 확정 / 복구 ---
                 lost_after=7,        # SUSPECT가 이만큼 지속되면 LOST(진짜 switch) 확정
                 motion_lag=12,       # motion 추정에 쓸 '오염 전' 시점 (SUSPECT 진입 기준 과거)
                 motion_span=15,      # 등속 속도 산출 구간 길이
                 recover_window=40,   # 복구 후 confidence 회복을 기다리는 최대 프레임
                 recover_ratio=0.75,  # 이 비율 이상으로 conf 회복되면 복구 성공
                 max_motion_speed=4.0,
                 feat_ratio=0.90, 
                 feat_warmup=30, 
                 feat_window=50): 
        # 감지 파라미터
        self.ratio = ratio; self.persist = persist
        self.warmup = warmup; self.update_thresh = update_thresh
        self.baseline = None
        self.conf_history = deque(maxlen=baseline_window)
        self.low_count = 0
        # 복구 파라미터
        self.lost_after = lost_after
        self.motion_lag = motion_lag
        self.motion_span = motion_span
        self.recover_window = recover_window
        self.recover_ratio = recover_ratio
        self.max_motion_speed = max_motion_speed
        self.feat_ratio = feat_ratio
        self.feat_warmup = feat_warmup
        from collections import deque as _dq
        self.feat_history = _dq(maxlen=feat_window)
        self.feat_ref = None
        self.feat_med = None
        # 상태
        self.state = "NORMAL"
        self.frame_id = 0
        self.suspect_count = 0
        self.recover_count = 0
        # NORMAL 동안의 위치/크기 히스토리: (frame, cx, cy, w, h)
        self.track_hist = deque(maxlen=max(motion_lag + motion_span + 5, 60))
        # 복구용 스냅샷
        self.snap_vel = None      # (vx, vy)
        self.snap_size = None     # (w, h)
        self.snap_pos = None      # 마지막 신뢰 위치 (cx, cy, frame)
        self._last_sim = None     # for debug
    def update(self, confidence, bbox, feat_vec=None):
        """
        Parameters
        ----------
        confidence : float   현재 프레임 response_max
        bbox : (x, y, w, h)  HIPTrack이 방금 낸 self.state

        Returns
        -------
        (corrected_bbox, info)
            corrected_bbox : 복구 개입 시 새 bbox, 아니면 입력 그대로
            info : 상태 디버그 dict
        """
        self.frame_id += 1
        #if 940 <= self.frame_id <= 945:
        #    print(f"  [update] f{self.frame_id} feat_vec is None? {feat_vec is None}, feat_ref is None? {self.feat_ref is None}")
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2

        cur_sim = None
        if feat_vec is not None:
            if self.frame_id <= self.feat_warmup:
                self.feat_history.append(feat_vec)
                self.feat_ref = sum(self.feat_history) / len(self.feat_history)
                if self.frame_id == self.feat_warmup:
                    sims = sorted(self._cos(f, self.feat_ref) for f in self.feat_history)
                    self.feat_med = sims[len(sims) // 2]
            elif self.feat_ref is not None:
                cur_sim = self._cos(feat_vec, self.feat_ref)
                self._last_sim = cur_sim  # for debug
                
        # --- warmup: baseline 구축 ---
        if self.frame_id <= self.warmup:
            self.conf_history.append(confidence)
            self.baseline = sum(self.conf_history) / len(self.conf_history)
            self.track_hist.append((self.frame_id, cx, cy, w, h))
            return bbox, self._info(rel=1.0)

        rel = confidence / self.baseline if self.baseline else 1.0
        low = rel < self.ratio

        # ===== 상태 기계 =====
        if self.state == "NORMAL":
            # 위치 히스토리 갱신 (NORMAL일 때만 신뢰)
            self.track_hist.append((self.frame_id, cx, cy, w, h))
            self.low_count = self.low_count + 1 if low else 0
            if self.low_count >= self.persist:
                # SUSPECT 진입: 이 시점 기준 '과거'에서 motion 스냅샷 확보
                self._take_snapshot()
                self.state = "SUSPECT"
                self.suspect_count = 0
            # baseline 갱신은 안정 프레임만
            if rel >= self.update_thresh:
                self.conf_history.append(confidence)
                self.baseline = sum(self.conf_history) / len(self.conf_history)
            return bbox, self._info(rel=rel)

        if self.state == "SUSPECT":
            self.suspect_count += 1
            conf_ok = rel >= self.recover_ratio
            feat_ok = (cur_sim is not None and self.feat_med is not None
                       and cur_sim >= self.feat_med * self.feat_ratio)
            if feat_ok:
                #print(f"  [SUSPECT종료] f{self.frame_id} conf_ok={conf_ok}(rel={rel:.2f}) "
                #      f"feat_ok={feat_ok}(sim={cur_sim if cur_sim else 'None'}) "
                #      f"개입지속={self.suspect_count}프레임")
                self.state = "NORMAL"; self.low_count = 0
                return bbox, self._info(rel)
            if self.suspect_count > self.recover_window:
                self.state = "NORMAL"; self.low_count = 0
                return bbox, self._info(rel)
            pred_box = self._predict_box()
            if pred_box is not None:
                return pred_box, self._info(rel, recovered=True)
            return bbox, self._info(rel)
          
        if self.state == "LOST":
            # motion 예측 위치로 self.state를 덮어써서 트래커를 끌어온다
            pred_box = self._predict_box()
            self.recover_count += 1
            if pred_box is not None:
                # 복구 후보 위치 제시
                if not low or self.recover_count == 1:
                    # conf가 회복되면(예측 위치에서 원객체 재포착) 성공
                    if rel >= self.recover_ratio:
                        self.state = "NORMAL"; self.low_count = 0
                        return bbox, self._info(rel=rel)  # 이미 회복, 보정 불필요
                if self.recover_count > self.recover_window:
                    # 복구 실패 -> 일단 NORMAL로 풀어주되 다음 흔들림 재감지에 맡김
                    self.state = "NORMAL"; self.low_count = 0
                    return bbox, self._info(rel=rel)
                return pred_box, self._info(rel=rel, recovered=True)
            else:
                self.state = "NORMAL"; self.low_count = 0
                return bbox, self._info(rel=rel)

        return bbox, self._info(rel=rel)

    # ----- 내부 헬퍼 -----
    def _take_snapshot(self):
        """SUSPECT 진입 시, 오염 전(motion_lag 이전) 구간으로 속도/크기 스냅샷."""
        hist = list(self.track_hist)
        if len(hist) < self.motion_lag + 2:
            self.snap_vel = (0.0, 0.0)
            f, cx, cy, w, h = hist[-1]
            self.snap_size = (w, h); self.snap_pos = (cx, cy, f)
            return
        # 오염 전 끝점 = lag 이전, 시작점 = 그보다 span 이전
        end = hist[-self.motion_lag]
        start_idx = max(0, len(hist) - self.motion_lag - self.motion_span)
        start = hist[start_idx]
        nf = end[0] - start[0]
        if nf <= 0:
            self.snap_vel = (0.0, 0.0)
        else:
            self.snap_vel = ((end[1] - start[1]) / nf, (end[2] - start[2]) / nf)
        self.snap_size = (end[3], end[4])
        self.snap_pos = (end[1], end[2], end[0])

    def _predict_box(self):
        """스냅샷 기준 등속 외삽으로 현재 프레임의 예측 bbox 반환."""
        #if self.snap_pos is None:
        #    if 940 <= self.frame_id <= 960:
        #        print(f"  [predict] f{self.frame_id} snap_pos=None")
        #    return None
        vx, vy = self.snap_vel
        speed = (vx*vx + vy*vy) ** 0.5
        #if 940 <= self.frame_id <= 960:
        #    print(f"  [predict] f{self.frame_id} speed={speed:.2f} snap_pos={self.snap_pos}")
        cx0, cy0, f0 = self.snap_pos
        vx, vy = self.snap_vel
        speed = (vx * vx + vy * vy) ** 0.5
        self._last_speed = speed                 # for debug
        if speed > self.max_motion_speed:
            return None
        w, h = self.snap_size
        dt = self.frame_id - f0
        pcx = cx0 + vx * dt
        pcy = cy0 + vy * dt
        return (pcx - w / 2, pcy - h / 2, w, h)

    def _cos(self, a, b):
        import numpy as np
        a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
        na = (a*a).sum()**0.5; nb = (b*b).sum()**0.5
        if na == 0 or nb == 0: return 0.0
        return float((a*b).sum() / (na*nb))
    
    def _info(self, rel, recovered=False):
        return {"frame": self.frame_id, "state": self.state, "rel": round(rel, 3),
                "baseline": round(self.baseline, 3) if self.baseline else None,
                "suspect_count": self.suspect_count, "recovered": recovered}
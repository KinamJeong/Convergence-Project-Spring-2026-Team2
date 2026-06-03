print(">>> switch_recovery loaded from:", __file__)
from collections import deque
import numpy as np


class BBoxKalmanFilter:
    """8차원 등속도 칼만 필터. 상태: [cx, cy, w, h, vx, vy, vw, vh]"""
    def __init__(self):
        self.x = np.zeros((8, 1), dtype=np.float32)
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = 1.0
        self.H = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self.H[i, i] = 1.0
        self.P = np.eye(8, dtype=np.float32) * 10.0
        self.Q = np.eye(8, dtype=np.float32) * 0.05
        self.R = np.eye(4, dtype=np.float32) * 8.0
        self.initialized = False

    def _tlwh_to_z(self, box):
        x, y, w, h = box
        return np.array([[x + w / 2], [y + h / 2], [w], [h]], dtype=np.float32)

    def _z_to_tlwh(self):
        cx, cy, w, h = self.x[:4].flatten()
        return [float(cx - w / 2), float(cy - h / 2), float(w), float(h)]

    def initiate(self, box):
        self.x[:4] = self._tlwh_to_z(box)
        self.x[4:] = 0.0
        self.P = np.eye(8, dtype=np.float32) * 10.0
        self.initialized = True

    def predict(self):
        if not self.initialized:
            return None
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self._z_to_tlwh()

    def update(self, box):
        if not self.initialized:
            self.initiate(box)
            return
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        z = self._tlwh_to_z(box)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8, dtype=np.float32) - K @ self.H) @ self.P


class SwitchRecoveryModule:
    def __init__(self,
                 ratio=0.65, persist=3, warmup=30,
                 baseline_window=50, update_thresh=0.85,
                 recover_window=40, recover_ratio=0.75,
                 feat_ratio=0.85, feat_warmup=30, feat_window=50):
        self.ratio = ratio
        self.persist = persist
        self.warmup = warmup
        self.update_thresh = update_thresh
        self.baseline = None
        self.conf_history = deque(maxlen=baseline_window)
        self.low_count = 0
        self.recover_window = recover_window
        self.recover_ratio = recover_ratio
        self.feat_ratio = feat_ratio
        self.feat_warmup = feat_warmup
        self.feat_history = deque(maxlen=feat_window)
        self.feat_ref = None
        self.feat_med = None
        self.state = "NORMAL"
        self.frame_id = 0
        self.suspect_count = 0
        self.kf = BBoxKalmanFilter()

    def update(self, confidence, bbox, feat_vec=None):
        self.frame_id += 1

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

        if self.frame_id <= self.warmup:
            self.conf_history.append(confidence)
            self.baseline = sum(self.conf_history) / len(self.conf_history)
            self.kf.update(bbox)
            return bbox, self._info(rel=1.0, sim=cur_sim)

        rel = confidence / self.baseline if self.baseline else 1.0
        low = rel < self.ratio

        if self.state == "NORMAL":
            if rel >= self.update_thresh:  
                self.kf.update(bbox)
            self.low_count = self.low_count + 1 if low else 0
            if self.low_count >= self.persist:
                self.state = "SUSPECT"
                self.suspect_count = 0
            if rel >= self.update_thresh:
                self.conf_history.append(confidence)
                self.baseline = sum(self.conf_history) / len(self.conf_history)
            return bbox, self._info(rel=rel, sim=cur_sim)

        if self.state == "SUSPECT":
            self.suspect_count += 1
            feat_ok = (cur_sim is not None and self.feat_med is not None
                       and cur_sim >= self.feat_med * self.feat_ratio)
            if feat_ok:
                self.state = "NORMAL"
                self.low_count = 0
                return bbox, self._info(rel=rel, sim=cur_sim)
            if self.suspect_count > self.recover_window:
                self.state = "NORMAL"
                self.low_count = 0
                return bbox, self._info(rel=rel, sim=cur_sim)
            pred_box = self.kf.predict()
            if pred_box is not None:
                return pred_box, self._info(rel=rel, sim=cur_sim, recovered=True)
            return bbox, self._info(rel=rel, sim=cur_sim)

        return bbox, self._info(rel=rel, sim=cur_sim)

    def _cos(self, a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _info(self, rel, sim=None, recovered=False):
        return {"frame": self.frame_id, "state": self.state,
                "rel": round(rel, 3), "sim": round(sim, 3) if sim is not None else None,
                "suspect_count": self.suspect_count, "recovered": recovered}
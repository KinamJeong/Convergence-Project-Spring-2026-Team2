"""
MVTD Online Data Augmentation Module
=====================================
HIPTrack Fine-Tuning을 위한 온라인 증강 모듈.
학습 중 매 배치마다 실시간으로 증강을 적용하며, 이미지를 디스크에 저장하지 않음.

augment_offline.py의 해양 특화 증강 기법을 온라인 방식으로 재구성:
  hflip               → 수평 반전 (좌우 방향 불변성)
  zoom_in             → 카메라 확대 시뮬레이션 (스케일 급변 대응)
  zoom_out            → 카메라 축소 시뮬레이션 (스케일 급변 대응)
  synthetic_reflection→ 수면 반사 합성 (대형 선박 반사상 대응)
  specular_highlight  → 수면 반사 하이라이트 주입 (소형 타겟 주변)
  wave_blur           → 파도 방향 모션 블러 (고속 이동 대응)
  maritime_haze       → 해상 안개/저대비 (돛-하늘 혼동 대응)
  wake_overlay        → 항적 텍스처 오버레이 (선미-항적 혼동 대응)
  foreground_occlusion→ 전경 물체 간섭 시뮬레이션 (부분 가려짐 대응)
  background_jitter   → 배경 색상 유사화 (배경 혼동 대응)
  color_jitter        → 색상 지터 (조명 변화 불변성)

사용법:
  from lib.train.data.mvtd_augmentation import MVTDAugmentor
  augmentor = MVTDAugmentor(aug_prob=0.5)
  aug_frames, aug_anno = augmentor.apply(frame_list, anno_frames)

오프라인 방식과의 차이:
  - 이미지를 디스크에 저장하지 않음
  - 매 배치마다 랜덤하게 다른 증강 적용 → 더 높은 다양성
  - numpy (OpenCV) ↔ PIL Image 변환 처리 포함
  - bbox 형식: torch.tensor [x, y, w, h] 자동 처리
"""

import cv2
import random
import numpy as np
import torch
from PIL import Image


# ============================================================
# 증강 기법 함수 모음 (입출력: numpy BGR image, tuple bbox)
# ============================================================

def hflip(img, bbox):
    """수평 반전 — 좌우 방향 불변성 학습"""
    h, w = img.shape[:2]
    flipped = cv2.flip(img, 1)
    x, y, bw, bh = bbox
    new_bbox = (w - x - bw, y, bw, bh)
    return flipped, new_bbox


def zoom_in(img, bbox, scale_range=(1.2, 1.8)):
    """
    카메라 급격한 확대 시뮬레이션 (스케일 급변 대응).
    타겟 중심으로 crop 후 원래 크기로 resize.
    """
    h, w = img.shape[:2]
    scale = random.uniform(*scale_range)
    x, y, bw, bh = bbox

    cx = x + bw / 2
    cy = y + bh / 2

    new_w = int(w / scale)
    new_h = int(h / scale)

    x0 = int(np.clip(cx - new_w / 2, 0, w - new_w))
    y0 = int(np.clip(cy - new_h / 2, 0, h - new_h))
    x1 = x0 + new_w
    y1 = y0 + new_h

    cropped = img[y0:y1, x0:x1]
    zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    ratio_x = w / new_w
    ratio_y = h / new_h
    new_bbox = (
        int((x - x0) * ratio_x),
        int((y - y0) * ratio_y),
        int(bw * ratio_x),
        int(bh * ratio_y),
    )
    new_bbox = _clip_bbox(new_bbox, w, h)
    return zoomed, new_bbox


def zoom_out(img, bbox, scale_range=(0.4, 0.7)):
    """
    카메라 급격한 축소 시뮬레이션 (스케일 급변 대응).
    타겟 이미지를 축소 후 검정 패딩.
    """
    h, w = img.shape[:2]
    scale = random.uniform(*scale_range)

    new_w = int(w * scale)
    new_h = int(h * scale)
    small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_x = random.randint(0, w - new_w)
    pad_y = random.randint(0, h - new_h)

    canvas = np.zeros_like(img)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = small

    x, y, bw, bh = bbox
    new_bbox = (
        int(x * scale + pad_x),
        int(y * scale + pad_y),
        int(bw * scale),
        int(bh * scale),
    )
    new_bbox = _clip_bbox(new_bbox, w, h)
    return canvas, new_bbox


def synthetic_reflection(img, bbox, alpha_range=(0.2, 0.55)):
    """
    수면 반사 합성 (대형 선박 반사상 대응).
    이미지 하단에 상단의 수평 반전 이미지를 alpha-blend.
    """
    h, w = img.shape[:2]
    alpha = random.uniform(*alpha_range)
    horizon = random.randint(int(h * 0.35), int(h * 0.65))

    reflected = cv2.flip(img[:horizon], 0)
    below_h = h - horizon
    reflected_r = cv2.resize(reflected, (w, below_h))

    result = img.copy()
    result[horizon:] = cv2.addWeighted(
        img[horizon:], 1.0 - alpha,
        reflected_r, alpha, 0
    )
    return result, bbox


def specular_highlight(img, bbox, n_range=(2, 6)):
    """
    수면 반사 하이라이트 주입 (소형 타겟 주변 반사 대응).
    HLS 색공간에서 고밝기 저채도 타원 패치 삽입.
    """
    h, w = img.shape[:2]
    result = img.copy()
    hls = cv2.cvtColor(result, cv2.COLOR_BGR2HLS).astype(np.int32)

    n = random.randint(*n_range)
    for _ in range(n):
        cx = random.randint(0, w)
        cy = random.randint(h // 2, h)
        rx = random.randint(8, 35)
        ry = random.randint(4, 16)
        angle = random.randint(0, 180)

        mask = np.zeros((h, w), np.uint8)
        cv2.ellipse(mask, (cx, cy), (rx, ry), angle, 0, 360, 255, -1)

        hls[:, :, 1][mask > 0] += random.randint(60, 90)
        hls[:, :, 2][mask > 0] -= random.randint(30, 50)

    hls = np.clip(hls, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hls, cv2.COLOR_HLS2BGR), bbox


def wave_blur(img, bbox, kernel_range=(5, 17), angle_range=(-35.0, 35.0)):
    """
    파도 방향 모션 블러 (고속 이동 모션블러 대응).
    수평에 가까운 방향으로 커널 제한.
    """
    k = random.randint(*kernel_range)
    k = k if k % 2 == 1 else k + 1
    angle = random.uniform(*angle_range)

    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k
    M = cv2.getRotationMatrix2D((k // 2, k // 2), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    s = kernel.sum()
    if s > 0:
        kernel /= s

    return cv2.filter2D(img, -1, kernel), bbox


def maritime_haze(img, bbox, intensity_range=(0.15, 0.45)):
    """
    해상 안개/저대비 (돛-하늘 혼동 대응).
    밝기 증가 + Gaussian blur 조합.
    """
    intensity = random.uniform(*intensity_range)
    haze = np.ones_like(img, dtype=np.float32) * 255.0
    result = cv2.addWeighted(
        img.astype(np.float32), 1.0 - intensity,
        haze, intensity, 0
    ).astype(np.uint8)

    blur_k = max(3, int(intensity * 12) * 2 + 1)
    result = cv2.GaussianBlur(result, (blur_k, blur_k), 0)
    return result, bbox


def wake_overlay(img, bbox):
    """
    항적(Wake) 텍스처 오버레이 (선미-항적 혼동 대응).
    선박 bbox 뒤쪽에 V자형 항적 패턴을 합성.
    """
    h, w = img.shape[:2]
    result = img.copy()
    x, y, bw, bh = bbox

    direction = random.choice([-1, 1])
    wake_start_x = x + bw if direction == 1 else x
    wake_start_y = y + bh // 2

    wake_len = random.randint(int(bw * 0.8), int(bw * 2.0))
    wake_angle = random.uniform(10, 25)

    overlay = result.copy()
    pts_upper = []
    pts_lower = []

    steps = 30
    for i in range(steps + 1):
        t = i / steps
        wx = int(wake_start_x + direction * wake_len * t)
        spread = int(wake_len * t * np.tan(np.radians(wake_angle)))
        pts_upper.append([wx, wake_start_y - spread])
        pts_lower.append([wx, wake_start_y + spread])

    pts = np.array(pts_upper + pts_lower[::-1], dtype=np.int32)
    pts = np.clip(pts, 0, [w - 1, h - 1])

    cv2.fillPoly(overlay, [pts], (220, 230, 235))
    alpha = random.uniform(0.25, 0.45)
    result = cv2.addWeighted(result, 1 - alpha, overlay, alpha, 0)

    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    noise = np.random.normal(0, 8, result.shape).astype(np.int16)
    result_int = result.astype(np.int16)
    result_int[mask > 0] += noise[mask > 0]
    result = np.clip(result_int, 0, 255).astype(np.uint8)

    return result, bbox


def foreground_occlusion(img, bbox):
    """
    전경 물체 간섭 시뮬레이션 (부분 가려짐 대응).
    랜덤 크기의 반투명 패치로 타겟 일부를 가림.
    (타겟 완전 소실 방지: min_visibility=0.4 체크 포함)
    """
    h, w = img.shape[:2]
    result = img.copy()
    x, y, bw, bh = bbox

    occ_w = random.randint(int(bw * 0.2), int(bw * 0.5))
    occ_h = random.randint(int(bh * 0.2), int(bh * 0.5))

    occ_x = random.randint(max(0, x - occ_w), min(w - occ_w, x + bw))
    occ_y = random.randint(max(0, y - occ_h), min(h - occ_h, y + bh))

    color_choices = [
        (int(random.uniform(30, 80)), int(random.uniform(60, 110)), int(random.uniform(50, 100))),
        (int(random.uniform(150, 220)), int(random.uniform(160, 230)), int(random.uniform(170, 240))),
        (int(random.uniform(180, 230)), int(random.uniform(185, 235)), int(random.uniform(190, 240))),
    ]
    color = random.choice(color_choices)

    occ_rect = (occ_x, occ_y, occ_w, occ_h)
    if _compute_visibility(bbox, occ_rect) < 0.4:
        return img, bbox

    alpha = random.uniform(0.5, 0.85)
    overlay = result.copy()
    cv2.rectangle(overlay,
                  (occ_x, occ_y),
                  (occ_x + occ_w, occ_y + occ_h),
                  color, -1)

    roi = overlay[occ_y:occ_y + occ_h, occ_x:occ_x + occ_w]
    k = max(3, min(occ_w, occ_h) // 4 * 2 + 1)
    roi_blur = cv2.GaussianBlur(roi, (k, k), 0)
    overlay[occ_y:occ_y + occ_h, occ_x:occ_x + occ_w] = roi_blur

    result = cv2.addWeighted(result, 1 - alpha, overlay, alpha, 0)
    return result, bbox


def background_jitter(img, bbox):
    """
    배경 색상 유사화 (배경 혼동 대응).
    타겟 주변 배경 영역의 색상을 타겟과 유사하게 만들어
    모델이 배경 구분 능력을 학습하도록 유도.
    """
    h, w = img.shape[:2]
    result = img.copy()
    x, y, bw, bh = bbox

    margin = random.randint(int(min(bw, bh) * 0.3), int(min(bw, bh) * 0.8))
    bg_x = max(0, x - margin)
    bg_y = max(0, y - margin)
    bg_x2 = min(w, x + bw + margin)
    bg_y2 = min(h, y + bh + margin)

    target_roi = img[y:y + bh, x:x + bw]
    if target_roi.size == 0:
        return img, bbox
    mean_color = target_roi.mean(axis=(0, 1))

    bg_region = result[bg_y:bg_y2, bg_x:bg_x2].astype(np.float32)
    noise = np.random.normal(0, 12, bg_region.shape)
    jitter = random.uniform(0.15, 0.35)
    bg_blended = bg_region * (1 - jitter) + mean_color * jitter + noise
    result[bg_y:bg_y2, bg_x:bg_x2] = np.clip(bg_blended, 0, 255).astype(np.uint8)

    result[y:y + bh, x:x + bw] = img[y:y + bh, x:x + bw]
    return result, bbox


def color_jitter(img, bbox, brightness=0.3, contrast=0.3, saturation=0.2):
    """일반 색상 지터 — 조명 변화 불변성"""
    result = img.copy().astype(np.float32)

    b = random.uniform(1 - brightness, 1 + brightness)
    result *= b

    c = random.uniform(1 - contrast, 1 + contrast)
    mean = result.mean()
    result = (result - mean) * c + mean

    result = np.clip(result, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
    s = random.uniform(1 - saturation, 1 + saturation)
    hsv[:, :, 1] *= s
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return result, bbox


# ============================================================
# 전략 딕셔너리
# ============================================================

STRATEGY_FN = {
    "hflip":                hflip,
    "zoom_in":              zoom_in,
    "zoom_out":             zoom_out,
    "synthetic_reflection": synthetic_reflection,
    "specular_highlight":   specular_highlight,
    "wave_blur":            wave_blur,
    "maritime_haze":        maritime_haze,
    "wake_overlay":         wake_overlay,
    "foreground_occlusion": foreground_occlusion,
    "background_jitter":    background_jitter,
    "color_jitter":         color_jitter,
}

# 전 클래스 공통 전략 풀 (모든 해양 특화 기법 포함)
ALL_STRATEGIES = list(STRATEGY_FN.keys())


# ============================================================
# 유틸리티 함수
# ============================================================

def _clip_bbox(bbox, w, h, min_size=10):
    """bbox가 이미지 경계를 벗어나지 않도록 클리핑"""
    x, y, bw, bh = bbox
    x = int(np.clip(x, 0, w - min_size))
    y = int(np.clip(y, 0, h - min_size))
    bw = int(np.clip(bw, min_size, w - x))
    bh = int(np.clip(bh, min_size, h - y))
    return (x, y, bw, bh)


def _compute_visibility(bbox, occ):
    """오클루전 후 타겟 가시성 비율 계산"""
    bx, by, bw, bh = bbox
    ox, oy, ow, oh = occ

    inter_x = max(0, min(bx + bw, ox + ow) - max(bx, ox))
    inter_y = max(0, min(by + bh, oy + oh) - max(by, oy))
    inter_area = inter_x * inter_y
    bbox_area = bw * bh
    if bbox_area == 0:
        return 1.0
    return 1.0 - (inter_area / bbox_area)


def _pil_to_bgr(pil_img):
    """PIL Image (RGB) → numpy BGR"""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _bgr_to_pil(bgr_img):
    """numpy BGR → PIL Image (RGB)"""
    return Image.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))


def _tensor_bbox_to_tuple(bbox_tensor):
    """torch.tensor [x, y, w, h] → tuple (x, y, w, h) int"""
    return tuple(int(v.item()) for v in bbox_tensor)


def _tuple_bbox_to_tensor(bbox_tuple):
    """tuple (x, y, w, h) → torch.tensor float"""
    return torch.tensor(bbox_tuple, dtype=torch.float32)


# ============================================================
# 메인 증강 클래스
# ============================================================

class MVTDAugmentor:
    """
    MVTD 온라인 증강 클래스.

    Parameters
    ----------
    aug_prob : float
        각 프레임에 증강을 적용할 확률 (0.0 ~ 1.0). 기본값 0.5.
    strategies : list[str] or None
        사용할 증강 전략 목록. None이면 ALL_STRATEGIES 전체 사용.
    min_bbox_size : int
        증강을 적용하기 위한 최소 bbox 크기 (px). 너무 작은 타겟은 건너뜀.

    Example
    -------
    >>> augmentor = MVTDAugmentor(aug_prob=0.5)
    >>> aug_frames, aug_anno = augmentor.apply(frame_list, anno_frames)
    """

    def __init__(self, aug_prob=0.5, strategies=None, min_bbox_size=10):
        self.aug_prob = aug_prob
        self.strategies = strategies if strategies is not None else ALL_STRATEGIES
        self.min_bbox_size = min_bbox_size

    def apply(self, frame_list, anno_frames):
        """
        프레임 리스트와 어노테이션에 온라인 증강 적용.

        Parameters
        ----------
        frame_list : list[PIL.Image]
            get_frames()에서 반환된 PIL 이미지 리스트
        anno_frames : dict
            {'bbox': [tensor, ...], 'valid': [...], 'visible': [...], ...}

        Returns
        -------
        aug_frame_list : list[PIL.Image]
            증강된 PIL 이미지 리스트
        aug_anno_frames : dict
            증강된 어노테이션 딕셔너리 (bbox 업데이트됨)
        """
        aug_frame_list = []
        aug_bboxes = []

        # 시퀀스 단위로 루프 밖에서 한 번만 결정
        apply_aug = random.random() <= self.aug_prob
        strategy_name = random.choice(self.strategies)
        strategy_fn = STRATEGY_FN[strategy_name]

        for i, frame in enumerate(frame_list):
            bbox_tensor = anno_frames['bbox'][i]  # torch.tensor [x, y, w, h]

            # 이번 시퀀스에 증강 미적용으로 결정된 경우
            if not apply_aug:
                aug_frame_list.append(frame)
                aug_bboxes.append(bbox_tensor)
                continue

            # bbox 유효성 체크 (너무 작은 타겟은 건너뜀)
            bbox_tuple = _tensor_bbox_to_tuple(bbox_tensor)
            if bbox_tuple[2] < self.min_bbox_size or bbox_tuple[3] < self.min_bbox_size:
                aug_frame_list.append(frame)
                aug_bboxes.append(bbox_tensor)
                continue

            # PIL → numpy BGR 변환
            bgr_img = _pil_to_bgr(frame)

            # 랜덤 전략 선택 및 적용
            strategy_name = random.choice(self.strategies)
            strategy_fn = STRATEGY_FN[strategy_name]

            try:
                aug_bgr, aug_bbox_tuple = strategy_fn(bgr_img, bbox_tuple)
                if aug_bbox_tuple is not None:
                    h, w = aug_bgr.shape[:2]
                    aug_bbox_tuple = _clip_bbox(aug_bbox_tuple, w, h)
                aug_frame_list.append(_bgr_to_pil(aug_bgr))
                aug_bboxes.append(_tuple_bbox_to_tensor(aug_bbox_tuple))
            except Exception:
                # 예외 발생 시 원본 유지
                aug_frame_list.append(frame)
                aug_bboxes.append(bbox_tensor)

        # anno_frames 업데이트 (bbox만 교체, 나머지는 원본 유지)
        aug_anno_frames = {k: v for k, v in anno_frames.items()}
        aug_anno_frames['bbox'] = aug_bboxes

        return aug_frame_list, aug_anno_frames
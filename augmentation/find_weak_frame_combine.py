import os
import cv2
import json
import numpy as np
from pathlib import Path

def calculate_metrics(gt_box, pred_box, img_wh):
    """
    논문 공식 지표 4가지 계산
    - Success (IoU)
    - Precision (Center Error)
    - Normalized Precision
    - Scale Ratio
    """
    if gt_box is None or pred_box is None:
        return 0.0, float('inf'), 0.0, 0.0
        
    gx, gy, gw, gh = gt_box
    px, py, pw, ph = pred_box
    img_w, img_h = img_wh

    # 1. Success (IoU) 계산
    inter_x1 = max(gx, px)
    inter_y1 = max(gy, py)
    inter_x2 = min(gx + gw, px + pw)
    inter_y2 = min(gy + gh, py + ph)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_gt = gw * gh
    area_pred = pw * ph
    union_area = area_gt + area_pred - inter_area
    iou = inter_area / union_area if union_area > 0 else 0.0

    # 2. Precision (Center Error in pixels) 계산
    g_cx, g_cy = gx + gw / 2.0, gy + gh / 2.0
    p_cx, p_cy = px + pw / 2.0, py + ph / 2.0
    center_error = np.sqrt((g_cx - p_cx) ** 2 + (g_cy - p_cy) ** 2)

    # 3. Normalized Precision 계산
    diag = np.sqrt(img_w ** 2 + img_h ** 2)
    norm_precision = max(0.0, 1.0 - (center_error / (diag * 0.1)))

    # 4. Scale Ratio (면적 비율)
    scale_ratio = area_pred / area_gt if area_gt > 0 else 0.0

    return round(iou, 4), round(center_error, 2), round(norm_precision, 4), round(scale_ratio, 4)

def read_bbox_file(file_path):
    bboxes = []
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                bboxes.append(None)
                continue
            parts = line.replace(',', ' ').split()
            if len(parts) >= 4:
                try:
                    bboxes.append(tuple(int(float(p)) for p in parts[:4]))
                except ValueError:
                    bboxes.append(None)
            else:
                bboxes.append(None)
    return bboxes

def analyze_weak_cases(seq_dir, pred_boxes, gt_boxes, frame_files):
    weak_report = []
    
    for i, (fpath, gt, pred) in enumerate(zip(frame_files, gt_boxes, pred_boxes)):
        if gt is None:
            continue
            
        img = cv2.imread(str(fpath))
        if img is None:
            continue
        h, w, _ = img.shape
        
        # 공식 지표 계산
        iou, center_error, norm_precision, scale_ratio = calculate_metrics(gt, pred, (w, h))
        
        # 취약 프레임 검출 조건 (IoU 0.5 미만 또는 Center Error 20픽셀 초과 또는 미탐지)
        is_weak_frame = (iou < 0.5) or (center_error > 20.0) or (pred is None)
        
        if is_weak_frame:
            gx, gy, gw, gh = gt
            roi = img[gy:gy+gh, gx:gx+gw]
            
            # 해상 물리 환경 특징 추출
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_contrast = gray.std()
            
            is_low_contrast = img_contrast < 30.0
            is_blur = False
            if roi.size > 0:
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                is_blur = cv2.Laplacian(roi_gray, cv2.CV_64F).var() < 100.0
            is_small = (gw * gh) < (w * h * 0.01)
            
            # 환경별 최적화 증강 전략 자동 매핑
            matched_strategies = []
            if is_low_contrast:
                matched_strategies.extend(["maritime_haze", "color_jitter"])
            if is_blur:
                matched_strategies.append("wave_blur")
            if is_small:
                matched_strategies.extend(["zoom_in", "background_jitter"])
                
            if scale_ratio > 1.8 or scale_ratio < 0.4:
                matched_strategies.append("zoom_out")
                
            # 기본 보완 전략 및 파일 주석 기반 특화 매핑 유도
            if not matched_strategies:
                matched_strategies.extend(["synthetic_reflection", "hflip"])
            
            # 특정 시퀀스 고유 취약점 보완 코드 주입 (예: 104-Boat의 항적 혼동 저격)
            if "104-Boat" in str(seq_dir):
                matched_strategies.append("wake_overlay")
            if any(s in str(seq_dir) for s in ["90-Boat", "118-Boat", "147-Boat"]):
                matched_strategies.append("foreground_occlusion")
                
            weak_report.append({
                "frame_idx": i,
                "frame_name": fpath.name,
                "metrics": {
                    "iou_success": iou,
                    "center_error_precision": center_error,
                    "normalized_precision": norm_precision,
                    "scale_ratio": scale_ratio
                },
                "gt_bbox": gt,
                "suggested_strategies": list(set(matched_strategies))
            })
            
    return weak_report

def main():
    # 기남 님의 환경에 맞춘 정확한 경로 세팅
    HOME_DIR = Path.home()
    DATA_ROOT = HOME_DIR / "hiptrack/HIPTrack/data/mvtd/train"
    PRED_ROOT = HOME_DIR / "hiptrack/HIPTrack/output/test/tracking_results/hiptrack/hiptrack"
    
    # 추론 결과 추출이 확인된 대상 시퀀스 목록
    TARGET_SEQUENCES = ["90-Boat", "104-Boat", "118-Boat", "135-Boat", "147-Boat"]
    
    print("==========================================================")
    print("  지정 경로 기반 다중 시퀀스 취약 프레임 마이닝 파이프라인")
    print("==========================================================")
    
    for seq_name in TARGET_SEQUENCES:
        seq_dir = DATA_ROOT / seq_name
        gt_path = seq_dir / "groundtruth.txt"
        pred_path = PRED_ROOT / f"{seq_name}.txt" # 예: 90-Boat.txt
        
        if not seq_dir.exists():
            print(f"[SKIP] 원본 데이터 폴더가 존재하지 않음: {seq_dir}")
            continue
        if not pred_path.exists():
            print(f"[SKIP] 추론 결과 파일이 존재하지 않음: {pred_path}")
            continue
            
        exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        frame_files = sorted([f for f in seq_dir.iterdir() if f.suffix.lower() in exts])
        
        gt_boxes = read_bbox_file(gt_path)
        pred_boxes = read_bbox_file(pred_path)
        
        # 프레임 수 싱크 맞추기
        if gt_boxes is None or pred_boxes is None:
            print(f"[SKIP] {seq_name} 라벨 파싱 실패.")
            continue
            
        min_len = min(len(frame_files), len(gt_boxes), len(pred_boxes))
        frame_files = frame_files[:min_len]
        gt_boxes = gt_boxes[:min_len]
        pred_boxes = pred_boxes[:min_len]
        
        print(f"--> [{seq_name}] 총 {min_len}개 프레임 정량 분석 중...")
        report = analyze_weak_cases(seq_dir, pred_boxes, gt_boxes, frame_files)
        
        # 시퀀스별 폴더 내부에 json 진단서 보관
        report_path = seq_dir / "weak_frames_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"    [완료] 취약 프레임 {len(report)}개 확정 -> 보고서 반영 완료.")
    
    print("\n==========================================================")
    print("  모든 대상 시퀀스 마이닝 완료. 다음 단계(증강) 진행 가능합니다.")
    print("==========================================================")

if __name__ == "__main__":
    main()
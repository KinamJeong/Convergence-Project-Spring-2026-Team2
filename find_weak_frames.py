import os
import numpy as np
import pandas as pd
import math

def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union_area = (w1 * h1) + (w2 * h2) - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def safe_load_boxes(filepath):
    try: return np.loadtxt(filepath, delimiter=',')
    except ValueError:
        try: return np.loadtxt(filepath, delimiter='\t')
        except ValueError: return np.loadtxt(filepath)

# 1. 경로 설정
pred_base_dir = '/home/ivl5/hiptrack/HIPTrack/output/test/tracking_results/hiptrack/hiptrack'
gt_base_dir = '/home/ivl5/hiptrack/HIPTrack/data/mvtd/train'

# 2. 🌟 논문 공식 평가 임계값 (SOT 표준)
IOU_THRESHOLD = 0.5          # Success Rate 기준: 겹침 비율이 50% 미만이면 실패
CENTER_ERR_THRESHOLD = 20.0  # Precision 기준: 중심점 오차가 20픽셀을 넘으면 실패
NORM_ERR_THRESHOLD = 0.2     # Normalized Precision 기준: 오차가 객체 대각선 길이의 20%를 넘으면 실패
SCALE_BLOAT_LIMIT = 1.5      # 면적 팽창 한계점: GT 대비 1.5배 이상 커지면 실패 (147-Boat 대응용 보조 지표)
SCALE_SHRINK_LIMIT = 0.5     # 면적 축소 한계점: GT 대비 절반 이하로 작아지면 실패

weak_cases_summary = []
processed_count = 0

print("⚠️ 논문 공식 지표 기반 Weak Case 정밀 분석을 시작합니다...")

for seq_name in sorted(os.listdir(pred_base_dir)):
    if not seq_name.endswith('.txt'): continue
        
    seq_id = seq_name.replace('.txt', '')
    pred_file = os.path.join(pred_base_dir, seq_name)
    gt_file = os.path.join(gt_base_dir, seq_id, 'groundtruth.txt')
    
    if not os.path.exists(gt_file): continue
        
    pred_boxes = safe_load_boxes(pred_file)
    gt_boxes = safe_load_boxes(gt_file)
    num_frames = min(len(pred_boxes), len(gt_boxes))
    
    in_weak_zone = False
    start_frame = -1
    current_cause = None
    
    for f_idx in range(num_frames):
        p_x, p_y, p_w, p_h = pred_boxes[f_idx]
        g_x, g_y, g_w, g_h = gt_boxes[f_idx]
        
        # [지표 1] Success (IoU)
        iou = calculate_iou(pred_boxes[f_idx], gt_boxes[f_idx])
        
        # [지표 2] Precision (Center Error in pixels)
        p_cx, p_cy = p_x + p_w/2, p_y + p_h/2
        g_cx, g_cy = g_x + g_w/2, g_y + g_h/2
        center_err = math.sqrt((p_cx - g_cx)**2 + (p_cy - g_cy)**2)
        
        # [지표 3] Normalized Precision
        gt_diag = math.sqrt(g_w**2 + g_h**2) if g_w > 0 and g_h > 0 else 1.0
        norm_center_err = center_err / gt_diag
        
        # [보조 지표] Scale Ratio (면적 비율)
        p_area, g_area = p_w * p_h, g_w * g_h
        scale_ratio = p_area / g_area if g_area > 0 else 1.0
        
        # 🚨 엄격한 실패 조건 (하나라도 위반하면 취약 프레임)
        is_weak = False
        fail_reason = ""
        
        if center_err > CENTER_ERR_THRESHOLD:
            is_weak, fail_reason = True, "Low Precision (Center Shift)"
        elif norm_center_err > NORM_ERR_THRESHOLD:
            is_weak, fail_reason = True, "Low Norm Precision"
        elif iou < IOU_THRESHOLD:
            is_weak, fail_reason = True, "Low Success (IoU)"
        elif scale_ratio > SCALE_BLOAT_LIMIT:
            is_weak, fail_reason = True, "Scale Bloat (>1.5x)"
        elif scale_ratio < SCALE_SHRINK_LIMIT:
            is_weak, fail_reason = True, "Scale Shrink (<0.5x)"
            
        if is_weak:
            if not in_weak_zone:
                in_weak_zone = True
                start_frame = f_idx + 1
                current_cause = fail_reason
        else:
            if in_weak_zone:
                in_weak_zone = False
                end_frame = f_idx
                weak_cases_summary.append({
                    'Sequence': seq_id,
                    'Start_Frame': start_frame,
                    'End_Frame': end_frame,
                    'Duration': end_frame - start_frame + 1,
                    'Main_Cause': current_cause
                })
                
    if in_weak_zone:
        weak_cases_summary.append({
            'Sequence': seq_id,
            'Start_Frame': start_frame,
            'End_Frame': num_frames,
            'Duration': num_frames - start_frame + 1,
            'Main_Cause': current_cause
        })
    processed_count += 1

# 결과 저장 로직
df_weak = pd.DataFrame(weak_cases_summary)
if not df_weak.empty:
    df_weak = df_weak[df_weak['Duration'] >= 10]
    df_weak = df_weak.sort_values(by=['Sequence', 'Start_Frame'])
    df_weak.to_csv('weak_frame_intervals.csv', index=False, encoding='utf-8-sig')
    print("✅ 분석 완료! 'weak_frame_intervals.csv'에 저장되었습니다.")
    print("\n📊 [발견된 주요 취약 구간 및 원인]")
    print(df_weak.head(15).to_string(index=False))
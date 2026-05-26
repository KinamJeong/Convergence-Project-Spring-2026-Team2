import os
import cv2
import json
import random
import numpy as np
from pathlib import Path

# 메인 오프라인 스크립트로부터 핵심 물리 Transform 함수들 임포트
from augment_offline import (
    STRATEGY_FN, 
    _clip_bbox, 
    read_groundtruth, 
    write_groundtruth
)

def get_local_frame_files(seq_dir):
    """시퀀스 폴더 내의 이미지 파일들을 정렬하여 반환"""
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    return sorted([f for f in Path(seq_dir).iterdir() if f.suffix.lower() in exts])

def execute_targeted_augmentation(data_root, seq_name, output_root, window_size=3):
    """개별 시퀀스의 보고서를 파싱하여 취약 구간에 정밀 타격 증강 주입"""
    data_root = Path(data_root)
    output_root = Path(output_root)
    seq_dir = data_root / seq_name
    
    report_path = seq_dir / "weak_frames_report.json"
    if not report_path.exists():
        print(f"  [SKIP] 취약점 보고서가 없음: {report_path.name}")
        return False
        
    with open(report_path, 'r') as f:
        weak_records = json.load(f)
        
    if not weak_records:
        print(f"  [SKIP] [{seq_name}] 마이닝된 취약 프레임이 없어 증강을 건너뜁니다.")
        return False

    frame_files = get_local_frame_files(seq_dir)
    gt_boxes = read_groundtruth(seq_dir / "groundtruth.txt")
    
    # 취약 프레임 인덱스별 추천 전략 맵 생성
    weak_map = {item["frame_idx"]: item["suggested_strategies"] for item in weak_records}
    
    # 시공간적(Temporal) 연속성을 위해 실패 프레임 앞뒤(window_size)로 가동 영역 확장
    extended_weak_targets = set()
    for idx in weak_map.keys():
        for w in range(-window_size, window_size + 1):
            if 0 <= idx + w < len(frame_files):
                extended_weak_targets.add(idx + w)

    # 스마트 증강 통합 출력 폴더 세팅 (_smart_aug 접미사 부여)
    aug_seq_dir = output_root / f"{seq_name}_smart_aug"
    aug_seq_dir.mkdir(parents=True, exist_ok=True)
    
    aug_gt_boxes = []
    aug_count = 0
    
    for i, (fpath, bbox) in enumerate(zip(frame_files, gt_boxes)):
        img = cv2.imread(str(fpath))
        if img is None:
            aug_gt_boxes.append(bbox)
            continue
            
        # 1. 정상 프레임 구간: 원본 이미지와 원본 BBox를 그대로 유지 (저장 용량 최적화)
        if i not in extended_weak_targets or bbox is None:
            cv2.imwrite(str(aug_seq_dir / fpath.name), img)
            aug_gt_boxes.append(bbox)
            continue
            
        # 2. 취약 프레임 및 주변 버퍼 구간: 대응하는 맞춤 전략 연산 적용
        # 주변 윈도우 프레임은 가장 가까운 실제 에러 프레임의 추천 전략을 탐색하여 승계
        ref_idx = i
        if ref_idx not in weak_map:
            # 좌우로 가장 가까운 취약 프레임의 전략 검색
            for step in range(1, window_size + 1):
                if (i - step) in weak_map:
                    ref_idx = i - step
                    break
                if (i + step) in weak_map:
                    ref_idx = i + step
                    break
                    
        strategies = weak_map.get(ref_idx, ["color_jitter"])
        selected_strategy = random.choice(strategies)
        strategy_fn = STRATEGY_FN.get(selected_strategy, STRATEGY_FN["color_jitter"])
        
        # 정밀 변환 행렬 및 바운딩 박스 업데이트 연산
        try:
            aug_img, aug_bbox = strategy_fn(img.copy(), bbox)
            if aug_bbox is not None:
                aug_bbox = _clip_bbox(aug_bbox, aug_img.shape[1], aug_img.shape[0])
            aug_count += 1
        except Exception:
            # 예외 발생 시 안전하게 원본 유지
            aug_img, aug_bbox = img, bbox
            
        cv2.imwrite(str(aug_seq_dir / fpath.name), aug_img)
        aug_gt_boxes.append(aug_bbox)
        
    # 새롭게 업데이트된 정답 라벨 파일 저장
    write_groundtruth(aug_seq_dir / "groundtruth.txt", aug_gt_boxes)
    print(f"  → [{seq_name}_smart_aug] 증강 완료: {aug_count}/{len(frame_files)} 프레임 변환됨.")
    return True

def main():
    HOME_DIR = Path.home()
    DATA_ROOT = HOME_DIR / "hiptrack/HIPTrack/data/mvtd/train"
    OUTPUT_ROOT = HOME_DIR / "hiptrack/HIPTrack/data/mvtd/train_smart"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    # 파이프라인 대상 다중 시퀀스 목록
    TARGET_SEQUENCES = ["118-Boat", "135-Boat", "147-Boat"]
    
    print("==========================================================")
    print("  취약점 진단서(JSON) 기반 정밀 타격형 다중 증강 파이프라인")
    print("==========================================================")
    
    success_count = 0
    for seq_name in TARGET_SEQUENCES:
        print(f"처리 중: {seq_name}")
        success = execute_targeted_augmentation(DATA_ROOT, seq_name, OUTPUT_ROOT, window_size=3)
        if success:
            success_count += 1
            
    print("\n==========================================================")
    print(f"  모든 공정 완료: 총 {success_count}개 시퀀스 스마트 증강 완료.")
    print(f"  출력 루트: {OUTPUT_ROOT.resolve()}")
    print("==========================================================")

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()
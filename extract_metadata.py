import os
import pandas as pd

# 1. 탐색할 데이터셋 경로 (앞서 분할한 전체 Train 폴더를 기준으로 탐색)
source_dir = '/home/ivl5/hiptrack/HIPTrack/data/mvtd/train'

metadata_list = []

print("전체 시퀀스의 길이와 속성 정보를 추출합니다...")

for seq_name in os.listdir(source_dir):
    seq_path = os.path.join(source_dir, seq_name)
    
    if not os.path.isdir(seq_path):
        continue
        
    # [카테고리] 추출
    category = seq_name.split('-')[-1] if '-' in seq_name else 'Unknown'
    
    # [시퀀스 길이] 추출: groundtruth.txt의 줄(line) 수 계산
    gt_path = os.path.join(seq_path, 'groundtruth.txt')
    seq_length = 0
    if os.path.exists(gt_path):
        with open(gt_path, 'r') as f:
            seq_length = sum(1 for line in f)
            
    # [추적 속성(Attributes)] 추출: .label 파일들 확인 및 프레임 수 대비 발생 비율 계산
    # (파일 안의 숫자가 1인 프레임이 얼마나 되는지 대략적으로 파악)
    attributes = []
    for file in os.listdir(seq_path):
        if file.endswith('.label'):
            attr_name = file.replace('.label', '')
            label_path = os.path.join(seq_path, file)
            
            # 라벨 파일 안에 '1'(해당 속성 발생)이 포함되어 있는지 확인
            try:
                with open(label_path, 'r') as f:
                    content = f.read()
                    if '1' in content:
                        attributes.append(attr_name)
            except Exception:
                pass
                
    # 데이터 저장
    metadata_list.append({
        'Sequence': seq_name,
        'Category': category,
        'Length (Frames)': seq_length,
        'Attributes': ", ".join(attributes) if attributes else "None"
    })

# DataFrame으로 변환
df = pd.DataFrame(metadata_list)

# 길이에 따라 Short / Medium / Long 범주화 (33%씩 균등 분할)
if not df.empty:
    df['Length_Group'] = pd.qcut(df['Length (Frames)'], 3, labels=['Short', 'Medium', 'Long'])

# 결과를 CSV 파일로 저장
output_csv = 'mvtd_metadata_summary.csv'
df.to_csv(output_csv, index=False, encoding='utf-8-sig')

print(f" 추출 완료! 총 {len(df)}개 시퀀스의 정보가 '{output_csv}'에 저장되었습니다.")

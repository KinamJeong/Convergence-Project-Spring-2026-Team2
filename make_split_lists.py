import pandas as pd
import math

# 1. 아까 추출해 둔 메타데이터 CSV 불러오기
csv_path = 'mvtd_metadata_summary.csv'
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"오류: '{csv_path}' 파일을 찾을 수 없습니다. 메타데이터 추출을 먼저 진행해주세요.")
    exit()

# 2. 검증 셋(Validation) 비율 설정 (15%)
VAL_RATIO = 0.15
val_sequences = []

# 3. 클래스(Category)와 길이(Length_Group) 2가지 조건을 동시에 만족하도록 그룹화 
# (예: 'Boat-Long' 그룹, 'USV-Short' 그룹 등 12개 조합 생성)
groups = df.groupby(['Category', 'Length_Group'], observed=True)

for (cat, length), group in groups:
    group_size = len(group)
    if group_size == 0:
        continue
        
    # 해당 세부 그룹에서 정확히 15%를 Validation으로 추출 (반올림)
    num_val = round(group_size * VAL_RATIO)
    
    # 세심한 보정: 특정 그룹의 데이터가 너무 적어 15%가 0명이 되는 것을 방지
    # (예: SailBoat-Short가 3개뿐이라면 15%는 0.45라 0개가 되므로, 강제로 최소 1개는 포함)
    if num_val == 0 and group_size >= 2:
        num_val = 1
        
    if num_val > 0:
        # random_state=42로 고정하여 언제 실행해도 항상 동일한 결과(재현성) 보장
        sampled = group.sample(n=num_val, random_state=42)
        val_sequences.extend(sampled['Sequence'].tolist())

# 4. Train / Validation 데이터프레임 분리
val_df = df[df['Sequence'].isin(val_sequences)]
train_df = df[~df['Sequence'].isin(val_sequences)]

# 5. 결과를 txt 파일로 저장 (팀원 공유용)
with open('train_list.txt', 'w') as f:
    f.write('\n'.join(train_df['Sequence'].tolist()))
    
with open('val_list.txt', 'w') as f:
    f.write('\n'.join(val_df['Sequence'].tolist()))

# 6. 분할 결과 요약 출력
print("데이터 분할 완료!\n")
print(f"전체 시퀀스: {len(df)}개")
print(f"Train Set : {len(train_df)}개 ({len(train_df)/len(df)*100:.1f}%)")
print(f"Valid Set : {len(val_df)}개 ({len(val_df)/len(df)*100:.1f}%)\n")

print("[Validation Set 상세 구성표]")
# 검증 셋 내의 클래스 vs 길이 교차표 출력
pivot = pd.crosstab(val_df['Category'], val_df['Length_Group'])
print("-" * 40)
print(pivot)
print("-" * 40)

print("\n'train_list.txt' 와 'val_list.txt' 파일이 현재 폴더에 생성되었습니다!")

import os
import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class MVTDDataset(BaseDataset):
    """ MVTD (Maritime Visual Tracking Dataset) 로더
    구조: 형태 B (이미지 및 groundtruth.txt가 시퀀스 디렉토리 최상단에 위치)
    """
    def __init__(self):
        super().__init__()
        # local.py에서 설정한 mvtd 경로를 불러옵니다.
        self.base_path = self.env_settings.mvtd_path
        # 데이터셋 폴더 내의 시퀀스 이름들을 동적으로 가져옵니다.
        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        valid_sequences = []
        for s in self.sequence_info_list:
            try:
                valid_sequences.append(self._construct_sequence(s))
            except FileNotFoundError as e:
                # 파일이 없는 시퀀스는 경고 메시지만 출력하고 건너뜀
                print(f"Warning: '{s}' 시퀀스의 일부 파일을 찾을 수 없어 제외합니다.")
                continue
            except Exception as e:
                # 그 외 형식 오류 등이 발생해도 프로그램이 멈추지 않도록 처리
                print(f"Warning: '{s}' 시퀀스 로드 중 오류 발생, 제외합니다. ({e})")
                continue
                
        return SequenceList(valid_sequences)

    def _construct_sequence(self, sequence_name):
        sequence_path = os.path.join(self.base_path, sequence_name)

        # 1. 정답(Ground Truth) 파일 경로 설정 및 로드
        anno_path = os.path.join(sequence_path, 'groundtruth.txt')
        
        # delimiter가 콤마(,)인지 탭(\t)인지에 따라 에러가 발생할 수 있습니다. 
        # MVTD 데이터셋이 기본적으로 콤마(,)를 사용한다고 가정합니다.
        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64, backend='numpy')

        # 2. 이미지 프레임 파일 목록 로드
        # 해당 폴더 내에서 이미지 확장자를 가진 파일들만 추출 후 이름순으로 정렬합니다.
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        frames = [
            os.path.join(sequence_path, f) 
            for f in os.listdir(sequence_path) 
            if f.lower().endswith(valid_exts)
        ]
        frames = sorted(frames)

        # 3. Sequence 객체로 반환
        return Sequence(sequence_name, frames, 'mvtd', ground_truth_rect)

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        # base_path 내부에 존재하는 폴더(시퀀스)들의 이름을 리스트로 반환합니다.
        # 예: ['1-Boat', '2-Boat', ...]
        if not os.path.exists(self.base_path):
            raise FileNotFoundError(f"MVTD dataset path not found: {self.base_path}")
            
        seq_list = [f for f in os.listdir(self.base_path) if os.path.isdir(os.path.join(self.base_path, f))]
        return sorted(seq_list)
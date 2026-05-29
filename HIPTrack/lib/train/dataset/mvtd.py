"""
MVTD Dataset Class for HIPTrack Fine-Tuning
=============================================
MVTD(Maritime Visual Tracking Dataset) 전용 데이터셋 클래스.

- GOT-10k 포맷을 따르므로 Got10k 클래스를 상속하여 구현
- get_frames()에서 MVTDAugmentor를 통해 온라인 증강 적용
- 원본 got10k.py는 전혀 수정하지 않음

MVTD 폴더 구조 (GOT-10k 포맷):
  <data_root>/train/
    1-Ship/
      frame0001.jpg
      frame0002.jpg
      ...
      groundtruth.txt
      absence.label
      cover.label
    2-Boat/
    ...

등록 방법:
  lib/train/dataset/__init__.py 에 아래 한 줄 추가:
    from .mvtd import MVTD

설정 방법:
  experiments/hiptrack/hiptrack_mvtd.yaml 의 DATASETS_NAME에 'MVTD' 추가
  lib/train/admin/local.py 의 mvtd_dir 경로 설정
"""

import os
import csv
import random
import numpy as np
import torch
import pandas
from collections import OrderedDict

from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings
from lib.train.data.mvtd_augmentation import MVTDAugmentor


class MVTD(BaseVideoDataset):
    """
    MVTD (Maritime Visual Tracking Dataset).

    GOT-10k 포맷을 따르며, HIPTrack Fine-Tuning을 위해
    온라인 증강(MVTDAugmentor)이 get_frames() 단계에서 적용됨.

    Parameters
    ----------
    root : str or None
        MVTD train 폴더 경로. None이면 env_settings().mvtd_dir 사용.
    image_loader : callable
        이미지 로더 함수. 기본값 jpeg4py_loader.
    split : str or None
        'train' 또는 'val'. data_specs/mvtd_train_split.txt 사용.
        None이면 전체 시퀀스 사용.
    seq_ids : list or None
        직접 시퀀스 이름 목록을 지정할 때 사용.
        split과 동시에 사용 불가.
    data_fraction : float or None
        데이터셋 중 사용할 비율 (0.0 ~ 1.0). None이면 전체 사용.
    aug_prob : float
        온라인 증강 적용 확률 (0.0 ~ 1.0). 기본값 0.5.
        0.0으로 설정하면 증강 없이 원본만 사용.
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader,
                 split=None, seq_ids=None, data_fraction=None,
                 aug_prob=0.5):

        # env_settings에 mvtd_dir이 없으면 got10k_dir 폴백
        if root is None:
            settings = env_settings()
            root = getattr(settings, 'mvtd_dir', None) or settings.got10k_dir

        super().__init__('MVTD', root, image_loader)

        # 시퀀스 목록 로드
        self.sequence_list = self._get_sequence_list()

        # split 또는 seq_ids로 시퀀스 필터링
        if split is not None:
            if seq_ids is not None:
                raise ValueError('split과 seq_ids를 동시에 설정할 수 없습니다.')
            ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
            if split == 'train':
                file_path = os.path.join(ltr_path, 'data_specs', 'mvtd_train_split.txt')
            elif split == 'val':
                file_path = os.path.join(ltr_path, 'data_specs', 'mvtd_val_split.txt')
            else:
                raise ValueError(f'알 수 없는 split: {split}')
            seq_ids = pandas.read_csv(
                file_path, header=None, dtype=str
            ).squeeze("columns").values.tolist()
            self.sequence_list = [s for s in self.sequence_list if s in seq_ids]
        elif seq_ids is not None:
            self.sequence_list = [s for s in self.sequence_list if s in seq_ids]
        # seq_ids도 split도 없으면 전체 시퀀스 사용

        # data_fraction 적용
        if data_fraction is not None:
            self.sequence_list = random.sample(
                self.sequence_list,
                int(len(self.sequence_list) * data_fraction)
            )

        # 메타 정보 및 클래스 정보 구성
        self.sequence_meta_info = self._load_meta_info()
        self.seq_per_class = self._build_seq_per_class()
        self.class_list = sorted(self.seq_per_class.keys())

        # 온라인 증강기 초기화
        # aug_prob=0.0 으로 설정하면 증강 없이 원본만 사용 (테스트 시 활용)
        self.augmentor = MVTDAugmentor(aug_prob=aug_prob)

    # ----------------------------------------------------------------
    # BaseVideoDataset 필수 메서드
    # ----------------------------------------------------------------

    def get_name(self):
        return 'mvtd'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    # ----------------------------------------------------------------
    # 시퀀스 목록 로드
    # ----------------------------------------------------------------

    def _get_sequence_list(self):
        """
        root/list.txt 에서 시퀀스 이름 목록 로드.
        list.txt가 없으면 폴더 목록에서 자동 생성.
        """
        list_path = os.path.join(self.root, 'list.txt')
        if os.path.exists(list_path):
            with open(list_path) as f:
                dir_list = [row[0] for row in csv.reader(f) if row]
        else:
            # list.txt가 없을 경우 폴더 목록으로 대체
            dir_list = sorted([
                d for d in os.listdir(self.root)
                if os.path.isdir(os.path.join(self.root, d))
            ])
        return dir_list

    # ----------------------------------------------------------------
    # 메타 정보
    # ----------------------------------------------------------------

    def _load_meta_info(self):
        return {s: self._read_meta(os.path.join(self.root, s))
                for s in self.sequence_list}

    def _read_meta(self, seq_path):
        """
        meta_info.ini 파싱. 없으면 시퀀스 이름에서 클래스 추출.
        예: '10-USV' → object_class_name = 'USV'
        """
        try:
            with open(os.path.join(seq_path, 'meta_info.ini')) as f:
                meta_info = f.readlines()
            object_meta = OrderedDict({
                'object_class_name': meta_info[5].split(': ')[-1].strip(),
                'motion_class':      meta_info[6].split(': ')[-1].strip(),
                'major_class':       meta_info[7].split(': ')[-1].strip(),
                'root_class':        meta_info[8].split(': ')[-1].strip(),
                'motion_adverb':     meta_info[9].split(': ')[-1].strip(),
            })
        except Exception:
            # meta_info.ini 없을 때: 폴더명에서 클래스 추출 (e.g. '10-USV' → 'USV')
            seq_name = os.path.basename(seq_path)
            parts = seq_name.split('-', 1)
            class_name = parts[1] if len(parts) == 2 else seq_name
            object_meta = OrderedDict({
                'object_class_name': class_name,
                'motion_class':      None,
                'major_class':       None,
                'root_class':        None,
                'motion_adverb':     None,
            })
        return object_meta

    def _build_seq_per_class(self):
        seq_per_class = {}
        for i, s in enumerate(self.sequence_list):
            cls = self.sequence_meta_info[s]['object_class_name']
            if cls in seq_per_class:
                seq_per_class[cls].append(i)
            else:
                seq_per_class[cls] = [i]
        return seq_per_class

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def get_class_name(self, seq_id):
        return self.sequence_meta_info[self.sequence_list[seq_id]]['object_class_name']

    # ----------------------------------------------------------------
    # 어노테이션 로드
    # ----------------------------------------------------------------

    def _read_bb_anno(self, seq_path):
        """groundtruth.txt → torch.tensor [N, 4] (x, y, w, h)"""
        bb_anno_file = os.path.join(seq_path, 'groundtruth.txt')
        gt = pandas.read_csv(
            bb_anno_file, delimiter=',', header=None,
            dtype=np.float32, na_filter=False, low_memory=False
        ).values
        return torch.tensor(gt)

    def _read_target_visible(self, seq_path):
        """absence.label + cover.label → visible, visible_ratio"""
        occlusion_file = os.path.join(seq_path, 'absence.label')
        cover_file = os.path.join(seq_path, 'cover.label')

        with open(occlusion_file, 'r', newline='') as f:
            occlusion = torch.ByteTensor([int(v[0]) for v in csv.reader(f)])
        with open(cover_file, 'r', newline='') as f:
            cover = torch.ByteTensor([int(v[0]) for v in csv.reader(f)])

        target_visible = ~occlusion & (cover > 0).byte()
        visible_ratio = cover.float() / 8
        return target_visible, visible_ratio

    # ----------------------------------------------------------------
    # 시퀀스 / 프레임 경로
    # ----------------------------------------------------------------

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible, visible_ratio = self._read_target_visible(seq_path)
        visible = visible & valid.byte()

        return {'bbox': bbox, 'valid': valid,
                'visible': visible, 'visible_ratio': visible_ratio}

    def _get_frame_path(self, seq_path, frame_id):
        """
        MVTD 프레임 파일명: 00000001.jpg (1-indexed, GOT-10k 형식과 동일)
        """
        return os.path.join(seq_path, '{:08d}.jpg'.format(frame_id + 1))

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    # ----------------------------------------------------------------
    # 핵심: get_frames() — 온라인 증강 적용
    # ----------------------------------------------------------------

    def get_frames(self, seq_id, frame_ids, anno=None):
        """
        프레임 로드 후 온라인 증강 적용.

        기존 got10k.py의 get_frames()와 인터페이스 동일.
        추가된 부분: self.augmentor.apply() 호출.

        Parameters
        ----------
        seq_id : int
        frame_ids : list[int]
        anno : dict or None

        Returns
        -------
        frame_list : list[PIL.Image]   증강된 이미지
        anno_frames : dict             증강된 어노테이션
        obj_meta : OrderedDict         시퀀스 메타 정보
        """
        seq_path = self._get_sequence_path(seq_id)
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]

        # 이미지 로드
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        # 어노테이션 프레임별 분리
        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        # ★ 온라인 증강 적용 (aug_prob=0.0이면 원본 그대로 반환)
        frame_list, anno_frames = self.augmentor.apply(frame_list, anno_frames)

        return frame_list, anno_frames, obj_meta
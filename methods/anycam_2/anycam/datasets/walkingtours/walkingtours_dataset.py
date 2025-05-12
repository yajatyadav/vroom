import os
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets._optical_flow import _read_flo

from anycam.common.io.io import load_flow
from anycam.datasets.common import get_flow_selector, get_target_size_and_crop, process_depth, process_flow, process_img, process_proj
from anycam.datasets.video_dataset import VideoDataset


class WalkingToursDataset(VideoDataset):
    NAME = "Walking Tours"

    @staticmethod
    def _get_sequences(data_path: str) -> Tuple[dict, dict]:
        sequences = {}
        id_to_file_id = {}

        for seq_dir in Path(data_path).iterdir():
            files = sorted(seq_dir.glob("*.jpg"))

            files = [f.stem for f in files]

            seq_len = len(files)

            seq_name = seq_dir.name

            sequences[seq_name] = seq_len
            id_to_file_id[seq_name] = files

        return sequences, id_to_file_id

    def make_img_path(self, seq, index) -> str:
        file_name = self._id_to_file_id[seq][index]
        return os.path.join(self.data_path, seq, f"{file_name}.jpg")

    def make_flow_path(self, seq, index, is_fwd) -> str:
        file_name = self._id_to_file_id[seq][index]
        return os.path.join(self.preprocessed_path, seq, f"{file_name}_{'fwd' if is_fwd else 'bwd'}.png")
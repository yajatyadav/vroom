import argparse
import os
import sys
import subprocess
from multiprocessing import Pool
from pathlib import Path
from time import sleep

import cv2
import numpy as np
from pytubefix import YouTube
from tqdm import tqdm
from subprocess import call

sys.path.extend([".", "..", "../..", "../../.."])

from anycam.datasets.aria_everyday_activities.aea_dataset import AriaEADataset


GOOD_SEQUENCES = {
3: [(400, 1040), (100, 2500)],
5: [(0, 640), (0, 2400)],
6: [(0, 640), (0, 2380)],
7: [(0, 640), (3280, 3920), (0, 3900)], 
8: [(500, 1140), (0, 3900)],
9: [(350, 990), (1400, 2040), (0, 2500)],
12: [(1100, 1740), (200, 2000)],
15: [(1240, 1880), (0, 1880)],
16: [(1240, 1880), (0, 2020)],
19: [(150, 790), (0, 4000)],
24: [(0, 640), (0, 960)],
25: [(200, 840), (0, 1380)],
28: [(450, 1090), (0, 1800)],
38: [(300, 940), (0, 1800)],
39: [(700, 1340), (1340, 1980), (0, 2100)],
48: [(1240, 1880), (0, 1880)],
49: [(1240, 1880), (0, 1880)],
50: [(0, 640), (0, 680)],
58: [(200, 840), (4100, 4740), (0, 4740)],
64: [(0, 640), (0, 1800)],
65: [(600, 1240), (0, 1800)],
73: [(1200, 1840), (4360, 5000), (0, 1800)],
79: [(0, 640), (0, 1800)],
84: [(100, 740), (0, 1500)],
89: [(200, 840), (0, 1260)],
94: [(300, 960), (740, 1360), (0, 2400)],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", type=str)
    parser.add_argument("-o", "--out_path", type=str)
    args = parser.parse_args()

    data_path = Path(args.data_path)
    out_path = Path(args.out_path)

    out_path.mkdir(exist_ok=True)

    for (seq, ranges) in tqdm(GOOD_SEQUENCES.items()):
        aea = AriaEADataset(
            data_path=data_path,
            split_path=None,
            image_size=None,
            frame_count=2,
            return_depth=False,
            return_flow=False,
            dilation=1,
            selected_sequences=[seq]
        )

        for i, (start, end) in enumerate(ranges):
            if end - start != 640:
                continue

            seq_name = list(aea._sequences.keys())[0]
            seq_folder = out_path / f"{seq_name}_{i}"

            seq_folder.mkdir(exist_ok=True)

            frame_folder = seq_folder / "frames"
            frame_folder.mkdir(exist_ok=True)

            poses = []
            projs = []

            for j in range(start, end):
                data = aea[j]
                
                if j == end - 1:
                    ids = [0, 1]
                else:
                    ids = [0]
                
                for k in ids:
                    img = data["imgs"][k]
                    img = np.transpose(img, (1, 2, 0))
                    img = (img * 255).astype(np.uint8)
                    cv2.imwrite(str(frame_folder / f"{j-start+k:05d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    
                    poses.append(data["poses"][k])
                    projs.append(data["projs"][k])

            poses = np.array(poses)
            projs = np.array(projs)

            np.save(seq_folder / "poses.npy", poses)
            np.save(seq_folder / "projs.npy", projs)
    
    print("Done")
            

if __name__ == "__main__":
    main()



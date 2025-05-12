import argparse
from pathlib import Path
from tqdm import tqdm
import sys
import os

sys.path.append(".")

from anycam.datasets.tum_rgbd.tumrgbd_dataset import TUMRGBDDataset

os.chdir(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

dynamic_scenes = [
        "sitting_halfsphere",
        'sitting_rpy',
        'sitting_static',
        'sitting_xyz',
        'walking_halfsphere',
        'walking_rpy',
        'walking_static',
        'walking_xyz'
    ]

dynamic_scenes = [f"rgbd_dataset_freiburg3_{scene}" for scene in dynamic_scenes]


def main(args):
    data_path = Path(args.data_path)
    out_path = args.out_path
    every_nth = args.every_nth

    tumrgbd = TUMRGBDDataset(
        data_path=data_path,
        split_path=None,
        image_size=384,
        frame_count=2,
        return_depth=False,
        return_flow=False,
        dilation=1,
    )

    datapoints = tumrgbd._datapoints
    
    filtered_datapoints = []

    for scene in dynamic_scenes:
        scene_dps = [dp for dp in datapoints if scene == dp[0]]
        if every_nth > 1:
            scene_dps = scene_dps[::every_nth]

        filtered_datapoints.extend(scene_dps)

    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "test_files.txt", "w") as f:
        for seq, idx in filtered_datapoints:
            f.write(f"{seq} {idx}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your script")
    parser.add_argument("--data_path", type=str, help="Path to the data")
    parser.add_argument("--out_path", type=str, help="Path to the output")
    parser.add_argument("--every_nth", type=int, default=1, help="Only use every nth frame")
    args = parser.parse_args()
    main(args)

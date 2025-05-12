import argparse
from pathlib import Path
from tqdm import tqdm
import sys
import os

sys.path.append(".")

from anycam.datasets.sintel.sintel_dataset import SintelDataset

os.chdir(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

scenes = [
        "alley_2",
        'ambush_4',
        'ambush_5',
        'ambush_6',
        'cave_2',
        'cave_4',
        'market_2',
        'market_5',
        'market_6',
        'shaman_3',
        'sleeping_1',
        'sleeping_2',
        'temple_2',
        'temple_3',
    ]


def main(args):
    data_path = Path(args.data_path)
    out_path = args.out_path

    sintel = SintelDataset(
        data_path=data_path,
        split_path=None,
        image_size=384,
        frame_count=2,
        return_depth=False,
        return_flow=False,
        dilation=1,
    )

    datapoints = sintel._datapoints

    filtered_datapoints = []

    for scene in scenes:
        scene_dps = [dp for dp in datapoints if scene == dp[0]]

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
    args = parser.parse_args()
    main(args)

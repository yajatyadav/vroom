import argparse
from pathlib import Path
from tqdm import tqdm
import sys
import os

sys.path.append(".")

from anycam.datasets.realestate10k.re10k_dataset import RealEstate10kDataset

os.chdir(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


SEQ_DATA = {}



def main(args):
    data_path = args.data_path
    out_path = args.out_path
    cap_len = args.cap_len
    stride = args.stride

    for seq in os.listdir(data_path):
        seq_path = Path(data_path) / seq / "frames"
        seq_len = len(list(seq_path.glob("*.png")))
        SEQ_DATA[seq] = seq_len
 
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    if cap_len > 0:
        for seq, seq_len in SEQ_DATA.items():
            SEQ_DATA[seq] = min(cap_len, seq_len)

    with open(out_path / "test_files.txt", "w") as f:
        for seq, seq_len in SEQ_DATA.items():
            for i in range(0, seq_len-1, stride):
                f.write(f"{seq} {i:010d}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your script")
    parser.add_argument("-d", "--data_path", type=str, help="Path to the data")
    parser.add_argument("--out_path", type=str, help="Path to the output")
    parser.add_argument("--cap_len", type=int, default=-1, help="Sequence length")
    parser.add_argument("--stride", type=int, default=1, help="Stride")
    args = parser.parse_args()
    main(args)

import argparse
from pathlib import Path
from tqdm import tqdm
import sys
import os

sys.path.append(".")

from anycam.datasets.realestate10k.re10k_dataset import RealEstate10kDataset

os.chdir(os.path.join(os.path.dirname(__file__), "..", "..", ".."))



def main(args):
    data_path = Path(args.data_path)
    out_path = args.out_path
    min_seq_len = args.min_seq_len

    re10k = RealEstate10kDataset(
        data_path=data_path,
        split_path="anycam/datasets/realestate10k/splits/720/test_files.txt",
        image_size=384,
        frame_count=2,
        return_depth=False,
        return_flow=False,
        dilation=1,
    )

    seq_data = {}

    last_seq = None
    last_new = 0

    for i in tqdm(range(len(re10k))):
        seq = re10k.get_sequence(i)

        if last_seq is None:
            last_seq = seq
            last_new = 0
        if last_seq != seq:
            seq_data[last_seq] = (last_new, i)
            last_seq = seq
            last_new = i

    seq_data[last_seq] = (last_new, len(re10k))

    seq_lens = {k: v[1] - v[0] for k, v in seq_data.items()}

    seq_minlen = [k for k, v in seq_lens.items() if v >= min_seq_len]

    print(f"Number of sequences: {len(seq_minlen)}")

    datapoints = []
    for k in seq_minlen:
        start, end = seq_data[k]

        datapoints += [re10k._datapoints[i] for i in range(start, end)]

    datapoints = [(seq, re10k._seq_data[seq]["timestamps"][i], re10k._seq_data[seq]["timestamps"][j]) for seq, (i, j) in datapoints]

    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "test_files.txt", "w") as f:
        for seq, t0, t1 in datapoints:
            f.write(f"{seq} {t0} {t1}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your script")
    parser.add_argument("--data_path", type=str, help="Path to the data")
    parser.add_argument("--out_path", type=str, help="Path to the output")
    parser.add_argument("--min_seq_len", type=int, default=24, help="Minimum sequence length")
    args = parser.parse_args()
    main(args)

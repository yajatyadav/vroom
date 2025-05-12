import argparse
from pathlib import Path
from tqdm import tqdm
import sys
import os

sys.path.append(".")

from anycam.datasets.realestate10k.re10k_dataset import RealEstate10kDataset

os.chdir(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


INTERESTING_SEQS = [
('10410418118434245359_5140_000_5160_000', 596, 793),
('10534368980139017457_4480_000_4500_000', 1158, 1357),
('10649066155322078676_1660_000_1680_000', 1357, 1553),
('10940141908690367388_4420_000_4440_000', 1750, 1949),
('10998289306141768318_1280_000_1300_000', 2147, 2345),
('11867874114645674271_600_000_620_000', 2940, 3127),
('11987368976578218644_1340_000_1360_000', 3325, 3523),
('13034900465317073842_1700_000_1720_000', 4516, 4714),
('13732041959462600641_720_000_740_000', 5111, 5310),
]


def main(args):
    out_path = args.out_path
 
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(out_path / "test_files.txt", "w") as f:
        for seq, t0, t1 in INTERESTING_SEQS:
            seq_len = t1 - t0
            for i in range(seq_len-1):
                f.write(f"{seq} {i:010d}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your script")
    parser.add_argument("--data_path", type=str, help="Path to the data")
    parser.add_argument("--out_path", type=str, help="Path to the output")
    parser.add_argument("--min_seq_len", type=int, default=24, help="Minimum sequence length")
    args = parser.parse_args()
    main(args)

import argparse
from pathlib import Path
from tqdm import tqdm
import sys
import os

sys.path.append(".")

from anycam.datasets.realestate10k.re10k_dataset import RealEstate10kDataset

os.chdir(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


SEQ_DATA = {
    '10940141908690367388_4420_000_4440_000': 199,
    '10998289306141768318_1280_000_1300_000': 198,
    '11436803605426256250_1720_000_1740_000': 199,
    '11867874114645674271_600_000_620_000': 187,
    '11987368976578218644_1340_000_1360_000': 198,
    '12056192874455954437_140_000_160_000': 199,
    '13585389505831587326_2560_000_2580_000': 199,
    '13748565785898537200_680_000_700_000': 197,
    '13887882285811432765_740_000_760_000': 198,
    '14188689528137485670_2660_000_2680_000': 198,
    '14386836877680112549_4460_000_4480_000': 198,
    '14470988792985854683_760_000_780_000': 199,
    '15370024704033662533_1240_000_1260_000': 197,
    '16942495693882305487_4340_000_4360_000': 198,
    '17212025549630306883_2500_000_2520_000': 198,
    '1765211916310163252_4400_000_4420_000': 198
}


def main(args):
    out_path = args.out_path
 
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(out_path / "test_files.txt", "w") as f:
        for seq, seq_len in SEQ_DATA.items():
            for i in range(seq_len-1):
                f.write(f"{seq} {i:010d}\n")

    out_path_64 = out_path.parent / (out_path.name + "_64")
    out_path_64.mkdir(parents=True, exist_ok=True)

    with open(out_path_64 / "test_files.txt", "w") as f:
        for part in range(3):
            for seq, seq_len in SEQ_DATA.items():
                for i in range(part * 64, min((part + 1) * 64, seq_len)):
                    f.write(f"{seq} {i:010d}\n")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your script")
    parser.add_argument("--data_path", type=str, help="Path to the data")
    parser.add_argument("--out_path", type=str, help="Path to the output")
    parser.add_argument("--min_seq_len", type=int, default=24, help="Minimum sequence length")
    args = parser.parse_args()
    main(args)

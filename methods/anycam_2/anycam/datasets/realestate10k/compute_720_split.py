import argparse
from pathlib import Path
from tqdm import tqdm

def main(args):
    data_path = Path(args.data_path)
    out_path = args.out_path

    # List all sequences in the data_path directory
    sequences = [seq.stem for seq in data_path.glob("*") if seq.is_dir()]

    print(f"Number of sequences: {len(sequences)}")

    # Count the number of files in each sequence folder
    file_counts = {}
    sequence_files = {}
    for sequence in tqdm(sequences):
        sequence_path = data_path / sequence
        files = list(sequence_path.glob("*"))
        file_count = len(files)
        file_counts[sequence] = file_count
        sequence_files[sequence] = sorted(files)

    # Sum up the count of all files
    total_file_count = sum(file_counts.values())
    print(f"Total number of files: {total_file_count}")

    # Split the sequences into train and test
    train_ratio = args.split_ratio
    train_count = int(len(sequences) * train_ratio)
    test_count = len(sequences) - train_count

    train_sequences = sequences[:train_count]
    test_sequences = sequences[train_count:]

    print(f"Number of train sequences: {len(train_sequences)}")
    print(f"Number of test sequences: {len(test_sequences)}")

    with open(Path(out_path) / "train_files.txt", "w") as f:
        for sequence in train_sequences:
            files = sequence_files[sequence]
            for i, file in enumerate(files[:-1]):
                f.write(f"{sequence} {file.stem} {files[i+1].stem}\n")

    with open(Path(out_path) / "test_files.txt", "w") as f:
        for sequence in test_sequences:
            files = sequence_files[sequence]
            for i, file in enumerate(files[:-1]):
                f.write(f"{sequence} {file.stem} {files[i+1].stem}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your script")
    parser.add_argument("--data_path", type=str, help="Path to the data")
    parser.add_argument("--out_path", type=str, help="Path to the output")
    parser.add_argument("--split_ratio", type=float, default=0.8, help="Split ratio for train-test split")
    args = parser.parse_args()
    main(args)

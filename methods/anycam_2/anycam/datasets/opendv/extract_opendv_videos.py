import logging
from pathlib import Path
import subprocess
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

logging.basicConfig(level = logging.INFO)

logger = logging.getLogger(__name__)


def get_length(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)


def seconds_to_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def index_videos(in_path):
    videos = {}

    if in_path.is_dir():
        for sub_path in in_path.iterdir():
            videos.update(index_videos(sub_path))
    else:
        try:
            videos[in_path] = get_length(str(in_path))
        except Exception as e:
            logger.warning(f"Error indexing {in_path}: {e} - Ignoring")

    return videos


def get_sub_sequences(videos, crop_start, crop_end, seq_len, stride):
    sequences = []

    for video_path, video_length in videos.items():
        for seq_start in range(crop_start, int(video_length - crop_end - seq_len), stride):
            seq_end = seq_start + seq_len
            sequences.append((video_path, seq_start, seq_end))

    return sequences


def extract_frames(video_path, start, end, fps, out_width, out_height, out_path):
    seq_name = f"{video_path.parent.name}_{video_path.stem}_{start}_{end}"
    seq_name = seq_name.strip().replace(" ", "_")

    seq_path = out_path / seq_name
    seq_path.mkdir(exist_ok=True, parents=True)

    logger.info(f"Extracting frames from {video_path} to {seq_path}")

    command = f"ffmpeg -i video_path -ss {seconds_to_timestamp(start)} -to {seconds_to_timestamp(end)} -vf fps={fps},scale={out_width}:{out_height} {seq_path / '%06d.jpg'}"

    command = command.split(" ")
    command[2] = str(video_path)

    logger.info(f"Running: {' '.join(command)}")

    subprocess.run(command)


def main(args):
    in_path = Path(args.in_path)
    crop_start = int(args.crop_start)
    crop_end = int(args.crop_end)
    seq_len = int(args.seq_len)
    stride = int(args.stride)
    fps = int(args.fps)
    out_width = int(args.out_width)
    out_height = int(args.out_height)
    out_path = Path(args.out_path)
    num_workers = int(args.num_workers)

    videos = index_videos(in_path)

    total_videos_length = sum(videos.values())

    logger.info(f"Found {len(videos)} videos. Total length: {seconds_to_timestamp(total_videos_length)}")

    sequences = get_sub_sequences(videos, crop_start, crop_end, seq_len, stride)

    total_length = sum([end - start for _, start, end in sequences])
    projected_frame_count = total_length * fps

    logger.info(f"Extracting {len(sequences)} sequences. Total length: {seconds_to_timestamp(total_length)}")
    logger.info(f"Projected frame count: {projected_frame_count}")

    logger.info(f"Extracting frames to {out_path}")

    def process_sequence(sequence):
        video_path, seq_start, seq_end = sequence
        extract_frames(video_path, seq_start, seq_end, fps, out_width, out_height, out_path)
        return True
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        tqdm(executor.map(process_sequence, sequences), total=len(sequences))

    logger.info("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract video information using ffprobe.")
    parser.add_argument("--in_path", type=str, help="Path to the videos")
    parser.add_argument("--out_path", type=str, help="Path to the output folder")

    parser.add_argument("--crop_start", type=str, default=120, help="Crop start")
    parser.add_argument("--crop_end", type=str, default=120, help="Crop end")
    parser.add_argument("--seq_len", type=str, default=20, help="Sequence length in seconds")
    parser.add_argument("--stride", type=str, default=180, help="Stride")
    parser.add_argument("--fps", type=str, default=10, help="Stride")
    parser.add_argument("--out_width", type=str, default=1280, help="Width")
    parser.add_argument("--out_height", type=str, default=720, help="Height")
    
    parser.add_argument("--num_workers", type=str, default=4, help="Number of workers")

    args = parser.parse_args()
    main(args)
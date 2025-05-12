import argparse
import os
import subprocess
from multiprocessing import Pool
from pathlib import Path
from time import sleep

from pytubefix import YouTube
import tqdm
from subprocess import call


class Data:
    def __init__(self, url, seqname, list_timestamps):
        self.url = url
        self.list_seqnames = []
        self.list_list_timestamps = []

        self.list_seqnames.append(seqname)
        self.list_list_timestamps.append(list_timestamps)

    def add(self, seqname, list_timestamps):
        self.list_seqnames.append(seqname)
        self.list_list_timestamps.append(list_timestamps)

    def __len__(self):
        return len(self.list_seqnames)


def process(data, seq_id, videoname, output_root, nth_frame=1):
    seqname = data.list_seqnames[seq_id]
    out_path = output_root / seqname
    if not out_path.exists():
        out_path.mkdir(exist_ok=True, parents=True)
    else:
        print("[INFO] Something Wrong, stop process")
        return True

    list_timestamp_str_timestamps = []
    for timestamp in data.list_list_timestamps[seq_id]:
        original_timestamp = timestamp
        timestamp = int(timestamp / 1000)
        str_hour = str(int(timestamp / 3600000)).zfill(2)
        str_min = str(int(int(timestamp % 3600000) / 60000)).zfill(2)
        str_sec = str(int(int(int(timestamp % 3600000) % 60000) / 1000)).zfill(2)
        str_mill = str(int(int(int(timestamp % 3600000) % 60000) % 1000)).zfill(3)
        _str_timestamp = str_hour + ":" + str_min + ":" + str_sec + "." + str_mill
        list_timestamp_str_timestamps.append((original_timestamp, _str_timestamp))

    if len(list_timestamp_str_timestamps) > nth_frame:
        list_timestamp_str_timestamps = list_timestamp_str_timestamps[::nth_frame]
    else:
        list_timestamp_str_timestamps = [list_timestamp_str_timestamps[0], list_timestamp_str_timestamps[-1]]

    # extract frames from a video
    for timestamp, str_timestamp in list_timestamp_str_timestamps:
        call(("ffmpeg", "-ss", str_timestamp, "-i", str(videoname), "-vframes", "1", "-f", "image2", str(out_path / f'{timestamp}.jpg')), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return False


def wrap_process(list_args):
    return process(*list_args)


class DataDownloader:
    def __init__(self, data_path: Path, out_path: Path, tmp_path: Path, mode='test', limit=None, resolution="360p", nth_frame=1):
        print("[INFO] Loading data list ... ", end='')
        self.data_path = data_path
        self.out_path = out_path
        self.tmp_path = tmp_path
        self.mode = mode
        self.limit = limit
        self.resolution = resolution
        self.nth_frame = nth_frame

        self.list_seqnames = sorted(self.data_path.glob('*.txt'))

        self.is_done = out_path.exists()

        if self.is_done:
            print("[INFO] The output dir has already existed.")

        self.is_done = False

        out_path.mkdir(exist_ok=True, parents=True)

        self.list_data = {}
        if not self.is_done:
            for txt_file in tqdm.tqdm(self.list_seqnames):
                dir_name = txt_file.parent.name
                seq_name = txt_file.stem

                # extract info from txt
                with open(txt_file, "r") as seq_file:
                    lines = seq_file.readlines()
                    youtube_url = ""
                    list_timestamps = []
                    for idx, line in enumerate(lines):
                        if idx == 0:
                            youtube_url = line.strip()
                        else:
                            timestamp = int(line.split(' ')[0])
                            list_timestamps.append(timestamp)

                if youtube_url in self.list_data:
                    self.list_data[youtube_url].add(seq_name, list_timestamps)
                else:
                    self.list_data[youtube_url] = Data(youtube_url, seq_name, list_timestamps)

            print(" Done! ")
            print("[INFO] {} movies are used in {} mode".format(len(self.list_data), self.mode))

        if limit is not None:
            self.list_data = dict(list(self.list_data.items())[:limit])

        self.list_data = dict(list(self.list_data.items())[-100:])

    def run(self):
        print("[INFO] Start downloading {} movies".format(len(self.list_data)))

        for global_count, data in enumerate(self.list_data.values()):
            print(f"[INFO] Downloading {data.url} ({global_count + 1}/{len(self.list_data)})")
            current_file = self.tmp_path / f"current_{self.mode}"

            call(("rm", "-r", str(current_file)))

            current_file.mkdir(exist_ok=True)

            try:
                # sometimes this fails because of known issues of pytube and unknown factors
                yt = YouTube(data.url)
                stream = yt.streams.filter(res=self.resolution).first()
                stream.download(str(current_file))
            except:
                with open(os.path.join(str(self.data_path.parent), 'failed_videos_' + self.mode + '.txt'), 'a') as f:
                    for seqname in data.list_seqnames:
                        f.writelines(seqname + '\n')
                continue

            sleep(1)

            current_file = next(current_file.iterdir())

            if len(data) == 1:  # len(data) is len(data.list_seqnames)
                process(data, 0, current_file, self.out_path)
            else:
                with Pool(processes=4) as pool:
                    pool.map(wrap_process, [(data, seq_id, current_file, self.out_path, self.nth_frame) for seq_id in range(len(data))])

            print(f"[INFO] Extracted {sum(map(len, data.list_list_timestamps))}")

            # remove videos
            call(("rm", str(current_file)))
            # os.system(command)

            if self.is_done:
                return False

        return True

    def show(self):
        print("########################################")
        global_count = 0
        for data in self.list_data.values():
            # print(" URL : {}".format(data.url))
            for idx in range(len(data)):
                # print(" SEQ_{} : {}".format(idx, data.list_seqnames[idx]))
                # print(" LEN_{} : {}".format(idx, len(data.list_list_timestamps[idx])))
                global_count = global_count + 1
            # print("----------------------------------------")

        print("TOTAL : {} sequnces".format(global_count))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str)
    parser.add_argument("-d", "--data_path", type=str)
    parser.add_argument("-o", "--out_path", type=str)
    parser.add_argument("-t", "--tmp_path", default="/dev/shm", type=str)
    parser.add_argument("-l", "--limit", default=None, type=int)
    parser.add_argument("-r", "--resolution", default="360p")
    parser.add_argument("-n", "--nth_frame", default=1, type=int)

    args = parser.parse_args()
    mode = args.mode
    data_path = Path(args.data_path)
    out_path = Path(args.out_path)
    tmp_path = Path(args.tmp_path)
    limit = args.limit
    resolution = args.resolution
    nth_frame = args.nth_frame

    if mode not in ["test", "train"]:
        raise ValueError(f"Invalid split mode: {mode}")

    data_path = data_path / mode
    out_path = out_path / mode
    downloader = DataDownloader(
        data_path=data_path,
        out_path=out_path,
        tmp_path=tmp_path,
        mode=mode,
        limit=limit,
        resolution=resolution,
        nth_frame=nth_frame,
    )

    downloader.show()
    is_ok = downloader.run()

    if is_ok:
        print("Done!")
    else:
        print("Failed")


if __name__ == "__main__":
    main()



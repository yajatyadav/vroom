#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 2 ]; then
  echo "Usage: $0 <CUDA_VISIBLE_DEVICES> <VIDEO_DIR>"
  exit 1
fi

DEVICE="$1"
DIR="$2"

for filepath in "$DIR"/*.mp4; do
  filename=$(basename "$filepath")
  seqname="${filename^^}"  # filename in ALL CAPS
  echo "=== Processing $filename with SEQ_NAME=$seqname on GPU $DEVICE ==="
  CUDA_VISIBLE_DEVICES="$DEVICE" python demo.py \
    --input "$filepath" \
    --num_frames 1000 \
    --window_size 16 \
    --seq_name "$seqname"
done
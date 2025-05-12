#!/bin/bash

# Function to display help message
show_help() {
  echo "Usage: $0 [-h] <model_name>"
  echo ""
  echo "Options:"
  echo "  -h, --help    Show this help message and exit"
  echo ""
  echo "Arguments:"
  echo "  <model_name>  Name of the model to download (e.g., anycam_seq2 or anycam_seq8)"
}

# Check if the help option is provided
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  show_help
  exit 0
fi

# Check if the model name is provided as an argument
if [ -z "$1" ]; then
  echo "Error: Model name is required."
  show_help
  exit 1
fi

# Define the base URL
BASE_URL="https://cvg.cit.tum.de/webshare/g/behindthescenes/anycam/checkpoints/"

# Get the model name from the command-line argument
model_name=$1

# Append .tar.gz to the model name
MODEL="${model_name}.tar.gz"

# Create the output directory if it doesn't exist
OUTPUT_DIR="pretrained_models"
mkdir -p $OUTPUT_DIR

# Download the selected model
wget -P $OUTPUT_DIR $BASE_URL$MODEL

# Unpack the tar.gz file
tar -xzf $OUTPUT_DIR/$MODEL -C $OUTPUT_DIR

# Remove the tar.gz file after unpacking
rm $OUTPUT_DIR/$MODEL

echo "Downloaded and unpacked $MODEL to $OUTPUT_DIR"
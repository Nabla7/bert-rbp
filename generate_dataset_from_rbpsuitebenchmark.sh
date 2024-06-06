#!/bin/bash

# Set the output directory
OUTPUT_PATH="./datasets"

# Create the output directory if it doesn't exist
mkdir -p $OUTPUT_PATH

# Define the benchmark directory
BENCHMARK_DIR="benchmark/train_dir"

# Run the Python script
python3 generate_datasets.py --path_to_benchmark $BENCHMARK_DIR --path_to_output $OUTPUT_PATH

# Print a message indicating the completion of the current file processing
echo "Processed all files in $BENCHMARK_DIR/train_dir"


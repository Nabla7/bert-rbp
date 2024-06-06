#!/bin/bash

# Base paths
ORIG_PATH="../datasets/"
PYTHON_PATH="../examples/run_finetune.py"
KMER=3

# Taking RBP and MODEL_PATH as arguments
RBP=$1
MODEL_PATH=$2

# Construct paths
DATA_PATH="${ORIG_PATH}${RBP}/training_sample_finetune/"
OUTPUT_PATH="${ORIG_PATH}${RBP}/finetuned_model/"

# Print paths for debugging
echo "Training model for RBP: $RBP"
echo "Using model path: $MODEL_PATH"
echo "Data path: $DATA_PATH"
echo "Output path: $OUTPUT_PATH"

# Ensure the MODEL_PATH is correctly set
if [ ! -f "${MODEL_PATH}/config.json" ]; then
  echo "Error: Model config.json not found in ${MODEL_PATH}"
  exit 1
fi

# Check if train.tsv and dev.tsv exist
if [ ! -f "${DATA_PATH}/train.tsv" ]; then
  echo "Error: train.tsv not found in ${DATA_PATH}"
  exit 1
fi

if [ ! -f "${DATA_PATH}/dev.tsv" ]; then
  echo "Error: dev.tsv not found in ${DATA_PATH}"
  exit 1
fi

# Check the size of the train.tsv file
TRAIN_SIZE=$(wc -l < "${DATA_PATH}/train.tsv")
if [ "$TRAIN_SIZE" -le 1 ]; then
  echo "Error: train.tsv is empty or contains only headers"
  exit 1
fi

# Train the model
python3 $PYTHON_PATH --model_type dna --tokenizer_name dna$KMER --model_name_or_path "$MODEL_PATH" --task_name dnaprom --data_dir "$DATA_PATH" --output_dir "$OUTPUT_PATH" --do_train --max_seq_length 101 --per_gpu_eval_batch_size 64 --per_gpu_train_batch_size 64 --gradient_accumulation_steps 4 --learning_rate 2e-4 --num_train_epochs 3 --logging_steps 200 --warmup_percent 0.1 --hidden_dropout_prob 0.1 --overwrite_output_dir --weight_decay 0.01 --n_process 8

# Update paths for evaluation and prediction
DATA_PATH="${ORIG_PATH}${RBP}/test_sample_finetune/"
PREDICT_PATH="$OUTPUT_PATH"

# Print paths for debugging
echo "Evaluating and predicting for RBP: $RBP"
echo "Data path: $DATA_PATH"
echo "Model path: $MODEL_PATH"
echo "Predict path: $PREDICT_PATH"

# Ensure dev.tsv exists in test_sample_finetune
if [ ! -f "${DATA_PATH}/dev.tsv" ]; then
  echo "Error: dev.tsv not found in ${DATA_PATH}"
  exit 1
fi

# Evaluate and predict
python3 $PYTHON_PATH --model_type dna --tokenizer_name dna$KMER --model_name_or_path "$MODEL_PATH" --task_name dnaprom --do_eval --do_predict --data_dir "$DATA_PATH" --output_dir "$MODEL_PATH" --predict_dir "$PREDICT_PATH" --max_seq_length 101 --per_gpu_eval_batch_size 16 --per_gpu_train_batch_size 16 --overwrite_output

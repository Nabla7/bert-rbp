#!/bin/bash

PYTHON_PATH=../examples/run_analysis_regiontype.py
ORIG_PATH=../datasets/

RBP=$1
echo "RBP: "$RBP

MODEL_PATH=$ORIG_PATH$RBP/finetuned_model
echo "Model path: "$MODEL_PATH

DATA_PATH=$ORIG_PATH$RBP/nontraining_sample_finetune
echo "Data path: "$DATA_PATH

PREDICT_PATH=$MODEL_PATH/analyze_regiontype/
echo "Predict path: "$PREDICT_PATH

# Verify the paths
if [ ! -d "$MODEL_PATH" ]; then
  echo "Error: Model path $MODEL_PATH does not exist."
  exit 1
fi

if [ ! -f "${MODEL_PATH}/config.json" ]; then
  echo "Error: config.json not found in $MODEL_PATH"
  exit 1
fi

if [ ! -d "$DATA_PATH" ]; then
  echo "Error: Data path $DATA_PATH does not exist."
  exit 1
fi

# Execute the analysis script
python3 $PYTHON_PATH --model_type dna --model_name_or_path $MODEL_PATH --task_name dnaprom --data_dir $DATA_PATH --predict_dir $PREDICT_PATH --do_analyze_regiontype --max_seq_length 101 --per_gpu_pred_batch_size 64 --n_process 8

echo "done"

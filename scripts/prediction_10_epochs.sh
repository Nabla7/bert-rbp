#!/bin/bash

# Arguments
PROTEINS=("HNRNPA1" "FUS" "MATR3" "ATXN2" "TAF15" "TIA1" "EWSR1" "TARDBP")
MODEL_PATH=$1

DATA_PATH=./
PYTHON_PATH=../examples/run_finetune.py
KMER=3

for RBP in "${PROTEINS[@]}"; do
    # Run prediction on dev.tsv
    python3 $PYTHON_PATH --model_type dna --tokenizer_name dna$KMER --model_name_or_path $MODEL_PATH/$RBP --task_name dnaprom --do_predict --data_dir $DATA_PATH --output_dir $MODEL_PATH/$RBP --predict_dir $MODEL_PATH/$RBP --max_seq_length 101 --per_gpu_eval_batch_size 32 --per_gpu_train_batch_size 32 --overwrite_output

    # Assuming the prediction results are saved in a file named "pred_results.npy"
    PRED_RESULTS=$MODEL_PATH/$RBP/pred_results.npy

    # Run the Python script to append the predictions to the TSV file
    python3 add_predictions_to_tsv.py binned_exon_sequences.tsv $PRED_RESULTS $RBP
done

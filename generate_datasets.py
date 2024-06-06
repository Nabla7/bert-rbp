import sys
import os
import argparse
import numpy as np
import pandas as pd
import re
from motif.motif_utils import seq2kmer

OUTPUT_FILE = 'original.tsv'

def createkmers(args):
    print("Starting createkmers function...")
    train_dir = args.path_to_benchmark
    
    
    # List all files in the train_dir
    files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, f))]
    print(f"Found {len(files)} files in {train_dir}")

    for file in files:
        filename = os.path.basename(file)
        pattern = r'([A-Z0-9]+).(negative|positive).fa'
        match = re.match(pattern, filename)
        
        if match:
            rbp_name = match.group(1)
            label = 1 if match.group(2) == 'positive' else 0
            output_dir = os.path.join(args.path_to_output, rbp_name)
            output_file = os.path.join(output_dir, OUTPUT_FILE)

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Created directory: {output_dir}")
            
            with open(file) as f:
                lines = f.readlines()
            
            mode = 'a' if os.path.exists(output_file) else 'w'
            with open(output_file, mode) as f:
                if mode == 'w':
                    f.write('sequence\tlabel\n')
                for line in lines:
                    if line.startswith('>'):
                        continue
                    sequence = re.sub('U', 'T', line.strip())
                    kmer_sequence = seq2kmer(sequence, args.kmer)
                    f.write(f"{kmer_sequence}\t{label}\n")
            print(f"Processed file: {file}")
        else:
            print(f"Skipping unrecognized file: {file}")

def preprocess(args):
    print("Starting preprocess function...")
    orig_dir = os.path.join(args.path_to_output)
    dirlist = os.listdir(orig_dir)
    dirlist.sort()

    len_sequence_list = []
    for rbp in dirlist:
        if rbp == '.ipynb_checkpoints':  # Skip .ipynb_checkpoints directory
            print(f"Skipping directory: {rbp}")
            continue
        print(f"Processing RBP: {rbp}")
        one_sequence_list = []
        rbp_dir = os.path.join(orig_dir, rbp, OUTPUT_FILE)
        df_rbp = pd.read_csv(rbp_dir, sep='\t')
        df_rbp = df_rbp.dropna(axis=0)
        query = 'sequence.str.match("([ATGC]{' + str(args.kmer) + '}\s)+")'
        df_rbp = df_rbp.query(query)
        df_rbp = df_rbp.drop_duplicates(subset='sequence')
        pos = df_rbp[df_rbp['label'] == 1]
        if len(pos) > args.max_num:
            pos = pos.sample(n=args.max_num, random_state=args.random_seed)
        neg = df_rbp[df_rbp['label'] == 0]
        if len(neg) > args.max_num:
            neg = neg.sample(n=args.max_num, random_state=args.random_seed)
        
        test_pos = pos.sample(frac=args.test_ratio, random_state=args.random_seed)
        test_neg = neg.sample(frac=args.test_ratio, random_state=args.random_seed)
        tr_pos = pos[~pos.sequence.isin(test_pos.sequence)].dropna()
        tr_neg = neg[~neg.sequence.isin(test_neg.sequence)].dropna()
        
        df_test = pd.merge(test_pos, test_neg, how='outer').sample(frac=1, random_state=args.random_seed)
        eval_pos = tr_pos.sample(frac=args.test_ratio, random_state=args.random_seed)
        eval_neg = tr_neg.sample(frac=args.test_ratio, random_state=args.random_seed)
        train_pos = tr_pos[~tr_pos.sequence.isin(eval_pos.sequence)].dropna()
        train_neg = tr_neg[~tr_neg.sequence.isin(eval_neg.sequence)].dropna()
        df_eval = pd.merge(eval_pos, eval_neg, how='outer').sample(frac=1, random_state=args.random_seed)
        df_train = pd.merge(train_pos, train_neg, how='outer').sample(frac=1, random_state=args.random_seed)
        one_sequence_list.append([rbp, len(train_pos), len(train_neg), len(eval_pos), len(eval_neg), len(test_pos), len(test_neg), len(pos), len(neg)])
        len_sequence_list.extend(one_sequence_list)
        df_nontrain = df_rbp[~df_rbp.sequence.isin(df_train.sequence)].dropna()

        test_dir = os.path.join(orig_dir, rbp, 'test_sample_finetune')
        train_dir = os.path.join(orig_dir, rbp, 'training_sample_finetune')
        nontrain_dir = os.path.join(orig_dir, rbp, "nontraining_sample_finetune")
        if not os.path.isdir(test_dir):
            os.makedirs(test_dir)
            print(f"Created directory: {test_dir}")
        if not os.path.isdir(train_dir):
            os.makedirs(train_dir)
            print(f"Created directory: {train_dir}")
        if not os.path.isdir(nontrain_dir):
            os.makedirs(nontrain_dir)
            print(f"Created directory: {nontrain_dir}")
        
        test_df_path = os.path.join(test_dir, "dev.tsv")
        df_test.to_csv(test_df_path, sep='\t', index=False)
        eval_df_path = os.path.join(train_dir, "dev.tsv")
        df_eval.to_csv(eval_df_path, sep='\t', index=False)
        train_df_path = os.path.join(train_dir, "train.tsv")
        df_train.to_csv(train_df_path, sep='\t', index=False)
        nontrain_df_path = os.path.join(nontrain_dir, "dev.tsv")
        df_nontrain.to_csv(nontrain_df_path, sep='\t', index=False)
        print(f"Finished processing RBP: {rbp}")

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--path_to_benchmark",
        default=None,
        type=str,
        required=True,
        help="path to the benchmark directory",
    )
    parser.add_argument(
        "--path_to_output",
        default=None,
        type=str,
        required=True,
        help="path to the output directory",
    )
    parser.add_argument(
        "--kmer",
        default=3,
        type=int,
        required=False,
        help="kmer of the output file",
    )
    parser.add_argument(
        "--max_num",
        default=15000,
        type=int,
        required=False,
        help="maximum number of samples to retrieve",
    )
    parser.add_argument(
        "--test_ratio",
        default=0.2,
        type=float,
        required=False,
        help="ratio of test data",
    )
    parser.add_argument(
        "--random_seed",
        default=0,
        type=int,
        required=False,
        help="seed number for random sampling",
    )
    
    args = parser.parse_args()
    
    print("Starting main process...")
    createkmers(args)
    preprocess(args)
    
    print("Finished all processes.")
    return
        
if __name__ == "__main__":
    main()

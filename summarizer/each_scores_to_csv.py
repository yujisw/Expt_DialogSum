#!/usr/bin/env python

import os
import argparse
import datetime
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List
import pandas as pd

import torch
from tqdm import tqdm

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from utils import cal_exact_rouge, calculate_rouge, chunks, parse_numeric_n_bool_cl_kwargs, use_task_specific_params
from rouge_score import rouge_scorer

ROUGE_KEYS = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

def run_generate(verbose=True):

    parser = argparse.ArgumentParser()
    parser.add_argument("gen_summary_path", type=str, help="summary file generated by models.")
    parser.add_argument("reference_path", type=str, help="like DATA_DIR/test.target")
    parser.add_argument("--csv_path", type=str, required=False, default=None, help="where to save metrics")
    parser.add_argument(
        "--prefix", type=str, required=False, default=None, help="will be added to the begininng of src examples"
    )
    parser.add_argument("--exact", action="store_true")

    # Unspecified args like --num_beams=2 --decoder_start_token_id=4 are passed to model.generate
    args, rest = parser.parse_known_args()
    parsed_args = parse_numeric_n_bool_cl_kwargs(rest)
    if parsed_args and verbose:
        print(f"parsed the following generate kwargs: {parsed_args}")

    # Compute scores
    score_fn = cal_exact_rouge if args.exact else calculate_rouge
    output_lns = [x.rstrip() for x in open(args.gen_summary_path).readlines()]
    reference_lns = [x.rstrip() for x in open(args.reference_path).readlines()]

    assert len(output_lns) == len(reference_lns)
    scorer = rouge_scorer.RougeScorer(ROUGE_KEYS, use_stemmer=True)
    sum_scores = {k:[] for k in ROUGE_KEYS}
    for gen, ref in zip(output_lns, reference_lns):
        scores = scorer.score(gen,ref)
        for k in ROUGE_KEYS:
            sum_scores[k].append(scores[k].fmeasure)
    
    df = pd.DataFrame(columns=['rouge1', 'rouge2', 'rougeL', 'rougeLsum'])
    for k in ROUGE_KEYS:
        df[k] = sum_scores[k]
    df.to_csv(args.csv_path)

    if verbose:
        print(scores)

if __name__ == "__main__":
    # Usage for Summarization:
    # python each_scores_to_csv.py GEN_SUMMARY $DATA_DIR/test.target --csv_path MODEL_DIR/scores.csv
    run_generate(verbose=False)
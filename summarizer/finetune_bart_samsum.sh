#!/usr/bin/env bash
export PYTHONPATH="../":"${PYTHONPATH}"
today=`date "+%Y-%m-%d-%H-%M-%S"`

# Set --gradient_accumulation_steps  so that effective batch size is 256 (2*128, 4*64, 8*32, 16*16)
python finetune.py \
    --learning_rate=3e-5 \
    --new_params_learning_rate=3e-5  \
    --do_train \
    --do_predict \
    --val_check_interval 0.25 \
    --max_source_length 512 --max_target_length 100 \
    --label_smoothing 0.1 --adafactor --task summarization \
    --train_batch_size 1 \
    --eval_batch_size 1 \
    --n_train -1 \
    --n_val -1 \
    --n_test -1 \
    --data_dir "/home/naraki/dialogsum/samsum_dataset" \
    --output_dir "output/${today}" \
    --model_name_or_path "facebook/bart-large-xsum" \
    --gpus 1 --logger_name wandb \
    --use_speaker_embeds \
    --partial_embed \
    --speaker_embed_scale 10 \
    --val_max_target_length 100 --test_max_target_length 100 \
    --max_length 100 --min_length 10 \
    --bart \
    "$@"

    # --bart \ # bartを使う時
    # --sinusoidal \ # bartを無理矢理sinusoidalにしたい時

    # --tokenizer_name facebook/bart-large \
    # --warmup_steps 500 \

    # --expand_vocab \ # dataset2の時に使う. dataset1の時は使わない
    # --use_speaker_embeds \
    # --partial_embed \
    # --speaker_embed_scale 10 \

    # --use_turn_embeds \

#     --model_name_or_path facebook/bart-base \
#     --tokenizer_name facebook/bart-base \


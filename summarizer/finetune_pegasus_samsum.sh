#!/usr/bin/env bash
export PYTHONPATH="../":"${PYTHONPATH}"
today=`date "+%Y-%m-%d-%H-%M-%S"`

# Set --gradient_accumulation_steps  so that effective batch size is 256 (2*128, 4*64, 8*32, 16*16)
python finetune.py \
    --learning_rate=3e-5 \
    --do_train \
    --do_predict \
    --val_check_interval 0.25 \
    --max_source_length 512 --max_target_length 128 \
    --label_smoothing 0.1 --adafactor --task summarization \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --n_train -1 \
    --n_val -1 \
    --n_test -1 \
    --data_dir "/home/acc12119do/dialogsum/samsum_dataset2" \
    --output_dir "output/${today}" \
    --model_name_or_path "google/pegasus-xsum" \
    --gpus 1 --logger_name wandb \
    --expand_vocab \
    --use_speaker_embeds \
    "$@"

#    --freeze_embeds \
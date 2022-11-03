# Expt_DialogSum
Experimental Environment for Dialog Summarization

## Training

```
python finetune.py --learning_rate=3e-5 --new_params_learning_rate=3e-5 --do_train --do_predict --val_check_interval 0.25 --max_source_length 512 --max_target_length 100 --label_smoothing 0.1 --adafactor --task summarization --train_batch_size 1 --eval_batch_size 1 --n_train -1 --n_val -1 --n_test -1 --data_dir /home/naraki/dialogsum/samsum_dataset_fixed --output_dir output/2021-05-12-16-13-26 --model_name_or_path google/pegasus-xsum --gpus 1 --logger_name wandb --use_speaker_embeds --partial_embed --speaker_embed_scale 10 --val_max_target_length 100 --test_max_target_length 100 --max_length 100 --min_length 10 --fixedspecialtoken
```

### Additional Arguments
- new_params_learning_rate: lr for the additional modules specified in the code.
- use_speaker_embeds: whether to use speaker embeddings
- partial_embed: whether the speaker embeddings are partial or not
- speaker_embed_scale: the scale parameter of the speaker embeddings
- fixedspecialtoken: whether to use the special tokens Naraki finally decided to use, like EOU (End-of-Utterance) or EOT (End-of-Turn)

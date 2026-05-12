#!/usr/bin/env bash
uv run main.py train queryner \
    --backend gliner2 \
    --pretrained_model fastino/gliner2-base-v1 \
    --epochs 30 \
    --batch_size 8 \
    --encoder_lr 1e-5 \
    --task_lr 5e-4 \
    --warmup_ratio 0.1 \
    --scheduler_type cosine \
    --threshold 0.5 \
    --best_metric f1 \
    --early_stopping_patience 3 \
    --wandb_run gliner2_train_queryner

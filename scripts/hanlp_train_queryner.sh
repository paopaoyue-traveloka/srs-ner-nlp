#!/usr/bin/env bash
uv run --group hanlp main.py train queryner \
    --transformer bert-base-cased \
    --epochs 30 \
    --batch_size 32 \
    --lr 1e-3 \
    --encoder_lr 5e-5 \
    --warmup_steps 0.1 \
    --grad_norm 5.0 \
    --gradient_accumulation 1 \
    --best_metric f1 \
    --early_stopping_patience 3 \
    --wandb_run hanlp_train_queryner

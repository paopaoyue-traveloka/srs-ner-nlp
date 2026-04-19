#!/usr/bin/env bash
uv run main.py train queryner \
    --pretrained_model CONLL03_NER_BERT_BASE_CASED_EN \
    --epochs 30 \
    --batch_size 32 \
    --lr 2e-5 \
    --warmup_steps 100 \
    --grad_norm 5.0 \
    --best_metric f1 \
    --early_stopping_patience 5 \
    --wandb_run hanlp_train_queryner

#!/usr/bin/env bash
uv run main.py train queryner \
    --pretrained_model EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_BASE \
    --epochs 10 \
    --batch_size 32 \
    --lr 2e-5 \
    --warmup_steps 100 \
    --grad_norm 5.0 \
    --best_metric f1 \
    --early_stopping_patience 5 \
    --wandb_run hanlp_train_queryner_exp1

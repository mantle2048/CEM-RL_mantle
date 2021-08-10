#! /usr/bin/env bash

logdir="data"

envs=("HalfCheetah-v3")
seeds=(1)

for env in ${envs[@]}; do
    for seed in ${seeds[@]}; do
    python es_grad.py \
        --env $env \
        --seed $seed \
        --use_td3 \
        --output logdir
    done
done

#! /usr/bin/env bash

logdir="data"

envs=("HalfCheetah-v3")
# envs=("BipedalWalker-v3")
# envs=("HalfCheetah-v3")
# envs=("Hopper-v3" "Walker2d-v3")
# envs=("Ant-v3")
# envs=("Swimmer-v3")
# envs=("AntBulletEnv-v0")
# envs=("HalfCheetah-v3" "Ant-v3")

# envs=("HalfCheetahBulletEnv-v0")
# envs=("AntBulletEnv-v0")
# envs=("HopperBulletEnv-v0")
# envs=("Walker2DBulletEnv-v0")
# envs=("HumanoidBulletEnv-v0")
# envs=("HalfCheetah-v3")
# envs=("Hopper-v3" "Walker2d-v3")
seeds=(4)

for env in ${envs[@]}; do
    for seed in ${seeds[@]}; do
    python es_grad.py \
        --env $env \
        --seed $seed \
        --use_td3 \
        --output logdir
    done
done

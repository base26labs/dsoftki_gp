#!/bin/bash

DEBUG=false

# Dataset
DATA_DIR=data/synthetic
DATASETS=(branin six-hump-camel styblinski-tang hartmann welch)
dims=(2 2 2 6 20)
# indices=(0 1 2 3 4)
indices2=(0 1 2)
N=20000

# Model
NOISE=0.1
value_noises=(0.1 0.1 0.1 0.1 0.1)
deriv_noises=(0.2 0.2 0.2 0.6 0.2)
# value_noises=(0.01 0.01 0.01 0.01 0.01)
# deriv_noises=(0.02 0.02 0.02 0.06 0.02)

NUM_INDUCING=512

# DEVICE="cuda:0"
DEVICE="cpu"

# Training
GP_LR=0.01
DSOFTKI_LR=0.02
DDSVGP_LR=0.03
learning_rates=(0.03 0.03 0.03 0.06 0.2)
batch_sizes=(1024 1024 1024 1024 1024)
EPOCHS=50
TRAIN_FRAC=0.5
SEEDS=(6535 8830 92357)
# SEEDS=(8830 92357)

# Wandb
PROJECT=dsoftki
GROUP=synthetic-0.1-rbf-b1024
# GROUP=synthetic-0.01-rbf-b1024

# Debug
if $DEBUG; then
    EPOCHS=1
    indices=(0)
    SEEDS=(6535)
    GROUP=test
fi

pushd ..
    for i in "${indices2[@]}"
    do
        for seed in "${SEEDS[@]}"
        do
            dataset=${DATASETS[$i]}
            batch_size=${batch_sizes[$i]}
            learning_rate=${learning_rates[$i]}
            value_noise=${value_noises[$i]}
            deriv_noise=${deriv_noises[$i]}
            echo $dataset
            echo $batch_size
            echo $learning_rate
            echo $deriv_noise
        
            python run.py \
                model=dexact_gp \
                draw=true \
                gp.dexact_gp.model.device=$DEVICE \
                gp.dexact_gp.model.use_ard=true \
                gp.dexact_gp.model.use_scale=true \
                gp.dexact_gp.model.noise=$NOISE \
                gp.dexact_gp.training.seed=$seed \
                gp.dexact_gp.training.epochs=$EPOCHS \
                gp.dexact_gp.training.learning_rate=$learning_rate \
                synthetic.N=$N \
                data_dir=$DATA_DIR \
                dataset.name=$dataset \
                dataset.train_frac=0.9 \
                dataset.val_frac=0 \
                wandb.project=$PROJECT \
                wandb.group=$GROUP \
                wandb.watch=true
        done
    done
popd


#!/bin/bash

DEBUG=false

# Dataset
DATA_DIR=data/synthetic
DATASETS=(branin six-hump-camel styblinski-tang hartmann welch)
dims=(2 2 2 6 20)
indices=(0 1 2 3 4)
N=20000

# Model
NOISE=0.1
value_noises=(0.01 0.01 0.01 0.01 0.01)
deriv_noises=(0.2 0.2 0.2 0.6 2.0)

NUM_INDUCING=512

DEVICE="cuda:0"

# Training
GP_LR=0.01
DSOFTKI_LR=0.02
DDSVGP_LR=0.03
learning_rates=(0.03 0.03 0.03 0.06 0.2)
batch_sizes=(1024 1024 1024 1024 1024)
EPOCHS=50
TRAIN_FRAC=0.5
SEEDS=(6535 8830 92357)

# Wandb
PROJECT=dsoftki
GROUP=synthetic-0.01-rbf-10x-b1024

# Debug
if $DEBUG; then
    EPOCHS=1
    indices=(0)
    SEEDS=(6535)
    GROUP=test
fi

pushd ..
    for seed in "${SEEDS[@]}"
    do
        for i in "${indices[@]}"
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
                model=dsoftki \
                draw=true \
                gp.dsoft_ki.model.interp_init=kmeans \
                gp.dsoft_ki.model.num_interp=$NUM_INDUCING \
                gp.dsoft_ki.model.fit_chunk_size=256 \
                gp.dsoft_ki.model.noise=$value_noise \
                gp.dsoft_ki.model.deriv_noise=$deriv_noise \
                gp.dsoft_ki.model.mll_approx=hutchinson_fallback \
                gp.dsoft_ki.model.device=$DEVICE \
                gp.dsoft_ki.model.fit_device=$DEVICE \
                gp.dsoft_ki.model.per_interp_T=true \
                gp.dsoft_ki.model.use_qr=true \
                gp.dsoft_ki.model.use_ard=true \
                gp.dsoft_ki.model.learn_noise=true \
                gp.dsoft_ki.model.use_scale=true \
                gp.dsoft_ki.model.kernel._target_=RBFKernel \
                gp.dsoft_ki.model.kernel.nu=1.5 \
                gp.dsoft_ki.training.seed=$seed \
                gp.dsoft_ki.training.epochs=$EPOCHS \
                gp.dsoft_ki.training.batch_size=$batch_size \
                gp.dsoft_ki.training.learning_rate=$DSOFTKI_LR \
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

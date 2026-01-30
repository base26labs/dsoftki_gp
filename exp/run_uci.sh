#!/bin/bash

DEBUG=false

# Dataset
DATA_DIR=data/uci_datasets/uci_datasets

# All UCI datasets with gradient support (all except houseelectric)
DATASETS=(pol elevators bike kin40k protein keggdirected ctslices keggundirected 3droad song buzz)
dims=(26 18 17 8 9 20 385 27 3 90 77)

indices=(0 1 2 3 4 5 6)

# Model parameters
noise=0.01
NOISE=0.1

# Per-dataset learning rates (tuned)
base_learning_rates=(0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01)
learning_rates=(0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02)

# Per-dataset batch sizes
batch_sizes=(1024 1024 1024 1024 1024 1024 1024 1024 1024 1024 1024)

NUM_INDUCING=512

DEVICE="cuda:0"

# Training - per dataset
EPOCHS=50

# Per-dataset train/val splits
train_fracs=(0.5 0.5 0.5 0.5 0.5 0.5 0.4 0.5 0.5 0.4 0.2)
val_fracs=(0.3 0.3 0.3 0.3 0.3 0.3 0.4 0.3 0.3 0.4 0.6)

SEEDS=(6535 8830 92357)

# Wandb
PROJECT=dsoftki
GROUP=uci-0.01-rbf-gradients

# Debug mode
if $DEBUG; then
    EPOCHS=1
    indices=(2)
    SEEDS=(6535)
    GROUP=test-uci
fi

pushd ..
    for seed in "${SEEDS[@]}"
    do
        for i in "${indices[@]}"
        do
            dataset=${DATASETS[$i]}
            batch_size=${batch_sizes[$i]}
            learning_rate=${learning_rates[$i]}
            base_learning_rate=${base_learning_rates[$i]}
            dim=${dims[$i]}
            train_frac=${train_fracs[$i]}
            val_frac=${val_fracs[$i]}
            deriv_noise=$(echo "$noise * $dim * 10" | bc -l)
            
            echo "Running $dataset (D=$dim) with seed $seed"
            echo "Batch size: $batch_size, LR: $learning_rate, Deriv noise: $deriv_noise"
            echo "Train frac: $train_frac, Val frac: $val_frac"
        
            # DSoft-KI with gradients
            python run.py \
                model=dsoftki \
                gp.dsoft_ki.model.interp_init=kmeans \
                gp.dsoft_ki.model.num_interp=$NUM_INDUCING \
                gp.dsoft_ki.model.fit_chunk_size=256 \
                gp.dsoft_ki.model.noise=$noise \
                gp.dsoft_ki.model.deriv_noise=$deriv_noise \
                gp.dsoft_ki.model.mll_approx=hutchinson_fallback \
                gp.dsoft_ki.model.device=$DEVICE \
                gp.dsoft_ki.model.fit_device=$DEVICE \
                gp.dsoft_ki.model.per_interp_T=true \
                gp.dsoft_ki.model.use_qr=true \
                gp.dsoft_ki.model.skip_nll=true \
                gp.dsoft_ki.model.use_ard=true \
                gp.dsoft_ki.model.learn_noise=true \
                gp.dsoft_ki.model.use_scale=true \
                gp.dsoft_ki.model.kernel._target_=RBFKernel \
                gp.dsoft_ki.model.kernel.nu=1.5 \
                gp.dsoft_ki.training.seed=$seed \
                gp.dsoft_ki.training.epochs=$EPOCHS \
                gp.dsoft_ki.training.batch_size=$batch_size \
                gp.dsoft_ki.training.learning_rate=$learning_rate \
                data_dir=$DATA_DIR \
                dataset.name=$dataset \
                dataset.train_frac=$train_frac \
                dataset.val_frac=$val_frac \
                uci.get_forces=true \
                uci.n_neighbors=3 \
                wandb.project=$PROJECT \
                wandb.group=$GROUP \
                wandb.watch=true

            # DDSVGP (derivative directions)
            ddsvgp_learning_rate=$(echo "($base_learning_rate * 3)" | bc -l)
            python run.py \
                model=ddsvgp \
                gp.ddsvgp.model.num_inducing=$NUM_INDUCING \
                gp.ddsvgp.model.induce_init=kmeans \
                gp.ddsvgp.model.mll_type=PLL \
                gp.ddsvgp.model.device=$DEVICE \
                gp.ddsvgp.model.use_scale=true \
                gp.ddsvgp.model.num_directions=2 \
                gp.ddsvgp.model.noise=$NOISE \
                gp.ddsvgp.training.seed=$seed \
                gp.ddsvgp.training.epochs=$EPOCHS \
                gp.ddsvgp.training.batch_size=$batch_size \
                gp.ddsvgp.training.learning_rate=$ddsvgp_learning_rate \
                data_dir=$DATA_DIR \
                dataset.name=$dataset \
                dataset.train_frac=$train_frac \
                dataset.val_frac=0 \
                uci.get_forces=true \
                uci.n_neighbors=3 \
                wandb.project=$PROJECT \
                wandb.group=$GROUP \
                wandb.watch=true                
        done
    done
popd

#!/bin/bash

# Test script for Deep Kernel Learning (DKL) on MD22 datasets
# This script tests DSoftKI with embed_dim != -1 to enable learned embeddings

DEBUG=false

# Dataset
DATA_DIR=data/md22
DATASETS=(Ac-Ala3-NHMe DHA AT-AT stachyose buckyball-catcher AT-AT-CG-CG double-walled-nanotube)
# Actual dimensions = n_atoms * 3 (x, y, z coordinates per atom)
dims=(126 168 180 261 444 354 1110)
indices=(6 5 4 3 2 1 0)

# Model
NUM_INDUCING=512
noise=0.01

# Deep Kernel Learning Parameters
# embed_dim should be less than input dimension
# For MD22 high-D data, use embeddings that compress ~5-10x
EMBED_DIMS=(24 24 24 24 24 24 24)  # Corresponding to each dataset

DEVICE="cuda:0"

# Training
DSOFTKI_LR=0.002
EMBED_LR=0.0005  # Separate learning rate for embedding networks (usually smaller)
learning_rates=(0.002 0.002 0.002 0.002 0.002 0.002 0.002)
batch_sizes=(256 256 256 256 128 128 128)
EPOCHS=50
SEEDS=(6535 8830 92357)


# Data splits
train_fracs=(0.9 0.9 0.9 0.9 0.9 0.9 0.9)
val_fracs=(0.0 0.0 0.0 0.0 0.0 0.0 0.0)

# Wandb
PROJECT=dsoftki
GROUP=md22-dkl

# Debug mode: quick test
if $DEBUG; then
    EPOCHS=5
    indices=(0)  # Just test Ac-Ala3-NHMe
    SEEDS=(6535)
    GROUP=test-md22-dkl
fi

echo "=========================================="
echo "Deep Kernel Learning Test on MD22 Data"
echo "=========================================="
echo "Testing DSoftKI with learned embeddings (embed_dim != -1)"
echo ""

pushd ..
    for seed in "${SEEDS[@]}"
    do
        for i in "${indices[@]}"
        do
            dataset=${DATASETS[$i]}
            dim=${dims[$i]}
            embed_dim=${EMBED_DIMS[$i]}
            batch_size=${batch_sizes[$i]}
            learning_rate=${learning_rates[$i]}
            train_frac=${train_fracs[$i]}
            val_frac=${val_fracs[$i]}
            deriv_noise=$(echo "$noise * $dim" | bc -l)

            echo "=========================================="
            echo "Dataset: $dataset"
            echo "Input dimension: $dim"
            echo "Embedding dimension: $embed_dim"
            echo "Batch size: $batch_size"
            echo "Learning rate (model): $DSOFTKI_LR"
            echo "Learning rate (embedding): $EMBED_LR"
            echo "Value noise: $noise"
            echo "Derivative noise: $deriv_noise"
            echo "Train/Val/Test: $train_frac/$val_frac/$(echo "1 - $train_frac - $val_frac" | bc -l)"
            echo "Seed: $seed"
            echo "=========================================="
            echo ""

            # DSoftKI with Deep Kernel Learning (embed_dim != -1)
            python run.py \
                model=dsoftki \
                gp.dsoft_ki.model.interp_init=random \
                gp.dsoft_ki.model.num_interp=$NUM_INDUCING \
                gp.dsoft_ki.model.fit_chunk_size=256 \
                gp.dsoft_ki.model.noise=$noise \
                gp.dsoft_ki.model.deriv_noise=$deriv_noise \
                gp.dsoft_ki.model.mll_approx=hutchinson_fallback \
                gp.dsoft_ki.model.device=$DEVICE \
                gp.dsoft_ki.model.fit_device=$DEVICE \
                gp.dsoft_ki.model.per_interp_T=true \
                gp.dsoft_ki.model.use_qr=true \
                gp.dsoft_ki.model.use_ard=true \
                gp.dsoft_ki.model.learn_noise=true \
                gp.dsoft_ki.model.use_scale=false \
                gp.dsoft_ki.model.kernel._target_=RBFKernel \
                gp.dsoft_ki.model.kernel.nu=1.5 \
                gp.dsoft_ki.model.embed_dim=$embed_dim \
                gp.dsoft_ki.model.use_dot=false \
                gp.dsoft_ki.model.skip_nll=true \
                gp.dsoft_ki.training.seed=$seed \
                gp.dsoft_ki.training.epochs=$EPOCHS \
                gp.dsoft_ki.training.batch_size=$batch_size \
                gp.dsoft_ki.training.learning_rate=$DSOFTKI_LR \
                gp.dsoft_ki.training.embed_lr=$EMBED_LR \
                data_dir=$DATA_DIR \
                dataset.name=$dataset \
                dataset.train_frac=$train_frac \
                dataset.val_frac=$val_frac \
                md22.unit_cube=false \
                wandb.project=$PROJECT \
                wandb.group=$GROUP \
                wandb.watch=true

            echo ""
            echo "Completed: $dataset with embed_dim=$embed_dim"
            echo ""
        done
    done
popd

echo "=========================================="
echo "Deep Kernel Learning Test Complete!"
echo "=========================================="
echo ""
echo "Results logged to W&B project: $PROJECT, group: $GROUP"
echo ""
echo "Key parameters tested:"
echo "  - embed_dim: ${EMBED_DIMS[@]}"
echo "  - use_dot: false (distance-based attention)"
echo "  - embed_lr: $EMBED_LR"
echo ""
echo "Compare these results with standard DSoftKI (embed_dim=-1)"
echo "to evaluate the benefit of learned embeddings on molecular dynamics."
echo ""

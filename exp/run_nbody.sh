#!/bin/bash

DEBUG=false

# Dataset
DATA_DIR=data
dataset=nbody

# N-body configuration
# State dimension = 2 * n_particles * n_dims
n_particles_arr=(2 4 6 8 10)  # Number of particles
n_dims=3        # Spatial dimensions (3D)
# Total state dimension: 2 * 10 * 3 = 60

indices=(0 1 2 3 4)

# Model
NOISE=0.1
value_noise=0.1
# Derivative noise scales with dimension


NUM_INDUCING=512

DEVICE="cuda:0"

# Training
GP_LR=0.01
DSOFTKI_LR=0.02
DDSVGP_LR=0.03
batch_size=1024
EPOCHS=50
TRAIN_FRAC=0.9
SEEDS=(6535 8830 92357)

# Wandb
PROJECT=dsoftki
GROUP=nbody

# Debug
if $DEBUG; then
    EPOCHS=1
    SEEDS=(6535)
    GROUP=test-nbody
fi

pushd ..
    for seed in "${SEEDS[@]}"
    do
        for i in "${indices[@]}"
        do
            n_particles=${n_particles_arr[$i]}

            dim=$((2 * n_particles * n_dims))
            deriv_noise=$(echo "$value_noise * $dim" | bc -l)

            echo "==========================================="
            echo "N-body Hamiltonian System"
            echo "==========================================="
            echo "Dataset: $dataset"
            echo "N particles: $n_particles"
            echo "Spatial dims: $n_dims"
            echo "State dimension: $dim"
            echo "Batch size: $batch_size"
            echo "Learning rate: $DSOFTKI_LR"
            echo "Value noise: $value_noise"
            echo "Derivative noise: $deriv_noise"
            echo "Seed: $seed"
            echo "==========================================="
            echo ""

            # DSoftKI with gradients
            python run.py \
                model=dsoftki \
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
                gp.dsoft_ki.model.use_scale=false \
                gp.dsoft_ki.model.kernel._target_=RBFKernel \
                gp.dsoft_ki.model.kernel.nu=1.5 \
                gp.dsoft_ki.training.seed=$seed \
                gp.dsoft_ki.training.epochs=$EPOCHS \
                gp.dsoft_ki.training.batch_size=$batch_size \
                gp.dsoft_ki.training.learning_rate=$DSOFTKI_LR \
                data_dir=$DATA_DIR \
                dataset.name=$dataset \
                dataset.train_frac=$TRAIN_FRAC \
                dataset.val_frac=0 \
                nbody.n_particles=$n_particles \
                nbody.n_dims=$n_dims \
                nbody.get_forces=true \
                nbody.standardize=true \
                nbody.unit_cube=true \
                wandb.project=$PROJECT \
                wandb.group=$GROUP \
                wandb.watch=true

            # Cannot use ARD ...
            python run.py \
                model=ddsvgp \
                draw=true \
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
                gp.ddsvgp.training.learning_rate=$DDSVGP_LR \
                synthetic.N=$N \
                data_dir=$DATA_DIR \
                dataset.name=$dataset \
                dataset.train_frac=$TRAIN_FRAC \
                dataset.val_frac=0 \
                nbody.n_particles=$n_particles \
                nbody.n_dims=$n_dims \
                nbody.get_forces=true \
                nbody.standardize=true \
                nbody.unit_cube=true \
                wandb.project=$PROJECT \
                wandb.group=$GROUP \
                wandb.watch=true
        done
    done
popd

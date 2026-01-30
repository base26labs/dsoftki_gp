#!/bin/bash

DEBUG=true

# Dataset
DATA_DIR=data/synthetic
DATASETS=(branin six-hump-camel styblinski-tang hartmann welch)
dims=(2 2 2 6 20)
indices=(0 1 2 3 4)
# indices=(4)
indices2=(0 1 2 3)
N=20000

# Model
NOISE=0.1
value_noises=(0.1 0.1 0.1 0.1 0.1)
deriv_noises=(0.2 0.2 0.2 0.6 0.2)

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
GROUP=synthetic-0.1-rbf-b1024

# Debug
if $DEBUG; then
    EPOCHS=1
    indices=(4)
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

            # python run.py \
            #     model=dsvgp \
            #     draw=true \
            #     gp.dsvgp.model.num_inducing=$NUM_INDUCING \
            #     gp.dsvgp.model.mll_type=PLL \
            #     gp.dsvgp.model.device=$DEVICE \
            #     gp.dsvgp.model.induce_init=kmeans \
            #     gp.dsvgp.model.use_ard=true \
            #     gp.dsvgp.model.use_scale=true \
            #     gp.dsvgp.model.noise=$NOISE \
            #     gp.dsvgp.training.seed=$seed \
            #     gp.dsvgp.training.epochs=$EPOCHS \
            #     gp.dsvgp.training.batch_size=$batch_size \
            #     gp.dsvgp.training.learning_rate=$learning_rate \
            #     synthetic.N=$N \
            #     data_dir=$DATA_DIR \
            #     dataset.name=$dataset \
            #     dataset.train_frac=0.9 \
            #     dataset.val_frac=0 \
            #     wandb.project=$PROJECT \
            #     wandb.group=$GROUP \
            #     wandb.watch=true

            # # Cannot use ARD ...
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
                dataset.train_frac=0.9 \
                dataset.val_frac=0 \
                wandb.project=$PROJECT \
                wandb.group=$GROUP \
                wandb.watch=true

            python run.py \
                model=softki \
                draw=true \
                gp.softki.model.induce_init=kmeans \
                gp.softki.model.num_inducing=$NUM_INDUCING \
                gp.softki.model.fit_chunk_size=1024 \
                gp.softki.model.device=$DEVICE \
                gp.softki.model.mll_approx=hutchinson_fallback \
                gp.softki.model.kernel._target_=RBFKernel \
                gp.softki.model.per_interp_T=true \
                gp.softki.model.use_qr=true \
                gp.softki.model.use_ard=true \
                gp.softki.model.noise=$NOISE \
                gp.softki.model.learn_noise=true \
                gp.softki.training.seed=$seed \
                gp.softki.training.epochs=$EPOCHS \
                gp.softki.training.batch_size=$batch_size \
                gp.softki.training.learning_rate=$GP_LR \
                synthetic.N=$N \
                data_dir=$DATA_DIR \
                dataset.name=$dataset \
                dataset.train_frac=0.9 \
                dataset.val_frac=0 \
                wandb.project=$PROJECT \
                wandb.group=$GROUP \
                wandb.watch=true

            python run.py \
                model=svgp \
                draw=true \
                gp.svgp.model.induce_init=kmeans \
                gp.svgp.model.num_inducing=$NUM_INDUCING \
                gp.svgp.model.device=$DEVICE \
                gp.svgp.model.mll_type=PLL \
                gp.svgp.model.use_scale=true \
                gp.svgp.model.use_ard=true \
                gp.svgp.model.learn_noise=true \
                gp.svgp.model.noise=$NOISE \
                gp.svgp.training.seed=$seed \
                gp.svgp.training.epochs=$EPOCHS \
                gp.svgp.training.batch_size=$batch_size \
                gp.svgp.training.learning_rate=$GP_LR \
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
                model=dsvgp \
                draw=true \
                gp.dsvgp.model.num_inducing=$NUM_INDUCING \
                gp.dsvgp.model.mll_type=PLL \
                gp.dsvgp.model.device=$DEVICE \
                gp.dsvgp.model.induce_init=kmeans \
                gp.dsvgp.model.use_ard=true \
                gp.dsvgp.model.use_scale=true \
                gp.dsvgp.model.noise=$NOISE \
                gp.dsvgp.training.seed=$seed \
                gp.dsvgp.training.epochs=$EPOCHS \
                gp.dsvgp.training.batch_size=$batch_size \
                gp.dsvgp.training.learning_rate=$learning_rate \
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

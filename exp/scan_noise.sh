#!/bin/bash

DEBUG=false

DATA_DIR=data/synthetic
DATASETS=(branin six-hump-camel styblinski-tang hartmann welch)
indices=(0 1 2 3 4)
GP_LR=0.01
dims=(2 2 2 6 20)
batch_sizes=(1024 1024 1024 1024 1024)
NOISE=0.01
deriv_noises=(0.02 0.02 0.02 0.06 0.2)
factors=(2.0 5.0 10.0 1.0)
DSOFTKI_LR=0.02
length=${#DATASETS[@]}

PROJECT=dsoftki
GROUP=scan-noise
EPOCHS=50
NUM_INDUCING=512
N=20000
TRAIN_FRAC=0.5
DEVICE="cuda:0"
SEEDS=(6535 8830 92357)

if $DEBUG; then
    EPOCHS=1
    DATASETS=(branin)
    length=${#DATASETS[@]}
    SEEDS=(6535)
    GROUP=test
fi

pushd ..
    for seed in "${SEEDS[@]}"
    do
        for factor in "${factors[@]}"
        do
            # for ((i=0; i<$length; i++));
            for i in "${indices[@]}"
            do
                dataset=${DATASETS[$i]}
                batch_size=${batch_sizes[$i]}
                deriv_noise=${deriv_noises[$i]}
                deriv_noise=$(echo "$factor * $deriv_noise" | bc)
                echo $dataset
                echo $batch_size
                echo $deriv_noise

                python run.py \
                    model=dsoftki \
                    gp.dsoft_ki.model.interp_init=kmeans \
                    gp.dsoft_ki.model.num_interp=$NUM_INDUCING \
                    gp.dsoft_ki.model.fit_chunk_size=256 \
                    gp.dsoft_ki.model.noise=$NOISE \
                    gp.dsoft_ki.model.deriv_noise=$deriv_noise \
                    gp.dsoft_ki.model.mll_approx=hutchinson_fallback \
                    gp.dsoft_ki.model.per_interp_T=true \
                    gp.dsoft_ki.model.kernel._target_=RBFKernel \
                    gp.dsoft_ki.model.kernel.nu=1.5 \
                    gp.dsoft_ki.model.device=$DEVICE \
                    gp.dsoft_ki.model.fit_device=$DEVICE \
                    gp.dsoft_ki.model.use_qr=true \
                    gp.dsoft_ki.model.use_ard=true \
                    gp.dsoft_ki.model.learn_noise=true \
                    gp.dsoft_ki.model.use_scale=true \
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
    done
popd

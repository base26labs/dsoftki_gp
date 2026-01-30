#!/bin/bash

# Parse command line arguments
DEBUG=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            DEBUG=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--debug]"
            exit 1
            ;;
    esac
done

DATA_DIR=data/md22
DATASETS=(double-walled-nanotube buckyball-catcher AT-AT-CG-CG stachyose AT-AT DHA Ac-Ala3-NHMe)
batch_sizes=(128 256 256 512 512 1024 1024)
dims=(1110 444 354 261 180 168 126)
noise=0.001
NOISE=0.1
base_lr=0.01
base_learning_rates=(0.0005 0.001 0.001 0.002 0.002 0.004 0.004)
learning_rates=(0.001 0.002 0.002 0.004 0.004 0.008 0.008)

length=${#DATASETS[@]}
indices=(0 1 2 3 4 5 6)

num_inducing=512

TRAIN_FRAC=0.9

PROJECT=dsoftki
GROUP=md22-0.1-rbf
EPOCHS=50
DEVICE="cuda:0"
SEEDS=(6535 8830 92357)

if $DEBUG; then
    EPOCHS=1
    indices=(0)
    SEEDS=(6535)
    GROUP=test
    echo "DEBUG MODE: Running with 1 epoch, 1 dataset, 1 seed"
fi

pushd ..
    for seed in "${SEEDS[@]}"
    do
        for i in "${indices[@]}"
        do
            dataset=${DATASETS[$i]}
            batch_size=${batch_sizes[$i]}
            learning_rate=${learning_rates[$i]}
            dim=${dims[$i]}
            deriv_noise=$(echo "$noise * $dim" | bc)
            base_learning_rate=${base_learning_rates[$i]}
            echo $dataset
            echo $batch_size
            echo $learning_rate
            echo $deriv_noise
        
            python run.py \
                model=dsoftki \
                gp.dsoft_ki.model.interp_init=kmeans \
                gp.dsoft_ki.model.num_interp=$num_inducing \
                gp.dsoft_ki.model.fit_chunk_size=256 \
                gp.dsoft_ki.model.noise=$noise \
                gp.dsoft_ki.model.mll_approx=hutchinson_fallback \
                gp.dsoft_ki.model.deriv_noise=$deriv_noise \
                gp.dsoft_ki.model.device=$DEVICE \
                gp.dsoft_ki.model.fit_device=$DEVICE \
                gp.dsoft_ki.model.use_qr=true \
                gp.dsoft_ki.model.per_interp_T=true \
                gp.dsoft_ki.model.use_ard=true \
                gp.dsoft_ki.model.learn_noise=true \
                gp.dsoft_ki.model.use_scale=true \
                gp.dsoft_ki.model.skip_nll=true \
                gp.dsoft_ki.model.kernel._target_=RBFKernel \
                gp.dsoft_ki.model.kernel.nu=1.5 \
                gp.dsoft_ki.training.seed=$seed \
                gp.dsoft_ki.training.epochs=$EPOCHS \
                gp.dsoft_ki.training.batch_size=$batch_size \
                gp.dsoft_ki.training.learning_rate=$learning_rate \
                data_dir=$DATA_DIR \
                dataset.name=$dataset \
                dataset.train_frac=$TRAIN_FRAC \
                dataset.val_frac=0 \
                md22.unit_cube=false \
                wandb.project=$PROJECT \
                wandb.group=$GROUP \
                wandb.watch=true

            python run.py \
                model=softki \
                gp.softki.model.induce_init=kmeans \
                gp.softki.model.num_inducing=$num_inducing \
                gp.softki.model.fit_chunk_size=1024 \
                gp.softki.model.device=$DEVICE \
                gp.softki.model.mll_approx=hutchinson_fallback \
                gp.softki.model.use_qr=true \
                gp.softki.model.use_ard=true \
                gp.softki.model.noise=$NOISE \
                gp.softki.model.learn_noise=true \
                gp.softki.model.kernel._target_=RBFKernel \
                gp.softki.model.kernel.nu=1.5 \
                gp.softki.training.seed=$seed \
                gp.softki.training.epochs=$EPOCHS \
                gp.softki.training.batch_size=$batch_size \
                gp.softki.training.learning_rate=$base_learning_rate \
                data_dir=$DATA_DIR \
                dataset.name=$dataset \
                dataset.train_frac=$TRAIN_FRAC \
                dataset.val_frac=0 \
                md22.unit_cube=false \
                wandb.project=$PROJECT \
                wandb.group=$GROUP \
                wandb.watch=true

            python run.py \
                model=svgp \
                gp.svgp.model.induce_init=kmeans \
                gp.svgp.model.num_inducing=$num_inducing \
                gp.svgp.model.device=$DEVICE \
                gp.svgp.model.mll_type=PLL \
                gp.svgp.model.use_scale=true \
                gp.svgp.model.use_ard=true \
                gp.svgp.model.learn_noise=true \
                gp.svgp.training.seed=$seed \
                gp.svgp.training.epochs=$EPOCHS \
                gp.svgp.training.batch_size=$batch_size \
                gp.svgp.training.learning_rate=$base_learning_rate \
                gp.svgp.model.noise=$NOISE \
                data_dir=$DATA_DIR \
                dataset.name=$dataset \
                dataset.train_frac=$TRAIN_FRAC \
                dataset.val_frac=0 \
                md22.unit_cube=false \
                wandb.project=$PROJECT \
                wandb.group=$GROUP \
                wandb.watch=true

            ddsvgp_learning_rate=$(echo "($base_learning_rate * 3)" | bc)
            python run.py \
                model=ddsvgp \
                gp.ddsvgp.model.num_inducing=$num_inducing \
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
                dataset.train_frac=$TRAIN_FRAC \
                dataset.val_frac=0 \
                md22.unit_cube=false \
                wandb.project=$PROJECT \
                wandb.group=$GROUP \
                wandb.watch=true
        done
    done
popd

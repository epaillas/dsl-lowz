#!/bin/bash


loss=mae
run_name=v3
statistic=gammat
abacus_dataset=dsl
train_test_split=/pscratch/sd/e/epaillas/sunbird/data/dsl_train_test_split.json
model_dir=/pscratch/sd/e/epaillas/sunbird/trained_models/optimise/dsl/

python /pscratch/sd/e/epaillas/sunbird/sunbird/emulators/optimise.py \
    --model_dir "$model_dir" \
    --run_name "$run_name" \
    --abacus_dataset "$abacus_dataset" \
    --statistic "$statistic" \
    --train_test_split "$train_test_split" \
    --loss "$loss" \
    --accelerator "gpu" \
    --data_reader "AbacusLightcone" \
    --output_transforms Normalize \
    # --independent_avg_scale True \

#!/bin/bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export STEPS=500


export OUTPUT_DIR="logs_ablation/vangogh_H"
accelerate launch train.py           --pretrained_model_name_or_path=$MODEL_NAME            --output_dir=$OUTPUT_DIR           --class_data_dir=./data/samples_painting/        --class_prompt="painting"         --caption_target "van gogh"         --concept_type style           --resolution=512            --train_batch_size=2            --learning_rate=7e-5           --max_train_steps=$STEPS            --scale_lr --hflip --noaug   --use_8bit_adam        --parameter_group cross-attn           --enable_xformers_memory_efficient_attention  --f_divergence_type hellinger  --gradient_accumulation_steps 2 --anchor_type "superclass"
accelerate launch evaluate.py --root $OUTPUT_DIR --filter delta*.bin --concept_type style --caption_target "van gogh" --eval_json ../assets/eval.json
accelerate launch evaluate.py --root $OUTPUT_DIR --filter delta*.bin --concept_type style --caption_target "van gogh" --eval_json ../assets/eval.json --eval_stage

export OUTPUT_DIR="logs_ablation/vangogh_P"
accelerate launch train.py           --pretrained_model_name_or_path=$MODEL_NAME            --output_dir=$OUTPUT_DIR           --class_data_dir=./data/samples_painting/        --class_prompt="painting"         --caption_target "van gogh"         --concept_type style           --resolution=512            --train_batch_size=2            --learning_rate=7e-5           --max_train_steps=$STEPS            --scale_lr --hflip --noaug   --use_8bit_adam        --parameter_group cross-attn           --enable_xformers_memory_efficient_attention  --f_divergence_type pearson_chi2  --gradient_accumulation_steps 2 --anchor_type "superclass"
accelerate launch evaluate.py --root $OUTPUT_DIR --filter delta*.bin --concept_type style --caption_target "van gogh" --eval_json ../assets/eval.json
accelerate launch evaluate.py --root $OUTPUT_DIR --filter delta*.bin --concept_type style --caption_target "van gogh" --eval_json ../assets/eval.json --eval_stage

export OUTPUT_DIR="logs_ablation/vangogh_MSE"
accelerate launch train.py           --pretrained_model_name_or_path=$MODEL_NAME            --output_dir=$OUTPUT_DIR           --class_data_dir=./data/samples_painting/        --class_prompt="painting"         --caption_target "van gogh"         --concept_type style           --resolution=512            --train_batch_size=2            --learning_rate=7e-5           --max_train_steps=$STEPS            --scale_lr --hflip --noaug   --use_8bit_adam        --parameter_group cross-attn           --enable_xformers_memory_efficient_attention  --f_divergence_type mse  --gradient_accumulation_steps 2 --anchor_type "superclass"
accelerate launch evaluate.py --root $OUTPUT_DIR --filter delta*.bin --concept_type style --caption_target "van gogh" --eval_json ../assets/eval.json
accelerate launch evaluate.py --root $OUTPUT_DIR --filter delta*.bin --concept_type style --caption_target "van gogh" --eval_json ../assets/eval.json --eval_stage




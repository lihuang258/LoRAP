#!/bin/bash
##compress model
CUDA_VISIBLE_DEVICES=0 \
python main.py \
  --model decapoda-research/llama-7b-hf \
  --dataset bookcorpus \
  --sparsity_ratio 0.2 \
  --save_model compressed_model/lorap_0.2 \
  --para_allocate 3 \
  --mlp_compress_method prune \
  --real_com False \
  --deco_method AWSVD \
  --sublayer mlp,self_attn


#finetune the compressed model
CUDA_VISIBLE_DEVICES=0 \
python post_training.py \
      --prune_model compressed_model/lorap_0.2/model-0.2.bin \
      --data_path yahma/alpaca-cleaned \
      --lora_r 8 \
      --num_epochs 2 \
      --learning_rate 1e-4 \
      --batch_size 64 \
      --output_dir lora_tuned_weights_7B_0.2 \
      --wandb_project 7B_0.2

# real decompose the compressed model
#Select the lora weight folder to load
lora_path="tuned_model_path"
CUDA_VISIBLE_DEVICES=0 \
python After_Training.py \
  --model_path compressed_model/lorap_0.2/model-0.2.bin \
  --use_lora True \
  --lora_path $lora_path \
  --para_allocate 3 \
  --sparsity_ratio 0.2 \
  --eval_seqlen 128 \
  --save_path tuned_model/lorap_0.2/model.bin
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="./data_example_captioned_small"
export OUTPUT_DIR="./exps/krk_without_pc_cap"

python inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate_ti=1e-4 \
  --color_jitter \
  --lr_scheduler="constant" \
  --lr_warmup_steps=100 \
  --placeholder_tokens="<s1>|<s2>" \
  --placeholder_token_at_data="<krk>|<s1><s2>"\
  --initializer_tokens="ra|nd" \
  --save_steps=100 \
  --max_train_steps_ti=2000 \
  --perform_inversion=True \
  --weight_decay_ti=0.01 \
  --device="cuda:0" \
  --log_wandb=True \
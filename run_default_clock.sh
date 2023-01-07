export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="./data/yc"
export OUTPUT_DIR="./exps/yc_noproj_1e-2"

python inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate_ti=4e-4 \
  --scale_lr\
  --color_jitter \
  --lr_scheduler="constant" \
  --lr_warmup_steps=100 \
  --placeholder_tokens="<s1>" \
  --placeholder_token_at_data="<s>|<s1>"\
  --initializer_tokens="clock" \
  --save_steps=50 \
  --max_train_steps_ti=5000 \
  --perform_inversion=True \
  --use_template="object"\
  --weight_decay_ti=0.01 \
  --device="cuda:0" \
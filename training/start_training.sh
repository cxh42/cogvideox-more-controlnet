#!/bin/bash
# 创建一个临时的空CSV文件（如果不存在）
touch dummy.csv
# 修改训练参数
accelerate launch --config_file accelerate_config_machine_single.yaml --multi_gpu \
  train_controlnet.py \
  --tracker_name "cogvideox-depth-controlnet" \
  --gradient_checkpointing \
  --pretrained_model_name_or_path THUDM/CogVideoX-2b \
  --enable_tiling \
  --enable_slicing \
  --validation_prompt "The video shows a street with cars parked on the side. The camera pans to the right, revealing more of the street and the surrounding area. The scene is overcast and foggy, creating a somewhat gloomy atmosphere. The camera movement is smooth and steady, allowing for a clear view of the surroundings." \
  --validation_video "../data/rgb/rgb (1).mp4" \
  --num_inference_steps 28 \
  --num_validation_videos 1 \
  --validation_steps 500 \
  --seed 42 \
  --mixed_precision bf16 \
  --output_dir "cogvideox-depth-controlnet" \
  --height 480 \
  --width 720 \
  --fps 8 \
  --max_num_frames 49 \
  --rgb_dir "../data/rgb" \
  --depth_dir "../data/depth" \
  --text_dir "../data/text" \
  --video_root_dir "../data" \
  --csv_path "./dummy.csv" \
  --stride_min 1 \
  --stride_max 3 \
  --hflip_p 0.5 \
  --controlnet_type "depth" \
  --use_custom_dataset \
  --controlnet_transformer_num_layers 8 \
  --controlnet_input_channels 3 \
  --downscale_coef 8 \
  --controlnet_weights 0.5 \
  --init_from_transformer \
  --train_batch_size 1 \
  --dataloader_num_workers 0 \
  --num_train_epochs 10 \
  --max_train_steps 5000 \
  --checkpointing_steps 500 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-5 \
  --lr_scheduler cosine_with_restarts \
  --lr_warmup_steps 250 \
  --lr_num_cycles 1 \
  --gradient_checkpointing \
  --optimizer AdamW \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --max_grad_norm 1.0 \
  --allow_tf32
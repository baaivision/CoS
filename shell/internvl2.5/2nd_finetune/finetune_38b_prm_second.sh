
OUTPUT_DIR='/output/internvl/prm_v2'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34221
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

# number of gpus: 8
# batch size per gpu: 2
# gradient accumulation steps: 8
# total batch size: 128
# epoch: 1
torchrun \
  --nnodes=2 \
  --node_rank=1 \
  --master_addr=172.24.188.208 \
  --nproc_per_node=8 \
  --master_port=${MASTER_PORT} \
  internvl/train/internvl_chat_prm.py \
  --model_name_or_path "/output/internvl/sft_38b_mpo_5e_6" \
  --conv_style "internvl2_5" \
  --use_fast_tokenizer False \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "/shell/data/prm-100K-train.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.4 \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone True \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 2 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 3000 \
  --save_total_limit 1 \
  --learning_rate 1e-6 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 8192 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage3_config_34b.json" \
  --report_to none \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"

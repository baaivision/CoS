
OUTPUT_DIR='/output/internvl/8b_dpo'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

# number of gpus: 8
# batch size per gpu: 4
# gradient accumulation steps: 8
# total batch size: 256
# epoch: 1
torchrun \
  --nnodes=2 \
  --node_rank=0 \
  --master_addr=172.24.188.208 \
  --nproc_per_node=8 \
  --master_port=${MASTER_PORT} \
  internvl/train/internvl_chat_dpo.py \
  --model_name_or_path "/output/internvl/sft_mpo_5e_6" \
  --conv_style "internvl2_5" \
  --use_fast_tokenizer False \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "/shell/data/dpo_v3.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.1 \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone False \
  --vision_select_layer -1 \
  --use_data_resampling False \
  --dataloader_num_workers 8 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 1200 \
  --save_total_limit 1 \
  --learning_rate 5e-6 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 8192 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length False \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage3_config.json" \
  --report_to none \
  --loss_type sigmoid,bco_pair \
  --sigmoid_loss_weight 1.0 \
  --bco_pair_loss_weight 0.0 \
  --rpo_alpha 1 \
  --use_liger False \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"


CHECKPOINT="/output/internvl/8b_dpo"

torchrun --nnodes=2 --node_rank=0 --master_addr=172.24.188.208 --nproc_per_node=8 --master_port=23357 eval/vqa/generate_dpo_data.py --checkpoint ${CHECKPOINT} \
    --datasets textvqa_val --dynamic \
    --temperature 1.0 --return-seqs 16 \
    --max_new_tokens 6144 \
    --input-json /dpo_data/v3/dpo_data_v3_pro_in.json \
    --out-dir /dpo_data/v3/internvl_mpo_sft_330K_dpo_v3_base/
 

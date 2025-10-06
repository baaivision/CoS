
CHECKPOINT="/official_models/internvl/InternVL2_5-8B"


torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=8 \
    --master_port=12356 \
    eval/scienceqa/evaluate_m3cot.py --checkpoint ${CHECKPOINT} --datasets m3cot_test --dynamic --o1_pred

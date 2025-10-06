
CHECKPOINT="/official_models/internvl/InternVL2_5-8B"


torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=8 \
    --master_port=12349 \
    eval/vqa/evaluate_vqa.py --checkpoint ${CHECKPOINT} --datasets ai2diagram_test --dynamic


CHECKPOINT="/official_models/internvl/InternVL2_5-8B"      


torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=8 \
    --master_port=12350 \
    eval/mathvista/evaluate_mathvista_mine.py --checkpoint ${CHECKPOINT} --datasets MathVista_testmini --dynamic --o1_pred

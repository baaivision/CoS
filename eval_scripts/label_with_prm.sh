
CHECKPOINT="/output/internvl/prm_v2"

torchrun --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=4 --master_port=23357 eval/vqa/score_with_prm.py --checkpoint ${CHECKPOINT} \
    --datasets textvqa_val --dynamic --auto \
    --input-json /dataset/LLaVA-NeXt/eval/m3cot/passn/llava-v1.6-8b-o1_330K/full_8/extract_full.json \
    --out-dir /dataset/LLaVA-NeXt/eval/m3cot/passn/llava-v1.6-8b-o1_330K/full_8/scores/prm_v2/ 

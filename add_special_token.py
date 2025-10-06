import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from internvl.model import *
import torch

model_path = "/official_models/internvl/InternVL2_5-38B-MPO"

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
model = InternVLChatModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)

special_tokens_dict = ["<|reasoning_start|>", "<|reasoning_end|>", "<|reasoning_proceed|>", "<|reasoning_backtrack|>", "<|reasoning_step_start|>", "<|reasoning_step_end|>",
                "<|reasoning_step_name_start|>", "<|reasoning_step_name_end|>", "<|reasoning_step_thought_start|>", "<|reasoning_step_thought_end|>", "<|reasoning_step_reflection_start|>", "<|reasoning_step_reflection_end|>",
                "<|answer_start|>", "<|answer_end|>", "ки", "к+и", "к-и"]   # , "ки", "к+и", "к-и"

descriptions = ["start of reasoning", "end of reasoning", "reasoning proceed", "reasoning backtrack", "start of reasoning step", "end of reasoning step",
            "start of reasoning step name", "end of reasoning step name", "start of reasoning step thought", "end of reasoning step thought", "start of reasoning step reflection", "end of reasoning step reflection",
            "start of answer", "end of answer",  "step tag to label", "correct step", "bad step"] #  "step tag to label", "correct step", "bad step"

# if you want only prm's special tokens
# special_tokens_dict = ["ки", "к+и", "к-и"]
# descriptions = ["step tag to label", "correct step", "bad step"]

# add special tokens
print(len(tokenizer))
num_added_toks = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_dict})
print("We have added", num_added_toks, "tokens")
print(len(tokenizer))

# resize token embeddings
print("checking input embedding diff:")
print(model.get_input_embeddings().weight.size()) 
model.resize_token_embeddings(len(tokenizer))
print(model.get_input_embeddings().weight.size())          #print(model.get_input_embeddings().weight.size())


# initialize added token embeddings with existing embeddings
with torch.no_grad():
    for i, token in enumerate(reversed(descriptions), start=1):
        tokenized = tokenizer.tokenize(token)
        tokenized_ids = tokenizer.convert_tokens_to_ids(tokenized)
        
        new_embedding = model.get_input_embeddings().weight[tokenized_ids].mean(axis=0)
        new_lmhead = model.lm_head.weight[tokenized_ids].mean(axis=0)

        model.get_input_embeddings().weight[-i, :] = new_embedding.clone().detach().requires_grad_(True)
        model.lm_head.weight[-i, :] = new_lmhead.clone().detach().requires_grad_(True)

print("success!")

# Save modified model, tokenizer and config.
output_dir = model_path + "_add_special_token"

model.config.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
model.save_pretrained(output_dir)

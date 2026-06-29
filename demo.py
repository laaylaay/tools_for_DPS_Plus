"""Demo: run weak2strong_tool on images in DPS+/examples."""

import json
import os

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from weak2strong_tool import weak2strong_tool


def main():
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    dataset = "Self-Gen"
    method = "weak2strong_tool"
    label = "Abyssinian"
    attack = "Somali"
    remote_model_name = "qwen3-vl-plus"
    mask_percentile = 80

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_token = os.getenv("HF_TOKEN")
    
    local_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        token=hf_token,
    ).eval().to(device)
    local_processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=hf_token,
    )

    files = os.listdir("./examples/")
    safe_remote_model_name = remote_model_name.replace("/", "-")
    output_path = f"./{safe_remote_model_name}_{dataset}_{method}.jsonl"
    
    question_prompt = f"Is it a '{label}' or a '{attack}'?"
    for i in files:
        image_path = "./examples/" + i
        result = weak2strong_tool(
            local_model=local_model,
            local_processor=local_processor,
            image_path=image_path,
            label=label,
            attack_text=attack,
            question_prompt=question_prompt,
            remote_model_name=remote_model_name,
            mask_percentile=mask_percentile,
        )
        result["image_path"] = image_path
        with open(output_path, "a", encoding="utf-8") as jsonl_file:
            jsonl_file.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

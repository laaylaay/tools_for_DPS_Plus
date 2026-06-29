import os

import numpy as np
import torch
from PIL import Image, ImageDraw


def build_local_inputs(local_processor, prompt, image_obj, device):
    """Build multimodal inputs for the local vision model."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_obj},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text_query = local_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return local_processor(text=[text_query], images=[image_obj], padding=True, return_tensors="pt").to(device)


def generate_heatmap_and_mask(
    local_model,
    local_processor,
    image_path,
    question_prompt,
    device,
    temp_image_dir,
    mask_percentile=80,
):
    original_image = Image.open(image_path).convert("RGB")

    target_size = (1024, 1024)
    resized_image_for_heatmap = original_image.resize(target_size, Image.Resampling.LANCZOS)
    english_instruction_suffix = "\nPlease answer in English."
    question_prompt_en = question_prompt + english_instruction_suffix
    general_prompt_en = "Describe the image in detail." + english_instruction_suffix

    with torch.no_grad():
        inputs_heatmap = build_local_inputs(local_processor, question_prompt_en, resized_image_for_heatmap, device)
        general_inputs = build_local_inputs(local_processor, general_prompt_en, resized_image_for_heatmap, device)
        output = local_model(**inputs_heatmap, output_attentions=True)
        general_output = local_model(**general_inputs, output_attentions=True)

        vision_start_token_id = local_processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end_token_id = local_processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        pos = inputs_heatmap["input_ids"].tolist()[0].index(vision_start_token_id) + 1
        pos_end = inputs_heatmap["input_ids"].tolist()[0].index(vision_end_token_id)

        image_inputs_aux = local_processor.image_processor(images=[resized_image_for_heatmap])
        grid_shape = image_inputs_aux["image_grid_thw"].numpy().squeeze(0)[1:]
        output_shape = (grid_shape / 2).astype(int)
        last_layer_att = output.attentions[-1][0, :, -1, pos:pos_end].mean(dim=0).to(torch.float32).cpu().numpy()
        last_layer_general_att = general_output.attentions[-1][0, :, -1, pos:pos_end].mean(dim=0).to(
            torch.float32
        ).cpu().numpy()
        last_layer_heatmap = (last_layer_att / (last_layer_general_att + 1e-9)).reshape(output_shape)

        patch_width = original_image.width / output_shape[1]
        patch_height = original_image.height / output_shape[0]

        masked_image = original_image.copy()
        draw = ImageDraw.Draw(masked_image)
        threshold = np.percentile(last_layer_heatmap, mask_percentile)
        for y, x in zip(*np.where(last_layer_heatmap > threshold)):
            draw.rectangle(
                [(x * patch_width, y * patch_height), ((x + 1) * patch_width, (y + 1) * patch_height)],
                fill="black",
            )

        base_name = os.path.basename(image_path)
        masked_image_path = os.path.join(temp_image_dir, f"masked_{base_name}")
        masked_image.save(masked_image_path)

        return {"masked_image_path": masked_image_path}, None

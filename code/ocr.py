from transformers import AutoModel, AutoTokenizer
import torch
import os
import sys


os.environ["CUDA_VISIBLE_DEVICES"] = '0'

model_name = 'deepseek-ai/DeepSeek-OCR-2'


tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, _attn_implementation='flash_attention_2', trust_remote_code=True, use_safetensors=True)
model = model.eval().cuda()

model.generation_config.repetition_penalty = 1.1

# prompt = "<image>\nFree OCR. "
# prompt = "<image>\n<|grounding|>Convert the document to markdown. "

prompt = "<image>\n<|grounding|>Convert the document to markdown. "


directory = '../data/amazon_video_games_item_images_v3'
n = 0
for filename in os.listdir(directory):
    n += 1
    if n > 100:
        break
    if filename.lower().endswith(".png"):
        full_path = os.path.join(directory, filename)


        image_file = full_path
        output_path = '../data/games/ocr_results_0719/' + filename.replace('.png', '')

        res = model.infer(tokenizer,
                        prompt=prompt,
                        image_file=image_file,
                        output_path = output_path,
                        base_size = 1024,
                        image_size = 768,
                        crop_mode=True,
                        save_results = True,
                        )


# Environment Setup
# conda create -n llm python=3.10
# conda activate llm
# pip install --pre --upgrade ipex-llm[xpu_2.6] --extra-index-url https://download.pytorch.org/whl/xpu
# pip install modelscope
# modelscope download --model Qwen/Qwen2.5-Omni-7B  --local_dir Qwen2.5-Omni-7B

#source /opt/intel/oneapi/2024.0/oneapi-vars.sh

import os
import time

os.environ["UR_L0_USE_IMMEDIATE_COMMANDLISTS"] = "0"                                    
os.environ["SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS"] = "1"

from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor
from ipex_llm import optimize_model
from qwen_omni_utils import process_mm_info
import torch

model_path = r"Qwen2.5-Omni-7B"

def get_xpu_memory_reserved_usage_mb():
    return torch.xpu.memory_stats()["reserved_bytes.all.current"] / 1024 / 1024

print("Loading model...")
t0 = time.time()
# torch.xpu.empty_cache()

model = Qwen2_5OmniModel.from_pretrained(model_path, enable_audio_output=False)
model = optimize_model(model, low_bit="sym_int4",
                       modules_to_not_convert=["audio_tower", "visual", "token2wav"])
model = model.half().to('xpu')

processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
print(f"Model loaded in {time.time() - t0:.2f} seconds.")

mem_load_model = get_xpu_memory_reserved_usage_mb()
print(f"XPU memory reserved during model load: {mem_load_model:.2f} MB")
# video input (use audio in video)
# conversation = [
#     {
#         "role": "system",
#         "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
#     },
#     {
#         "role": "user",
#         "content": [
#             {"type": "video", "video": r"test.mp4"},
#         ],
#     },
# ]

#image input
conversation = [
    {
        "role": "system",
        "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": r"resources/01_landsacpe.jpg"},
            {"type": "text", "text": "描述这张图片"},
        ],
    },
]

# Preparation for inference
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
inputs = processor(text=text, audios=audios, images=images, videos=videos, return_tensors="pt", padding=True)
inputs = inputs.to(model.device).to(model.dtype)

print("inputs: ", inputs)

print("Pixel Values Shape:", inputs.get("pixel_values", None).shape)

t0 = time.time()
# note: use `thinker_max_new_tokens` instead of `max_new_tokens`
text_ids = model.generate(**inputs, use_audio_in_video=True, thinker_max_new_tokens=1024)
text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
t1 = time.time()

mem_execute_model = get_xpu_memory_reserved_usage_mb()

print(f"Generation time:   {t1 - t0:.2f} s")
# print(f"Peak XPU memory this run:    {peak_mem:.2f} MB")

print(f"XPU memory reserved after model load: {mem_execute_model:.2f} MB")
print(text)

# https://github.com/QwenLM/Qwen2.5-Omni/tree/main/cookbooks
# inference with video screen recording and test, do not use audio in video and no audio output 
import soundfile as sf
import os
import subprocess
import tempfile
import argparse

from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from ipex_llm import optimize_model
import torch

parser = argparse.ArgumentParser(description="Voice Chating using Qwen2.5 7B Omni model")
parser.add_argument("--input", type=str, default="input.mp4", help="Path to the audio video file")
parser.add_argument("--query", help="描述一下音频中讲的什么")
args = parser.parse_args()

def get_xpu_memory_reserved_usage_mb():
    return torch.xpu.memory_stats()["reserved_bytes.all.current"] / 1024 / 1024

def inference(video_path, prompt):
    conversation = [
        {"role": "system", "content": [
                {"type": "text", "text": "You are a helpful assistant."}
            ],},
        {"role": "user", "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": prompt},
            ]
        },
    ]
    USE_AUDIO_IN_VIDEO = False
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    # image_inputs, video_inputs = process_vision_info([messages])
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = processor(text=text, audios=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = inputs.to(model.device).to(model.dtype)

    output = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, return_audio=False)

    text = processor.batch_decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return text


model_path = "Qwen2.5-Omni-7B"
model = Qwen2_5OmniModel.from_pretrained(model_path, enable_audio_output=True)

model = optimize_model(model, low_bit="sym_int4",
                       modules_to_not_convert=["audio_tower", "visual", "token2wav"])
model = model.half().to('xpu')

processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

# pls Notice it can use up to XPU memory reserved: 28426.00 MB
## summarize:
video_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/screen.mp4"
prompt = "Who is the authors of this paper?"
# video_path = args.input

## Use a local HuggingFace model to inference.
response = inference(video_path, prompt)

print("Results: ", response[0])

mem_execute_model = get_xpu_memory_reserved_usage_mb()
print(f"XPU memory reserved during model running: {mem_execute_model:.2f} MB")

## Assistant
video_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/screen.mp4"
prompt = "Please trranslate the abstract of paper into Chinese."

response = inference(video_path, prompt)

print("Results: ", response[0])






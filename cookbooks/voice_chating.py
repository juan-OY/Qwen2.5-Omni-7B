# https://github.com/QwenLM/Qwen2.5-Omni/tree/main/cookbooks

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

def inference(audio_path):
    conversation = [
        {"role": "system", "content": [
                {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
            ],},
        {"role": "user", "content": [
                {"type": "audio", "audio": audio_path},
            ]
        },
    ]
    USE_AUDIO_IN_VIDEO = True
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    # image_inputs, video_inputs = process_vision_info([messages])
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = processor(text=text, audios=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = inputs.to(model.device).to(model.dtype)

    output = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, return_audio=True)

    text = processor.batch_decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    audio = output[1]
    return text, audio


model_path = "Qwen2.5-Omni-7B"
model = Qwen2_5OmniModel.from_pretrained(model_path, enable_audio_output=True)

model = optimize_model(model, low_bit="sym_int4",
                       modules_to_not_convert=["audio_tower", "visual", "token2wav"])
model = model.half().to('xpu')

processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

audio_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/guess_age_gender.wav"

audio_path = args.input

## Use a local HuggingFace model to inference.
response = inference(audio_path)
# print(response[0][0])
# display(Audio(response[1], rate=24000))
print("Results: ", response[0])
audio_file = "output.wav"
sf.write(
        audio_file,
        response[1].reshape(-1).detach().cpu().numpy(),
        samplerate=24000,)

mem_execute_model = get_xpu_memory_reserved_usage_mb()
print(f"XPU memory reserved during model running: {mem_execute_model:.2f} MB")

print("VOice chat complete. Output saved as output.wav")



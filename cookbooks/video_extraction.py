import soundfile as sf
import os
import subprocess
import tempfile
import argparse

from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from ipex_llm import optimize_model
import torch

os.environ["UR_L0_USE_IMMEDIATE_COMMANDLISTS"] = "0"                                    
os.environ["SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS"] = "1"

def truncate_video(input_path, max_seconds=30):
    # temp_dir=tempfile.mkdtemp()
    # output_path = os.path.join(temp_dir, "truncated_video.mp4")
    output_path = os.path.join(os.getcwd(), "truncated_video.mp4")  # 当前目录
    print("save truncated video")
    try:
        cmd = f"ffmpeg -i {input_path} -t {max_seconds} -c:v copy -c:a copy {output_path} -y"
        subprocess.run(cmd, shell=True, check=True, stderr=subprocess.PIPE)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error truncating video: {e}")
        return None
    
def get_xpu_memory_reserved_usage_mb():
    return torch.xpu.memory_stats()["reserved_bytes.all.current"] / 1024 / 1024

parser = argparse.ArgumentParser(description="Transcribe video using Qwen2.5 model")
parser.add_argument("--input", type=str, default="input.mp4", help="Path to the input video file")
parser.add_argument("--max_seconds", type=int, default=20, help="Maximum duration of the truncated video in seconds")   
parser.add_argument("--truncate", action="store_true", help=" truncate video")
parser.add_argument("--query", default="描述一下视频中讲的什么", help="Text input")
args = parser.parse_args()

video_path = args.input
if args.truncate:
    print(f"Truncating video to {video_path} ")
    video_path = truncate_video(video_path, args.max_seconds)

model_path = "Qwen2.5-Omni-7B"
model = Qwen2_5OmniModel.from_pretrained(model_path, enable_audio_output=True)

model = optimize_model(model, low_bit="sym_int4",
                       modules_to_not_convert=["audio_tower", "visual", "token2wav"])
model = model.half().to('xpu')

processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

conversation = [
        {"role": "system",
         "content":"You are a helpful assistant."},
        {"role": "user", "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": args.query},
            ]
        },
    ]

USE_AUDIO_IN_VIDEO = True
text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    # image_inputs, video_inputs = process_vision_info([messages])
audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = processor(text=text, audios=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = inputs.to(model.device).to(model.dtype)

#Generate both text and audio outputs
# text_ids, output = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, return_audio=False)
text_ids, output = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO)

text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print("model response: ", text[0])

mem_execute_model = get_xpu_memory_reserved_usage_mb()
print(f"XPU memory reserved during model running: {mem_execute_model:.2f} MB")
# Save the audio output

# audio_file = "output.wav"
# sf.write(
#         audio_file,
#         output.reshape(-1).detach().cpu().numpy(),
#         samplerate=24000,)

print("Transcription complete. Output saved as output.wav")

# if args.truncate and os.path.dirname(video_path) != os.path.dirname(args.input):
#     try:
#         os.remove(video_path)   
#         os.rmdir(os.path.dirname(video_path)) 
#         print
#     except OSError as e:
#         print(f"Error deleting temporary files: {e}")


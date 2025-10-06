#!/usr/bin/env python3
# qwen_probe_q1.py
# Qwen-Audio span detection on MSP-Podcast (50 examples for demo)

import json
import numpy as np
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "Qwen/Qwen2-Audio-7B-Instruct"


import os

os.environ["HF_DATASETS_CACHE"] = "/N/slate/mealieff/hf_cache/datasets"
os.environ["HF_MODULES_CACHE"] = "/N/slate/mealieff/hf_cache/modules"
os.environ["HF_METRICS_CACHE"] = "/N/slate/mealieff/hf_cache/metrics"


print(f"Loading {MODEL_NAME} on {device} ...")
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)


prompt_shift = (
    "Listen carefully to the audio. Locate the *first emotional shift* in this segment. "
    "Return JSON with the approximate second and the excerpt of transcript where it occurs, "
    "in this format: {'shift_time': X, 'span': '...'}"
)

prompt_intensity = (
    "Identify where emotional intensity changes within this segment. "
    "Return JSON: {'start_time': X, 'end_time': Y, 'span': '...'}"
)

prompt_expressive = (
    "Mark the emotionally expressive portion in the transcript with [emote/] ... [/emote]. "
    "Return only the modified transcript."
)

prompt_emotion = (
    "For the span tagged [emote/] ... [/emote], what is the dominant emotion? "
    "Choose one: angry, happy, sad, neutral, disgust, fear, surprise."
)

def ask_qwen(audio, transcript, prompt):
    text_input = f"{prompt}\n\nTranscript:\n{transcript}"
    inputs = processor(
        text=text_input,
        audios=audio,
        sampling_rate=16000,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=512)
    result = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    return result


dataset = load_dataset("narad/ravdess", split="train[:50]")
print("Loading IEMOCAP/MSP-Podcast subset...")

results = []

for spk_id in set(dataset["speaker_id"]):
    # Select up to 10 turns per speaker
    speaker_turns = [ex for ex in dataset if ex["speaker_id"] == spk_id][:10]
    if not speaker_turns:
        continue

    for seg_length in [30, 120]:  # 30s and 2min
        # Concatenate audio
        audio_concat = np.concatenate([turn["audio"]["array"] for turn in speaker_turns])
        sr = speaker_turns[0]["audio"]["sampling_rate"]
        duration = len(audio_concat) / sr
        if duration > seg_length:
            audio_concat = audio_concat[: seg_length * sr]

        transcript_concat = " ".join([
            turn.get("transcription") or turn.get("text", "") for turn in speaker_turns
        ])

        print(f"🎧 Speaker {spk_id}, {seg_length}s segment")

        # Qwen prompts
        qwen_shift = ask_qwen(audio_concat, transcript_concat, prompt_shift)
        qwen_intensity = ask_qwen(audio_concat, transcript_concat, prompt_intensity)
        qwen_expressive = ask_qwen(audio_concat, transcript_concat, prompt_expressive)
        qwen_emotion = ask_qwen(audio_concat, qwen_expressive, prompt_emotion)

        results.append({
            "speaker_id": spk_id,
            "segment_duration": f"{seg_length}s",
            "qwen_shift": qwen_shift,
            "qwen_intensity": qwen_intensity,
            "expressive_span": qwen_expressive,
            "emotion_label": qwen_emotion
        })
with open("msp_span_detection.json", "w") as f:
    json.dump(results, f, indent=2)

print("✅ Saved results to msp_span_detection.json")


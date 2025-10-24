#!/usr/bin/env python3
"""
Extract per-frame ground-truth VAD labels for MSP-Podcast segments.
Outputs 3 new JSON files with frame-level continuous emotion annotations.
"""

import json, os
import torch, torchaudio
from transformers import AutoFeatureExtractor

# ----------------------
# CONFIG
# ----------------------
MODEL_NAME = "Qwen/Qwen2-Audio-7B"
DATA_DIR = "../data"
INPUT_FILES = {
    "train": "msp_train_20251024.json",
    "dev": "msp_dev_20251024.json",
    "test": "msp_test1_20251024.json",
}
OUTPUT_SUFFIX = "_frameVAD.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def process_dataset(split_name, filename):
    src_path = os.path.join(DATA_DIR, filename)
    out_path = os.path.join(DATA_DIR, split_name + OUTPUT_SUFFIX)

    print(f"\n▶ Processing {split_name}: {src_path}")
    data = load_json(src_path)
    output = []

    for entry in data:
        if not all(k in entry for k in ("valence", "arousal", "audio_path")):
            continue

        wav_path = entry["audio_path"]
        if not os.path.exists(wav_path):
            print(f"⚠ Missing file: {wav_path}")
            continue

        wav, sr = torchaudio.load(wav_path)
        wav = wav.squeeze(0)

        with torch.no_grad():
            feats = extractor(wav, sampling_rate=sr, return_tensors="pt", padding=True)
        n_frames = feats["input_values"].shape[1]  # frame count from feature extractor

        frame_labels = [[entry["valence"], entry["arousal"]]] * n_frames

        output.append({
            "segment_id": entry["segment_id"],
            "audio_path": wav_path,
            "frame_labels": frame_labels
        })

    save_json(out_path, output)
    print(f"Saved frame VAD dataset: {out_path} ({len(output)} segments)")


if __name__ == "__main__":
    for split, file in INPUT_FILES.items():
        process_dataset(split, file)
    print("\nFrame-level VAD extraction complete!")


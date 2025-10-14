import os
import json
from pathlib import Path
from datasets import load_dataset
from transformers import pipeline, AutoProcessor, AutoModelForMultimodal
import torchaudio
import numpy as np

# ---------------------------
# Configuration
# ---------------------------
DATA_DIR = Path.home() / "datasets/seamless_interaction/improvised/dev"
OUTPUT_DIR = Path.home() / "media/volume/data"
PROMPT_FILE = Path("prompts/af3_prompt.txt")

# HF model to cache
MODEL_NAME = "nvidia/audio-flamingo-3"

# ---------------------------
# Load prompt
# ---------------------------
with open(PROMPT_FILE, "r") as f:
    PROMPT_TEXT = f.read()

# ---------------------------
# Load model and processor
# ---------------------------
print("Loading Flamingo-3 model...")
processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=OUTPUT_DIR)
model = AutoModelForMultimodal.from_pretrained(MODEL_NAME, cache_dir=OUTPUT_DIR)
pipe = pipeline(
    task="audio-to-text",
    model=model,
    feature_extractor=processor,
    tokenizer=processor,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# ---------------------------
# Helper: load speaker segments
# ---------------------------
def load_speaker_segments(json_path):
    """
    Return a list of dicts: [{"speaker": ..., "start": ..., "end": ..., "text": ...}, ...]
    Assumes JSON has 'segments' with speaker turn info.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    
    segments = []
    for seg in data.get("segments", []):
        speaker = seg.get("speaker", "unknown")
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        text = seg.get("text", "")
        segments.append({"speaker": speaker, "start": start, "end": end, "text": text})
    return segments

# ---------------------------
# Helper: extract audio segment
# ---------------------------
def extract_audio_segment(wav_path, start, end):
    waveform, sr = torchaudio.load(wav_path)
    start_frame = int(start * sr)
    end_frame = int(end * sr)
    return waveform[:, start_frame:end_frame]

# ---------------------------
# Main processing loop
# ---------------------------
for file_prefix in sorted(DATA_DIR.glob("*/*.wav")):
    wav_path = file_prefix
    json_path = wav_path.with_suffix(".json")
    
    if not json_path.exists():
        print(f"Skipping {wav_path}, no JSON found")
        continue

    segments = load_speaker_segments(json_path)
    results = []

    for seg in segments:
        audio_segment = extract_audio_segment(wav_path, seg["start"], seg["end"])
        
        # Convert waveform to float32 and CPU if needed
        audio_segment = audio_segment.float().cpu()
        
        # Run Flamingo-3 audio reasoning
        try:
            output = pipe({
                "text": PROMPT_TEXT,
                "audio": audio_segment
            })
            # Pipe output may be string -> parse as JSON
            output_json = json.loads(output[0]["generated_text"])
            results.append({
                "speaker": seg["speaker"],
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "analysis": output_json
            })
        except Exception as e:
            print(f"Error processing segment {seg}: {e}")
            continue

    # Save results
    out_path = OUTPUT_DIR / f"{wav_path.stem}_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved analysis to {out_path}")

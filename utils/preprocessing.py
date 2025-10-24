#!/usr/bin/env python3
"""
Preprocess MSP-Podcast using consensus labels (single ground truth per segment).
Output JSON: msp_podcast_preprocessed.json
"""

import os
import json

# ----------------------------------------------------------
# File paths — edit BASE_DIR if needed
# ----------------------------------------------------------
BASE_DIR = "/media/volume/data/MSP-PODCAST-Publish-1.12"
AUDIO_DIR = os.path.join(BASE_DIR, "Audios")
TRANSCRIPT_DIR = os.path.join(BASE_DIR, "Transcripts")
LABELS_FILE = os.path.join(BASE_DIR, "Labels", "labels_consensus.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "msp_podcast_preprocessed.json")

# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def read_transcript(seg_id):
    txt_path = os.path.join(TRANSCRIPT_DIR, f"{seg_id}.txt")
    if os.path.exists(txt_path):
        return open(txt_path, "r", encoding="utf-8").read().strip()
    return ""


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    print("🔄 Loading consensus annotations...")
    labels_dict = load_json(LABELS_FILE)

    dataset = []

    for wav_name, info in labels_dict.items():
        seg_id = wav_name.replace(".wav", "")
        audio_path = os.path.join(AUDIO_DIR, wav_name)

        if not os.path.exists(audio_path):
            continue  # Some have labels but no released audio

        dataset.append({
            "segment_id": seg_id,
            "partition": info.get("Split_Set"),
            "audio_path": audio_path,
            "transcript": read_transcript(seg_id),
            "speaker_id": info.get("SpkrID"),
            "gender": info.get("Gender"),
            "emotion_discrete": info.get("EmoClass"),
            "valence": float(info.get("EmoVal", 0)),
            "arousal": float(info.get("EmoAct", 0)),
            "dominance": float(info.get("EmoDom", 0))
        })

    print(f"Final dataset size: {len(dataset)}")
    print(f"Writing → {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print("✅ Done!.")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Preprocess MSP-Podcast data into a unified JSON file for multimodal reasoning.
Creates msp_podcast_preprocessed.json
"""

import os
import csv
import json
from glob import glob

# ----------------------------------------------------------------------
# File paths (edit if necessary)
# ----------------------------------------------------------------------
BASE_DIR = "media/volume/data/MSP-PODCAST-Publish-1.12/"
AUDIO_DIR = os.path.join(BASE_DIR, "Audios")
LABELS_FILE = os.path.join(BASE_DIR, "Labels", "labels_consensus.csv")
TRANSCRIPT_DIR = os.path.join(BASE_DIR, "Transcripts")
ALIGN_DIR = os.path.join(BASE_DIR, "ForceAligned")
SPEAKER_FILE = os.path.join(BASE_DIR, "Speaker_ids.txt")
PARTITION_FILE = os.path.join(BASE_DIR, "Partition.txt")
OUTPUT_FILE = os.path.join(BASE_DIR, "msp_podcast_preprocessed.json")

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def read_partition_file(path):
    partitions = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                seg_id, split = parts
                partitions[seg_id] = split
    return partitions

def read_speaker_file(path):
    speakers = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                spk_id = parts[0]
                info = parts[1:]
                speakers[spk_id] = {"meta": info}
    return speakers

def read_labels_csv(path):
    labels = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seg_id = row["FileName"].split(".")[0]
            labels[seg_id] = {
                "emotion_discrete": row.get("Emotion", ""),
                "emotion_continuous": {
                    "valence": float(row.get("Valence", 0)),
                    "arousal": float(row.get("Arousal", 0)),
                    "dominance": float(row.get("Dominance", 0)),
                }
            }
    return labels

def read_transcript(seg_id):
    path = os.path.join(TRANSCRIPT_DIR, f"{seg_id}.txt")
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def read_alignment(seg_id):
    align_path = os.path.join(ALIGN_DIR, f"{seg_id}.csv")
    if not os.path.exists(align_path):
        return []
    alignment = []
    with open(align_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            alignment.append({
                "word": row.get("word"),
                "start": float(row.get("start")),
                "end": float(row.get("end"))
            })
    return alignment

# ----------------------------------------------------------------------
# Main merging logic
# ----------------------------------------------------------------------
def main():
    partitions = read_partition_file(PARTITION_FILE)
    speakers = read_speaker_file(SPEAKER_FILE)
    labels = read_labels_csv(LABELS_FILE)
    
    data = []
    for seg_id, split in partitions.items():
        audio_path = os.path.join(AUDIO_DIR, f"{seg_id}.wav")
        if not os.path.exists(audio_path):
            continue
        
        entry = {
            "segment_id": seg_id,
            "partition": split,
            "audio_path": audio_path,
            "transcript": read_transcript(seg_id),
            "alignment": read_alignment(seg_id),
        }
        
        # Merge in emotion info
        if seg_id in labels:
            entry.update(labels[seg_id])
        else:
            entry["emotion_discrete"] = None
            entry["emotion_continuous"] = None
        
        # Speaker info (if extractable from ID)
        spk_id = seg_id.split("_")[0]
        entry["speaker_id"] = spk_id
        entry["speaker_meta"] = speakers.get(spk_id, {})
        
        data.append(entry)
    
    print(f"Writing {len(data)} entries to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()


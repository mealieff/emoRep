#!/usr/bin/env python3
import json
from pathlib import Path
from datetime import datetime

# ----------------------------------------------------
# Config
# ----------------------------------------------------
INPUT_FILE = "msp_podcast_preprocessed.json"
OUTPUT_DIR = Path("../output")
OUTPUT_DIR.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d")
SUMMARY_FILE = OUTPUT_DIR / f"split_summary_{timestamp}.txt"

# ----------------------------------------------------
def main():
    print("Loading merged dataset...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    train, dev, test1 = [], [], []

    for entry in data:
        split = entry.get("partition", "").lower()
        if split == "train":
            train.append(entry)
        elif split == "development" or split == "dev":
            dev.append(entry)
        elif split == "test1":
            test1.append(entry)

    # Save output files
    train_file = OUTPUT_DIR / f"msp_train_{timestamp}.json"
    dev_file = OUTPUT_DIR / f"msp_dev_{timestamp}.json"
    test1_file = OUTPUT_DIR / f"msp_test1_{timestamp}.json"

    print(f"Writing Train → {train_file}")
    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(train, f, indent=2, ensure_ascii=False)

    print(f"Writing Dev → {dev_file}")
    with open(dev_file, "w", encoding="utf-8") as f:
        json.dump(dev, f, indent=2, ensure_ascii=False)

    print(f"Writing Test1 → {test1_file}")
    with open(test1_file, "w", encoding="utf-8") as f:
        json.dump(test1, f, indent=2, ensure_ascii=False)

    # Summary report
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write("MSP Podcast Split Summary\n")
        f.write("========================\n")
        f.write(f"Train: {len(train)}\n")
        f.write(f"Dev: {len(dev)}\n")
        f.write(f"Test1: {len(test1)}\n")

    print("\n✅ Done!")
    print(f"Summary written to: {SUMMARY_FILE}")

# ----------------------------------------------------
if __name__ == "__main__":
    main()


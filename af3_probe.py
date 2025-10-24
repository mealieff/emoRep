import os
import json
import torch
import librosa
import soundfile as sf
from pathlib import Path
import argparse
from predict import Predictor  # Your AF3 Cog predictor

class SegmentAndAnalyze:
    def __init__(self):
        self.predictor = None

    def setup(self):
        """Initialize the AF3 Predictor."""
        self.predictor = Predictor()
        self.predictor.setup()

    def segment_audio(self, audio_path: str, silence_thresh: float = 40.0, min_silence_len: int = 1000):
        """Segment audio into non-silent intervals using librosa."""
        y, sr = librosa.load(audio_path, sr=None)
        intervals = librosa.effects.split(y, top_db=silence_thresh, frame_length=min_silence_len)
        segments = []
        for start, end in intervals:
            segment = y[start:end]
            segment_path = f"{Path(audio_path).stem}_segment_{start}_{end}.wav"
            sf.write(segment_path, segment, sr)
            segments.append(segment_path)
        return segments

    def safe_predict(self, audio_path: str, prompt: str, **kwargs):
        """Call AF3 Predictor safely with plain Python arguments."""
        return self.predictor.predict(
            audio=audio_path,
            prompt=prompt,
            start_time=kwargs.get("start_time", None),
            end_time=kwargs.get("end_time", None),
            enable_thinking=kwargs.get("enable_thinking", False),
            temperature=kwargs.get("temperature", 0.0),
            max_length=kwargs.get("max_length", 0),
            system_prompt=kwargs.get("system_prompt", "")
        )

    def analyze_segments(self, segments: list, prompt: str):
        """Analyze each audio segment using AF3."""
        analyses = []
        for segment_path in segments:
            analysis = self.safe_predict(audio_path=segment_path, prompt=prompt)
            analyses.append({"segment": segment_path, "analysis": analysis})
        return analyses

    def predict(self, audio: str, prompt: str):
        """Segment the audio and run AF3 analysis on each segment."""
        segments = self.segment_audio(audio)
        analyses = self.analyze_segments(segments, prompt)
        return json.dumps(analyses, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Segment audio and analyze with Audio Flamingo 3")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file")
    parser.add_argument("--prompt", type=str, required=True, help="Path to prompt text file")
    args = parser.parse_args()

    # Load prompt text
    with open(args.prompt, "r") as f:
        prompt_text = f.read()

    analyzer = SegmentAndAnalyze()
    analyzer.setup()
    result = analyzer.predict(audio=args.audio, prompt=prompt_text)

    # Save results
    output_path = f"{Path(args.audio).stem}_analysis.json"
    with open(output_path, "w") as f:
        f.write(result)
    print(f"Saved analysis to {output_path}")


if __name__ == "__main__":
    main()


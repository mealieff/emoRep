import torch, json
import soundfile as sf
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForSeq2SeqLM

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B")
model = AutoModelForSeq2SeqLM.from_pretrained("Qwen/Qwen2-Audio-7B")

dataset_name = "iemocap"

# Load first 20 examples
dataset = load_dataset(dataset_name, split="train[:20]")

def run_qwen_analysis(audio, transcript, file_id):
    inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").to(model.device)

    def ask_qwen(prompt):
        generated_ids = model.generate(**inputs, max_new_tokens=512, decoder_input_ids=processor.tokenizer(prompt, return_tensors="pt").input_ids.to(model.device))
        return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Prompts
    fast_half_prompt = "What do you hear in the fast half (second half) of the audio?"
    prompt_bme = "Summarize what you hear at beginning, middle, and end of the audio. Respond as JSON with keys beginning, middle, end."
    prompt_shift = "Mark the beginning of an emotional shift in the transcript. Return transcript span and approximate second."
    prompt_intensity = 'Mark the beginning and end of intensity change. Format: {"start_time": X, "end_time": Y, "span": "..."}'
    prompt_expressive = "Mark the emotionally expressive span in transcript with [emote/] ... [/emote]. Return only the modified transcript."

    return {
        "file": file_id,
        "transcript": transcript,
        "fast_half": ask_qwen(fast_half_prompt),
        "begin_mid_end": ask_qwen(prompt_bme),
        "emotional_shift": ask_qwen(prompt_shift),
        "intensity_span": ask_qwen(prompt_intensity),
        "expressive_span": ask_qwen(prompt_expressive),
    }

results = []

for i, ex in enumerate(dataset):
    audio = ex["audio"]
    transcript = ex.get("transcription") or ex.get("text") or ""
    result = run_qwen_analysis(audio, transcript, f"sample_{i}")
    results.append(result)

with open("span_detection_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("✅ Saved results to span_detection_results.json"

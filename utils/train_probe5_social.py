# train_probe5_social.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from model_setup import *

MODEL_NAME = "Qwen-Audio"
DATA_PATH = "msp_podcast_preprocessed.json"
PROBE = "probe5_social"

data = [d for d in load_json_dataset(DATA_PATH) if "social_label" in d]
texts = [d["dialogue_context"] for d in data]
labels = [d["social_label"] for d in data]
num_labels = len(set(labels))

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-5)
ce = torch.nn.CrossEntropyLoss()

for e in range(3):
    for i in range(0, len(texts), 4):
        toks = tokenizer(texts[i:i+4], padding=True, truncation=True, return_tensors="pt").to(device)
        y = torch.tensor(labels[i:i+4]).to(device)
        out = model(**toks, labels=y)
        out.loss.backward(); opt.step(); opt.zero_grad()
    print(f"Epoch {e+1}: {out.loss.item():.4f}")

pred = model(**toks).logits.argmax(-1)
metrics = eval_classification(pred, y, num_labels)
save_outputs(PROBE, {"FinalLoss": out.loss.item(), **metrics})


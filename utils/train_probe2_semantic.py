# train_probe2_semantic.py
import torch
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from model_setup import *

MODEL_NAME = "roberta-base"
DATA_PATH = "msp_podcast_preprocessed.json"
PROBE = "probe2_semantic"

data = [d for d in load_json_dataset(DATA_PATH) if d["emotion_discrete"]]
texts = [d["transcript"] for d in data]
labels = [d["emotion_discrete"] for d in data]

le = LabelEncoder().fit(labels)
num_labels = len(le.classes_)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-5)
ce = torch.nn.CrossEntropyLoss()

for epoch in range(3):
    for i in range(0, len(texts), 8):
        toks = tokenizer(texts[i:i+8], padding=True, truncation=True, return_tensors="pt").to(device)
        y = torch.tensor(le.transform(labels[i:i+8])).to(device)
        out = model(**toks, labels=y)
        out.loss.backward(); opt.step(); opt.zero_grad()
    print(f"Epoch {epoch+1}: {out.loss.item():.4f}")

pred = model(**toks).logits.argmax(-1)
metrics = eval_classification(pred, y, num_labels)
save_outputs(PROBE, {"FinalLoss": out.loss.item(), **metrics})


# train_probe3_temporal.py
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel
from model_setup import *

MODEL_NAME = "Qwen-Audio"
DATA_PATH = "msp_podcast_preprocessed.json"
PROBE = "probe3_temporal"

class TemporalDataset(Dataset):
    def __init__(self):
        self.data = [d for d in load_json_dataset(DATA_PATH) if d["alignment"]]
    def __getitem__(self, i):
        d = self.data[i]
        x = torch.tensor(d["emotion_continuous"])
        y = torch.tensor(d["alignment"]["span_boundaries"])  # [start, end]
        return x, y
    def __len__(self): return len(self.data)

class SpanRegressor(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.enc = AutoModel.from_pretrained(MODEL_NAME)
        self.reg = nn.Linear(dim, 2)
    def forward(self, x): return self.reg(self.enc(x).last_hidden_state.mean(1))

model = SpanRegressor().to(device)
opt = optim.Adam(model.parameters(), lr=2e-5)
loss_fn = nn.SmoothL1Loss()
dl = DataLoader(TemporalDataset(), batch_size=4, shuffle=True)

for e in range(3):
    for feat, tgt in dl:
        pred = model(feat.to(device))
        loss = loss_fn(pred, tgt.to(device))
        loss.backward(); opt.step(); opt.zero_grad()
    print(f"Epoch {e+1}: {loss.item():.4f}")

metrics = eval_regression(pred, tgt.to(device))
save_outputs(PROBE, {"FinalLoss": loss.item(), **metrics})


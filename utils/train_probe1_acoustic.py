# train_probe1_acoustic.py
import torch, torchaudio
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoFeatureExtractor, AutoModel
from model_setup import *

MODEL_NAME = "Qwen-Audio"
DATA_PATH = "msp_podcast_preprocessed.json"
PROBE = "probe1_acoustic"

class AcousticDataset(Dataset):
    def __init__(self):
        self.data = [d for d in load_json_dataset(DATA_PATH) if d["emotion_continuous"]]
    def __getitem__(self, i):
        d = self.data[i]
        wav, sr = torchaudio.load(d["audio_path"])
        return wav.squeeze(0), torch.tensor(d["emotion_continuous"])
    def __len__(self): return len(self.data)

extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
regressor = nn.Linear(model.config.hidden_size, 2).to(device)
opt = optim.Adam(list(model.parameters()) + list(regressor.parameters()), lr=1e-5)
loss_fn = nn.MSELoss()

dl = DataLoader(AcousticDataset(), batch_size=4, shuffle=True)
for epoch in range(3):
    for wav, label in dl:
        feats = extractor(wav, return_tensors="pt", sampling_rate=16000, padding=True).to(device)
        with torch.no_grad(): enc = model(**feats).last_hidden_state.mean(1)
        pred = regressor(enc)
        loss = loss_fn(pred, label.to(device))
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"Epoch {epoch+1}: {loss.item():.4f}")

metrics = eval_regression(pred, label.to(device))
save_outputs(PROBE, {"FinalLoss": loss.item(), **metrics})


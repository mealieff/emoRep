# train_probe4_affective.py
import torch, torchaudio
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, AutoFeatureExtractor
from model_setup import *

AUDIO_MODEL = "Qwen-Audio"
TEXT_MODEL = "roberta-base"
DATA_PATH = "msp_podcast_preprocessed.json"
PROBE = "probe4_affective"

class AffectiveDataset(Dataset):
    def __init__(self):
        self.data = [d for d in load_json_dataset(DATA_PATH) if d["emotion_continuous"]]
    def __getitem__(self, i):
        d = self.data[i]
        wav, sr = torchaudio.load(d["audio_path"])
        return wav, d["transcript"], torch.tensor(d["emotion_continuous"])
    def __len__(self): return len(self.data)

ae = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL)
am = AutoModel.from_pretrained(AUDIO_MODEL).to(device)
tt = AutoTokenizer.from_pretrained(TEXT_MODEL)
tm = AutoModel.from_pretrained(TEXT_MODEL).to(device)

class Fusion(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.cross = nn.TransformerEncoderLayer(d_model=dim, nhead=8)
        self.out = nn.Linear(dim, 2)
    def forward(self, a, t): return self.out(self.cross(a + t))

fusion = Fusion().to(device)
opt = optim.Adam(fusion.parameters(), lr=1e-5)
loss_fn = nn.MSELoss()
dl = DataLoader(AffectiveDataset(), batch_size=2, shuffle=True)

for e in range(3):
    for wav, txt, target in dl:
        af = ae(wav.squeeze(1), return_tensors="pt", sampling_rate=16000, padding=True).to(device)
        tf = tt(txt, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            ae_out = am(**af).last_hidden_state.mean(1)
            te_out = tm(**tf).last_hidden_state.mean(1)
        pred = fusion(ae_out, te_out)
        loss = loss_fn(pred, target.to(device))
        loss.backward(); opt.step(); opt.zero_grad()
    print(f"Epoch {e+1}: {loss.item():.4f}")

metrics = eval_regression(pred, target.to(device))
save_outputs(PROBE, {"FinalLoss": loss.item(), **metrics})


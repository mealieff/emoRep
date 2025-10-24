# model_setup.py
import os, json, torch, datetime
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoFeatureExtractor,
)
from torchmetrics import ConcordanceCorrCoef, PearsonCorrCoef, Accuracy, F1Score

device = "cuda" if torch.cuda.is_available() else "cpu"

# ======== Shared Utilities ========

def timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def save_outputs(probe_name, results_dict):
    """Save evaluation summary to ../output/"""
    os.makedirs("../output", exist_ok=True)
    base_name = f"{probe_name}_{timestamp()}"
    json_path = f"../output/{base_name}.json"
    txt_path = f"../output/{base_name}.txt"
    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=4)
    with open(txt_path, "w") as f:
        for k, v in results_dict.items():
            f.write(f"{k}: {v}\n")
    print(f"✅ Saved {probe_name} results to {json_path} and {txt_path}")

def load_json_dataset(path):
    with open(path) as f: return json.load(f)

# ======== Metric Helpers ========

def eval_regression(pred, gold):
    ccc = ConcordanceCorrCoef().to(device)
    r = PearsonCorrCoef().to(device)
    return {"CCC": float(ccc(pred, gold)), "Pearson_r": float(r(pred, gold))}

def eval_classification(pred, gold, num_labels):
    acc = Accuracy(task="multiclass", num_classes=num_labels).to(device)
    f1 = F1Score(task="multiclass", num_classes=num_labels).to(device)
    return {"Accuracy": float(acc(pred, gold)), "F1": float(f1(pred, gold))}


from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
import torch

MODEL_PATH = "models/phobert_sentiment"

def load_model(use_cuda=True):
    device = torch.device(
        "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        use_fast=False
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )
    model.to(device)
    model.eval()

    return tokenizer, model, device

import argparse
import json
import sys
import warnings

from src.model_loader import load_model
from src.predictor import predict_sentiment


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--use-pyvi", action="store_true")
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=Warning)

    tokenizer, model, device = load_model(use_cuda=args.use_cuda)
    result = predict_sentiment(
        args.text,
        model,
        tokenizer,
        device,
        use_pyvi=args.use_pyvi,
    )
    result["device"] = str(device)

    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

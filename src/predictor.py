import torch

from src.preprocessing import preprocess_text

id2label = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

def predict_sentiment(
    text,
    model,
    tokenizer,
    device,
    use_pyvi=True
):

    text = preprocess_text(text, use_pyvi=use_pyvi)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=48
    )
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

        probs = torch.softmax(
            outputs.logits,
            dim=1
        )

        pred_id = torch.argmax(
            probs,
            dim=1
        ).item()

    return {
        "label": id2label[pred_id],
        "probs": probs[0].tolist()
    }

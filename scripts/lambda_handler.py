#!/usr/bin/env python3
"""
BiasGuard - AWS Lambda inference handler

Deploy with: package model + this script + dependencies into a Lambda layer.
Expects MODEL_PATH env or /opt/ml/model/ for SageMaker endpoint.
"""
import json
import os


def load_model():
    """Load model once (Lambda reuses container)."""
    if "model" not in load_model.__dict__:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        path = os.environ.get("MODEL_PATH", "/opt/ml/model")
        load_model.model = AutoModelForSequenceClassification.from_pretrained(path)
        load_model.tokenizer = AutoTokenizer.from_pretrained(path)
        load_model.model.eval()
        load_model.device = "cuda" if torch.cuda.is_available() else "cpu"
        load_model.model.to(load_model.device)
    return load_model.model, load_model.tokenizer, load_model.device


def lambda_handler(event, context):
    """Handle API Gateway or direct invoke. Expects: {"text": "..."} or {"body": "{\"text\":\"...\"}"}."""
    try:
        if isinstance(event.get("body"), str):
            body = json.loads(event["body"])
        else:
            body = event
        text = body.get("text", "")
        if not text:
            return {"statusCode": 400, "body": json.dumps({"error": "Missing text"})}

        model, tokenizer, device = load_model()
        enc = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        import torch
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0].tolist()
        pred = int(logits.argmax(dim=-1).cpu().item())
        label = "biased" if pred == 1 else "neutral"

        return {
            "statusCode": 200,
            "body": json.dumps({
                "label": label,
                "prediction": pred,
                "probabilities": probs,
            }),
        }
    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}

#!/usr/bin/env python3
"""
BiasGuard - Real-time inference API server

Usage: python -m scripts.api_server --model outputs/biasguard --port 8000
"""
import argparse
from pathlib import Path

# Lazy imports to keep startup fast
_model = _tokenizer = _device = None


def get_model(model_path: str):
    global _model, _tokenizer, _device
    if _model is None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        _tokenizer = AutoTokenizer.from_pretrained(path)
        try:
            from peft import PeftModel
            adapter_cfg = path / "adapter_config.json"
            if adapter_cfg.exists():
                import json
                cfg = json.loads(adapter_cfg.read_text())
                base = cfg.get("base_model_name_or_path", str(path))
                _model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=2)
                _model = PeftModel.from_pretrained(_model, path)
            else:
                _model = AutoModelForSequenceClassification.from_pretrained(path)
        except Exception:
            _model = AutoModelForSequenceClassification.from_pretrained(path)
        _model.eval()
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _model.to(_device)
    return _model, _tokenizer, _device


def predict(text: str, model_path: str) -> dict:
    model, tokenizer, device = get_model(model_path)
    import torch
    enc = tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0].tolist()
    pred = int(logits.argmax(dim=-1).cpu().item())
    return {"label": "biased" if pred == 1 else "neutral", "prediction": pred, "probabilities": probs}


def create_app(model_path: str):
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
    except ImportError:
        raise ImportError("Install fastapi and uvicorn: pip install fastapi uvicorn")

    app = FastAPI(title="BiasGuard API", version="1.0")

    class Request(BaseModel):
        text: str

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/predict")
    def predict_endpoint(req: Request):
        return predict(req.text, model_path)

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="outputs/biasguard")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    app = create_app(args.model)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

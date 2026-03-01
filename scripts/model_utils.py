"""
Model utilities for BiasGuard - LLaMA2 seq classification + LoRA support.
"""
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput


def _is_llama_like(model_name: str) -> bool:
    """Check if model is decoder-only (LLaMA, Mistral, Phi, etc.)."""
    name_lower = model_name.lower()
    return any(x in name_lower for x in ["llama", "mistral", "phi", "tinyllama", "qwen"])


def create_model_and_tokenizer(model_name: str, num_labels: int = 2, use_lora: bool = True):
    """Create model + tokenizer. Uses LoRA for LLaMA-like models."""
    if _is_llama_like(model_name):
        return _create_llama_classifier(model_name, num_labels, use_lora)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return model, tokenizer


def _create_llama_classifier(model_name: str, num_labels: int, use_lora: bool):
    """Create LLaMA-based seq classifier with optional LoRA."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
    except (Exception, TypeError, ValueError):
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = num_labels
        base = AutoModel.from_pretrained(model_name)
        model = _LlamaSeqClsWrapper(base, config)

    if use_lora:
        try:
            from peft import LoraConfig, get_peft_model, TaskType

            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj"],
                bias="none",
            )
            model = get_peft_model(model, peft_config)
        except ImportError:
            pass  # Train full model if peft not installed

    return model, tokenizer


class _LlamaSeqClsWrapper(nn.Module):
    """Wrapper to add classification head to LLaMA base."""

    def __init__(self, base_model, config):
        super().__init__()
        self.model = base_model
        self.score = nn.Linear(config.hidden_size, config.num_labels)
        self.num_labels = config.num_labels

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.model(input_ids, attention_mask=attention_mask, **kwargs)
        hidden = outputs.last_hidden_state[:, -1, :]
        logits = self.score(hidden)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)

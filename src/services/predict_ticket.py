
import torch

from transformers import AutoTokenizer
from src.models.multitask_model import EthanMultiTaskBERT

from src.config.paths import MODEL_RESOLVER,MODEL_BASE
from src.config.logger import get_logger


logger = get_logger("predict_ticket")

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and multitask model once
tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE,use_fast=True)

model = EthanMultiTaskBERT.from_pretrained(MODEL_RESOLVER)
model.to(device)
model.eval()


def predict_ticket_trained(description: str):
    try:
        # Tokenize input
        inputs = tokenizer(
            description,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        # Run both classification heads
        with torch.no_grad():
            resolver_logits, priority_logits = model(**inputs)

        # Resolver prediction
        resolver_probs = torch.softmax(resolver_logits, dim=-1)
        resolver_id = resolver_probs.argmax(dim=-1).item()

        # Priority prediction
        priority_probs = torch.softmax(priority_logits, dim=-1)
        priority_id = priority_probs.argmax(dim=-1).item()

        return {
            "resolver_group": model.config.resolver_id2label[str(resolver_id)],
            "resolver_confidence": round(
                resolver_probs[0, resolver_id].item(), 4
            ),
            "priority": model.config.priority_id2label[str(priority_id)],
            "priority_confidence": round(
                priority_probs[0, priority_id].item(), 4
            )
        }

    except Exception as e:
        logger.error(f"predict_ticket_trained: {e}")
        raise

import json
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification,pipeline
from src.config.paths import MODEL_RESOLVER,MODEL_UNTRAINED
from src.config.logger import get_logger

logger = get_logger("predict_ticket")


tokenizer = AutoTokenizer.from_pretrained(MODEL_UNTRAINED)
untrained_model = AutoModelForSequenceClassification.from_pretrained(MODEL_UNTRAINED)
classifier = pipeline("text-classification",model=MODEL_UNTRAINED)

## Original Model
def predict_ticket_untrained(description: str):
    try:

        prediction = classifier(description)[0]

        return {
            "resolver_group": prediction["label"],
            "confidence": round(prediction["score"], 4)
        }

    except Exception as e:
        logger.error(f"predict_ticket_untrained: {e}")
        raise

## Resolver team trained Model
def predict_ticket_trained(description: str):
    try:
        trained_tokenizer = AutoTokenizer.from_pretrained(MODEL_RESOLVER)

        trained_model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_RESOLVER
        )

        trained_classifier = pipeline(
            "text-classification",
            model=trained_model,
            tokenizer=trained_tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )

        prediction = trained_classifier(description)[0]

        return {
            "resolver_group": prediction["label"],
            "confidence": round(prediction["score"], 4)
        }

    except Exception as e:
        logger.error(f"predict_ticket_trained: {e}")
        raise
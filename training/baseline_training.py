import pandas as pd

from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,Trainer, TrainingArguments)

from src.config.paths import CHAT_DATA,GEM_DATA
from src.config.logger import get_logger

logger = get_logger("baseline_training.py")

def train_model(dataset_dir):
    try:
        logger.info("Training started")

        df = pd.read_csv(dataset_dir) # text, resolver_group, priority, parent_incident_id, client_id, caller_name
        logger.info(df.sample(20))
        labels = sorted(df.resolver_group.unique())
        df["y"] = df.resolver_group.map({l: i for i, l in enumerate(labels)})
        train, test = train_test_split(df, test_size=0.2)
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        # def enc(d): return Dataset.from_pandas(d).map(
        # lambda b: tok(b["text"], truncation=True, max_length=32, padding="max_length"), batched=True)

        def enc(d): return Dataset.from_dict({"text": d["text"].astype(str).tolist(),"labels": d["y"].astype(int).tolist()
            }).map(lambda b: tok(b["text"],truncation=True,max_length=64,padding="max_length"),batched=True).remove_columns(["text"])

        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(labels))
        args = TrainingArguments("out", num_train_epochs=10, learning_rate=5e-3, eval_strategy="epoch",
        save_strategy="epoch", load_best_model_at_end=True)
        trainer = Trainer(model=model, args=args, train_dataset=enc(train), eval_dataset=enc(test),
        compute_metrics=lambda p: {"accuracy": (p.predictions.argmax(-1) == p.label_ids).mean()})
        trainer.train()
        logger.info("Test accuracy:", trainer.evaluate()["eval_accuracy"])
        trainer.save_model("prod_model")

        logger.info("Training ended")


    except Exception as e:
        logger.error(f"train_model: {type(e).__name__}: {e}")
        print(f"train_model: {type(e).__name__}: {e}")
        raise

if __name__ == "__main__":
    train_model(CHAT_DATA)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments
)
from collections import Counter
import pandas as pd
from datasets import DatasetDict
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch

class ModelTrainer:
    
    @staticmethod
    def compute_metrics(eval_pred):
        """Compute accuracy and F1 score"""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, average="weighted")
        }
    
    @staticmethod
    def inspect_dataset(tokenized_ds, split="train", tokenizer=None, n_samples=5):
        """
        Inspect a tokenized dataset:
        - First few rows
        - Label distribution
        """
        dataset = tokenized_ds[split]

        # Show first few samples
        print(f"\n📦 First {n_samples} samples in '{split}' set:")
        for i in range(n_samples):
            sample = dataset[i]
            tokens = sample["input_ids"]
            if tokenizer:
                decoded = tokenizer.decode(tokens, skip_special_tokens=True)
            else:
                decoded = tokens
            print(f"\nExample {i+1}:")
            print(f"Text: {decoded[:200]}{'...' if len(decoded) > 200 else ''}")
            print(f"Label: {sample['labels']}")

        # Label distribution
        labels = [x["labels"] for x in dataset]
        label_counts = Counter(labels)
        total = sum(label_counts.values())
        print("\n📊 Label distribution:")
        for label, count in label_counts.items():
            print(f"  Label {label}: {count} samples ({count / total:.2%})")


    @staticmethod
    def train(
        model_name: str,
        dataset: DatasetDict,
        inspect: bool = False,
        num_labels: int =2,
        **training_kwargs
    ) -> dict:
        """Complete training pipeline"""

        # Initialize model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )

        max_length = training_kwargs.pop("max_length", 128)

        # Tokenization
        def tokenize(batch):
            return tokenizer(
                batch["text"],
                padding="max_length",
                truncation=True,
                max_length=max_length
            )
        
        tokenized_ds = dataset.map(tokenize, batched=True)
        tokenized_ds = tokenized_ds.rename_column("label", "labels")

        if inspect:
            ModelTrainer.inspect_dataset(tokenized_ds, split="train", tokenizer=tokenizer)

        # Training setup
        args = TrainingArguments(
            output_dir=f"./results/{model_name.replace('/', '-')}",
            per_device_train_batch_size=training_kwargs.pop("batch_size", 16),
            per_device_eval_batch_size=training_kwargs.pop("batch_size", 16),
            num_train_epochs=training_kwargs.pop("epochs", 1),
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_dir="./logs",
            logging_steps=1000,
            learning_rate=training_kwargs.pop("learning_rate", 2e-5),
            weight_decay=training_kwargs.pop("weight_decay", 0.01),
            load_best_model_at_end=True,
            report_to="none",
            optim="adamw_torch",  # Better optimizer
            fp16=torch.cuda.is_available(),  # Auto GPU
            **training_kwargs
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized_ds["train"],
            eval_dataset=tokenized_ds["test"],
            compute_metrics=ModelTrainer.compute_metrics,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        )

        # Training and evaluation
        trainer.train()
        metrics = trainer.evaluate()
        
        return metrics
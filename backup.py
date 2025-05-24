from transformers import BertTokenizer, AutoTokenizer, AutoModelForSequenceClassification, BertForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict, Dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import evaluate


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted")
    }


def train_model(model_name, dataset, num_labels=None, epochs=1, batch_size=16):
    """
    Args:
        model_name: HF model path or custom model
        dataset: Must have 'train' and 'test' splits
        num_labels: Auto-detected if None
        epochs: Training epochs
        batch_size: Per-device batch size
    Returns:
        metrics: Dictionary of evaluation metrics
        trainer: Trainer object for further analysis
    """
    # Auto-detect num_labels if not specified
    if num_labels is None:
        num_labels = len(set(dataset["train"]["label"]))

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )

    # Tokenization with proper text/label columns
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.rename_column("label", "labels")  # HF expects 'labels'

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"./results/{model_name.replace('/', '-')}",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=100,
        load_best_model_at_end=True,
        report_to="none",
        optim="adamw_torch",  # Better optimizer
        fp16=True,  # Enable mixed precision if GPU available
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    # Training and evaluation
    trainer.train()
    metrics = trainer.evaluate()

    return metrics, trainer  # Return both metrics and trainer object


def flatten_dual_source_dataset(dataset_split, human_field, ai_field, human_label=0, ai_label=1):
    texts = []
    labels = []

    for example in dataset_split:
        for text in example.get(human_field, []):
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
                labels.append(human_label)
        for text in example.get(ai_field, []):
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
                labels.append(ai_label)

    return Dataset.from_dict({"text": texts, "label": labels})


# Define which fields to flatten for each dataset
flatten_fields = {
    "HC3": ("human_answers", "chatgpt_answers"),
}

# Different datasets
datasets = {
    "HC3": load_dataset("Hello-SimpleAI/HC3", "all"),
    "ai-human-gen": load_dataset("likhithasapu/ai-human-gen")
      .rename_column("response", "text")
      .map(lambda x: {"label": 1 - x["human-generated"]}),  # Flip labels (0→1, 1→0),
    "AI-and-Human-Generated-Text": load_dataset("Ateeqq/AI-and-Human-Generated-Text")
      .rename_column("abstract", "text")
}

prepared_datasets = {}

for name, raw_dataset in datasets.items():
    print(f"\n📦 Processing dataset: {name}")

    if name in flatten_fields:
        human_field, ai_field = flatten_fields[name]
        flattened = flatten_dual_source_dataset(raw_dataset["train"], human_field, ai_field)
        dataset = flattened.train_test_split(test_size=0.2)
    else:
        print(f"⚠️ No flattening logic for '{name}', skipping.")
        prepared_datasets[name] = raw_dataset
        continue

    prepared_datasets[name] = dataset
    print(f"✅ Flattened and split '{name}' — {len(dataset['train'])} train / {len(dataset['test'])} test samples.")

# Models with different architectures
models = {
    "distilbert-base-uncased", # Transformer (Small)
    "huawei-noah/TinyBERT_General_4L_312D", # Transformer (Tiny)
    "roberta-base", # Transformer (Big)
    "microsoft/deberta-v3-base",  # State-of-the-art detector
    "xlm-roberta-base",
    #"facebook/rag-sequence-base",  # Retrieval-augmented (Requires a diff training setup)
    # "unum-cloud/ufo-llama2-13b-r16",  # Quantized LLM
    # "salesforce/stylogan",  # Style-based detection
    # "gpt-4",  # API-based zero-shot
}


# for name, data in datasets.items():
#     print(f"name: {name}\n")
#     print(f"data: {data}\n\n\n")
#     # print(f"\n{name} label distribution:")
#     # print(data["text"].features["label"])
#     # print(data["text"][:3]["label"])  # Show first 3 labels


def show_label_distribution(dataset_dict, dataset_name):
    print(f"\n🔍 {dataset_name} Label Distribution:")
    for split_name, split_data in dataset_dict.items():
        # Extract labels (handles different dataset formats)
        if "label" in split_data.features:
            labels = split_data["label"]
        # elif "human-generated" in split_data.features:
        #     labels = [0 if x else 1 for x in split_data["human-generated"]]  # Flip for consistency
        # else:
        #     labels = []
        
        # Calculate distribution
        if labels:
            ai_percent = 100 * sum(labels) / len(labels)
            print(f"  {split_name}:")
            print(f"    AI (1): {ai_percent:.1f}%")
            print(f"    Human (0): {100 - ai_percent:.1f}%")
            print(f"    First 3 labels: {labels[:3]}")  # Show sample labels
        else:
            print(f"  {split_name}: No labels found")

# Check all datasets
for name, data in prepared_datasets.items():
    show_label_distribution(data, name)

results = {}
for ds_name, dataset in datasets.items():
    results[ds_name] = {}
    for model_name in models:
        metrics, _ = train_model(model_name, prepared_datasets[ds_name], epochs=1)
        results[ds_name][model_name] = metrics
from datasets import load_dataset
from transformers import (
    RobertaTokenizer,
    RobertaForMultipleChoice,
    Trainer,
    TrainingArguments,
)
import numpy as np
import torch
from sklearn.metrics import accuracy_score
import os

# Load dataset
dataset = load_dataset("tau/commonsense_qa")
train_ds = dataset["train"]
val_ds = dataset["validation"]

# Load tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Label map
label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

# Preprocessing
def preprocess(example):
    question = example["question"]
    choices = example["choices"]["text"]
    encodings = tokenizer(
        [(question, c) for c in choices],
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    return {
        "input_ids": [encodings["input_ids"]],
        "attention_mask": [encodings["attention_mask"]],
        "labels": label_map[example["answerKey"]],
    }

train_ds = train_ds.map(preprocess)
val_ds = val_ds.map(preprocess)

# Format for PyTorch
columns = ["input_ids", "attention_mask", "labels"]
train_ds.set_format(type="torch", columns=columns)
val_ds.set_format(type="torch", columns=columns)

# Load model
model = RobertaForMultipleChoice.from_pretrained("roberta-base")

# Training config
training_args = TrainingArguments(
    output_dir="./models/roberta-finetuned",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=1,
)

# Evaluation metric
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

# Final evaluation
metrics = trainer.evaluate()
print(f"📊 Final Validation Accuracy: {metrics['eval_accuracy']:.2%}")

# Save final model
trainer.save_model("models/roberta-finetuned")
tokenizer.save_pretrained("models/roberta-finetuned")

import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    RobertaTokenizer,
    RobertaForMultipleChoice,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score

# Optional: disable wandb prompts
os.environ["WANDB_DISABLED"] = "true"

# Load dataset
dataset = load_dataset("tau/commonsense_qa")
train_ds = dataset["train"]
val_ds = dataset["validation"]

# Load tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

# Preprocessing function
def preprocess(example):
    question = example["question"]
    choices = example["choices"]["text"]

    encoding = tokenizer(
        [(question, choice) for choice in choices],
        padding="max_length",
        truncation=True,
        max_length=128,
    )

    return {
        "input_ids": encoding["input_ids"],              # shape: [5, seq_len]
        "attention_mask": encoding["attention_mask"],    # shape: [5, seq_len]
        "labels": label_map[example["answerKey"]],       # int
    }

# Apply preprocessing
train_ds = train_ds.map(preprocess)
val_ds = val_ds.map(preprocess)

# Set format for PyTorch
train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Load model
model = RobertaForMultipleChoice.from_pretrained("roberta-base")
model.config.num_labels = 5  # Just to be safe

# Training arguments
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

# Metric function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train!
trainer.train()

# Final evaluation
metrics = trainer.evaluate()
print(f"📊 Final Validation Accuracy: {metrics['eval_accuracy']:.2%}")

# Save model
trainer.save_model("models/roberta-finetuned")
tokenizer.save_pretrained("models/roberta-finetuned")

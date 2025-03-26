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

os.environ["WANDB_DISABLED"] = "true"

#  Load and slice dataset
dataset = load_dataset("tau/commonsense_qa")
train_ds = dataset["train"].select(range(6000))      
val_ds = dataset["validation"].select(range(1000))   

#  Setup tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

def preprocess(example):
    question = example["question"]
    choices = example["choices"]["text"]

    encoding = tokenizer(
        [(question, choice) for choice in choices],
        padding="max_length",
        truncation=True,
        max_length=96,  
    )

    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "labels": label_map[example["answerKey"]],
    }

#  Preprocess and format
train_ds = train_ds.map(preprocess)
val_ds = val_ds.map(preprocess)

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

#  Model
model = RobertaForMultipleChoice.from_pretrained("roberta-base")
model.config.num_labels = 5

#  Trainer config
training_args = TrainingArguments(
    output_dir="./models/roberta-fast-v2",
    evaluation_strategy="epoch",
    save_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,  
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    report_to="none",
)

# ✅ Metric
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}

# ✅ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

#  Train
trainer.train()

# Evaluate
metrics = trainer.evaluate()
print(f" Final Validation Accuracy: {metrics['eval_accuracy']:.2%}")

# Save model
trainer.save_model("models/roberta-fast-v2")
tokenizer.save_pretrained("models/roberta-fast-v2")

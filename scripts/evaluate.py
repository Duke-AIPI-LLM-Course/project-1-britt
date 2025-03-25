import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForMultipleChoice
from src.data import load_commonsenseqa
from src.train import CommonsenseQADataset
import json
import os

def evaluate():
    _, val_ds = load_commonsenseqa()
    tokenizer = RobertaTokenizer.from_pretrained('models/roberta-finetuned')
    model = RobertaForMultipleChoice.from_pretrained('models/roberta-finetuned')
    dataset = CommonsenseQADataset(val_ds, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    predictions = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for pred, label in zip(preds, labels):
                predictions.append({
                    "predicted": ["A", "B", "C", "D", "E"][pred.item()],
                    "actual": ["A", "B", "C", "D", "E"][label.item()],
                    "correct": pred.item() == label.item()
                })

    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy:.2%}")

    os.makedirs("results", exist_ok=True)
    with open("results/results.json", "w") as f:
        json.dump({
            "accuracy": accuracy,
            "samples": predictions[:10]  # save sample outputs
        }, f, indent=2)

if __name__ == "__main__":
    evaluate()

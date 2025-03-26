import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer,
    RobertaForMultipleChoice,
    get_scheduler
)
from torch.optim import AdamW
from tqdm import tqdm
from data import load_commonsenseqa


class CommonsenseQADataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = ["A", "B", "C", "D", "E"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        question = row["question"]
        choices = row["choices"]["text"]
        label = self.label_map.index(row["answerKey"])

        encodings = self.tokenizer(
            [(question, choice) for choice in choices],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return encodings["input_ids"], encodings["attention_mask"], torch.tensor(label)


def train():
    train_ds, _ = load_commonsenseqa()
    
    # ✅ FAST DEV: only use 3000 examples
    train_ds = train_ds.select(range(3000))

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    dataset = CommonsenseQADataset(train_ds, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = RobertaForMultipleChoice.from_pretrained('roberta-base')
    optimizer = AdamW(model.parameters(), lr=2e-5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    num_epochs = 3  # ✅ also reduce epochs for fast testing
    total_steps = len(dataloader) * num_epochs

    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    print(f"🚀 Training on {device} for {num_epochs} epochs with 3,000 examples")

    for epoch in range(num_epochs):
        print(f"\n📘 Epoch {epoch + 1}/{num_epochs}")
        running_loss = 0.0

        for step, (input_ids, attention_mask, labels) in enumerate(tqdm(dataloader)):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            if step % 100 == 0:
                print(f"Step {step}: Loss = {loss.item():.4f}")

        avg_loss = running_loss / len(dataloader)
        print(f"✅ Epoch {epoch + 1} complete. Avg Loss: {avg_loss:.4f}")

    model.save_pretrained("models/roberta-finetuned")
    tokenizer.save_pretrained("models/roberta-finetuned")
    print("🎉 Model saved to models/roberta-finetuned")


if __name__ == "__main__":
    train()

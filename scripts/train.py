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
    def __init__(self, dataset, tokenizer, max_length=64):
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

        # Build list of (question, choice) pairs
        encoding = self.tokenizer(
            [(question, choice) for choice in choices],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        return input_ids, attention_mask, torch.tensor(label)


def train():
    train_ds, _ = load_commonsenseqa()
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    dataset = CommonsenseQADataset(train_ds, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = RobertaForMultipleChoice.from_pretrained('roberta-base')
    optimizer = AdamW(model.parameters(), lr=2e-5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    num_epochs = 5
    num_training_steps = num_epochs * len(dataloader)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    for epoch in range(num_epochs):
        print(f"\n🔥 Epoch {epoch + 1}/{num_epochs}")
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

            if step % 100 == 0:
                print(f"Step {step}: Loss = {loss.item():.4f}")

    model.save_pretrained("models/roberta-finetuned")
    tokenizer.save_pretrained("models/roberta-finetuned")
    print("✅ Model saved to models/roberta-finetuned")

if __name__ == "__main__":
    train()

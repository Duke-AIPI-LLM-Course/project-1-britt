import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForMultipleChoice
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from src.data import load_commonsenseqa

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
        inputs = self.tokenizer(
            [question] * 5,
            choices,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        label = self.label_map.index(row["answerKey"])
        return input_ids, attention_mask, torch.tensor(label)

def train():
    train_ds, _ = load_commonsenseqa()
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    dataset = CommonsenseQADataset(train_ds, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = RobertaForMultipleChoice.from_pretrained('roberta-base')
    optimizer = AdamW(model.parameters(), lr=5e-5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    for epoch in range(3):
        print(f"Epoch {epoch + 1}")
        for input_ids, attention_mask, labels in tqdm(dataloader):
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
            optimizer.zero_grad()

    model.save_pretrained("models/roberta-finetuned")
    tokenizer.save_pretrained("models/roberta-finetuned")
    print("Model saved to models/roberta-finetuned")

if __name__ == "__main__":
    train()

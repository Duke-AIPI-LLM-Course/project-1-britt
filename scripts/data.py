from datasets import load_dataset

def load_commonsenseqa():
    ds = load_dataset("tau/commonsense_qa")
    return ds["train"], ds["validation"]

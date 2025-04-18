from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from ..config import DATASET_NAME, TASK_PROMPT, QUESTION


def load_meter_dataset():
    dataset = load_dataset(DATASET_NAME)
    return dataset["train"], dataset["test"]


class DocVQADataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        question = f"{TASK_PROMPT}{QUESTION}"
        answer = example["label"]
        image = example["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        return question, answer, image


def create_data_loaders(
    train_data, test_data, processor, device, batch_size=1, num_workers=0
):
    def collate_fn(batch):
        questions, answers, images = zip(*batch)
        inputs = processor(
            text=list(questions), images=list(images), return_tensors="pt", padding=True
        ).to(device)
        return inputs, answers

    # Create datasets
    train_dataset = DocVQADataset(train_data)
    val_dataset = DocVQADataset(test_data)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    return train_loader, val_loader

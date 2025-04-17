"""
Dataset classes and data loading utilities.
"""

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from ..config import DATASET_NAME, TASK_PROMPT, QUESTION

def load_meter_dataset():
    """Load and return the energy meter dataset."""
    ds = load_dataset(DATASET_NAME)
    return ds['train'], ds['test']

class DocVQADataset(Dataset):
    """Dataset class for DocVQA tasks on meter images."""
    
    def __init__(self, data):
        """
        Initialize the dataset.
        
        Args:
            data: The dataset split from HuggingFace datasets
        """
        self.data = data

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single example from the dataset.
        
        Args:
            idx: Index of the example
            
        Returns:
            tuple: (question, answer, image)
        """
        example = self.data[idx]
        question = f"{TASK_PROMPT}{QUESTION}"
        answer = example['label']
        image = example['image']
        if image.mode != "RGB":
            image = image.convert("RGB")
        return question, answer, image

def create_data_loaders(train_data, test_data, processor, device, batch_size=1, num_workers=0):
    """
    Create DataLoaders for training and validation.
    
    Args:
        train_data: Training dataset
        test_data: Test/validation dataset
        processor: Florence processor for tokenization
        device: Compute device (CPU/GPU)
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    def collate_fn(batch):
        questions, answers, images = zip(*batch)
        inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True).to(device)
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
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn, 
        num_workers=num_workers
    )
    
    return train_loader, val_loader

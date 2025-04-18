"""
Training utilities for Florence model.
"""

import os
import torch
from tqdm import tqdm
from torch.optim import AdamW
from transformers import get_scheduler
from huggingface_hub import login
from ..config import MODEL_CHECKPOINTS_DIR, OUTPUT_MODEL_ID, RESULTS_DIR
from ..utils.visualization import plot_training_metrics

import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models.layers")


def train_model(train_loader, val_loader, model, processor, device, epochs=10, lr=1e-6):
    """
    Train the Florence model on data.

    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        model: The Florence model
        processor: The Florence processor
        device: Device to train on (CPU/GPU)
        epochs: Number of training epochs
        lr: Learning rate

    Returns:
        model: The trained model
    """
    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    os.makedirs(MODEL_CHECKPOINTS_DIR, exist_ok=True)

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            inputs, answers = batch

            # Get input tensors
            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"]

            # Tokenize the ground truth answers
            labels = processor.tokenizer(
                text=answers,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False,
            ).input_ids.to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids, pixel_values=pixel_values, labels=labels
            )
            loss = outputs.loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Average Training Loss: {avg_train_loss}")

        # Validation phase
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in tqdm(
                val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"
            ):
                inputs, answers = batch

                input_ids = inputs["input_ids"]
                pixel_values = inputs["pixel_values"]
                labels = processor.tokenizer(
                    text=answers,
                    return_tensors="pt",
                    padding=True,
                    return_token_type_ids=False,
                ).input_ids.to(device)

                outputs = model(
                    input_ids=input_ids, pixel_values=pixel_values, labels=labels
                )
                loss = outputs.loss

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Average Validation Loss: {avg_val_loss}")

        # Save the best model based on validation loss
        if epoch == 0 or avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_dir = os.path.join(MODEL_CHECKPOINTS_DIR, "best_model")
            os.makedirs(best_model_dir, exist_ok=True)
            model.save_pretrained(best_model_dir)
            processor.save_pretrained(best_model_dir)
            print(
                f"Saved best model at epoch {epoch + 1} with validation loss: {avg_val_loss}"
            )

    plot_training_metrics(train_losses, val_losses, save_dir=RESULTS_DIR)
    print("Training completed.")
    return model, processor


def push_model_to_hub(model, processor, model_id=OUTPUT_MODEL_ID):
    login()
    model.push_to_hub(model_id)
    processor.push_to_hub(model_id)
    print(f"Model pushed to {model_id}")

import argparse
import torch

from src.data.dataset import load_meter_dataset, create_data_loaders
from src.modeling.florence import load_florence_model_and_processor, freeze_vision_tower
from src.modeling.trainer import train_model, push_model_to_hub
from src.config import DEFAULT_BATCH_SIZE, DEFAULT_NUM_WORKERS, DEFAULT_LEARNING_RATE, DEFAULT_EPOCHS

def parse_args():
    parser = argparse.ArgumentParser(description="Train Florence model for meter reading")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS, help="Number of data loader workers")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument("--push_to_hub", action="store_true", help="Push model to HuggingFace Hub after training")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    model, processor, device = load_florence_model_and_processor()    
    model = freeze_vision_tower(model)
    
    train_data, test_data = load_meter_dataset()
    
    train_loader, val_loader = create_data_loaders(
        train_data, 
        test_data, 
        processor, 
        device, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )
    
    model, processor = train_model(
        train_loader, 
        val_loader, 
        model, 
        processor, 
        device,
        epochs=args.epochs, 
        lr=args.learning_rate
    )
    
    if args.push_to_hub:
        push_model_to_hub(model, processor)

if __name__ == "__main__":
    main()
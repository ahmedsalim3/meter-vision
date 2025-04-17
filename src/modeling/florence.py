"""
Florence model initialization and configuration.
"""

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from ..config import MODEL_NAME, MODEL_REVISION, TRUST_REMOTE_CODE

def load_florence_model_and_processor(device=None):
    """
    Load the Florence model and processor.
    
    Args:
        device: The device to load the model on. If None, will use CUDA if available.
        
    Returns:
        tuple: (model, processor)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading model on device: {device}")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        trust_remote_code=TRUST_REMOTE_CODE, 
        revision=MODEL_REVISION
    ).to(device)
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME, 
        trust_remote_code=TRUST_REMOTE_CODE, 
        revision=MODEL_REVISION
    )
    
    return model, processor, device

def freeze_vision_tower(model):
    """
    Freeze the vision tower parameters to save memory and speed up training.
    
    Args:
        model: The Florence model
        
    Returns:
        model: The model with frozen vision tower
    """
    for param in model.vision_tower.parameters():
        param.requires_grad = False
    
    return model

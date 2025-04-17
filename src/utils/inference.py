"""
Inference utilities for the Florence model.
"""

import os
from PIL import Image
import torch
from ..config import TASK_PROMPT, MAX_NEW_TOKENS, NUM_BEAMS, RESULTS_DIR
from ..utils.visualization import visualize_inference

def run_inference(model, processor, question, image, device):
    """
    Run inference on a single image.
    
    Args:
        model: The Florence model
        processor: The Florence processor
        question: Question to ask about the image
        image: PIL Image object
        device: Device to run inference on
        
    Returns:
        str: The model's answer
    """
    # Ensure the prompt format is correct
    if not question.startswith(TASK_PROMPT):
        prompt = f"{TASK_PROMPT}{question}"
    else:
        prompt = question

    # Ensure the image is in RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Prepare inputs
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    
    # Generate
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=MAX_NEW_TOKENS,
        num_beams=NUM_BEAMS
    )
    
    # Decode and post-process
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=TASK_PROMPT, 
        image_size=(image.width, image.height)
    )

    visualize_inference(
        image, 
        prompt, 
        parsed_answer, 
        save_path=os.path.join(RESULTS_DIR, "inference_visualization.png")
    )
    
    return parsed_answer

def load_and_process_image(image_path):
    """
    Load an image from a file and prepare it for inference.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        PIL.Image: The loaded image
    """
    try:
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except Exception as e:
        raise ValueError(f"Error loading image: {e}")

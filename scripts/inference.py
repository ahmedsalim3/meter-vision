#!/usr/bin/env python3
"""
Script to run inference with the fine-tuned Florence model.
"""

import argparse
from PIL import Image

from src.modeling.florence import load_florence_model_and_processor
from src.utils.inference import run_inference, load_and_process_image
from src.config import QUESTION, OUTPUT_MODEL_ID

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on meter images")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image file")
    parser.add_argument("--model_id", type=str, default=OUTPUT_MODEL_ID, 
                        help="Model ID on HuggingFace Hub or local path")
    parser.add_argument("--question", type=str, default=QUESTION, 
                        help="Question to ask about the image")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load model and processor
    model, processor, device = load_florence_model_and_processor()
    
    # Load and process image
    image = load_and_process_image(args.image_path)
    
    # Run inference
    answer = run_inference(model, processor, args.question, image, device)
    
    # Display results
    print(f"Question: {args.question}")
    print(f"Answer: {answer}")
    
    # Optional: Display the image
    try:
        image.show()
    except Exception as e:
        print(f"Could not display image: {e}")

if __name__ == "__main__":
    main()
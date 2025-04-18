import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig
from PIL import Image
import sys
import matplotlib.pyplot as plt

def run_inference(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint_path = "ahmed-salim/meter-vision"
    config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)
    config.vision_config.model_type = "davit"
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, config=config, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)
    
    torch.cuda.empty_cache()
    
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    prompt = "DocVQA" + "What is the meter values?"
    
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task="DocVQA", image_size=(image.width, image.height))
    meter_value = list(parsed_answer.values())[0]
    
    return image, meter_value

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python infer.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    image, meter_value = run_inference(image_path)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Model Response: {meter_value}", fontsize=16)
    plt.show()
    
    print(f"Meter reading: {meter_value}")
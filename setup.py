from setuptools import setup, find_packages

setup(
    name="meter-vision",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "timm>=0.9.0",
        "einops>=0.6.0",
        "Pillow>=9.0.0",
        "tqdm>=4.65.0",
        "huggingface-hub>=0.16.0",
        "matplotlib>=3.5.0",
    ],
    author="Ahmed Salim",
    author_email="realahmedsalim@gmail.com",
    description="Fine-tuning Florence-2 for meter value reading",
    keywords="computer vision, transformers, OCR, meter reading",
    python_requires=">=3.8",
)
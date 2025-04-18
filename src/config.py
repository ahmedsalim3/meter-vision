"""
Configuration parameters for the project.
"""

import os

# Model configuration
MODEL_NAME = "microsoft/Florence-2-base-ft"
MODEL_REVISION = "main"
TRUST_REMOTE_CODE = True

# Dataset configuration
DATASET_NAME = "henrik-dra/energy-meter"

# Training configuration
DEFAULT_BATCH_SIZE = 4
DEFAULT_NUM_WORKERS = 0
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_EPOCHS = 15

# Inference configuration
MAX_NEW_TOKENS = 20
NUM_BEAMS = 3

# Task prompt
TASK_PROMPT = "<DocVQA>"
QUESTION = "What is the meter values?"

# Paths
MODEL_CHECKPOINTS_DIR = "./model_checkpoints2"
RESULTS_DIR = "./results"
os.makedirs(MODEL_CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

OUTPUT_MODEL_ID = "ahmed-salim/meter-vision"

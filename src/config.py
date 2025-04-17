"""
Configuration parameters for the project.
"""

# Model configuration
MODEL_NAME = "microsoft/Florence-2-base-ft"
MODEL_REVISION = "refs/pr/6"
TRUST_REMOTE_CODE = True

# Dataset configuration
DATASET_NAME = "henrik-dra/energy-meter"

# Training configuration
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 0
DEFAULT_LEARNING_RATE = 1e-6
DEFAULT_EPOCHS = 10

# Inference configuration
MAX_NEW_TOKENS = 1024
NUM_BEAMS = 3

# Task prompt
TASK_PROMPT = "<DocVQA>"
QUESTION = "What is the meter values?"

# Paths
MODEL_CHECKPOINTS_DIR = "./model_checkpoints"
RESULTS_DIR = "./results"
OUTPUT_MODEL_ID = "ahmed-salim/meter-vision"
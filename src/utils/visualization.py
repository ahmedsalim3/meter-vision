"""
Visualization utilities for model metrics and results.
"""

import matplotlib
matplotlib.use('Agg')  # For headless environments
import matplotlib.pyplot as plt
import os
from PIL import Image
from matplotlib import rcParams

rcParams.update({
    'axes.edgecolor': 'white',
    'axes.facecolor': '#EAEAF2',
    'axes.grid': True,
    'grid.color': 'white',
    'grid.linestyle': '--',
    'grid.alpha': 0.6,
    'xtick.color': 'black',
    'ytick.color': 'black',
    'axes.labelcolor': 'black',
    'axes.titleweight': 'bold',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.frameon': True,
    'legend.fontsize': 11,
    'figure.facecolor': 'white'
})

# Custom color palette
colors = {
    'train': '#13034d',
    'val': '#084d02'
}

def plot_training_metrics(train_losses, val_losses, save_dir):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses (list): Training losses per epoch
        val_losses (list): Validation losses per epoch
        save_dir (str): Directory to save the plot
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label='Training Loss', marker='o', color=colors['train'], linewidth=2)
    plt.plot(epochs, val_losses, label='Validation Loss', marker='s', color=colors['val'], linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Metrics', fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'training_metrics.png')
    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Metrics plot saved to {save_path}")

def visualize_inference(image, question, answer, save_path=None):
    """
    Visualize inference results with image and text.
    
    Args:
        image (PIL.Image): Input image
        question (str): Asked question
        answer (str): Model's answer
        save_path (str): Path to save visualization
    """
    plt.figure(figsize=(12, 6))
    plt.imshow(image)
    plt.axis('off')
    text_str = f"Question: {question}\nAnswer: {answer}"
    plt.text(10, 30, text_str, 
             fontsize=12, 
             color='white', 
             bbox=dict(facecolor='black', alpha=0.8))
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    plt.close()
import os

import torch
from tqdm import tqdm

from model import get_model
from dataset import get_dataloaders

DEVICE     = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
BATCH_SIZE = 32


def evaluate():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    archive_dir  = os.path.join(project_root, 'archive')
    model_path   = os.path.join(project_root, 'models', 'best_model.pth')

    if not os.path.exists(model_path):
        print(f"No trained model at {model_path} — run train.py first.")
        return

    try:
        # The official test set has no labels; evaluate on the held-out val split.
        _, val_loader, _ = get_dataloaders(archive_dir, BATCH_SIZE)
    except FileNotFoundError as e:
        print(e)
        return

    print(f"Loading trained model from: {model_path}")
    ckpt        = torch.load(model_path, map_location=DEVICE)
    class_names = ckpt['class_names']
    model       = get_model(num_classes=len(class_names)).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f"Evaluating on {len(val_loader.dataset)} validation images ({DEVICE})...")
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Evaluating'):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs        = model(images)
            predicted      = outputs.argmax(1)
            total         += labels.size(0)
            correct       += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"\nFinal Validation Accuracy: {accuracy:.2f}%")


if __name__ == '__main__':
    evaluate()

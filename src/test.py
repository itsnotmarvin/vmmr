import torch
import os
from tqdm import tqdm
from model import get_model
from dataset import get_dataloaders

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
BATCH_SIZE = 32

def find_dataset(start_path):
    for root, dirs, files in os.walk(start_path):
        if 'train' in dirs and 'test' in dirs:
            return os.path.join(root, 'train'), os.path.join(root, 'test')
    return None, None

def evaluate():
    print("Locating test data and saved model...")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_dir, test_dir = find_dataset(project_root)
    if train_dir is None:
        train_dir, test_dir = find_dataset(os.path.expanduser('~/Downloads'))

    model_path = os.path.join(project_root, 'models', 'best_model.pth')
    print(f"Loading trained model from: {model_path}")

    ckpt        = torch.load(model_path, map_location=DEVICE)
    class_names = ckpt['class_names']
    model       = get_model(num_classes=len(class_names)).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    _, test_loader, _ = get_dataloaders(train_dir, test_dir, BATCH_SIZE)

    print(f"Starting evaluation on {DEVICE}...")
    correct, total = 0, 0
    with torch.no_grad():
        loop = tqdm(test_loader, desc='Evaluating')
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs        = model(images)
            _, predicted   = torch.max(outputs.data, 1)
            total         += labels.size(0)
            correct       += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"\nFinal Test Accuracy: {accuracy:.2f}%")

if __name__ == '__main__':
    evaluate()

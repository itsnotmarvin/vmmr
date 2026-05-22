import torch
import torch.nn as nn
import torch.optim as optim
import os
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

from model import get_model
from dataset import get_dataloaders

DEVICE     = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
BATCH_SIZE = 32
EPOCHS_S1  = 5
EPOCHS_S2  = 20
LR_S1      = 1e-3
LR_S2      = 1e-4

def find_dataset(start_path):
    for root, dirs, files in os.walk(start_path):
        if 'train' in dirs and 'test' in dirs:
            return os.path.join(root, 'train'), os.path.join(root, 'test')
    return None, None

def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct    += outputs.argmax(1).eq(labels).sum().item()
            total      += images.size(0)
    torch.mps.empty_cache()
    return total_loss / total, correct / total

def train():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_dir, test_dir = find_dataset(project_root)
    if train_dir is None:
        train_dir, test_dir = find_dataset(os.path.expanduser('~/Downloads'))
    if train_dir is None:
        print("Dataset not found.")
        return

    print(f"Found data at: {train_dir}")
    train_loader, test_loader, class_names = get_dataloaders(train_dir, test_dir, BATCH_SIZE)

    model      = get_model(num_classes=len(class_names), freeze_backbone=True).to(DEVICE)
    criterion  = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_acc   = 0.0
    model_path = os.path.join(project_root, 'models', 'best_model.pth')
    os.makedirs(os.path.join(project_root, 'models'), exist_ok=True)

    # STAGE 1: head only
    print(f"\n=== Stage 1: Head only ({EPOCHS_S1} epochs) ===")
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_S1, weight_decay=1e-4
    )
    for epoch in range(1, EPOCHS_S1 + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc = run_epoch(model, test_loader, criterion)
        print(f"  Epoch {epoch:02d} | train_acc={tr_acc:.4f} | val_acc={vl_acc:.4f} | loss={tr_loss:.4f}")
        if vl_acc > best_acc:
            best_acc = vl_acc
            torch.save({'model_state_dict': model.state_dict(), 'class_names': class_names, 'best_acc': best_acc}, model_path)
            print(f"  Saved new best: {best_acc:.4f}")

    # STAGE 2: unfreeze everything
    print(f"\n=== Stage 2: Full fine-tune ({EPOCHS_S2} epochs) ===")
    for param in model.parameters():
        param.requires_grad = True
    optimizer = optim.AdamW(model.parameters(), lr=LR_S2, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_S2, eta_min=1e-6)

    for epoch in range(1, EPOCHS_S2 + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc = run_epoch(model, test_loader, criterion)
        scheduler.step()
        print(f"  Epoch {epoch:02d} | train_acc={tr_acc:.4f} | val_acc={vl_acc:.4f} | loss={tr_loss:.4f}")
        if vl_acc > best_acc:
            best_acc = vl_acc
            torch.save({'model_state_dict': model.state_dict(), 'class_names': class_names, 'best_acc': best_acc}, model_path)
            print(f"  Saved new best: {best_acc:.4f}")

    print(f"\nTraining Complete! Best val accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")

if __name__ == '__main__':
    train()

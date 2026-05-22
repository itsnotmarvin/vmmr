import os
import random

import scipy.io as sio
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

TRAIN_TFM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

VAL_TFM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class StanfordCarsDataset(Dataset):
    """Stanford Cars: a flat folder of images + .mat annotations.

    The Stanford Cars dataset does not ship class subfolders, so torchvision's
    ImageFolder cannot read it. Each sample's label lives in cars_train_annos.mat.
    """

    def __init__(self, images_dir, samples, transform=None):
        self.images_dir = images_dir
        self.samples = samples            # list of (fname, label) with 0-indexed label
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        img = Image.open(os.path.join(self.images_dir, fname)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def _locate(archive_dir):
    """Resolve the dataset paths inside the extracted Kaggle archive/ folder."""
    images_dir = os.path.join(archive_dir, 'cars_train', 'cars_train')
    devkit     = os.path.join(archive_dir, 'car_devkit', 'devkit')
    annos_mat  = os.path.join(devkit, 'cars_train_annos.mat')
    meta_mat   = os.path.join(devkit, 'cars_meta.mat')
    for p in (images_dir, annos_mat, meta_mat):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Dataset file missing: {p}\n"
                f"Download Stanford Cars from Kaggle and extract it to {archive_dir}"
            )
    return images_dir, annos_mat, meta_mat


def get_dataloaders(archive_dir, batch_size=32, val_split=0.15, seed=42):
    """Build train/val dataloaders from the Stanford Cars archive.

    The official test set ships without class labels, so we carve a validation
    split out of the 8,144 labelled training images instead.

    Returns: (train_loader, val_loader, class_names)
    """
    images_dir, annos_mat, meta_mat = _locate(archive_dir)

    # cars_meta.mat -> 196 class names, indexed 1..196 in the annotations.
    class_names = [str(c[0]) for c in sio.loadmat(meta_mat)['class_names'][0]]

    # cars_train_annos.mat -> records with fields fname and class (1-indexed).
    annos = sio.loadmat(annos_mat)['annotations'][0]
    samples = [(str(a['fname'][0]), int(a['class'][0][0]) - 1) for a in annos]

    rng = random.Random(seed)
    rng.shuffle(samples)
    n_val = int(len(samples) * val_split)
    val_samples   = samples[:n_val]
    train_samples = samples[n_val:]

    train_ds = StanfordCarsDataset(images_dir, train_samples, TRAIN_TFM)
    val_ds   = StanfordCarsDataset(images_dir, val_samples,   VAL_TFM)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)

    return train_loader, val_loader, class_names

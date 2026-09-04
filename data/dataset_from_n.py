import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from PIL import Image


def get_severstal_classes(ann_dir):
    """Собирает все классы из аннотаций."""
    classes = set()
    for fname in os.listdir(ann_dir):
        if fname.endswith('.json'):
            with open(os.path.join(ann_dir, fname), 'r') as f:
                data = json.load(f)
                for obj in data.get('objects', []):
                    classes.add(obj['classTitle'])
    return sorted(classes)


def load_severstal_data(root, train_part=0.7, val_part=0.15, random_state=42):
    """
    Загружает пути к изображениям и метки (multi-label).
    Возвращает кортеж: (all_paths, all_labels, train_paths, train_labels, val_paths, val_labels,
                        test_paths, test_labels, class_to_idx, num_classes)
    """
    img_dir = os.path.join(root, 'train', 'img')
    ann_dir = os.path.join(root, 'train', 'ann')

    classes = get_severstal_classes(ann_dir)
    num_classes = len(classes)
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    image_paths = []
    labels = []

    for ann_file in os.listdir(ann_dir):
        if not ann_file.endswith('.json'):
            continue
        img_name = ann_file.replace('.json', '')
        img_path = os.path.join(img_dir, img_name)
        with open(os.path.join(ann_dir, ann_file), 'r') as f:
            data = json.load(f)
        label_vec = [0] * num_classes
        for obj in data.get('objects', []):
            cls_name = obj['classTitle']
            label_vec[class_to_idx[cls_name]] = 1
        image_paths.append(img_path)
        labels.append(label_vec)

    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, labels, train_size=train_part, random_state=random_state
    )
    test_ratio = 1 - val_part / (1 - train_part)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_ratio, random_state=random_state
    )

    return (image_paths, np.array(labels),
            X_train, np.array(y_train),
            X_val, np.array(y_val),
            X_test, np.array(y_test),
            class_to_idx, num_classes)


class SeverstalDataset(Dataset):
    """Датасет для Severstal с multi-label."""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('L')
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        if self.transform:
            img = self.transform(img)
        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = zip(*batch)
        return torch.stack(images), torch.stack(labels)
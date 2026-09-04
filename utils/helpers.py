import os
import random
import numpy as np
import torch
from qiskit_machine_learning.utils import algorithm_globals

def set_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    algorithm_globals.random_seed = seed
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def print_dataset_stats(train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, class_names):
    """Выводит статистику по выборкам (без визуализации)."""
    splits = {
        'Train': (train_paths, train_labels),
        'Val': (val_paths, val_labels),
        'Test': (test_paths, test_labels)
    }
    for split_name, (paths, labels) in splits.items():
        num_samples = len(paths)
        multilabel_count = sum(1 for lbl in labels if np.sum(lbl) > 1)
        class_counts = np.sum(labels, axis=0)
        print(f"\n{'='*50}")
        print(f"{split_name} set: {num_samples} images")
        print(f"  - Multi-label images (≥2 defects): {multilabel_count} ({multilabel_count/num_samples*100:.1f}%)")
        print(f"  - Class distribution:")
        for i, cls_name in enumerate(class_names):
            count = class_counts[i]
            pct = count / num_samples * 100
            print(f"      {cls_name}: {count} ({pct:.1f}%)")
        print(f"{'='*50}")
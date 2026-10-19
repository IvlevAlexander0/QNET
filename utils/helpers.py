"""Helper utilities for seeding, RNG state capture and dataset statistics."""

import os
import random

import numpy as np
import torch
from qiskit_machine_learning.utils import algorithm_globals


def set_seed(seed, deterministic=True):
    """Seed all relevant random number generators.

    Args:
        seed: Integer seed value.
        deterministic: If True, also enables deterministic PyTorch algorithms
            and disables cuDNN benchmarking.

    Returns:
        None.
    """
    random.seed(seed)
    np.random.seed(seed)
    algorithm_globals.random_seed = seed
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """Seed a DataLoader worker reproducibly.

    Args:
        worker_id: Worker index supplied by PyTorch (unused).

    Returns:
        None.
    """
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _as_device(device):
    """Normalize a device specifier to ``torch.device``.

    Args:
        device: ``torch.device`` instance or a string specifier.

    Returns:
        ``torch.device`` instance.
    """
    return device if isinstance(device, torch.device) else torch.device(device)


def capture_rng_state(train_generator, device):
    """Capture RNG and sampler state for exact resume.

    Args:
        train_generator: ``torch.Generator`` used for training shuffling.
        device: Device whose CUDA RNG state should also be captured when
            applicable.

    Returns:
        Dictionary with keys ``python``, ``numpy``, ``torch_cpu``,
        ``train_sampler`` and, when ``device`` is CUDA, ``torch_cuda``.
    """
    device = _as_device(device)
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
        'train_sampler': train_generator.get_state(),
    }
    if device.type == 'cuda':
        state['torch_cuda'] = torch.cuda.get_rng_state(device)
    return state


def restore_rng_state(state, train_generator, device):
    """Restore RNG and sampler state captured by ``capture_rng_state``.

    Args:
        state: Dictionary produced by ``capture_rng_state``; may be ``None``.
        train_generator: ``torch.Generator`` used for training shuffling.
        device: Device whose CUDA RNG state should also be restored when
            applicable.

    Returns:
        True when the state was restored, False when ``state`` is empty.
    """
    if not state:
        return False
    device = _as_device(device)
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'].cpu())
    train_generator.set_state(state['train_sampler'].cpu())
    if device.type == 'cuda' and 'torch_cuda' in state:
        torch.cuda.set_rng_state(state['torch_cuda'].cpu(), device)
    return True


def print_dataset_stats(train_paths, train_labels, val_paths, val_labels,
                        test_paths, test_labels, class_names):
    """Print per-split dataset statistics.

    Args:
        train_paths: Training image paths.
        train_labels: Training multi-label matrix of shape
            ``(len(train_paths), num_classes)``.
        val_paths: Validation image paths.
        val_labels: Validation multi-label matrix.
        test_paths: Test image paths.
        test_labels: Test multi-label matrix.
        class_names: List of class names aligned with the label columns.

    Returns:
        None.
    """
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
        print(f"  - Multi-label images (≥2 defects): {multilabel_count} "
              f"({multilabel_count/num_samples*100:.1f}%)")
        print(f"  - Class distribution:")
        for i, cls_name in enumerate(class_names):
            count = class_counts[i]
            pct = count / num_samples * 100
            print(f"      {cls_name}: {count} ({pct:.1f}%)")
        print(f"{'='*50}")
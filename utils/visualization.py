"""Visualization utilities for training metrics and model predictions."""

import os

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image


def save_training_plots(history, save_dir, filename='training_metrics.png'):
    """Save loss, accuracy and F1 curves for a training run.

    Args:
        history: Dictionary with keys ``train_loss``, ``val_loss``,
            ``train_acc``, ``val_acc``, ``train_f1`` and ``val_f1``, each
            mapping to a list of per-epoch values.
        save_dir: Directory where the plot is written. When ``None`` the
            plot is shown interactively instead.
        filename: Output file name inside ``save_dir``.

    Returns:
        None.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history['train_loss'], label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, history['train_acc'], label='Train Acc')
    axes[1].plot(epochs, history['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(epochs, history['train_f1'], label='Train F1')
    axes[2].plot(epochs, history['val_f1'], label='Val F1')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1 (macro)')
    axes[2].set_title('F1-score')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, filename), dpi=150)
        plt.close(fig)
    else:
        plt.show()


def _denormalize(img_tensor, transform):
    """Undo normalization applied by a ``torchvision`` transform.

    Args:
        img_tensor: Image tensor of shape ``(1, C, H, W)``.
        transform: Transformation pipeline that may contain a
            ``transforms.Normalize`` instance.

    Returns:
        Displayable image ``np.ndarray`` of shape ``(H, W)`` for single-channel
        inputs or ``(H, W, C)`` for multi-channel inputs, with values in
        ``[0, 1]``.
    """
    norm = None
    if transform is not None:
        candidates = getattr(transform, 'transforms', [transform])
        for t in candidates:
            if isinstance(t, torchvision.transforms.Normalize):
                norm = t
                break
    if norm is not None:
        mean = torch.tensor(norm.mean).view(-1, 1, 1)
        std = torch.tensor(norm.std).view(-1, 1, 1)
        img = img_tensor.squeeze(0).cpu() * std + mean
        return torch.clamp(img, 0.0, 1.0).squeeze(0).numpy()
    return img_tensor.squeeze(0).squeeze(0).cpu().numpy()


def save_predictions_from_folder(model, folder, class_names, device, transform,
                                 save_dir, threshold=0.5):
    """Run a model over a folder of images and save annotated visualizations.

    Folder files are expected to be named as binary strings of length
    ``len(class_names)`` (for example ``'0000.jpg'``, ``'1000.jpg'``), where
    character ``i`` indicates whether class ``i`` is present.

    Args:
        model: Trained model returning per-class probabilities in ``[0, 1]``.
        folder: Directory containing the images to process.
        class_names: List of class names aligned with the label positions.
        device: Device on which the model runs.
        transform: Transformation pipeline applied to each input image.
        save_dir: Directory where visualization files are written.
        threshold: Decision threshold applied to predicted probabilities.

    Returns:
        List of paths to the saved visualization files.
    """
    model.eval()
    num_classes = len(class_names)

    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(valid_ext))
    if not files:
        print(f"[save_predictions_from_folder] В папке {folder} нет изображений.")
        return []

    os.makedirs(save_dir, exist_ok=True)
    saved_files = []

    with torch.no_grad():
        for i, fname in enumerate(files):
            img_path = os.path.join(folder, fname)
            stem = os.path.splitext(fname)[0]
            true_vec = None
            if len(stem) == num_classes and all(c in '01' for c in stem):
                true_vec = np.array([int(c) for c in stem], dtype=np.float32)
            else:
                print(f"[save_predictions_from_folder] Пропуск {fname}: "
                      f"имя не соответствует формату бинарной строки длины {num_classes}.")
                continue

            img = Image.open(img_path).convert('L')
            img_tensor = transform(img).unsqueeze(0).to(device)
            probs = model(img_tensor)
            pred_vec = (probs > threshold)[0].int().cpu().numpy()

            img_disp = _denormalize(img_tensor, transform)

            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.imshow(img_disp, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')

            true_classes = [class_names[j] for j, v in enumerate(true_vec) if v == 1]
            pred_classes = [class_names[j] for j, v in enumerate(pred_vec) if v == 1]

            true_str = ", ".join(true_classes) if true_classes else "Нет дефектов"
            pred_str = ", ".join(pred_classes) if pred_classes else "Нет дефектов"

            title = (
                f"{fname}\n"
                f"Истина [{stem}]: {true_str}\n"
                f"Предск: {pred_str}\n"
                f"{np.round(probs[0].cpu().numpy(), 2)}"
            )
            ax.set_title(title, fontsize=10)

            save_path = os.path.join(save_dir, f"pred_{i:02d}_{stem}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            saved_files.append(save_path)

    return saved_files
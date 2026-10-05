import os
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image

def save_training_plots(history, save_dir, filename='training_metrics.png'):
    """
    Сохраняет графики обучения (loss, accuracy, f1) в указанную директорию.
    Если save_dir is None, показывает графики интерактивно.
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
    """Обратное преобразование Normalize для отображения."""
    norm = None
    if transform is not None:
        # Ищем Normalize как в Compose, так и среди одиночных трансформаций
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
    """
    Загружает изображения из папки, созданной copy_class_combinations.py,
    и сохраняет визуализации предсказаний модели.

    Имя каждого файла (без расширения) — бинарная строка длиной num_classes:
        позиция i (слева направо) соответствует i-му классу из class_names.
        1 — класс присутствует, 0 — отсутствует.

    Аргументы:
        model       : обученная модель (возвращает вероятности [0..1]).
        folder      : путь к папке с изображениями вида '0000.jpg', '1000.jpg', ...
        class_names : список имён классов в порядке, использованном при копировании.
        device      : torch.device.
        transform   : трансформация для входа в модель (та же, что в тесте/валидации).
        save_dir    : папка для сохранения визуализаций.
        threshold   : порог бинаризации предсказаний.

    Возвращает:
        list[str]: пути к сохранённым файлам.
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

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

# need rework
# def save_predictions(model, image_paths, true_labels, class_names, device,
#                      transform, save_dir, num_samples=5, threshold=0.5):
#     """
#     Визуализирует предсказания модели на случайных изображениях и сохраняет их.
#     Возвращает список путей к сохранённым файлам.
#     """
#     model.eval()
#     indices = np.random.choice(len(image_paths), min(num_samples, len(image_paths)), replace=False)
#     saved_files = []

#     with torch.no_grad():
#         for i, idx in enumerate(indices):
#             img_path = image_paths[idx]
#             img_name = os.path.basename(img_path)
#             img = Image.open(img_path).convert('L')
#             true_label = torch.tensor(true_labels[idx], dtype=torch.float32)

#             img_tensor = transform(img).unsqueeze(0).to(device)
#             probs = model(img_tensor)
#             pred_labels = (probs > threshold)[0].int()

#             # Обратное преобразование для отображения
#             norm = None
#             if transform is not None:
#                 for t in transform.transforms:
#                     if isinstance(t, torchvision.transforms.Normalize):
#                         norm = t
#                         break
#             if norm is not None:
#                 mean = torch.tensor(norm.mean).view(-1, 1, 1)
#                 std = torch.tensor(norm.std).view(-1, 1, 1)
#                 img_disp = img_tensor.squeeze(0).cpu() * std + mean
#                 img_disp = torch.clamp(img_disp, 0.0, 1.0).squeeze(0).numpy()
#             else:
#                 img_disp = img_tensor.squeeze(0).squeeze(0).cpu().numpy()

#             fig, ax = plt.subplots(1, 1, figsize=(5, 5))
#             ax.imshow(img_disp, cmap='gray', vmin=0, vmax=1)
#             ax.axis('off')

#             true_classes = [class_names[j] for j, v in enumerate(true_label) if v == 1]
#             pred_classes = [class_names[j] for j, v in enumerate(pred_labels) if v == 1]
#             true_str = ", ".join(true_classes) if true_classes else "Нет дефектов"
#             pred_str = ", ".join(pred_classes) if pred_classes else "Нет дефектов"
#             ax.set_title(f"{img_name}\nИстина: {true_str}\nПредск: {pred_str}\n{np.round(probs[0].cpu().numpy(), 2)}", fontsize=10)

#             if save_dir:
#                 os.makedirs(save_dir, exist_ok=True)
#                 save_path = os.path.join(save_dir, f"pred_{i:02d}_{img_name.replace('.jpg', '.png')}")
#                 plt.savefig(save_path, dpi=150, bbox_inches='tight')
#                 saved_files.append(save_path)
#                 plt.close(fig)
#             else:
#                 plt.show()

#     return saved_files

import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from PIL import Image

def get_severstal_classes(ann_dir):
    """
    Собирает все классы из аннотаций (файла train.csv).
    Параметр ann_dir — путь к папке, в которой находится train.csv.
    Возвращает отсортированный список идентификаторов классов (например, [1, 2, 3, 4]).
    """
    classes = set()
    csv_path = os.path.join(ann_dir, 'train.csv')
    if not os.path.exists(csv_path):
        return []  # или можно выбросить исключение

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_id = int(row['ClassId'])
            encoded_pixels = row.get('EncodedPixels', '').strip()
            if encoded_pixels:   # если есть разметка, то класс присутствует
                classes.add(class_id)

    return sorted(classes)


def load_severstal_data(root, train_part=0.7, val_part=0.15, random_state=42):
    """
    Загружает пути к изображениям и метки (multi-label) из структуры:
        root/
            train_images/      - папка с изображениями (файлы .jpg)
            train.csv          - файл с разметкой (ImageId, ClassId, EncodedPixels)

    Возвращает кортеж: (all_paths, all_labels, train_paths, train_labels, val_paths, val_labels,
                        test_paths, test_labels, class_to_idx, num_classes)
    """
    # Путь к CSV и папке с изображениями
    csv_path = os.path.join(root, 'train.csv')
    img_dir = os.path.join(root, 'train_images')

    # Чтение CSV и сбор классов для каждого изображения
    image_to_classes = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row['ImageId']
            class_id = int(row['ClassId'])
            encoded_pixels = row['EncodedPixels']
            # Если есть разметка (не пустая строка), добавляем класс
            if encoded_pixels and encoded_pixels.strip():
                image_to_classes.setdefault(image_id, set()).add(class_id)

    # Определяем все классы (обычно 1..4)
    all_classes = set()
    for classes in image_to_classes.values():
        all_classes.update(classes)
    class_list = sorted(all_classes)  # ['1','2','3','4']
    num_classes = len(class_list)
    class_to_idx = {cls: i for i, cls in enumerate(class_list)}

    # Сбор всех файлов изображений
    image_paths = []
    labels = []
    # Ищем .jpg файлы в папке train_images
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Папка с изображениями не найдена: {img_dir}")
    for fname in os.listdir(img_dir):
        if fname.lower().endswith('.jpg'):
            img_path = os.path.join(img_dir, fname)
            image_id = fname  # Имя файла соответствует ImageId в CSV
            classes = image_to_classes.get(image_id, set())
            # Строим мультилейбл вектор
            label_vec = [0] * num_classes
            for cls in classes:
                if cls in class_to_idx:
                    label_vec[class_to_idx[cls]] = 1
            image_paths.append(img_path)
            labels.append(label_vec)

    # Разделение на train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, labels, train_size=train_part, random_state=random_state
    )
    # Доля val от оставшейся части
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
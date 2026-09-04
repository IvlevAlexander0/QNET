import os
import torch
from PIL import Image
import numpy as np
from torchvision import transforms

from models import create_model
from config import Config
from utils import set_seed


def return_classifier(config_dict):
    """
    Загружает модель классификации и возвращает функцию predict и список имён классов.

    Args:
        config_dict (dict): словарь с параметрами:
            - device: 'cuda' или 'cpu'
            - num_classes: int
            - in_size: int
            - model_name: str
            - model_mode: str ('classical' или 'quantum')
            - h_load_path: str (путь к весам)
            - transform_mean: float
            - transform_std: float
            - threshold: float (по умолчанию 0.5)
            - class_names: list of str (имена классов)

    Returns:
        predict (callable): функция, принимающая PIL.Image и возвращающая
            (probs: np.ndarray, labels: np.ndarray) размерности (num_classes,)
        class_names (list): список имён классов
    """
    config = Config(**config_dict)

    set_seed(config.seed, config.det)

    model, _ = create_model(config, config.num_classes, (config.in_height, config.in_width))

    if not os.path.exists(config.h_load_path):
        raise FileNotFoundError(f"File {config.h_load_path} not found")
    state_dict = torch.load(config.h_load_path, map_location=config.device)
    model.load_state_dict(state_dict)
    model = model.to(config.device)
    model.eval()

    mean = torch.tensor(config.transform_mean)
    std = torch.tensor(config.transform_std)
    transform = transforms.Compose([
        transforms.Resize((config.in_height, config.in_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    threshold = config.threshold#getattr(config, 'threshold', 0.5)
    class_names = getattr(config, 'class_names', None)
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(config.num_classes)]

    def predict(image):
        """
        Принимает PIL.Image, возвращает вероятности и бинарные метки.
        """
        # Важно: конвертируем в grayscale, т.к. модель обучена на 1-канальных изображениях
        if image.mode != 'L':
            image = image.convert('L')
        img_tensor = transform(image).unsqueeze(0).to(config.device)
        with torch.no_grad():
            probs = model(img_tensor)  # (1, num_classes)
        probs = probs.squeeze(0).cpu().numpy()  # (num_classes,)
        labels = (probs >= threshold).astype(int)
        return probs, labels

    return predict, class_names
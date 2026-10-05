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
            - batch_size: int (опционально, по умолчанию 32) — размер порции
              при пакетном инференсе

    Returns:
        predict (callable): функция, принимающая:
            * одиночный PIL.Image -> (probs: np.ndarray (num_classes,),
                                       labels: np.ndarray (num_classes,))
            * итерируемое PIL.Image -> (probs: np.ndarray (N, num_classes),
                                         labels: np.ndarray (N, num_classes))
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

    threshold = config.threshold
    class_names = getattr(config, 'class_names', None)
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(config.num_classes)]

    # Размер порции для инференса (можно задать через config_dict['batch_size'])
    batch_size = int(getattr(config, 'batch_size', 32) or 32)
    if batch_size <= 0:
        batch_size = 1

    def _prepare(image):
        """PIL.Image -> нормализованный тензор (C, H, W)."""
        if not isinstance(image, Image.Image):
            raise TypeError(
                f"Ожидается PIL.Image, получено {type(image).__name__}"
            )
        # Модель обучена на 1-канальных изображениях
        if image.mode != 'L':
            image = image.convert('L')
        return transform(image)

    @torch.no_grad()
    def predict(images):
        """
        Принимает одиночное PIL.Image или итерируемое PIL.Image.
        Возвращает вероятности и бинарные метки.
        """
        single = isinstance(images, Image.Image)
        if single:
            images = [images]
        else:
            images = list(images)

        if len(images) == 0:
            probs = np.empty((0, config.num_classes), dtype=np.float32)
            labels = np.empty((0, config.num_classes), dtype=np.int64)
            return probs, labels

        chunks_out = []
        for start in range(0, len(images), batch_size):
            chunk = images[start:start + batch_size]
            batch_tensor = torch.stack([_prepare(im) for im in chunk], dim=0)
            batch_tensor = batch_tensor.to(config.device)
            logits = model(batch_tensor)               # (B, num_classes)
            chunks_out.append(logits.detach().cpu().numpy())

        probs = np.concatenate(chunks_out, axis=0)     # (N, num_classes)
        labels = (probs >= threshold).astype(np.int64) # (N, num_classes)

        if single:
            return probs[0], labels[0]
        return probs, labels

    return predict, class_names
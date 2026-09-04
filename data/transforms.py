import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import ImageFilter, Image
import io
import numpy as np
import random

class AddGaussianNoise:
    """Гауссовский шум (аддитивный)."""
    def __init__(self, mean=0.0, std=0.05):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class AddPoissonNoise:
    """Пуассоновский шум (характерен для фотонов в слабом свете)."""
    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, tensor):
        scaled = tensor * 255.0
        noisy = torch.poisson(scaled * self.scale) / self.scale
        noisy = torch.clamp(noisy, 0, 255) / 255.0
        return noisy

    def __repr__(self):
        return f"{self.__class__.__name__}(scale={self.scale})"


class AddSaltPepperNoise:
    """Шум 'соль и перец' (импульсный)."""
    def __init__(self, prob=0.01):
        self.prob = prob

    def __call__(self, tensor):
        salt_pepper = torch.rand_like(tensor)
        tensor[salt_pepper < self.prob/2] = 0.0
        tensor[salt_pepper > 1 - self.prob/2] = 1.0
        return tensor

    def __repr__(self):
        return f"{self.__class__.__name__}(prob={self.prob})"


class AddSpeckleNoise:
    """Мультипликативный спекл-шум (моделирует текстуру)."""
    def __init__(self, std=0.1):
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std
        return tensor + tensor * noise

    def __repr__(self):
        return f"{self.__class__.__name__}(std={self.std})"


class JPEGCompression:
    """Артефакты сжатия JPEG (имитация через уровень качества)."""
    def __init__(self, quality=75):
        self.quality = quality

    def __call__(self, tensor):
        img = transforms.ToPILImage()(tensor)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=self.quality)
        buffer.seek(0)
        img_jpeg = Image.open(buffer)
        tensor_jpeg = transforms.ToTensor()(img_jpeg)
        return tensor_jpeg

    def __repr__(self):
        return f"{self.__class__.__name__}(quality={self.quality})"


# class AdjustBrightnessContrast:
#     """Случайное изменение яркости и контраста (имитация освещения)."""
#     def __init__(self, brightness_range=(-0.2, 0.2), contrast_range=(0.8, 1.2)):
#         self.brightness_range = brightness_range
#         self.contrast_range = contrast_range

#     def __call__(self, tensor):
#         brightness_factor = random.uniform(*self.brightness_range)
#         contrast_factor = random.uniform(*self.contrast_range)
#         tensor = tensor + brightness_factor
#         tensor = (tensor - 0.5) * contrast_factor + 0.5
#         return torch.clamp(tensor, 0.0, 1.0)

#     def __repr__(self):
#         return f"{self.__class__.__name__}(brightness={self.brightness_range}, contrast={self.contrast_range})"


class Blur:
    """Размытие (имитация вибрации камеры/объекта)."""
    def __init__(self, degree=10):
        self.degree = degree

    def __call__(self, tensor):
        img = transforms.ToPILImage()(tensor)
        img = img.filter(ImageFilter.GaussianBlur(radius=self.degree/10))
        return transforms.ToTensor()(img)

    def __repr__(self):
        return f"{self.__class__.__name__}(degree={self.degree})"


# --- Основная функция ---

def get_transforms(in_height, in_width, mean, std, noise_config=None):
    """
    Возвращает словарь трансформаций.
    Если noise_config задан, добавляет шумы в тестовую трансформацию.

    Параметры шумов (в конфиге):
    - gaussian_std: float (default None)
    - poisson_scale: float (default None)
    - salt_pepper_prob: float (default None)
    - speckle_std: float (default None)
    - jpeg_quality: int (default None)
    - brightness_range: list (default None)
    - contrast_range: list (default None)
    - motion_degree: int (default None)

    Пример noise_config:
    {
        "gaussian_std": 0.05,
        "contrast_range": [0.9, 1.1]
    }
    """
    base = [
        transforms.Resize((in_height, in_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]

    train_transform = transforms.Compose(base)

    val_transform = transforms.Compose(base)

    test_transforms = base.copy()

    if noise_config:
        if 'jpeg_quality' in noise_config and noise_config['jpeg_quality'] is not None:
            test_transforms.append(JPEGCompression(quality=noise_config['jpeg_quality']))
        if 'gaussian_std' in noise_config and noise_config['gaussian_std'] is not None:
            test_transforms.append(AddGaussianNoise(std=noise_config['gaussian_std']))
        if 'poisson_scale' in noise_config and noise_config['poisson_scale'] is not None:
            test_transforms.append(AddPoissonNoise(scale=noise_config['poisson_scale']))
        if 'salt_pepper_prob' in noise_config and noise_config['salt_pepper_prob'] is not None:
            test_transforms.append(AddSaltPepperNoise(prob=noise_config['salt_pepper_prob']))
        if 'speckle_std' in noise_config and noise_config['speckle_std'] is not None:
            test_transforms.append(AddSpeckleNoise(std=noise_config['speckle_std']))
        # None check
        # if 'brightness_range' in noise_config or 'contrast_range' in noise_config:
        #     b_range = noise_config.get('brightness_range', (-0.1, 0.1))
        #     c_range = noise_config.get('contrast_range', (0.9, 1.1))
        #     test_transforms.append(AdjustBrightnessContrast(brightness_range=b_range, contrast_range=c_range))
        if 'motion_degree' in noise_config and noise_config['motion_degree'] is not None:
            test_transforms.append(Blur(degree=noise_config['motion_degree']))

    test_transform = transforms.Compose(test_transforms)

    return {
        'train': train_transform,
        'val': val_transform,
        'test': test_transform
    }
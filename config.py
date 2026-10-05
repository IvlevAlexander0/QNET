import os
import json
import torch

class Config:
    """Конфигурация проекта, загружаемая из JSON-файла."""

    # Значения по умолчанию
    dataset_path = './Dataset_severstal'
    c_load_path = "__.pth"
    q_load_path = "__.pth"
    h_load_path = "__.pth"

    train_part = 0.6
    val_part = 0.2
    in_height = 256
    in_width = 1600
    threshold = 0.5

    epochs = 16
    initial_lr = 0.001
    w_decay = 0.0
    batch_size = 32

    shots = 128
    backend_type = 'aer_simulator'  # 'aer_simulator', 'noisy_simulator'
    transpile_before = True
    encoding_type = 'angle'
    ansatz_type = 'ry'
    entanglement = 'circular'
    reps = 4

    C_train_flag = True
    Q_train_flag = True
    C_load_params = False
    Q_load_params = False

    seed = 42
    device = "cpu"
    det = True
    eps = 1e-7

    # Параметры DataLoader
    # num_workers = 8
    # prefetch_factor = 2

    model_name = 'effnetv0'       
    model_mode = 'quantum'         # 'classical', 'quantum', 'hybrid'

    noise_config = None

    log_dir = "./logs"                # базовая папка для логов
    enable_logging = True             # включить логирование
    save_visualizations = True
    enable_tensorboard = True
    resume_from = None

    num_classes = 4
    class_weights = [0.17, 0.6, 0.03, 0.2]
    transform_mean = 0.343
    transform_std = 0.134
    class_names = ["царапины", "трещины", "расслоения", "пятна"]
    class_colors = ['#FF4136', '#2ECC40', '#0074D9', '#FF851B']

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @classmethod
    def from_json(cls, json_path='config.json'):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Файл {json_path} не найден. Используются параметры по умолчанию.")
            return cls()

        instance = cls()
        for key, value in data.items():
            if key in ['C_train_flag', 'Q_train_flag', 'C_load_params', 'Q_load_params',
                       'det', 'transpile_before', 'enable_logging', 'save_visualizations', 'use_compile']:
                if isinstance(value, str):
                    value = value.lower() == 'true'
            if key == 'device' and isinstance(value, str):
                value = torch.device(value)
            setattr(instance, key, value)

        return instance

    def get_noise_param(self, param_name, default=None):
        """
        Удобный метод для получения параметра шума.
        Пример: config.get_noise_param('gaussian_std', 0.0)
        """
        if self.noise_config is None:
            return default
        return self.noise_config.get(param_name, default)

    def has_noise(self):
        """Проверяет, задана ли конфигурация шумов."""
        return self.noise_config is not None and len(self.noise_config) > 0
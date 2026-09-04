import os
import json
import torch
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Конфигурация проекта, загружаемая из JSON-файла."""

    # Значения по умолчанию
    dataset_path = './Dataset_severstal'
    c_load_path = "__.pth"
    q_load_path = "__.pth"
    h_load_path = "__.pth"

    train_part = 0.7
    val_part = 0.1
    batch_size = 128
    in_height = 256
    in_width = 1600
    threshold = 0.5

    epochs = 64
    initial_lr = 0.002
    w_decay = 0.0

    shots = 512
    backend_type = 'aer_simulator'  # 'aer_simulator', 'octillion_remote', 'octillion_local'
    octillion_token = "---"
    octillion_chip = "Snowdrop 4q ver2"
    transpile_before = True

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

    model_name = 'clp_net'       
    model_mode = 'hybrid'         # 'classical', 'quantum', 'hybrid'

    noise_config = None

    log_dir = "./logs"                # базовая папка для логов
    enable_logging = True             # включить логирование
    save_visualizations = True
    enable_tensorboard = True
    resume_from = None

    num_classes = 4
    class_weights = [2, 7, 0.34, 2.2]
    transform_mean = 0.343
    transform_std = 0.134
    class_names = ["1", "2", "3", "4"]
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
            # Преобразование типов для некоторых ключей
            if key in ['C_train_flag', 'Q_train_flag', 'C_load_params', 'Q_load_params',
                       'det', 'transpile_before', 'enable_logging', 'save_visualizations', 'use_compile']:
                if isinstance(value, str):
                    value = value.lower() == 'true'
            if key == 'device' and isinstance(value, str):
                value = torch.device(value)
            # Устанавливаем атрибут без проверки на существование
            setattr(instance, key, value)

        env_token = os.environ.get('OCTILLION_TOKEN')
        if env_token is not None:
            setattr(instance, 'octillion_token', env_token)
            # Логируем предупреждение, если в конфиге был другой токен
            if hasattr(instance, 'octillion_token') and getattr(instance, 'octillion_token') != env_token:
                print("INFO: OCTILLION_TOKEN environment variable overrides config value.")
        else:
            # Если токен не задан в окружении, но в конфиге стоит "---", выдаём предупреждение
            if hasattr(instance, 'octillion_token') and instance.octillion_token == "---":
                print("WARNING: octillion_token is set to placeholder '---'. "
                      "Set OCTILLION_TOKEN environment variable for real usage.")

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
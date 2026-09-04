import numpy as np
from PIL import Image


def return_segmenter(config_dict=None):
    """
    Заглушка сегментатора – возвращает исходное изображение в виде numpy-массива
    (значения от 0 до 1, dtype=float).
    """
    def segment(image):
        # image: PIL.Image
        # Преобразуем в RGB (на случай, если изображение ч/б) и в массив float [0,1]
        if image.mode != 'RGB':
            image = image.convert('RGB')
        arr = np.array(image, dtype=np.float32) / 255.0
        return arr

    return segment
import numpy as np
from PIL import Image


def return_segmenter(config_dict=None):
    """
    Заглушка сегментатора – возвращает исходное изображение в виде numpy-массива
    (значения от 0 до 1, dtype=float32).

    Поддерживает одиночное изображение и пакет:
        * одиночный PIL.Image       -> np.ndarray (H, W, 3)
        * итерируемое PIL.Image     -> np.ndarray (N, H, W, 3), если размеры
          всех изображений совпадают; иначе список массивов (H_i, W_i, 3)

    Args:
        config_dict (dict | None): необязательный словарь параметров.
            Поддерживается ключ 'batch_size' (int) — размер порции обработки.
    """
    batch_size = 32
    if config_dict:
        batch_size = int(config_dict.get('batch_size', batch_size) or batch_size)
    if batch_size <= 0:
        batch_size = 32

    def _segment_one(image):
        """PIL.Image -> np.ndarray (H, W, 3), float32, [0, 1]."""
        if not isinstance(image, Image.Image):
            raise TypeError(
                f"Ожидается PIL.Image, получено {type(image).__name__}"
            )
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return np.asarray(image, dtype=np.float32) / 255.0

    def segment(images):
        single = isinstance(images, Image.Image)
        if single:
            images = [images]
        else:
            images = list(images)

        results = []
        # Обрабатываем порциями — полезно, если сегментатор станет «тяжёлым»
        for start in range(0, len(images), batch_size):
            chunk = images[start:start + batch_size]
            results.extend(_segment_one(im) for im in chunk)

        if single:
            return results[0]
        if not results:
            return np.empty((0,), dtype=np.float32)

        # Пробуем собрать единый массив; если размеры разные — отдаём список
        shapes = {arr.shape for arr in results}
        if len(shapes) == 1:
            return np.stack(results, axis=0)
        return results

    return segment
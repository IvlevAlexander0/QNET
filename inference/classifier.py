"""Classifier inference factory for Severstal defect classification."""

import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models import create_model, freeze_for_inference
from config import Config
from utils import set_seed


def return_classifier(config_dict):
    """Load a classification model and return a prediction callable.

    Args:
        config_dict: Mapping with configuration keys:
            - ``device``: ``'cuda'`` or ``'cpu'``.
            - ``num_classes``: Number of classes.
            - ``in_height``: Model input height.
            - ``in_width``: Model input width.
            - ``model_name``: Registered model name.
            - ``model_mode``: ``'classical'`` or ``'quantum'``.
            - ``h_load_path``: Path to the model weights.
            - ``transform_mean``: Normalization mean.
            - ``transform_std``: Normalization standard deviation.
            - ``threshold``: Decision threshold (default ``0.5``).
            - ``class_names``: Optional list of class names.
            - ``batch_size``: Optional inference batch size (default ``32``).

    Returns:
        Tuple ``(predict, class_names)`` where ``predict`` is a callable and
        ``class_names`` is a list of class names.

        The ``predict`` callable accepts either a single ``PIL.Image`` or an
        iterable of ``PIL.Image`` objects and returns:
            - for a single image: ``(probs, labels)`` where ``probs`` is a
              ``np.ndarray`` of shape ``(num_classes,)`` and ``labels`` is an
              ``np.ndarray`` of shape ``(num_classes,)``;
            - for an iterable: ``(probs, labels)`` where ``probs`` is a
              ``np.ndarray`` of shape ``(N, num_classes)`` and ``labels`` is
              an ``np.ndarray`` of shape ``(N, num_classes)``.

    Raises:
        FileNotFoundError: If the model weights file does not exist.
    """
    config = Config(**config_dict)

    set_seed(config.seed, config.det)

    model, info = create_model(config, config.num_classes,
                               (config.in_height, config.in_width))

    if not os.path.exists(config.h_load_path):
        raise FileNotFoundError(f"File {config.h_load_path} not found")
    try:
        state_dict = torch.load(config.h_load_path, map_location=config.device, weights_only=True)
    except:
        state_dict = (torch.load(config.h_load_path, map_location=config.device, weights_only=False))['model_state_dict']
    model.load_state_dict(state_dict)

    if getattr(config, "freeze_quantum", True):
        freeze_for_inference(model, info=info, transpile=True, inplace=True)

    model = model.to(config.device)
    model.eval()

    mean = torch.tensor(config.transform_mean)
    std = torch.tensor(config.transform_std)
    transform = transforms.Compose([
        transforms.Resize((config.in_height, config.in_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    threshold = config.threshold
    class_names = getattr(config, 'class_names', None)
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(config.num_classes)]

    batch_size = int(getattr(config, 'batch_size', 32) or 32)
    if batch_size <= 0:
        batch_size = 1

    def _prepare(image):
        """Convert a ``PIL.Image`` to a normalized tensor.

        Args:
            image: Input ``PIL.Image`` object.

        Returns:
            Normalized image tensor of shape ``(C, H, W)``.

        Raises:
            TypeError: If ``image`` is not a ``PIL.Image``.
        """
        if not isinstance(image, Image.Image):
            raise TypeError(
                f"Ожидается PIL.Image, получено {type(image).__name__}"
            )
        if image.mode != 'L':
            image = image.convert('L')
        return transform(image)

    @torch.no_grad()
    def predict(images):
        """Predict class probabilities and binary labels.

        Args:
            images: A single ``PIL.Image`` or an iterable of ``PIL.Image``.

        Returns:
            Tuple ``(probs, labels)``. For a single image, ``probs`` has
            shape ``(num_classes,)`` and ``labels`` has shape
            ``(num_classes,)``. For an iterable of ``N`` images, ``probs``
            has shape ``(N, num_classes)`` and ``labels`` has shape
            ``(N, num_classes)``. Both are ``np.ndarray`` objects; ``labels``
            is integer-valued.
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
            logits = model(batch_tensor)
            chunks_out.append(logits.detach().cpu().numpy())

        probs = np.concatenate(chunks_out, axis=0)
        labels = (probs >= threshold).astype(np.int64)

        if single:
            return probs[0], labels[0]
        return probs, labels

    return predict, class_names
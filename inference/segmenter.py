"""Segmentation inference factory for single images and image batches."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from segmantation_base.inference import _resolve_config, SegmentationInference


def return_segmenter(config_dict=None):
    """Build a ``segment(images)`` callable backed by the segmentation model.

    Default output is an RGB overlay ``[H, W, 3]`` ``uint8`` — the original
    image with predicted defect masks blended in class colors. This is
    suitable for ``PIL.Image.fromarray`` and the PySide6 GUI in
    ``local_interface.py``.

    Interface:
        ``segment(single_pil)`` returns ``np.ndarray`` of shape ``[H, W, 3]``
        with dtype ``uint8`` (overlay).
        ``segment([pil_1, pil_2, ...])`` returns ``np.ndarray`` of shape
        ``[N, H, W, 3]`` when all images share the same size, otherwise a
        list of ``[H_i, W_i, 3]`` arrays.

    Setting ``output_type`` to ``"probabilities"`` or ``"masks"`` restores the
    metric-oriented API, with channel layout controlled by ``output_layout``.

    Args:
        config_dict: ``InferenceConfig`` instance, ``dict`` or ``None``.
            - ``None`` reads ``config.json`` from the current working
              directory if present.
            - A ``dict`` must contain a checkpoint under ``checkpoint`` or one
              of ``c_load_path``, ``q_load_path``, ``h_load_path``; other
              ``InferenceConfig`` keys are optional.
            Segmenter-specific keys:
                ``output_type``: one of ``"overlay"``, ``"probabilities"``,
                    ``"masks"`` (default ``"overlay"``).
                ``output_layout``: one of ``"chw"``, ``"hwc"``; ignored for
                    overlay output. Defaults to ``"chw"`` for probabilities
                    and ``"hwc"`` for masks.
                ``overlay_alpha``: Foreground opacity in ``[0, 1]`` used when
                    blending class colors over the original image (default
                    ``0.58``).
                ``batch_size``: Number of input images per batch (default
                    ``32``). Model forwards are capped at its existing tile
                    batch size to limit memory use.

    Returns:
        Callable ``segment(images)`` that accepts a single ``PIL.Image`` or an
        iterable of ``PIL.Image`` and returns ``np.ndarray`` or ``list`` as
        described above.

    Raises:
        ValueError: If no checkpoint is provided in the configuration, if
            ``output_type`` or ``output_layout`` is invalid, or if
            ``overlay_alpha`` is outside ``[0, 1]``.
    """
    config = _resolve_config(config_dict)
    if not config.checkpoint:
        raise ValueError(
            "No checkpoint in config. Set 'checkpoint' (or one of "
            "'c_load_path', 'q_load_path', 'h_load_path') in config_dict or config.json"
        )

    output_type = "overlay"
    output_layout = None
    overlay_alpha = 0.58
    if isinstance(config_dict, dict):
        output_type = str(config_dict.get("output_type", output_type)).lower()
        layout_value = config_dict.get("output_layout")
        if layout_value is not None:
            output_layout = str(layout_value).lower()
        if config_dict.get("overlay_alpha") is not None:
            overlay_alpha = float(config_dict["overlay_alpha"])

    if output_type not in ("overlay", "probabilities", "masks"):
        raise ValueError(
            f"output_type must be 'overlay', 'probabilities', or 'masks'; "
            f"got {output_type!r}"
        )
    if output_layout is None:
        output_layout = "hwc" if output_type == "masks" else "chw"
    if output_type != "overlay" and output_layout not in ("chw", "hwc"):
        raise ValueError(
            f"output_layout must be 'chw' or 'hwc', got {output_layout!r}"
        )
    if not 0.0 <= overlay_alpha <= 1.0:
        raise ValueError(
            f"overlay_alpha must be in [0,1], got {overlay_alpha!r}"
        )

    inference = SegmentationInference(
        checkpoint_path=config.checkpoint,
        device=config.device,
        target_size=config.target_size,
        transform_type=config.transform_type,
        config=config,
    )
    default_threshold = float(config.threshold)
    configured_batch_size = (config_dict.get("batch_size") if isinstance(config_dict, dict)
                             else getattr(config_dict, "batch_size", None))
    batch_size = max(1, int(configured_batch_size or 32))
    model_batch_size = min(batch_size, inference._tile_batch_size)

    @torch.inference_mode()
    def _predict_batch(images):
        """Reuse the V3 preprocessing and tiled predictor for image batches."""
        groups = {}
        for index, image in enumerate(images):
            if inference.transform_type == "resize":
                size = tuple(inference.target_size)
            elif isinstance(image, Image.Image):
                size = (image.height, image.width)
            else:
                with Image.open(image) as source:
                    size = (source.height, source.width)
            groups.setdefault(size, []).append(index)

        predictions = [None] * len(images)
        crop_size = (inference.target_size if inference.transform_type == "crop"
                     else inference._training_crop_size)
        overlap = min(inference._tile_overlap, crop_size[0] // 2, crop_size[1] // 2)

        for indices in groups.values():
            for start in range(0, len(indices), model_batch_size):
                offsets = indices[start:start + model_batch_size]
                if len(offsets) == 1:
                    predictions[offsets[0]] = inference.predict_probabilities(images[offsets[0]])
                    continue

                prepared = [inference.preprocess_image(images[i]) for i in offsets]
                inputs = torch.cat([item["input_tensor"] for item in prepared], dim=0)
                with torch.autocast(
                    device_type=inference.device.type,
                    dtype=inference._amp_dtype or torch.float16,
                    enabled=inference._amp_dtype is not None,
                ):
                    logits = inference.training_module.predict_full_logits(
                        inference.model, inputs, crop_size=crop_size, overlap=overlap,
                        tile_batch_size=inference._tile_batch_size,
                    )
                    probs = torch.sigmoid(logits.float())
                probs = probs.cpu().numpy().astype(np.float32, copy=False)

                for position, index in enumerate(offsets):
                    prediction = probs[position]
                    if inference.transform_type == "resize":
                        meta = prepared[position]
                        prediction = inference.remove_padding_from_mask(
                            prediction, meta["original_size"], meta["padding"],
                            meta["scale_factor"],
                        )
                        prediction = np.clip(prediction, 0.0, 1.0)
                    predictions[index] = prediction
        return predictions

    def _make_overlay(image, probs_chw):
        """Blend class colors over the original RGB image.

        Args:
            image: Source ``PIL.Image`` whose RGB pixels are used as the base.
            probs_chw: Per-class probability array of shape
                ``[num_classes, H, W]``.

        Returns:
            Overlay ``np.ndarray`` of shape ``[H, W, 3]`` and dtype ``uint8``.
        """
        original = np.asarray(inference._load_rgb(image), dtype=np.uint8)
        binary_mask = (probs_chw > default_threshold).astype(np.uint8)
        color_mask = inference._mask_to_color(binary_mask)
        foreground = binary_mask.any(axis=0)
        overlay = original.astype(np.float32)
        if foreground.any():
            overlay[foreground] = (
                (1.0 - overlay_alpha) * overlay[foreground]
                + overlay_alpha * color_mask[foreground].astype(np.float32)
            )
        return np.clip(overlay, 0.0, 255.0).astype(np.uint8)

    def segment(images):
        """Run segmentation inference on one or more images.

        Args:
            images: A single ``PIL.Image`` or an iterable of ``PIL.Image``.

        Returns:
            Depending on ``output_type`` and the input count:
            - single image, ``"overlay"``: ``np.ndarray`` of shape
              ``[H, W, 3]`` with dtype ``uint8``;
            - iterable, ``"overlay"``: ``np.ndarray`` of shape
              ``[N, H, W, 3]`` when all sizes match, else a ``list`` of
              ``[H_i, W_i, 3]`` arrays;
            - single image, ``"probabilities"`` or ``"masks"``:
              ``np.ndarray`` of shape ``[C, H, W]`` (``"chw"``) or
              ``[H, W, C]`` (``"hwc"``); dtype ``float32`` for probabilities
              and ``uint8`` for masks;
            - iterable: stacked ``np.ndarray`` of shape ``[N, ...]`` when all
              sizes match, else a ``list`` of per-image arrays;
            - empty input: an empty ``np.ndarray`` with the appropriate shape
              for the selected output configuration.
        """
        single = isinstance(images, (Image.Image, str, Path))
        if single:
            images = [images]
        else:
            images = list(images)

        results = []
        for start in range(0, len(images), batch_size):
            chunk = images[start:start + batch_size]
            for image, probs in zip(chunk, _predict_batch(chunk)):
                if output_type == "overlay":
                    results.append(_make_overlay(image, probs))
                    continue
                if output_type == "masks":
                    output = (probs > default_threshold).astype(np.uint8)
                else:
                    output = probs.astype(np.float32)
                if output_layout == "hwc":
                    output = np.transpose(output, (1, 2, 0))
                results.append(output)

        if single:
            return results[0]
        if not results:
            if output_type == "overlay":
                return np.empty((0, 0, 0, 3), dtype=np.uint8)
            if output_layout == "chw":
                empty = (inference.num_classes, 0, 0)
            else:
                empty = (0, 0, inference.num_classes)
            dtype = np.uint8 if output_type == "masks" else np.float32
            return np.empty((0, *empty), dtype=dtype)
        if len({arr.shape for arr in results}) == 1:
            return np.stack(results, axis=0)
        return results

    return segment
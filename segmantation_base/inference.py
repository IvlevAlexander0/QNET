"""Inference for Severstal V3 EfficientNet-B0 FPN (classic and hybrid QuFeX).

Two integration styles are provided:

1. Class API (backward compatible with the previous inference module):
       inference = SegmentationInference("checkpoints/best_model.pth")
       image, mask, probs, _ = inference.predict("sample.jpg")
       inference.visualize_prediction("sample.jpg", 0.5, "out.png")

2. Segmenter factory matching ``data/segmenter.py`` (drop-in replacement):
       from inference import return_segmenter
       segment = return_segmenter({"checkpoint": "checkpoints/best_model.pth"})
       probs = segment(pil_image)                    # [4,H,W] float32 [0,1]
       probs_batch = segment([pil_image_1, pil_image_2])  # [N,4,H,W] or list

   Both the real model and the placeholder stub share the same
   ``return_segmenter(config_dict) -> segment(images)`` contract, so you can
   swap them without touching training/evaluation code. The stub returns
   ``[H,W,3]`` because it returns the raw image; the real segmenter returns
   ``[4,H,W]`` probability maps (use ``output_layout="hwc"`` for ``[H,W,4]``).

Config keys accepted by ``return_segmenter`` (``dict`` or ``InferenceConfig``):
    checkpoint | c_load_path | q_load_path | h_load_path   -> model .pth
    target_size: [h, w]          or in_height + in_width    -> tile/crop size
    transform_type: "crop" | "resize"
    threshold: float 0..1
    device: "cpu" | "cuda" | "cuda:1"
    num_classes: int (validated against checkpoint)
    class_names: list[str]       (legend labels, in class_ids order)
    class_colors: list["#RRGGBB" | [r,g,b]]  (visualization colors)
    output_type: "probabilities" (default) | "masks"
    output_layout: "chw" (default) | "hwc"
    batch_size: int              (accepted for segmenter.py interface parity;
                                  image-level batching is not yet exploited)

The default "crop" mode performs native-resolution sliding-window inference,
matching validation in V3. "resize" retains aspect-ratio-preserving resize +
padding. Four classes are independent sigmoid channels and can overlap.

Keep this file beside severstal_V3_FPN_QuFeX_phase_readout.py (or its
"(1).py" variant). The trained model definition is imported from that file,
so inference cannot silently drift from the current architecture.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
from PIL import Image, ImageOps
from torch.nn import functional as F


# Explicit, stable colors: background is black; each class remains an
# independent binary mask even where its display color overlaps other classes.
CLASS_COLORS = np.asarray(
    ((230, 57, 70), (33, 158, 88), (52, 121, 220), (245, 164, 50)),
    dtype=np.uint8,
)
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
DEFAULT_TARGET_SIZE = (256, 512)  # V3 DEFAULT_IMAGE_SIZE: height, width.


def _parse_rgb_color(color):
    """Convert '#RRGGBB', '#RGB', or [R,G,B] to a 3-tuple of ints in [0,255]."""
    if isinstance(color, str):
        text = color.strip().lstrip("#")
        if len(text) == 3:
            text = "".join(ch * 2 for ch in text)
        if len(text) != 6 or any(ch not in "0123456789abcdefABCDEF" for ch in text):
            raise ValueError(f"Invalid color string: {color!r}")
        return tuple(int(text[i:i + 2], 16) for i in (0, 2, 4))
    if isinstance(color, (list, tuple)) and len(color) == 3:
        rgb = tuple(int(c) for c in color)
        if any(not 0 <= c <= 255 for c in rgb):
            raise ValueError(f"Color component out of range: {color!r}")
        return rgb
    raise ValueError(f"Invalid color value: {color!r}")


class InferenceConfig:
    """Inference-side settings mirroring the training ``Config`` object.

    Exposes only what inference consumes; unknown training keys in
    ``config.json`` (dataset_path, epochs, noise_config, ...) are ignored.
    """

    checkpoint = None
    target_size = tuple(DEFAULT_TARGET_SIZE)
    transform_type = "crop"
    threshold = 0.5
    device = None
    num_classes = 4
    class_names = None
    class_colors = tuple(tuple(int(c) for c in color) for color in CLASS_COLORS)

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @classmethod
    def from_json(cls, json_path="config.json"):
        """Load settings from a JSON file (same file as training ``config.json``)."""
        json_path = Path(json_path)
        if not json_path.is_file():
            raise FileNotFoundError(f"Config file not found: {json_path}")
        with json_path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, dict):
            raise ValueError(f"Config must be a JSON object: {json_path}")
        return cls._from_dict(data)

    @classmethod
    def from_dict(cls, data):
        """Load settings from a Python dict (see class docstring for keys)."""
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data):
        if not isinstance(data, dict):
            raise TypeError(f"Config must be a dict, got {type(data).__name__}")
        instance = cls()

        # Checkpoint: prefer the generic key; then fall back to training keys.
        # "__.pth" is the training-script sentinel for "not set".
        for key in ("checkpoint", "c_load_path", "q_load_path", "h_load_path"):
            value = data.get(key)
            if not value:
                continue
            text = str(value).strip()
            if text.startswith("__") or text == "__.pth":
                continue
            instance.checkpoint = text
            break

        # target_size: explicit key wins. Otherwise accept in_height/in_width
        # (the training config's names) with a warning when they diverge from
        # the V3 training crop size.
        if data.get("target_size") is not None:
            instance.target_size = tuple(int(v) for v in data["target_size"])
        elif data.get("in_height") is not None and data.get("in_width") is not None:
            instance.target_size = (int(data["in_height"]), int(data["in_width"]))
            if instance.target_size != tuple(DEFAULT_TARGET_SIZE):
                print(
                    f"WARNING: config has in_height={data['in_height']} "
                    f"in_width={data['in_width']}; using {instance.target_size} "
                    f"as the sliding-window size. The V3 model was trained on "
                    f"{DEFAULT_TARGET_SIZE[0]}x{DEFAULT_TARGET_SIZE[1]} crops; "
                    "add an explicit 'target_size' key to silence this."
                )

        if data.get("transform_type"):
            instance.transform_type = str(data["transform_type"]).strip().lower()
        if data.get("threshold") is not None:
            instance.threshold = float(data["threshold"])
        if data.get("device"):
            instance.device = str(data["device"]).strip()
        if data.get("num_classes") is not None:
            instance.num_classes = int(data["num_classes"])

        names = data.get("class_names")
        if names:
            instance.class_names = [str(name) for name in names]

        colors = data.get("class_colors")
        if colors:
            instance.class_colors = tuple(_parse_rgb_color(color) for color in colors)

        return instance


def _supports_plot_window():
    """Detect non-graphical backends without treating TkAgg/QtAgg as Agg."""
    backend = plt.get_backend().lower()
    non_graphical = {
        "agg", "cairo", "pdf", "pgf", "ps", "svg", "template",
        "module://matplotlib_inline.backend_inline",
        "module://ipykernel.pylab.backend_inline",
    }
    return backend not in non_graphical


def _load_training_module():
    """Load the exact architecture/prediction routines from the nearby V3 file."""
    search_dirs = [Path(__file__).resolve().parent, Path.cwd().resolve()]
    candidates = []
    for folder in search_dirs:
        for name in (
            "severstal_V3_FPN_QuFeX_phase_readout.py",
            "severstal_V3_FPN_QuFeX_phase_readout(1).py",
        ):
            candidates.append(folder / name)
        candidates.extend(sorted(folder.glob("severstal_V3_FPN_QuFeX_phase_readout*.py")))

    seen = set()
    for path in candidates:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        spec = importlib.util.spec_from_file_location("_severstal_v3_inference_model", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        for symbol in ("build_model", "predict_full_logits", "IMAGENET_MEAN", "IMAGENET_STD"):
            if not hasattr(module, symbol):
                raise ImportError(f"Training module {path} does not define {symbol}")
        return module
    raise FileNotFoundError(
        "Severstal V3 training module not found. Put "
        "severstal_V3_FPN_QuFeX_phase_readout.py alongside this inference script."
    )


class SegmentationInference:
    """Infer four independent Severstal defect classes from one or many images.

    Public constructor/method names and positional arguments are compatible
    with the old inference module. Masks and probabilities now have [4,H,W]
    shape, as required by the four-class network.

    ``predict``, ``preprocess_image`` and ``predict_probabilities`` accept
    either a path or a ``PIL.Image``; ``visualize_prediction``,
    ``_save_masks`` and ``batch_predict`` accept both too, with PIL images
    only used for the visualization stem/label.
    """

    def __init__(
        self,
        checkpoint_path,
        device=None,
        target_size=None,
        transform_type=None,
        config=None,
    ):
        """Load a V3 model; use the V3 256x512 training crop by default.

        Args:
            checkpoint_path: Path to a training checkpoint (.pth).
            device: torch device or string; auto-selects CUDA if available.
            target_size: (height, width); if None, taken from config/checkpoint.
            transform_type: 'crop' (full image via tiles) or 'resize' (letterbox).
            config: Optional ``InferenceConfig`` providing defaults and class metadata.
        """
        config = config or InferenceConfig()

        resolved_device = device if device is not None else config.device
        self.device = torch.device(
            resolved_device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but it is not available")

        self.checkpoint_path = checkpoint_path
        self.transform_type = transform_type or config.transform_type or "crop"
        self.transform_types = ["crop", "resize"]
        if self.transform_type not in self.transform_types:
            raise ValueError(
                f"Unknown transform type {self.transform_type!r}, "
                f"available transforms: {self.transform_types}"
            )

        # Training checkpoints contain numpy RNG state and are not always loadable
        # with weights_only=True. Only load .pth files from trusted sources.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError("Checkpoint must contain the V3 model_state_dict")
        self.training_module = _load_training_module()
        model_type = str(checkpoint.get("model_type", "")).strip().lower()
        if model_type not in ("classic", "hybrid"):
            raise ValueError(f"Checkpoint has unsupported model_type: {model_type!r}")
        self.model_type = model_type

        architecture = checkpoint.get("architecture")
        current_architecture = self.training_module.ARCHITECTURE_ID
        legacy_architecture = getattr(self.training_module, "LEGACY_ARCHITECTURE_ID", None)
        if architecture != current_architecture:
            if not (model_type == "classic" and architecture == legacy_architecture):
                if architecture == legacy_architecture and model_type == "hybrid":
                    raise ValueError(
                        "This is a hybrid checkpoint with the old YY readout. "
                        "Current hybrid XY readout requires a retrained checkpoint."
                    )
                raise ValueError(
                    f"Checkpoint architecture {architecture!r} does not match "
                    f"current V3 architecture {current_architecture!r}."
                )

        self.num_classes = int(checkpoint.get("num_classes", self.training_module.NUM_CLASSES))
        if self.num_classes != self.training_module.NUM_CLASSES or self.num_classes != 4:
            raise ValueError(f"Expected four defect channels, got {self.num_classes}")
        if config.num_classes is not None and int(config.num_classes) != self.num_classes:
            raise ValueError(
                f"config num_classes={config.num_classes} disagrees with checkpoint "
                f"num_classes={self.num_classes}"
            )

        self.class_ids = tuple(int(i) for i in checkpoint.get("class_ids", (1, 2, 3, 4)))
        if len(self.class_ids) != self.num_classes:
            raise ValueError("Checkpoint class_ids and num_classes disagree")

        checkpoint_size = tuple(int(v) for v in checkpoint.get("image_size", DEFAULT_TARGET_SIZE))
        self.training_module.validate_image_geometry(checkpoint_size)
        self._training_crop_size = checkpoint_size
        effective_target_size = (
            target_size if target_size is not None else config.target_size
        )
        if effective_target_size is None:
            effective_target_size = checkpoint_size
        self.target_size = tuple(int(v) for v in effective_target_size)
        if len(self.target_size) != 2 or any(v <= 0 for v in self.target_size):
            raise ValueError("target_size must be (positive height, positive width)")
        if self.transform_type == "crop":
            self.training_module.validate_image_geometry(self.target_size)
        if tuple(self.target_size) != checkpoint_size:
            print(
                f"WARNING: target size {self.target_size} differs from training "
                f"crop size {checkpoint_size}; use training size for best comparability."
            )

        self.model = self.training_module.build_model(
            model_type=model_type,
            in_channels=3,
            base_channels=int(checkpoint.get("base_channels", 64)),
            out_classes=self.num_classes,
            batch_circuit_size=4096,
            pretrained_encoder=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.to(self.device)
        if self.device.type == "cuda":
            self.model.to(memory_format=torch.channels_last)
        self.model.eval()

        self.default_threshold = float(config.threshold)
        self.class_colors = self._resolve_class_colors(config)
        self.class_names = self._resolve_class_names(config)

        self._image_mean = self.training_module.IMAGENET_MEAN
        self._image_std = self.training_module.IMAGENET_STD
        self._tile_batch_size = 4
        self._tile_overlap = 128
        self._amp_dtype = None
        if self.device.type == "cuda":
            self._amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"Model loaded from {checkpoint_path}")
        print(f"Model type: {model_type}; classes: {self.class_ids}")
        print(f"Target size: {self.target_size}; transform type: {self.transform_type}")

    def _resolve_class_colors(self, config):
        colors = np.asarray(config.class_colors, dtype=np.uint8)
        if colors.shape != (len(self.class_ids), 3):
            raise ValueError(
                f"config class_colors has shape {colors.shape}; "
                f"expected ({len(self.class_ids)}, 3)"
            )
        return colors

    def _resolve_class_names(self, config):
        names = config.class_names
        if names is None:
            return tuple(f"Class {class_id}" for class_id in self.class_ids)
        if len(names) != len(self.class_ids):
            raise ValueError(
                f"config class_names has {len(names)} entries; "
                f"expected {len(self.class_ids)}"
            )
        return tuple(str(name) for name in names)

    @staticmethod
    def _load_rgb(image):
        """Accept a path or PIL.Image; always return RGB PIL.Image."""
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        with Image.open(image) as source:
            return source.convert("RGB")

    @staticmethod
    def _display_stem(image):
        """Filename stem for display/save; 'image' for PIL images."""
        if isinstance(image, Image.Image):
            return "image"
        return Path(image).stem

    def resize_image_with_padding(self, image, target_size):
        """Aspect-preserving letterbox; returns image, scale, (top,bottom,left,right)."""
        source_w, source_h = image.size
        target_h, target_w = (int(target_size[0]), int(target_size[1]))
        if min(source_w, source_h, target_h, target_w) < 1:
            raise ValueError("Source and target image dimensions must be positive")
        scale = min(target_w / source_w, target_h / source_h)
        new_w = max(1, min(target_w, round(source_w * scale)))
        new_h = max(1, min(target_h, round(source_h * scale)))
        resized_image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        pad_left = (target_w - new_w) // 2
        pad_right = target_w - new_w - pad_left
        pad_top = (target_h - new_h) // 2
        pad_bottom = target_h - new_h - pad_top
        padded_image = ImageOps.expand(
            resized_image,
            border=(pad_left, pad_top, pad_right, pad_bottom),
            fill=0,
        )
        return padded_image, (scale, scale), (pad_top, pad_bottom, pad_left, pad_right)

    def remove_padding_from_mask(self, mask, original_size, padding, scale_factor):
        """Restore a [H,W] or [C,H,W] mask, preserving float probabilities."""
        del scale_factor  # Cropping by exact padding is more accurate than inverse scaling.
        array = np.asarray(mask)
        single_channel = array.ndim == 2
        if not single_channel and array.ndim != 3:
            raise ValueError("Mask must be [H,W] or [C,H,W]")
        original_width, original_height = (int(original_size[0]), int(original_size[1]))
        top, bottom, left, right = (int(v) for v in padding)
        h, w = array.shape[-2:]
        if min(top, bottom, left, right) < 0 or top + bottom >= h or left + right >= w:
            raise ValueError(f"Padding {padding} does not fit mask shape {array.shape}")
        array = array[..., top:h - bottom if bottom else h,
                      left:w - right if right else w]
        data = torch.from_numpy(np.ascontiguousarray(array)).float()
        if single_channel:
            data = data.unsqueeze(0)
        data = data.unsqueeze(0)
        discrete = not np.issubdtype(array.dtype, np.floating)
        if discrete:
            restored = F.interpolate(data, size=(original_height, original_width), mode="nearest")
        else:
            restored = F.interpolate(
                data, size=(original_height, original_width), mode="bilinear", align_corners=False
            )
        result = restored.squeeze(0).numpy()
        if single_channel:
            result = result[0]
        return result.astype(array.dtype, copy=False)

    def _normalize_image(self, image_array):
        """Exact non-augmented V3 normalization: RGB uint8 / 255, ImageNet stats."""
        rgb = torch.from_numpy(np.ascontiguousarray(image_array.transpose(2, 0, 1))).float()
        rgb.div_(255.0).sub_(self._image_mean).div_(self._image_std)
        return rgb.unsqueeze(0).to(self.device)

    def preprocess_image(self, image):
        """Return the same preprocessing dictionary keys as the former API.

        ``image`` may be a filesystem path or a ``PIL.Image``.
        """
        original_image = self._load_rgb(image)
        original_size = original_image.size  # (width, height)
        original_array = np.asarray(original_image).copy()

        if self.transform_type == "crop":
            processed_image = original_array
            return {
                "input_tensor": self._normalize_image(processed_image),
                "processed_image": processed_image,
                "original_image": original_array,
                "original_size": original_size,
            }

        if self.transform_type == "resize":
            resized_image, scale_factor, padding = self.resize_image_with_padding(
                original_image, self.target_size
            )
            image_array = np.asarray(resized_image).copy()
            return {
                "input_tensor": self._normalize_image(image_array),
                "processed_image": image_array,
                "original_image": original_array,
                "original_size": original_size,
                "scale_factor": scale_factor,
                "padding": padding,
            }
        raise ValueError(f"Unknown transform type, available transforms: {self.transform_types}")

    def predict(self, image_path, threshold=None):
        """Predict four independent binary masks and float32 probabilities.

        ``image_path`` may be a filesystem path or a ``PIL.Image`` (the name is
        kept for backward compatibility).

        Returns (original_image [H,W,3], prediction_mask [4,H,W],
        confidence_scores [4,H,W], preprocessed_data).
        """
        if threshold is None:
            threshold = self.default_threshold
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        preprocessed_data = self.preprocess_image(image_path)
        input_tensor = preprocessed_data["input_tensor"]
        crop_size = self.target_size if self.transform_type == "crop" else self._training_crop_size
        overlap = min(self._tile_overlap, crop_size[0] // 2, crop_size[1] // 2)
        with torch.inference_mode():
            with torch.autocast(
                device_type=self.device.type,
                dtype=self._amp_dtype or torch.float16,
                enabled=self._amp_dtype is not None,
            ):
                logits = self.training_module.predict_full_logits(
                    self.model,
                    input_tensor,
                    crop_size=crop_size,
                    overlap=overlap,
                    tile_batch_size=self._tile_batch_size,
                )
                probs = torch.sigmoid(logits.float())
            confidence_scores = probs[0].cpu().numpy().astype(np.float32, copy=False)
        if self.transform_type == "resize":
            confidence_scores = self.remove_padding_from_mask(
                confidence_scores,
                preprocessed_data["original_size"],
                preprocessed_data["padding"],
                preprocessed_data["scale_factor"],
            )
            confidence_scores = np.clip(confidence_scores, 0.0, 1.0)
        prediction_mask = (confidence_scores > threshold).astype(np.uint8)
        return (preprocessed_data["original_image"], prediction_mask,
                confidence_scores, preprocessed_data)

    def predict_probabilities(self, image, threshold=None):
        """Return only the [C,H,W] float32 probabilities (path or PIL.Image)."""
        _, _, confidence_scores, _ = self.predict(image, threshold)
        return confidence_scores

    def _mask_to_color(self, prediction_mask):
        """Display-only RGB encoding of overlapping class masks."""
        masks = np.asarray(prediction_mask)
        if masks.ndim != 3 or masks.shape[0] != len(self.class_ids):
            raise ValueError(
                f"Expected [{len(self.class_ids)},H,W] masks, got {masks.shape}"
            )
        totals = masks.astype(np.float32).sum(axis=0)[..., None]
        weighted = np.einsum("chw,cd->hwd", masks.astype(np.float32),
                             self.class_colors.astype(np.float32))
        return np.clip(weighted / np.maximum(totals, 1), 0, 255).astype(np.uint8)

    def visualize_prediction(self, image_path, threshold=None, save_path=None):
        """Preserve the original 2x2 visualization, now color-coded for 4 classes."""
        original_image, prediction_mask, confidence_scores, preprocessed_data = self.predict(
            image_path, threshold
        )
        color_mask = self._mask_to_color(prediction_mask)
        original_float = original_image.astype(np.float32)
        foreground = prediction_mask.any(axis=0)
        overlay = original_float.copy()
        overlay[foreground] = (
            0.42 * original_float[foreground] + 0.58 * color_mask[foreground]
        )

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title("Original Image", fontsize=12, fontweight="bold")
        axes[0, 1].imshow(preprocessed_data["processed_image"])
        mode_text = "tiled at native resolution" if self.transform_type == "crop" else "letterboxed"
        axes[0, 1].set_title(
            f"Preprocessed ({mode_text}, {self.target_size[0]}x{self.target_size[1]})",
            fontsize=12, fontweight="bold"
        )
        axes[1, 0].imshow(color_mask)
        axes[1, 0].set_title("Prediction Mask (4 classes)", fontsize=12, fontweight="bold")
        axes[1, 1].imshow(np.clip(overlay, 0, 255).astype(np.uint8))
        axes[1, 1].set_title("Overlay on Original", fontsize=12, fontweight="bold")
        for ax in axes.ravel():
            ax.axis("off")
        fig.legend(
            handles=[Patch(facecolor=self.class_colors[index] / 255.0,
                           label=self.class_names[index])
                     for index in range(len(self.class_ids))],
            loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.01),
        )
        display_name = (
            "image" if isinstance(image_path, Image.Image) else Path(image_path).name
        )
        fig.suptitle(
            f"Segmentation Results - {display_name}\n"
            f"Original: {original_image.shape[1]}x{original_image.shape[0]} "
            f"| Model tiles: {self.target_size[0]}x{self.target_size[1]}",
            fontsize=14, fontweight="bold",
        )
        fig.tight_layout(rect=(0, 0.055, 1, 0.93))

        has_plot_window = _supports_plot_window()
        stem = self._display_stem(image_path)
        if save_path is None and not has_plot_window:
            save_path = Path.cwd() / f"result_{stem}.png"
        if save_path is not None:
            save_path = Path(save_path)
            if not save_path.suffix:
                save_path = save_path / f"result_{stem}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight", dpi=300)
            print(f"Visualization saved to {save_path}")
        if has_plot_window:
            plt.show()
        plt.close(fig)
        return prediction_mask, confidence_scores

    def _save_masks(self, prediction_mask, image_path, masks_dir):
        masks_dir = Path(masks_dir)
        masks_dir.mkdir(parents=True, exist_ok=True)
        stem = self._display_stem(image_path)
        Image.fromarray(self._mask_to_color(prediction_mask), mode="RGB").save(
            masks_dir / f"mask_{stem}.png"
        )
        for index, class_id in enumerate(self.class_ids):
            Image.fromarray((prediction_mask[index] * 255).astype(np.uint8)).save(
                masks_dir / f"mask_{stem}_class{class_id}.png"
            )

    def batch_predict(self, image_dir, output_dir=None, threshold=None):
        """Process a directory with the same result records and output layout."""
        if threshold is None:
            threshold = self.default_threshold
        image_dir = Path(image_dir)
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        image_paths = sorted(
            path for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            masks_dir = output_dir / "masks"
            masks_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for image_path in image_paths:
            print(f"Processing: {image_path.name}")
            save_path = (output_dir / f"result_{image_path.stem}.png"
                         if output_dir is not None else None)
            pred_mask, confidence = self.visualize_prediction(
                str(image_path), threshold, save_path
            )
            if output_dir is not None:
                self._save_masks(pred_mask, image_path, masks_dir)
            results.append({
                "image_path": image_path,
                "prediction_mask": pred_mask,
                "confidence_scores": confidence,
            })
        return results


def _resolve_config(config_dict):
    """Normalize the ``return_segmenter`` argument into an ``InferenceConfig``."""
    if config_dict is None:
        config_path = Path("config.json")
        if config_path.is_file():
            return InferenceConfig.from_json(config_path)
        raise ValueError(
            "return_segmenter requires config_dict with at least 'checkpoint' "
            "(or 'c_load_path'/'q_load_path'/'h_load_path'), "
            "or a config.json file in the working directory."
        )
    if isinstance(config_dict, InferenceConfig):
        return config_dict
    if isinstance(config_dict, dict):
        return InferenceConfig.from_dict(config_dict)
    raise TypeError(
        f"config_dict must be dict, InferenceConfig, or None; "
        f"got {type(config_dict).__name__}"
    )

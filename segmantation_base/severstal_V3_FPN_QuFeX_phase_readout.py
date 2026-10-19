"""Severstal 4-class segmentation: matched EfficientNet-B0 FPN and phase-aware QuFeX 6-1-2.

Data source and splitting:
    ./data_severstal/train.csv       # original pixel-accurate Severstal RLE
    ./data_severstal/train/images/   # original training image pool, 256 x 1600

Create seeded 80/10/10 train/validation/test splits of complete image IDs
at runtime. Split manifests go to checkpoint_dir/data_split.json; the original
CSV and images are never moved or modified. Images without a positive CSV row
are treated as negatives, as in the original Severstal training set.
Separate YOLO val/images, test_images, and YOLO *.txt labels are NEVER used as
segmentation ground truth.

Two strictly matched models have the same EfficientNet-B0 encoder, FPN decoder,
12-channel feature projector, residual fusion, and trainable parameter count.
Only one shared four-parameter operator changes:
    classic: nonlinear classical 6-value mixer, unchanged from V2
    hybrid:  phase-aware six-qubit statevector circuit (one shared filter)
Both act on two groups of 6 channels across a compact 4x8 feature map: '6-1-2'.
This is a cross-channel adaptation inspired by QuFeX, not an assertion that its
circuit/processing is identical to the original QuFeX paper. The updated quantum
operator re-uploads the input through a phase encoding, and uses complementary
Pauli-Z, mixed (X+Y)/sqrt(2), and Pauli-X measurements while retaining four
trainable parameters and six outputs. This is a hypothesis, not a guaranteed
improvement of Severstal validation IoU.

Train on native-resolution 256x512 crops; validate on entire images by merging
sliding-window logits. CUDA AMP and channels_last accelerate the CNN; the quantum
branch uses vectorized six-qubit GPU statevectors in float32/complex64.

Usage: adjust the familiar configuration at the end of this file and run it.
       python severstal_V3_FPN_QuFeX_phase_readout.py --self-test
       python severstal_V3_FPN_QuFeX_phase_readout.py --benchmark  # compares train steps
       (self-test needs no data or network access)
Requires: torch, torchvision, numpy, pillow, tqdm.
Pretrained ImageNet weights apply only to EfficientNet-B0, not FPN or head.
Pretrained encoder weights are downloaded once by torchvision if not cached.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler, get_worker_info
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from tqdm import tqdm


DEFAULT_IMAGE_SIZE = (256, 512)  # crop height/width, not resize target
NUM_CLASSES = 4
CLASS_IDS = (1, 2, 3, 4)
HARD_IOU_THRESHOLD = 0.5
DEFAULT_SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
ARCHITECTURE_ID = "efficientnet_b0_fpn_crosschannel_621_severstal_v3_xy_readout"
LEGACY_ARCHITECTURE_ID = "efficientnet_b0_fpn_crosschannel_621_severstal_v3_phase_readout"


def _normalize_model_type(model_type: str) -> str:
    mode = str(model_type).lower().strip()
    if mode not in ("classic", "hybrid"):
        raise ValueError("model_type must be 'classic' or 'hybrid'")
    return mode


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Seed both experiments. Fast cuDNN kernels by default; opt in to strict determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.allow_tf32 = not deterministic
        torch.backends.cuda.matmul.allow_tf32 = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    info = get_worker_info()
    if info is not None:
        transforms = getattr(info.dataset, "transforms", None)
        if transforms is not None and hasattr(transforms, "set_random_seed"):
            transforms.set_random_seed(worker_seed)


def validate_image_geometry(image_size: Sequence[int]) -> tuple[int, int]:
    """Ensure crops fit stride-32 EfficientNet and return the deepest feature size."""
    if len(image_size) != 2:
        raise ValueError("image_size must be (height, width)")
    h, w = (int(image_size[0]), int(image_size[1]))
    if h < 32 or w < 32 or h % 32 or w % 32:
        raise ValueError(f"Crop size {h}x{w} must have both dimensions divisible by 32")
    return h // 32, w // 32


def decode_severstal_rle(encoded_pixels: str, height: int, width: int) -> np.ndarray:
    """Decode 1-indexed, column-major Kaggle Severstal RLE into an HxW mask."""
    encoded_pixels = str(encoded_pixels).strip()
    if not encoded_pixels or encoded_pixels.lower() == "nan":
        return np.zeros((height, width), dtype=np.uint8)
    pairs = np.fromstring(encoded_pixels, sep=" ", dtype=np.int64)
    if len(pairs) % 2:
        raise ValueError("Malformed RLE: odd number of start/length entries")
    starts, lengths = pairs[::2] - 1, pairs[1::2]
    ends = starts + lengths
    if np.any(starts < 0) or np.any(lengths <= 0) or np.any(ends > height * width):
        raise ValueError(f"RLE indices outside image bounds {height}x{width}")
    flat = np.zeros(height * width, dtype=np.uint8)
    for start, end in zip(starts, ends):
        flat[int(start):int(end)] = 1
    return flat.reshape((width, height)).T


def load_severstal_annotations(
    csv_path: str | os.PathLike,
) -> dict[str, tuple[tuple[str, ...], ...]]:
    """Index positive RLEs by filename and defect class (two common CSV formats).

    Accept ImageId,ClassId,EncodedPixels and Kaggle ImageId_ClassId,EncodedPixels.
    Empty RLE records and images absent from the CSV mean no defect, not a box.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Annotation CSV not found: {path}")
    annotations: dict[str, list[list[str]]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        try:
            columns = [name.strip() for name in next(reader)]
        except StopIteration as exc:
            raise ValueError(f"Empty annotation file: {path}") from exc
        wide_format = all(col in columns for col in ("ImageId", "ClassId", "EncodedPixels"))
        combined_format = all(col in columns for col in ("ImageId_ClassId", "EncodedPixels"))
        if not wide_format and not combined_format:
            raise ValueError(
                "train.csv needs ImageId,ClassId,EncodedPixels or "
                f"ImageId_ClassId,EncodedPixels columns; found {columns}"
            )
        rle_idx = columns.index("EncodedPixels")
        image_idx = columns.index("ImageId" if wide_format else "ImageId_ClassId")
        class_idx = columns.index("ClassId") if wide_format else None
        for line_no, fields in enumerate(reader, start=2):
            if not fields or all(not item.strip() for item in fields):
                continue
            if len(fields) <= max(image_idx, rle_idx, class_idx or 0):
                raise ValueError(f"Malformed CSV row {line_no}")
            raw_name = fields[image_idx].strip()
            if wide_format:
                class_text = fields[class_idx].strip()
            else:
                if "_" not in raw_name:
                    raise ValueError(f"Cannot extract ClassId from CSV row {line_no}: {raw_name}")
                raw_name, class_text = raw_name.rsplit("_", 1)
            image = raw_name.replace("\\", "/").rsplit("/", 1)[-1]
            rle = fields[rle_idx].strip()
            if not image or not class_text:
                raise ValueError(f"ImageId/ClassId missing on CSV row {line_no}")
            try:
                class_id = int(class_text)
            except ValueError as exc:
                raise ValueError(f"Invalid ClassId on CSV row {line_no}: {class_text}") from exc
            if class_id not in CLASS_IDS:
                raise ValueError(f"Unexpected ClassId={class_id} on CSV row {line_no}")
            if rle and rle.lower() != "nan":
                annotations.setdefault(image, [[] for _ in CLASS_IDS])[class_id - 1].append(rle)
    return {image: tuple(tuple(items) for items in per_class)
            for image, per_class in annotations.items()}


def _apportion(total: int, ratios: np.ndarray, ensure_nonempty: bool = False) -> np.ndarray:
    """Round target split counts while keeping the sum exactly equal to total."""
    ideal = total * ratios
    counts = np.floor(ideal).astype(np.int64)
    for i in np.argsort(-(ideal - counts), kind="stable")[:total - int(counts.sum())]:
        counts[i] += 1
    if ensure_nonempty:
        if total < len(ratios):
            raise ValueError("At least three images are needed for train/val/test")
        for i in range(len(ratios)):
            if counts[i] == 0:
                donor = int(np.argmax(counts))
                if counts[donor] < 2:
                    raise ValueError("Cannot create three nonempty splits")
                counts[donor] -= 1
                counts[i] += 1
    return counts


def _validate_image_split(split: dict[str, list[str]], image_names: Sequence[str]) -> None:
    """All three splits must partition only the original training image pool."""
    if set(split) != {"train", "val", "test"}:
        raise ValueError("Require train, val and test image splits")
    seen: set[str] = set()
    for kind in ("train", "val", "test"):
        names = split[kind]
        if not names or len(names) != len(set(names)):
            raise ValueError(f"{kind} split is empty or contains duplicate images")
        if seen.intersection(names):
            raise ValueError("Train, validation and test image IDs must not overlap")
        seen.update(names)
    if seen != set(image_names):
        raise ValueError("Splits must cover exactly the original train image pool")


def split_image_pool(
    image_names: Sequence[str],
    annotations: dict[str, tuple[tuple[str, ...], ...]],
    seed: int = DEFAULT_SEED,
    val_fraction: float = 0.10,
    test_fraction: float = 0.10,
) -> dict[str, list[str]]:
    """Deterministic multi-label-stratified image split; never split CSV RLE rows.

    The five labels used for stratification are the four defect classes plus the
    all-zero group. Rare labels are assigned first; target split capacities and
    class frequencies are balanced with a fast greedy allocation (no sklearn).
    All RLE rows of one image remain exclusively in a single split.
    """
    if not (math.isfinite(val_fraction) and math.isfinite(test_fraction)
            and val_fraction > 0 and test_fraction > 0
            and val_fraction + test_fraction < 1):
        raise ValueError("Require val_fraction > 0, test_fraction > 0, and sum < 1")
    names = sorted(image_names)
    if len(names) != len(set(names)):
        raise ValueError("Image pool contains duplicate filenames")
    if len(names) < 3:
        raise ValueError("At least three images are required for train/val/test")
    n = len(names)
    ratios = np.array((1 - val_fraction - test_fraction,
                       val_fraction, test_fraction), dtype=np.float64)
    capacities = _apportion(n, ratios, ensure_nonempty=True)
    labels = np.zeros((n, NUM_CLASSES + 1), dtype=np.uint8)
    for i, name in enumerate(names):
        rles = annotations.get(name, ((),) * NUM_CLASSES)
        for c in range(NUM_CLASSES):
            labels[i, c] = int(bool(rles[c]))
        labels[i, -1] = int(not labels[i, :NUM_CLASSES].any())
    rng = np.random.default_rng(seed)
    priorities = [rng.permutation(np.flatnonzero(labels[:, c])).tolist()
                  for c in range(labels.shape[1])]
    remaining = labels.sum(axis=0, dtype=np.int64)
    target_labels = np.column_stack([
        _apportion(int(count), ratios) for count in remaining
    ])
    assigned_labels = np.zeros_like(target_labels)
    assigned_sizes = np.zeros(3, dtype=np.int64)
    assignment = np.full(n, -1, dtype=np.int8)
    # Break ties in a seed-controlled order rather than always favoring train.
    split_rank = np.empty(3, dtype=np.int64)
    split_rank[rng.permutation(3)] = np.arange(3)
    for _ in range(n):
        existing = np.flatnonzero(remaining)
        label = int(existing[np.argmin(remaining[existing])])
        pool = priorities[label]
        while pool and assignment[pool[-1]] != -1:
            pool.pop()
        if not pool:
            raise RuntimeError("Internal split allocation error")
        sample = int(pool.pop())
        available = np.flatnonzero(assigned_sizes < capacities)
        active = np.flatnonzero(labels[sample])
        primary_deficit = target_labels[available, label] - assigned_labels[available, label]
        context_deficit = np.maximum(
            0, target_labels[available][:, active] - assigned_labels[available][:, active]
        ).sum(axis=1)
        space = (capacities[available] - assigned_sizes[available]) / capacities[available]
        score = (10.0 * primary_deficit + 0.10 * context_deficit
                 + 0.01 * space - 0.0001 * split_rank[available])
        chosen = int(available[np.argmax(score)])
        assignment[sample] = chosen
        assigned_sizes[chosen] += 1
        assigned_labels[chosen, active] += 1
        remaining[active] -= 1
    if not np.array_equal(assigned_sizes, capacities):
        raise RuntimeError("Internal split size mismatch")
    result = {kind: [names[i] for i in np.flatnonzero(assignment == index)]
              for index, kind in enumerate(("train", "val", "test"))}
    _validate_image_split(result, names)
    return result


def save_split_manifest(
    checkpoint_dir: str | os.PathLike,
    csv_path: str | os.PathLike,
    image_dir: str | os.PathLike,
    split: dict[str, list[str]],
    seed: int,
    val_fraction: float,
    test_fraction: float,
) -> str:
    """Record split and CSV fingerprint; reject accidental changes on resumed runs."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    csv_digest = hashlib.sha256(Path(csv_path).read_bytes()).hexdigest()
    payload = {
        "seed": int(seed),
        "val_fraction": float(val_fraction),
        "test_fraction": float(test_fraction),
        "csv_sha256": csv_digest,
        "images": split,
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest = {
        "description": "Severstal RLE train/val/test image-level split; original train.csv unchanged",
        "image_dir": str(Path(image_dir).resolve()),
        "train_csv": str(Path(csv_path).resolve()),
        "split_fingerprint": fingerprint,
        **payload,
    }
    manifest_file = checkpoint_dir / "data_split.json"
    if manifest_file.exists():
        existing = json.loads(manifest_file.read_text(encoding="utf-8"))
        if existing.get("split_fingerprint") != fingerprint:
            raise RuntimeError(
                f"Data split differs from existing {manifest_file}. "
                "Use a fresh checkpoint_dir for a different seed/split/CSV."
            )
    else:
        manifest_file.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
        )
    print(f"Split manifest: {manifest_file} | sha256={fingerprint[:12]}")
    return fingerprint


def list_image_files(images_dir: str | os.PathLike) -> list[str]:
    folder = Path(images_dir)
    if not folder.is_dir():
        raise FileNotFoundError(f"Image directory not found: {folder}")
    files = sorted(p.name for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise RuntimeError(f"No images in {folder}")
    return files


class _SeverstalTransforms:
    """Lightweight aligned augmentations and ImageNet normalization, without Albumentations."""

    def __init__(self, is_train: bool):
        self.is_train = is_train

    def __call__(self, *, image: np.ndarray, mask: np.ndarray) -> dict[str, torch.Tensor]:
        if self.is_train:
            if random.random() < 0.5:
                image = np.flip(image, axis=1)
                mask = np.flip(mask, axis=1)
            if random.random() < 0.2:
                gain = random.uniform(0.85, 1.15)
                offset = random.uniform(-0.10, 0.10) * 255
                image = np.clip(image.astype(np.float32) * gain + offset, 0, 255).astype(np.uint8)
        image_tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
        image_tensor.div_(255.0).sub_(IMAGENET_MEAN).div_(IMAGENET_STD)
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask.transpose(2, 0, 1))).float()
        return {"image": image_tensor, "mask": mask_tensor}


def get_transforms(
    image_size: Sequence[int] = DEFAULT_IMAGE_SIZE,
    is_train: bool = True,
    seed: int = DEFAULT_SEED,
) -> _SeverstalTransforms:
    """Normalization is independent of crop geometry; cropping happens in Dataset."""
    validate_image_geometry(image_size)
    del seed  # DataLoader worker_init_fn seeds Python random for reproducible augmentation
    return _SeverstalTransforms(is_train)


class SeverstalSegmentationDataset(Dataset):
    """RLE masks; one native-resolution, optionally defect-aware crop per train image."""

    def __init__(
        self,
        images_list: Sequence[str],
        images_dir: str | os.PathLike,
        annotations: dict[str, tuple[tuple[str, ...], ...]],
        transforms=None,
        crop_size: Sequence[int] | None = None,
        positive_crop_prob: float = 0.7,
    ):
        self.images_list = list(images_list)
        self.images_dir = Path(images_dir)
        self.annotations = annotations
        self.transforms = transforms
        self.crop_size = tuple(int(v) for v in crop_size) if crop_size is not None else None
        self.positive_crop_prob = float(positive_crop_prob)
        self._augmentation_seed: int | None = None
        self._augmentation_epoch: torch.Tensor | None = None
        if not 0.0 <= self.positive_crop_prob <= 1.0:
            raise ValueError("positive_crop_prob must lie in [0,1]")
        if self.crop_size is not None:
            validate_image_geometry(self.crop_size)

    def __len__(self) -> int:
        return len(self.images_list)

    def _enable_reproducible_augmentation(self, seed: int) -> None:
        # A shared CPU scalar lets persistent DataLoader workers see the epoch.
        self._augmentation_seed = int(seed)
        self._augmentation_epoch = torch.zeros((), dtype=torch.int64, device="cpu").share_memory_()

    def _set_epoch(self, epoch: int) -> None:
        if self._augmentation_epoch is not None:
            self._augmentation_epoch.fill_(int(epoch))

    def _crop(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self.crop_size is not None
        crop_h, crop_w = self.crop_size
        h, w = image.shape[:2]
        pad_h = max(crop_h - h, 0)
        pad_w = max(crop_w - w, 0)
        if pad_h or pad_w:
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
            mask = np.pad(mask, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
            h, w = image.shape[:2]

        target = None
        if random.random() < self.positive_crop_prob:
            present_classes = [c for c in range(NUM_CLASSES) if mask[..., c].any()]
            if present_classes:
                selected = random.choice(present_classes)
                foreground_locations = np.flatnonzero(mask[..., selected])
                pixel = int(foreground_locations[random.randrange(len(foreground_locations))])
                target = divmod(pixel, w)

        if target is None:
            top = random.randint(0, h - crop_h)
            left = random.randint(0, w - crop_w)
        else:
            row, col = target
            top = min(max(row - random.randrange(crop_h), 0), h - crop_h)
            left = min(max(col - random.randrange(crop_w), 0), w - crop_w)
        return (
            image[top:top + crop_h, left:left + crop_w],
            mask[top:top + crop_h, left:left + crop_w],
        )

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self._augmentation_seed is None:
            return self._load_item(idx)
        assert self._augmentation_epoch is not None
        # The crop/augmentation distribution is unchanged. Its random draw is
        # stable across worker assignments, prefetching, and resumed runs.
        key = f"{self._augmentation_seed}:{int(self._augmentation_epoch.item())}:{self.images_list[idx]}"
        item_seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little")
        python_state, numpy_state = random.getstate(), np.random.get_state()
        try:
            random.seed(item_seed)
            np.random.seed(item_seed % (2**32))
            return self._load_item(idx)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)

    def _load_item(self, idx: int) -> dict[str, torch.Tensor]:
        filename = self.images_list[idx]
        with Image.open(self.images_dir / filename) as pil_image:
            image = np.asarray(pil_image.convert("RGB"), dtype=np.uint8)
        h, w = image.shape[:2]
        mask = np.zeros((h, w, NUM_CLASSES), dtype=np.uint8)
        for class_idx, rles in enumerate(self.annotations.get(filename, ((),) * NUM_CLASSES)):
            for rle in rles:
                mask[..., class_idx] |= decode_severstal_rle(rle, h, w)
        if self.crop_size is not None:
            image, mask = self._crop(image, mask)
        if self.transforms is not None:
            processed = self.transforms(image=image, mask=mask)
            image_tensor, mask_tensor = processed["image"], processed["mask"]
        else:
            image_tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
            mask_tensor = torch.from_numpy(np.ascontiguousarray(mask.transpose(2, 0, 1))).float()
        if mask_tensor.ndim != 3 or mask_tensor.shape[0] != NUM_CLASSES:
            raise RuntimeError(f"Four-channel mask required, received {tuple(mask_tensor.shape)}")
        return {"image": image_tensor, "mask": mask_tensor}


class Classical6ChannelOperator(nn.Module):
    """Four-parameter nonlinear cross-channel control, paired with the 6-qubit operator."""

    n_qubits = 6

    def __init__(self, batch_circuit_size: int = 4096):
        super().__init__()
        del batch_circuit_size
        self.weights = nn.Parameter(torch.tensor([0.21, -0.19, 0.17, 0.23]))

    @staticmethod
    def _pairs(x: torch.Tensor, theta: torch.Tensor, pairs: tuple[tuple[int, int], ...]) -> torch.Tensor:
        c, s = torch.cos(theta), torch.sin(theta)
        out = x.clone()
        for left, right in pairs:
            a, b = x[:, left], x[:, right]
            out[:, left] = c * a - s * b
            out[:, right] = s * a + c * b
        return out

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[-1] != 6:
            raise ValueError("Classical operator expects [N,6]")
        theta = self.weights
        y = torch.tanh(inputs)
        y = self._pairs(y, theta[0], ((0, 1), (2, 3), (4, 5)))
        y = torch.tanh(y)
        y = self._pairs(y, theta[1], ((1, 2), (3, 4)))
        y = torch.sin(0.5 * math.pi * y + theta[2])
        y = self._pairs(y, theta[2], ((0, 2),))
        return torch.tanh(y + torch.sin(theta[3]) * y.mean(dim=1, keepdim=True))


class QuFeX6QuantumOperator(nn.Module):
    """Phase-aware six-qubit QuFeX-inspired operator with four trainable angles.

    The six input features are angle-encoded with RY(pi*x). After the first
    entangling CNOT layer, phase is re-encoded with RZ(theta_2 + 0.25*pi*x), so
    the circuit can express input-dependent phase correlations. Instead of six
    Pauli-Z marginals (which are initially dominated by a nearly constant
    population bias for small x), use two Pauli-Z, two (X+Y)/sqrt(2) and two
    Pauli-X expectations. The fixed mixed axes retain phase sensitivity while
    adding X sensitivity; there are still six bounded outputs and four angles.

    The circuit shares four trainable angles over two groups of six features;
    there are no new learnable weights, intermediate maps or trainable gates.
    State-vector operations are batched torch complex64 GEMMs and index_select;
    no CPU/GPU copies occur in forward or backward.
    """

    n_qubits = 6

    def __init__(self, batch_circuit_size: int = 4096):
        super().__init__()
        if int(batch_circuit_size) < 1:
            raise ValueError("batch_circuit_size must be positive")
        self.batch_circuit_size = int(batch_circuit_size)
        self.weights = nn.Parameter(torch.tensor([0.21, -0.19, 0.17, 0.23]))
        basis = torch.arange(1 << self.n_qubits, dtype=torch.long)
        for suffix, pairs in (
            ("primary", ((0, 1), (2, 3), (4, 5))),
            ("bridge", ((1, 2), (3, 4))),
            ("coarse", ((0, 2),)),
        ):
            perm = basis.clone()
            for control, target in pairs:
                cbit, tbit = (1 << (5 - control)), (1 << (5 - target))
                perm ^= ((perm & cbit) != 0).long() * tbit
            inverse = torch.empty_like(perm)
            inverse[perm] = basis
            self.register_buffer(f"_cnot_{suffix}", inverse, persistent=False)
        z_signs = [torch.where((basis & (1 << (5 - wire))) == 0, 1., -1.)
                   for wire in range(6)]
        self.register_buffer("_z_signs", torch.stack(z_signs), persistent=False)
        # One gather supplies mixed X/Y(q=2,3) and X(q=4,5) observables.
        # Basis indexing is big-endian, with q=0 the most significant bit.
        flips = torch.stack([basis ^ (1 << (5 - wire)) for wire in (2, 3, 4, 5)])
        self.register_buffer("_readout_flips", flips.reshape(-1), persistent=False)
        self.phase_reupload_strength = 0.25  # fixed hyperparameter, not trainable

    @property
    def quantum_parameter_count(self) -> int:
        return self.weights.numel()

    @staticmethod
    def _one_qubit_matrix(theta: torch.Tensor, gate: str) -> torch.Tensor:
        """2x2 unitary with gradients intact; used to build 64x64 block matrices."""
        c, s = torch.cos(theta * 0.5), torch.sin(theta * 0.5)
        if gate == "ry":
            return torch.stack((torch.stack((c, -s)), torch.stack((s, c)))).to(torch.complex64)
        if gate == "rx":
            minus_is = torch.complex(torch.zeros_like(s), -s)
            cc = torch.complex(c, torch.zeros_like(c))
            return torch.stack((torch.stack((cc, minus_is)),
                                torch.stack((minus_is, cc))))
        raise ValueError(f"Unsupported rotation gate {gate}")

    @staticmethod
    def _six_qubit_unitary(single_qubit_gates: list[torch.Tensor]) -> torch.Tensor:
        """Kronecker product of six simultaneous single-qubit rotations."""
        matrix = single_qubit_gates[0]
        for gate in single_qubit_gates[1:]:
            matrix = torch.kron(matrix, gate)
        return matrix

    def _simulate_chunk(self, inputs: torch.Tensor) -> torch.Tensor:
        """Fused statevector: two complex GEMMs and three CNOT permutations.

        RY(pi*x) -> RY/RX -> CNOT -> RZ(theta+0.25*pi*x) -> CNOT ->
        RY -> CNOT -> six complementary Z/(X+Y)/sqrt(2)/X expectations.
        Each operation is differentiable and executed on the input tensor device.
        """
        batch = len(inputs)
        half = (0.5 * math.pi) * inputs
        initial_qubits = torch.stack((torch.cos(half), torch.sin(half)), dim=-1)
        # Tensor product of the six data-encoded |0> states, with wire 0 MSB.
        state = initial_qubits[:, 0, :]
        for wire in range(1, 6):
            state = (state.unsqueeze(-1) * initial_qubits[:, wire, :].unsqueeze(1)).reshape(batch, -1)
        state = state.to(torch.complex64)

        ry0 = self._one_qubit_matrix(self.weights[0], "ry")
        rx1 = self._one_qubit_matrix(self.weights[1], "rx")
        first_layer = self._six_qubit_unitary([ry0, rx1, ry0, rx1, ry0, rx1])
        state = state @ first_layer.T
        state = state.index_select(1, self._cnot_primary)

        # Fixed data re-uploading adds an input-dependent RZ phase, in addition
        # to theta_2. All six phase rotations commute: evaluate the diagonal
        # jointly in one batched matrix multiplication, without gate loops.
        phase = (
            0.5 * self.weights[2] * self._z_signs.sum(dim=0)
            + (0.5 * math.pi * self.phase_reupload_strength)
            * (inputs @ self._z_signs)
        )
        state = state * torch.complex(torch.cos(phase), -torch.sin(phase))
        state = state.index_select(1, self._cnot_bridge)

        ry3 = self._one_qubit_matrix(self.weights[3], "ry")
        final_layer = self._six_qubit_unitary([ry3] * 6)
        state = state @ final_layer.T
        state = state.index_select(1, self._cnot_coarse)

        # q0,q1: <Z>. q2,q3: <(X+Y)/sqrt(2)>. q4,q5: <X>.
        # For bit-sign s(i) and bit-flipped index flip(i):
        # <Y> = sum_i s(i) Im(conj(psi_i) * psi_flip(i)),
        # <X> = sum_i Re(conj(psi_i) * psi_flip(i)).
        z = (state.abs().square()) @ self._z_signs[:2].T
        flip_states = state.index_select(1, self._readout_flips)
        overlap = state.conj().unsqueeze(1) * flip_states.reshape(batch, 4, -1)
        y = (overlap[:, :2].imag * self._z_signs[2:4].unsqueeze(0)).sum(dim=-1)
        x_readouts = overlap.real.sum(dim=-1)
        mixed_xy = (x_readouts[:, :2] + y) * (1.0 / math.sqrt(2.0))
        return torch.cat((z, mixed_xy, x_readouts[:, 2:]), dim=-1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[-1] != 6:
            raise ValueError("Quantum operator expects [N,6]")
        # Complex64 has no fp16 autocast implementation; preserve differentiability
        # by promoting only this small branch to float32, not the CNN.
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            x = inputs.float()
            parts = [self._simulate_chunk(x[start:start + self.batch_circuit_size])
                     for start in range(0, len(x), self.batch_circuit_size)]
            return torch.cat(parts, dim=0).to(dtype=inputs.dtype)


class CrossChannelMixer621(nn.Module):
    """Project 64->12, mix two six-channel groups, project 12->64 and add residual."""

    def __init__(self, width: int, model_type: str, batch_circuit_size: int = 4096):
        super().__init__()
        model_type = _normalize_model_type(model_type)
        self.pre = nn.Conv2d(width, 12, kernel_size=1, bias=False)
        operator = QuFeX6QuantumOperator if model_type == "hybrid" else Classical6ChannelOperator
        self.operator = operator(batch_circuit_size=batch_circuit_size)
        self.post = nn.Conv2d(12, width, kernel_size=1, bias=False)
        self.gain = nn.Parameter(torch.tensor(0.20))
        self.pool_size = (4, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        pooled = F.adaptive_avg_pool2d(x, (min(h, self.pool_size[0]), min(w, self.pool_size[1])))
        features = torch.tanh(self.pre(pooled))
        _, _, ph, pw = features.shape
        packed = features.permute(0, 2, 3, 1).reshape(-1, 6)
        mixed = self.operator(packed)
        unpacked = mixed.reshape(b, ph, pw, 12).permute(0, 3, 1, 2).contiguous()
        residual = self.post(unpacked)
        residual = F.interpolate(residual, size=(h, w), mode="bilinear", align_corners=False)
        return x + torch.tanh(self.gain) * residual


class DepthwiseSeparableFPNBlock(nn.Module):
    """GPU-friendly lightweight FPN smoothing at each scale."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MatchedEfficientFPN(nn.Module):
    """Same FPN architecture in both modes; operator inside P5 is the only switch."""

    def __init__(
        self,
        model_type: str,
        in_channels: int = 3,
        base_channels: int = 64,
        out_classes: int = NUM_CLASSES,
        batch_circuit_size: int = 4096,
        pretrained_encoder: bool = True,
    ):
        super().__init__()
        self.model_type = _normalize_model_type(model_type)
        if in_channels != 3:
            raise ValueError("ImageNet-pretrained EfficientNet-B0 requires RGB in_channels=3")
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained_encoder else None
        try:
            # Stage endpoints: strides 4,8,16,32 with channels 24,40,112,320.
            self.encoder = efficientnet_b0(weights=weights).features[:8]
        except Exception as exc:
            if pretrained_encoder:
                raise RuntimeError(
                    "Cannot obtain ImageNet EfficientNet-B0 weights. Download/cache them "
                    "once or explicitly set pretrained_encoder=False (lower accuracy)."
                ) from exc
            raise
        self.stage_indices = (2, 3, 5, 7)
        self.lateral = nn.ModuleList([nn.Conv2d(ch, base_channels, 1)
                                      for ch in (24, 40, 112, 320)])
        self.bottleneck = CrossChannelMixer621(
            base_channels, self.model_type, batch_circuit_size=batch_circuit_size
        )
        self.smooth = nn.ModuleList([DepthwiseSeparableFPNBlock(base_channels)
                                     for _ in range(4)])
        self.fusion = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.SiLU(inplace=True),
        )
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1, groups=base_channels, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels, out_classes, kernel_size=1),
        )
        # Do not re-initialize the pretrained EfficientNet encoder.

    @staticmethod
    def _resize(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(x, size=size, mode="nearest")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_size = x.shape[-2:]
        if original_size[0] % 32 or original_size[1] % 32:
            raise ValueError(f"Model input {original_size} must be divisible by 32")
        stages = []
        for index, layer in enumerate(self.encoder):
            x = layer(x)
            if index in self.stage_indices:
                stages.append(x)
        c2, c3, c4, c5 = stages
        l2, l3, l4, l5 = (projection(features)
                           for projection, features in zip(self.lateral, (c2, c3, c4, c5)))
        p5 = self.bottleneck(l5)
        p4 = l4 + self._resize(p5, l4.shape[-2:])
        p3 = l3 + self._resize(p4, l3.shape[-2:])
        p2 = l2 + self._resize(p3, l2.shape[-2:])
        pyramids = [smoothing(p) for smoothing, p in zip(self.smooth, (p2, p3, p4, p5))]
        p2_size = pyramids[0].shape[-2:]
        fused = self.fusion(torch.cat([self._resize(p, p2_size) for p in pyramids], dim=1))
        logits = self.segmentation_head(fused)
        return F.interpolate(logits, size=original_size, mode="bilinear", align_corners=False)


def build_model(
    model_type: str,
    in_channels: int = 3,
    base_channels: int = 64,
    out_classes: int = NUM_CLASSES,
    batch_circuit_size: int = 4096,
    pretrained_encoder: bool = True,
) -> nn.Module:
    return MatchedEfficientFPN(
        model_type=model_type,
        in_channels=in_channels,
        base_channels=base_channels,
        out_classes=out_classes,
        batch_circuit_size=batch_circuit_size,
        pretrained_encoder=pretrained_encoder,
    )


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_keras_equivalent_parameters(model: nn.Module) -> int:
    return count_trainable_parameters(model) + sum(
        2 * m.num_features for m in model.modules()
        if isinstance(m, nn.BatchNorm2d) and m.track_running_stats
    )


class BCEDiceLoss(nn.Module):
    """Supervise foreground and explicitly penalize predictions on empty masks."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight, self.dice_weight = float(bce_weight), float(dice_weight)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape != targets.shape:
            raise ValueError(f"Logits/targets mismatch: {logits.shape} != {targets.shape}")
        logits, targets = logits.float(), targets.float()
        bce = self.bce(logits, targets)
        probs = logits.sigmoid()
        tp = (probs * targets).sum(dim=(0, 2, 3))
        volume = probs.sum(dim=(0, 2, 3)) + targets.sum(dim=(0, 2, 3))
        # Keep this fully on GPU: Python bool(present.any()) would synchronize
        # CUDA every iteration and slow down both training variants.
        present = (targets.sum(dim=(0, 2, 3)) > 0).to(dtype=probs.dtype)
        dice_per_class = 1.0 - (2.0 * tp + 1.0) / (volume + 1.0)
        dice = (dice_per_class * present).sum() / present.sum().clamp_min(1)
        return self.bce_weight * bce + self.dice_weight * dice


def _multilabel_hard_stats(
    predictions: torch.Tensor, targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if predictions.shape != targets.shape:
        raise ValueError("Prediction/target masks must have same shape")
    pred, truth = predictions.bool(), targets.bool()
    dims = (0, 2, 3)
    tp = (pred & truth).sum(dims).float()
    fp = (pred & ~truth).sum(dims).float()
    fn = (~pred & truth).sum(dims).float()
    return tp, fp, fn


def _nanmean(values: torch.Tensor) -> float:
    present = ~torch.isnan(values)
    return float(values[present].mean().item()) if bool(present.any()) else float("nan")


def _finalize_epoch_metrics(
    loss: float, tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor,
) -> dict[str, float]:
    iou_den = tp + fp + fn
    dice_den = 2 * tp + fp + fn
    iou = torch.where(iou_den > 0, tp / iou_den.clamp_min(1), float("nan"))
    dice = torch.where(dice_den > 0, 2 * tp / dice_den.clamp_min(1), float("nan"))
    result = {"loss": float(loss), "dice": _nanmean(dice), "iou": _nanmean(iou)}
    result["f1"] = result["dice"]  # Same hard-mask metric; preserve both CSV columns.
    return result


def _amp_context(device: torch.device, dtype: torch.dtype | None):
    return torch.autocast(device_type=device.type,
                          dtype=dtype if dtype is not None else torch.float16,
                          enabled=(device.type == "cuda" and dtype is not None))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    threshold: float,
    epoch: int,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    amp_dtype: torch.dtype | None = None,
) -> dict[str, float]:
    model.train()
    loss_sum = torch.zeros((), device=device)
    tp_sum = torch.zeros(NUM_CLASSES, device=device)
    fp_sum, fn_sum = torch.zeros_like(tp_sum), torch.zeros_like(tp_sum)
    seen = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch} Training", leave=False)
    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device, non_blocking=True)
        if device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
        masks = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _amp_context(device, amp_dtype):
            logits = model(images)
            loss = loss_fn(logits, masks)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            tp, fp, fn = _multilabel_hard_stats(logits.sigmoid() > threshold, masks > .5)
            tp_sum += tp
            fp_sum += fp
            fn_sum += fn
            seen += len(images)
            loss_sum += loss.detach() * len(images)
            if batch_idx % 25 == 0:
                pbar.set_postfix(loss=f"{loss_sum.item() / seen:.4f}")
    metrics = _finalize_epoch_metrics(loss_sum.item() / max(seen, 1), tp_sum, fp_sum, fn_sum)
    print(f"Epoch {epoch} TRAIN loss: {metrics['loss']:.4f} "
          f"Dice: {metrics['dice']:.4f} IoU@0.5: {metrics['iou']:.4f} F1: {metrics['f1']:.4f}")
    return metrics


def _tile_starts(full_size: int, crop_size: int, overlap: int) -> list[int]:
    if not 0 <= overlap < crop_size:
        raise ValueError("Tile overlap must be >=0 and smaller than crop size")
    last = max(0, full_size - crop_size)
    if last == 0:
        return [0]
    starts = list(range(0, last + 1, crop_size - overlap))
    if starts[-1] != last:
        starts.append(last)
    return starts


@torch.no_grad()
def predict_full_logits(
    model: nn.Module,
    images: torch.Tensor,
    crop_size: Sequence[int] = DEFAULT_IMAGE_SIZE,
    overlap: int = 128,
    tile_batch_size: int = 4,
) -> torch.Tensor:
    """Overlap-and-average logits over entire images. No validation resizing."""
    ch, cw = tuple(int(v) for v in crop_size)
    validate_image_geometry((ch, cw))
    if tile_batch_size < 1:
        raise ValueError("tile_batch_size must be >= 1")
    n, _, height, width = images.shape
    if (height, width) == (ch, cw):
        return model(images).float()
    if height < ch or width < cw:
        images = F.pad(images, (0, max(0, cw - width), 0, max(0, ch - height)))
    padded_h, padded_w = images.shape[-2:]
    y_positions = _tile_starts(padded_h, ch, min(overlap, ch - 1))
    x_positions = _tile_starts(padded_w, cw, min(overlap, cw - 1))
    result = torch.zeros((n, NUM_CLASSES, padded_h, padded_w), device=images.device)
    counts = torch.zeros((1, 1, padded_h, padded_w), device=images.device)
    positions = [(row, col) for row in y_positions for col in x_positions]
    for index in range(n):
        accumulated = torch.zeros_like(result[:1])
        counts.zero_()
        for start in range(0, len(positions), tile_batch_size):
            points = positions[start:start + tile_batch_size]
            tiles = torch.cat([images[index:index + 1, :, row:row + ch, col:col + cw]
                               for row, col in points], dim=0)
            if tiles.is_cuda:
                tiles = tiles.contiguous(memory_format=torch.channels_last)
            batch_logits = model(tiles).float()
            for tile_logits, (row, col) in zip(batch_logits, points):
                accumulated[:, :, row:row + ch, col:col + cw] += tile_logits.unsqueeze(0)
                counts[:, :, row:row + ch, col:col + cw] += 1
        result[index:index + 1] = accumulated / counts.clamp_min(1)
    return result[..., :height, :width]


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    threshold: float,
    epoch: int,
    device: torch.device,
    image_size: Sequence[int] = DEFAULT_IMAGE_SIZE,
    overlap: int = 128,
    tile_batch_size: int = 4,
    amp_dtype: torch.dtype | None = None,
    split_name: str = "VAL",
) -> dict[str, float]:
    model.eval()
    loss_sum = torch.zeros((), device=device)
    tp_sum = torch.zeros(NUM_CLASSES, device=device)
    fp_sum, fn_sum = torch.zeros_like(tp_sum), torch.zeros_like(tp_sum)
    seen = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch} {split_name}", leave=False)
    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with _amp_context(device, amp_dtype):
            logits = predict_full_logits(model, images, image_size, overlap, tile_batch_size)
            loss = loss_fn(logits, masks)
        tp, fp, fn = _multilabel_hard_stats(logits.sigmoid() > threshold, masks > .5)
        tp_sum += tp
        fp_sum += fp
        fn_sum += fn
        seen += len(images)
        loss_sum += loss.detach() * len(images)
        if batch_idx % 25 == 0:
            pbar.set_postfix(loss=f"{loss_sum.item() / seen:.4f}")
    metrics = _finalize_epoch_metrics(loss_sum.item() / max(seen, 1), tp_sum, fp_sum, fn_sum)
    print(f"Epoch {epoch} {split_name:>5} loss: {metrics['loss']:.4f} "
          f"Dice: {metrics['dice']:.4f} IoU@0.5: {metrics['iou']:.4f} F1: {metrics['f1']:.4f}")
    return metrics


def _print_split_summary(
    split_name: str,
    image_names: Sequence[str],
    annotations: dict[str, tuple[tuple[str, ...], ...]],
) -> list[int]:
    positives = [0] * NUM_CLASSES
    with_defect = 0
    for name in image_names:
        per_class = annotations.get(name, ((),) * NUM_CLASSES)
        is_positive = False
        for cls, rles in enumerate(per_class):
            if rles:
                positives[cls] += 1
                is_positive = True
        with_defect += is_positive
    print(f"{split_name}: images={len(image_names)} | positive_images={with_defect} "
          f"| all-zero_images={len(image_names) - with_defect} | "
          + ", ".join(f"c{i}={value}" for i, value in zip(CLASS_IDS, positives)))
    return positives


def _capture_rng_state(train_rng: torch.Generator, device: torch.device) -> dict:
    state = {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(), "train_sampler": train_rng.get_state(),
    }
    if device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state(device)
    return state


def _restore_rng_state(state: dict | None, train_rng: torch.Generator,
                       device: torch.device) -> bool:
    if state is None:
        return False
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # map_location may have moved saved ByteTensors to CUDA; RNG APIs need CPU.
    torch.set_rng_state(state["torch_cpu"].cpu())
    train_rng.set_state(state["train_sampler"].cpu())
    if device.type == "cuda" and "torch_cuda" in state:
        torch.cuda.set_rng_state(state["torch_cuda"].cpu(), device)
    return True


def _validate_checkpoint(checkpoint: dict, model_type: str, split_fingerprint: str) -> None:
    saved_mode = str(checkpoint.get("model_type", "")).lower().strip()
    if saved_mode != model_type:
        raise ValueError("Checkpoint model_type differs from requested model_type")
    architecture = checkpoint.get("architecture")
    legacy_classic = architecture == LEGACY_ARCHITECTURE_ID and model_type == "classic"
    if architecture != ARCHITECTURE_ID and not legacy_classic:
        if architecture == LEGACY_ARCHITECTURE_ID and model_type == "hybrid":
            raise ValueError(
                "This hybrid checkpoint uses the old YY readout. The new fixed "
                "(X+Y)/sqrt(2) readout requires a fresh hybrid training run."
            )
        raise ValueError("Checkpoint architecture does not match this model")
    if checkpoint.get("split_fingerprint") != split_fingerprint:
        raise ValueError(
            "Checkpoint was trained on a different (or unrecorded) data split. "
            "Do not resume across train/validation/test splits."
        )


def _checkpoint_iou(checkpoint: dict) -> float:
    value = checkpoint.get("val_metrics", {}).get("iou")
    return float(value) if value is not None and math.isfinite(float(value)) else -math.inf


def _load_resume_checkpoints(
    start_checkpoint_dir: str | os.PathLike, model_type: str,
    split_fingerprint: str, device: torch.device,
) -> tuple[Path, dict, dict]:
    """Resume the latest available epoch and retain the best available VAL model."""
    folder = Path(start_checkpoint_dir)
    candidates = []
    for path in folder.glob("ckpt_epoch_*.pth"):
        number = path.stem.removeprefix("ckpt_epoch_")
        if path.is_file() and number.isdigit():
            candidates.append((int(number), path))
    best_path = folder / "best_model.pth"
    if candidates:
        ckpt_path = max(candidates, key=lambda entry: entry[0])[1]
    elif best_path.is_file():
        ckpt_path = best_path
    else:
        raise FileNotFoundError(f"No epoch checkpoint or best_model.pth in {folder}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    _validate_checkpoint(checkpoint, model_type, split_fingerprint)
    best_checkpoint = checkpoint
    if best_path.is_file() and best_path.resolve() != ckpt_path.resolve():
        saved_best = torch.load(best_path, map_location=device, weights_only=False)
        _validate_checkpoint(saved_best, model_type, split_fingerprint)
        if _checkpoint_iou(saved_best) >= _checkpoint_iou(checkpoint):
            best_checkpoint = saved_best
        # A retained best file can be newer than the remaining epoch files.
        if int(saved_best["epoch"]) > int(checkpoint["epoch"]):
            ckpt_path, checkpoint = best_path, saved_best
    if not math.isfinite(_checkpoint_iou(best_checkpoint)):
        raise RuntimeError("No available checkpoint has a finite VAL IoU")
    return ckpt_path, checkpoint, best_checkpoint


def _prepare_metrics_csv(path: Path, header: tuple[str, ...], resume_epoch: int | None) -> None:
    """Keep the existing CSV schema and align rows with the resumed checkpoint."""
    rows: dict[int, list[str]] = {}
    if resume_epoch is not None and path.is_file():
        with path.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.reader(stream)
            if tuple(next(reader, ())) != header:
                raise ValueError(f"Cannot append metrics with a different CSV header: {path}")
            for line_no, fields in enumerate(reader, start=2):
                if not fields:
                    continue
                if len(fields) != len(header):
                    raise ValueError(f"Malformed metrics CSV row {line_no}: {path}")
                epoch = int(fields[0])
                if epoch < 1:
                    raise ValueError(f"Invalid epoch on metrics CSV row {line_no}: {path}")
                if epoch <= resume_epoch:
                    rows[epoch] = fields  # Retain the last row if an old log duplicated an epoch.
    # Atomic replacement also handles a CSV row written just before a crash,
    # when its corresponding epoch checkpoint was not saved yet.
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows[epoch] for epoch in sorted(rows))
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def run_training(
    dataset_root: str | os.PathLike = "./data_severstal",
    in_channels: int = 3,
    epochs: int = 40,
    batch_size: int = 8,
    lr: float = 3e-4,
    model_type: str = "hybrid",
    image_size: Sequence[int] = DEFAULT_IMAGE_SIZE,
    checkpoint_dir: str = "checkpoints",
    start_checkpoint_dir: str | None = None,
    threshold: float = HARD_IOU_THRESHOLD,
    num_workers: int = 4,
    device: torch.device | str = DEVICE,
    seed: int = DEFAULT_SEED,
    savefile_name: str = "testsave",
    base_channels: int = 64,
    batch_circuit_size: int = 4096,
    pretrained_encoder: bool = True,
    positive_crop_prob: float = 0.7,
    val_overlap: int = 128,
    val_tile_batch_size: int = 4,
    use_amp: bool = True,
    deterministic: bool = False,
    val_fraction: float = 0.10,
    test_fraction: float = 0.10,
) -> nn.Module:
    """Train on a fixed train.csv-based image split; evaluate holdout test once.

    Preserve existing run_training parameters. `image_size` is a crop size, NOT
    an image resize; `epochs` is total epochs. All three splits come from the
    original train image pool and its RLEs, never external YOLO val/test.
    Resume the latest epoch; return the model selected only by best VAL IoU.
    """
    model_type = _normalize_model_type(model_type)
    deep_size = validate_image_geometry(image_size)
    if threshold != HARD_IOU_THRESHOLD:
        raise ValueError("The primary comparable IoU threshold must be 0.5")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if batch_size < 1 or epochs < 1:
        raise ValueError("batch_size and epochs must be positive")
    set_global_seed(seed, deterministic=deterministic)

    root = Path(dataset_root)
    # Only the original training image pool has segmentation ground truth in
    # train.csv; never mistake YOLO val/images or test_images for labeled masks.
    train_dir = root / "train" / "images"
    if not train_dir.is_dir() and (root / "train").is_dir():
        train_dir = root / "train"  # also support Kaggle train/*.jpg layout
    csv_path = root / "train.csv"
    annotations = load_severstal_annotations(csv_path)
    available_images = list_image_files(train_dir)
    image_set = set(available_images)
    matched_positives = image_set.intersection(annotations)
    if not matched_positives:
        csv_examples = sorted(annotations)[:4]
        image_examples = available_images[:4]
        raise RuntimeError(
            "No positive RLE masks match the available original training images. "
            f"train.csv has {len(annotations)} positive filenames, "
            f"train/images has {len(available_images)} image files. "
            f"Examples from CSV: {csv_examples}; image folder: {image_examples}. "
            "Check image filenames and dataset_root; empty YOLO boxes are not masks."
        )
    absent_positive = len(annotations.keys() - image_set)
    print(f"Dataset root: {root.resolve()}")
    print(f"Segmentation labels: {csv_path}; image pool: {train_dir}")
    print("YOLO val/labels, val/images and test_images are excluded from split")
    print(f"CSV positive images matched: {len(matched_positives)}/{len(annotations)}; "
          f"positive CSV filenames missing from pool: {absent_positive}")
    if absent_positive:
        print("WARNING: some positive RLE filenames are not in the selected train image pool. "
              f"Examples: {sorted(annotations.keys() - image_set)[:5]}")
    print("Images without positive CSV RLEs are assumed defect-free, as in the "
          "standard Severstal annotations.")
    split = split_image_pool(
        available_images, annotations, seed=seed,
        val_fraction=val_fraction, test_fraction=test_fraction,
    )
    train_images, val_images, test_images = (
        split["train"], split["val"], split["test"]
    )
    for name, imgs in split.items():
        _print_split_summary(name.upper(), imgs, annotations)
    if not any(any(annotations.get(name, ((),) * NUM_CLASSES)[c]
                   for c in range(NUM_CLASSES)) for name in val_images):
        raise RuntimeError(
            "The generated validation split has no positive defects; use a "
            "larger dataset or different split fractions/seed."
        )
    for cls in range(NUM_CLASSES):
        if any(not any(annotations.get(name, ((),) * NUM_CLASSES)[cls]
                       for name in imgs) for imgs in split.values()):
            print(f"WARNING: class {cls + 1} absent from at least one split; "
                  "interpret that split's macro scores carefully")

    split_fingerprint = save_split_manifest(
        checkpoint_dir, csv_path, train_dir, split,
        seed=seed, val_fraction=val_fraction, test_fraction=test_fraction,
    )
    train_ds = SeverstalSegmentationDataset(
        train_images, train_dir, annotations,
        transforms=get_transforms(image_size, True, seed), crop_size=image_size,
        positive_crop_prob=positive_crop_prob,
    )
    train_ds._enable_reproducible_augmentation(seed)
    val_ds = SeverstalSegmentationDataset(
        val_images, train_dir, annotations,
        transforms=get_transforms(image_size, False, seed), crop_size=None,
    )
    test_ds = SeverstalSegmentationDataset(
        test_images, train_dir, annotations,
        transforms=get_transforms(image_size, False, seed), crop_size=None,
    )
    train_rng = torch.Generator().manual_seed(seed)
    # Shuffle and worker seeds have separate streams. Recreating persistent
    # workers on resume must not consume an extra draw from the shuffle stream.
    worker_rng = torch.Generator().manual_seed(seed)
    val_rng = torch.Generator().manual_seed(seed)
    test_rng = torch.Generator().manual_seed(seed)
    common_kwargs = dict(
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        common_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=RandomSampler(train_ds, generator=train_rng),
                              generator=worker_rng, **common_kwargs)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            generator=val_rng, **common_kwargs)
    # Construct the test DataLoader only after training for a truly held-out test.

    set_global_seed(seed, deterministic=deterministic)
    model = build_model(
        model_type=model_type, in_channels=in_channels, base_channels=base_channels,
        out_classes=NUM_CLASSES, batch_circuit_size=batch_circuit_size,
        pretrained_encoder=pretrained_encoder,
    ).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    param_count = count_trainable_parameters(model)
    print(f"Model={model_type} | encoder=EfficientNet-B0 "
          f"| FPN_width={base_channels} | input_crop={tuple(image_size)} "
          f"| deep_size={deep_size} | trainable_params={param_count:,} "
          f"| keras_equivalent_total={count_keras_equivalent_parameters(model):,}")
    operator_backend = "torch-statevector" if model_type == "hybrid" else "torch-classical"
    print(f"Mixer=6-1-2 cross-channel | operator params=4 "
          f"| pooling=4x8 max | backend={device.type.upper()} {operator_backend}")
    if model_type == "hybrid":
        print("Quantum feature map=RY input + RZ data re-upload (0.25*pi*x) "
              "| Pauli readout=Z,Z,(X+Y)/sqrt(2),(X+Y)/sqrt(2),X,X | fast fused statevector")

    loss_fn = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    amp_dtype = None
    if use_amp and device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
    print(f"Device={device} | amp={str(amp_dtype)} | channels_last={device.type == 'cuda'} "
          f"| val_tile_batch_size={val_tile_batch_size} | val_overlap={val_overlap}")

    first_epoch = 1
    best_val_iou = -1.0
    if start_checkpoint_dir is not None:
        ckpt_path, checkpoint, best_checkpoint = _load_resume_checkpoints(
            start_checkpoint_dir, model_type, split_fingerprint, device,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint and "scheduler_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if checkpoint.get("scaler_state_dict") and scaler.is_enabled():
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            first_epoch = int(checkpoint["epoch"]) + 1
            if not _restore_rng_state(checkpoint.get("rng_state"), train_rng, device):
                print("WARNING: legacy checkpoint has no RNG state; exact replay is unavailable.")
        best_val_iou = _checkpoint_iou(best_checkpoint)
        retained_best = dict(best_checkpoint)
        retained_best["best_val_iou"] = best_val_iou
        # Seed the destination even if subsequent epochs never beat the old best.
        torch.save(retained_best, Path(checkpoint_dir) / "best_model.pth")
        print(f"Resumed {ckpt_path}: next epoch={first_epoch}")

    os.makedirs(checkpoint_dir, exist_ok=True)
    csv_path = Path(f"{savefile_name}.csv")
    csv_header = (
        "Epoch", "TRAIN_loss", "VAL_loss", "TRAIN_Dice", "VAL_Dice",
        "TRAIN_IoU@0.5", "VAL_IoU@0.5", "TRAIN_F1", "VAL_F1",
    )
    _prepare_metrics_csv(csv_path, csv_header,
                         first_epoch - 1 if start_checkpoint_dir is not None else None)
    for epoch in range(first_epoch, epochs + 1):
        train_ds._set_epoch(epoch)
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_fn, threshold, epoch, device,
            scaler=scaler, amp_dtype=amp_dtype,
        )
        val_metrics = validate_one_epoch(
            model, val_loader, loss_fn, threshold, epoch, device,
            image_size=image_size, overlap=val_overlap,
            tile_batch_size=val_tile_batch_size, amp_dtype=amp_dtype,
        )
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow((
                epoch, f"{train_metrics['loss']:.6f}", f"{val_metrics['loss']:.6f}",
                f"{train_metrics['dice']:.6f}", f"{val_metrics['dice']:.6f}",
                f"{train_metrics['iou']:.6f}", f"{val_metrics['iou']:.6f}",
                f"{train_metrics['f1']:.6f}", f"{val_metrics['f1']:.6f}",
            ))
        scheduler.step(val_metrics["loss"])
        val_iou = float(val_metrics["iou"])
        improved = not math.isnan(val_iou) and val_iou > best_val_iou
        if improved:
            best_val_iou = val_iou
        checkpoint = {
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
            "train_metrics": train_metrics, "val_metrics": val_metrics,
            "best_val_iou": best_val_iou, "model_type": model_type,
            "architecture": ARCHITECTURE_ID,
            "num_classes": NUM_CLASSES, "class_ids": CLASS_IDS,
            "image_size": tuple(image_size), "deep_size": deep_size,
            "base_channels": base_channels, "seed": seed,
            "dataset_root": str(root), "pretrained_encoder": pretrained_encoder,
            "split_fingerprint": split_fingerprint,
            "val_fraction": val_fraction, "test_fraction": test_fraction,
            "rng_state": _capture_rng_state(train_rng, device),
        }
        epoch_path = Path(checkpoint_dir) / f"ckpt_epoch_{epoch:03d}.pth"
        torch.save(checkpoint, epoch_path)
        print(f"Saved checkpoint: {epoch_path}")
        if improved:
            best_path = Path(checkpoint_dir) / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"New best (VAL IoU@0.5: {best_val_iou:.4f}): {best_path}")
        print(f"Epoch {epoch} finished. LR={optimizer.param_groups[0]['lr']:.6g}\n")
    print("Training complete. Evaluating held-out TEST once with best VAL checkpoint.")
    best_path = Path(checkpoint_dir) / "best_model.pth"
    if not best_path.is_file():
        raise RuntimeError("No valid best VAL checkpoint: cannot evaluate TEST")
    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             generator=test_rng, **common_kwargs)
    test_metrics = validate_one_epoch(
        model, test_loader, loss_fn, threshold, int(best_checkpoint["epoch"]), device,
        image_size=image_size, overlap=val_overlap, tile_batch_size=val_tile_batch_size,
        amp_dtype=amp_dtype, split_name="TEST",
    )
    result_path = Path(checkpoint_dir) / "test_metrics.json"
    result_path.write_text(json.dumps({
        "best_val_epoch": int(best_checkpoint["epoch"]),
        "split_fingerprint": split_fingerprint,
        # JSON has no NaN: absent classes may yield undefined macro metrics.
        "metrics": {key: (value if math.isfinite(value) else None)
                    for key, value in test_metrics.items()},
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Final TEST metrics: {result_path}; returning best VAL model")
    return model


def self_test() -> None:
    """CPU-only smoke tests including multi-class shapes, classical parity and gradients."""
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    set_global_seed(42, deterministic=True)
    assert validate_image_geometry((256, 512)) == (8, 16)
    for h, w, row, col in ((5, 7, 0, 3), (256, 512, 22, 499)):
        position = col * h + row + 1
        pixels = decode_severstal_rle(f"{position} 1", h, w)
        assert int(pixels.sum()) == 1 and pixels[row, col] == 1
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "images").mkdir()
        Image.fromarray(np.zeros((64, 192, 3), np.uint8)).save(td / "images" / "a.jpg")
        Image.fromarray(np.zeros((64, 192, 3), np.uint8)).save(td / "images" / "b.jpg")
        with (td / "train.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ImageId", "ClassId", "EncodedPixels"])
            writer.writerow(["a.jpg", 2, "1 2"])
            writer.writerow(["a.jpg", 4, "100 3"])
        ann = load_severstal_annotations(td / "train.csv")
        ds = SeverstalSegmentationDataset(["a.jpg", "b.jpg"], td / "images", ann,
                                          transforms=get_transforms((64, 128), True),
                                          crop_size=(64, 128), positive_crop_prob=1)
        item = ds[0]
        assert item["image"].shape == (3, 64, 128)
        assert item["mask"].shape == (4, 64, 128)
        assert item["mask"].sum() > 0
        assert ds[1]["mask"].sum() == 0
        ds._enable_reproducible_augmentation(42)
        ds._set_epoch(3)
        python_state, numpy_state = random.getstate(), np.random.get_state()
        first_item, repeated_item = ds[0], ds[0]
        torch.testing.assert_close(first_item["image"], repeated_item["image"], atol=0, rtol=0)
        torch.testing.assert_close(first_item["mask"], repeated_item["mask"], atol=0, rtol=0)
        assert random.getstate() == python_state
        np.testing.assert_equal(np.random.get_state(), numpy_state)
        with (td / "combined.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ImageId_ClassId", "EncodedPixels"])
            writer.writerow(["a.jpg_4", "12 3"])
            writer.writerow(["b.jpg_1", ""])
        assert load_severstal_annotations(td / "combined.csv")["a.jpg"][3] == ("12 3",)
        fake_names = [f"{i:05d}.jpg" for i in range(150)]
        fake_labels = {}
        for i, filename in enumerate(fake_names):
            groups = [[] for _ in CLASS_IDS]
            if i % 3 == 0:
                groups[0].append("1 2")
            if i % 5 == 0:
                groups[1].append("2 2")
            if i % 7 == 0:
                groups[2].append("3 2")
            if i % 11 == 0:
                groups[3].append("4 2")
            if any(groups):
                fake_labels[filename] = tuple(tuple(group) for group in groups)
        first = split_image_pool(fake_names, fake_labels, seed=42)
        second = split_image_pool(fake_names[::-1], fake_labels, seed=42)
        assert first == second
        assert tuple(len(first[key]) for key in ("train", "val", "test")) == (120, 15, 15)
        assert not (set(first["train"]) & set(first["val"]))
        assert not (set(first["train"]) & set(first["test"]))
        assert not (set(first["val"]) & set(first["test"]))
        assert set().union(*map(set, first.values())) == set(fake_names)
        assert all(all(any(fake_labels.get(image, ((),) * NUM_CLASSES)[cls]
                           for image in names) for cls in range(NUM_CLASSES))
                   for names in first.values())
        fp = save_split_manifest(td / "ckpt", td / "train.csv",
                                 td / "images", first, seed=42,
                                 val_fraction=.1, test_fraction=.1)
        assert fp == save_split_manifest(td / "ckpt", td / "train.csv",
                                         td / "images", first, seed=42,
                                         val_fraction=.1, test_fraction=.1)
        assert len(fp) == 64

        metrics_path = td / "metrics.csv"
        with metrics_path.open("w", newline="") as stream:
            csv.writer(stream).writerows([
                ("Epoch", "metric"), (1, "old"), (2, "keep"), (1, "new"), (3, "discard"),
            ])
        _prepare_metrics_csv(metrics_path, ("Epoch", "metric"), resume_epoch=2)
        with metrics_path.open(newline="") as stream:
            assert list(csv.reader(stream)) == [["Epoch", "metric"], ["1", "new"], ["2", "keep"]]
        for epoch, score in ((1, .6), (2, .5)):
            checkpoint = {"epoch": epoch, "val_metrics": {"iou": score},
                          "architecture": ARCHITECTURE_ID, "model_type": "classic",
                          "split_fingerprint": fp}
            torch.save(checkpoint, td / "ckpt" / f"ckpt_epoch_{epoch:03d}.pth")
            if epoch == 1:
                torch.save(checkpoint, td / "ckpt" / "best_model.pth")
        path, last_checkpoint, best_checkpoint = _load_resume_checkpoints(
            td / "ckpt", "classic", fp, torch.device("cpu"),
        )
        assert path.name == "ckpt_epoch_002.pth"
        assert last_checkpoint["epoch"] == 2 and best_checkpoint["epoch"] == 1

    train_rng = torch.Generator().manual_seed(17)
    rng_state = _capture_rng_state(train_rng, torch.device("cpu"))
    expected = (random.random(), np.random.random(), torch.rand(3),
                torch.randperm(9, generator=train_rng))
    assert _restore_rng_state(rng_state, train_rng, torch.device("cpu"))
    replayed = (random.random(), np.random.random(), torch.rand(3),
                torch.randperm(9, generator=train_rng))
    assert expected[:2] == replayed[:2]
    torch.testing.assert_close(expected[2], replayed[2], atol=0, rtol=0)
    torch.testing.assert_close(expected[3], replayed[3], atol=0, rtol=0)
    _restore_rng_state(rng_state, train_rng, torch.device("cpu"))

    q = QuFeX6QuantumOperator()
    c = Classical6ChannelOperator()
    x = torch.randn(5, 6, requires_grad=True)
    q_out = q(x)
    assert q_out.shape == (5, 6) and torch.isfinite(q_out).all()
    assert q_out.abs().max() <= 1.00001
    q_out.square().mean().backward()
    assert torch.isfinite(q.weights.grad).all() and (q.weights.grad.abs() > 1e-8).all(), q.weights.grad
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert q.weights.numel() == c.weights.numel() == 4
    assert torch.isfinite(c(x.detach())).all()

    # Check phase re-uploading is active, not accidentally cancelled by the
    # CNOT/measurement layout. All inputs/weights are held fixed.
    with torch.no_grad():
        baseline = q(x.detach())
        q.phase_reupload_strength = 0.0
        without_reupload = q(x.detach())
        q.phase_reupload_strength = 0.25
    assert (baseline - without_reupload).abs().max() > 1e-4

    # Independent analytic sanity checks for Pauli-Y and Pauli-X readouts.
    # A qubit in (|0> + i|1>) / sqrt(2) has <Y> = +1;
    # a qubit in (|0> + |1>) / sqrt(2) has <X> = +1.
    amp = 1.0 / math.sqrt(2.0)
    state_y = torch.zeros((1, 64), dtype=torch.complex64)
    state_y[0, 0] = amp
    state_y[0, 8] = 1j * amp  # qubit 2, big-endian bit 3
    y_gather = state_y.index_select(1, q._readout_flips).reshape(1, 4, 64)
    y_overlap = state_y.conj().unsqueeze(1) * y_gather
    expected_y = (y_overlap[:, 0].imag * q._z_signs[2]).sum(dim=1)
    torch.testing.assert_close(expected_y, torch.ones_like(expected_y), atol=1e-6, rtol=0)
    # The same |+Y> state has <(X+Y)/sqrt(2)> = 1/sqrt(2).
    expected_xy = (y_overlap[:, 0].real.sum(dim=1) + expected_y) / math.sqrt(2.0)
    torch.testing.assert_close(expected_xy, torch.ones_like(expected_xy) / math.sqrt(2.0),
                               atol=1e-6, rtol=0)
    state_xy = torch.zeros((1, 64), dtype=torch.complex64)
    state_xy[0, 0], state_xy[0, 8] = amp, complex(.5, .5)
    xy_gather = state_xy.index_select(1, q._readout_flips).reshape(1, 4, 64)
    xy_overlap = state_xy.conj().unsqueeze(1) * xy_gather
    mixed = (xy_overlap[:, 0].real.sum(dim=1)
             + (xy_overlap[:, 0].imag * q._z_signs[2]).sum(dim=1)) / math.sqrt(2.0)
    torch.testing.assert_close(mixed, torch.ones_like(mixed), atol=1e-6, rtol=0)
    state_x = torch.zeros((1, 64), dtype=torch.complex64)
    state_x[0, 0] = amp
    state_x[0, 2] = amp  # qubit 4, big-endian bit 1
    x_gather = state_x.index_select(1, q._readout_flips).reshape(1, 4, 64)
    expected_x = (state_x.conj().unsqueeze(1) * x_gather)[:, 2].real.sum(dim=1)
    torch.testing.assert_close(expected_x, torch.ones_like(expected_x), atol=1e-6, rtol=0)

    shapes, params = [], []
    for mode in ("classic", "hybrid"):
        set_global_seed(42, deterministic=True)
        net = build_model(mode, base_channels=32, pretrained_encoder=False)
        net.eval()
        params.append(count_trainable_parameters(net))
        synthetic = torch.rand((1, 3, 64, 128))
        with torch.no_grad():
            logits = net(synthetic)
        assert logits.shape == (1, NUM_CLASSES, 64, 128), logits.shape
        assert torch.isfinite(logits).all()
        with torch.no_grad():
            window = predict_full_logits(net, torch.rand(1, 3, 64, 192),
                                         (64, 128), overlap=32, tile_batch_size=2)
        assert window.shape == (1, 4, 64, 192)
        shapes.append(tuple(logits.shape))
    assert params[0] == params[1], params
    loss_fn = BCEDiceLoss()
    empty_logits = torch.zeros((1, 4, 16, 16), requires_grad=True)
    empty_target = torch.zeros_like(empty_logits)
    loss = loss_fn(empty_logits, empty_target)
    assert loss.item() > 0
    loss.backward()
    assert empty_logits.grad.abs().sum() > 0
    print(f"SELF-TEST PASSED: shapes={shapes}, matched_parameters={params[0]:,}, "
          f"quantum_gradients={q.weights.grad.tolist()}, empty_mask_loss={loss.item():.5f}")


def benchmark_training_steps(
    device: torch.device | str = DEVICE,
    image_size: Sequence[int] = DEFAULT_IMAGE_SIZE,
    base_channels: int = 64,
    batch_size: int = 2,
    iterations: int = 8,
) -> None:
    """Local, dataset-free timing of equal-shape classic and hybrid train steps.

    Forward + BCE/Dice + backward + AdamW. Pretrained weights are intentionally
    disabled for offline timing; CNN architecture and parameter counts match.
    The classical and quantum operators have different operation counts.
    """
    device = torch.device(device)
    validate_image_geometry(image_size)
    if device.type == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    amp_dtype = None
    if device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    images = torch.randn((batch_size, 3, *image_size), device=device)
    masks = (torch.rand((batch_size, 4, *image_size), device=device) > .97).float()
    if device.type == "cuda":
        images = images.contiguous(memory_format=torch.channels_last)
    loss_fn = BCEDiceLoss()
    for mode in ("classic", "hybrid"):
        set_global_seed(42)
        net = build_model(mode, base_channels=base_channels, pretrained_encoder=False).to(device)
        if device.type == "cuda":
            net = net.to(memory_format=torch.channels_last)
            torch.cuda.reset_peak_memory_stats(device)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
        net.train()
        elapsed = []
        for step in range(iterations + 2):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            with _amp_context(device, amp_dtype):
                loss = loss_fn(net(images), masks)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if step >= 2:
                elapsed.append(time.perf_counter() - start)
        mean = sum(elapsed) / len(elapsed)
        memory = (f", peak_cuda={torch.cuda.max_memory_allocated(device) / 1024**2:.0f} MiB"
                  if device.type == "cuda" else "")
        print(f"BENCHMARK {mode}: device={device}, batch={batch_size}, "
              f"crop={tuple(image_size)}, mean_train_step={mean:.4f} s{memory}")



if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
        sys.exit(0)
    if "--benchmark" in sys.argv:
        benchmark_training_steps()
        sys.exit(0)

    # Same simple user-facing configuration as the original script.
    seed = 64
    model_type = "classic"  # set to "classic" for strictly matched reference
    prefix = "severstal_FPN_crosschannel621_phase_v3"

    run_training(
        dataset_root="./data_severstal",
        model_type=model_type,
        in_channels=3,
        epochs=40,
        batch_size=8,
        lr=3e-4,
        image_size=DEFAULT_IMAGE_SIZE,
        start_checkpoint_dir=None,
        checkpoint_dir=f"checkpoints_{prefix}_{model_type}_seed{seed}",
        savefile_name=f"{prefix}_{model_type}_seed{seed}",
        threshold=0.5,
        num_workers=6,
        seed=seed,
        base_channels=64,
        batch_circuit_size=4096,
        pretrained_encoder=True,
        val_overlap=128,
        val_tile_batch_size=4,
        use_amp=True,
        val_fraction=0.10,      # split the original train.csv image pool
        test_fraction=0.10,     # final held-out evaluation, only once
    )

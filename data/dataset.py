"""Dataset utilities for multi-label Severstal defect classification."""

import os
import csv
import hashlib
import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image


def get_severstal_classes(ann_dir):
    """Collect all class identifiers from ``train.csv`` annotations.

    Args:
        ann_dir: Directory containing ``train.csv``.

    Returns:
        Sorted list of integer class identifiers that occur in non-empty
        annotations.
    """
    classes = set()
    csv_path = os.path.join(ann_dir, 'train.csv')
    if not os.path.exists(csv_path):
        return []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_id = int(row['ClassId'])
            encoded_pixels = row.get('EncodedPixels', '').strip()
            if encoded_pixels:
                classes.add(class_id)

    return sorted(classes)


def _apportion(total, ratios, ensure_nonempty=False):
    """Allocate ``total`` items across splits according to ``ratios``.

    Args:
        total: Total number of items to distribute.
        ratios: Sequence of split ratios summing to approximately 1.
        ensure_nonempty: If True, guarantees every split receives at least
            one item when the total permits.

    Returns:
        Integer numpy array with per-split item counts.
    """
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


def _validate_image_split(split, image_names):
    """Validate that image splits are non-empty, disjoint, and complete.

    Args:
        split: Mapping with keys ``train``, ``val`` and ``test`` whose values
            are lists of image identifiers.
        image_names: Original image pool used to check full coverage.

    Returns:
        None.

    Raises:
        ValueError: If the split structure or image coverage is invalid.
    """
    if set(split) != {"train", "val", "test"}:
        raise ValueError("Require train, val and test image splits")
    seen = set()
    for kind in ("train", "val", "test"):
        names = split[kind]
        if not names or len(names) != len(set(names)):
            raise ValueError(f"{kind} split is empty or contains duplicate images")
        if seen.intersection(names):
            raise ValueError("Train, validation and test image IDs must not overlap")
        seen.update(names)
    if seen != set(image_names):
        raise ValueError("Splits must cover exactly the original image pool")


def split_image_pool(image_names, image_to_classes, num_classes, seed=42,
                     val_fraction=0.15, test_fraction=0.15):
    """Create a deterministic multi-label-stratified image-level split.

    Stratification groups are the presence of each class plus a separate
    ``all-zero`` group for images without defects.

    Args:
        image_names: Iterable of image identifiers.
        image_to_classes: Mapping from image identifier to a set of class
            indices.
        num_classes: Number of classes.
        seed: Random seed.
        val_fraction: Fraction of images assigned to validation.
        test_fraction: Fraction of images assigned to test.

    Returns:
        Dictionary with keys ``train``, ``val`` and ``test``, each mapping to
        a list of image identifiers.

    Raises:
        ValueError: If split fractions or the image pool are invalid.
        RuntimeError: If internal split allocation fails.
    """
    if not (val_fraction > 0 and test_fraction > 0
            and val_fraction + test_fraction < 1):
        raise ValueError("Require val_fraction > 0, test_fraction > 0, sum < 1")
    names = sorted(image_names)
    if len(names) != len(set(names)):
        raise ValueError("Image pool contains duplicate filenames")
    if len(names) < 3:
        raise ValueError("At least three images are required for train/val/test")
    n = len(names)
    ratios = np.array((1 - val_fraction - test_fraction, val_fraction, test_fraction),
                      dtype=np.float64)
    capacities = _apportion(n, ratios, ensure_nonempty=True)
    labels = np.zeros((n, num_classes + 1), dtype=np.uint8)
    for i, name in enumerate(names):
        cls_set = image_to_classes.get(name, set())
        for c in cls_set:
            if 0 <= c < num_classes:
                labels[i, c] = 1
        labels[i, -1] = int(not labels[i, :num_classes].any())
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


def save_split_manifest(checkpoint_dir, csv_path, split, seed, val_fraction,
                        test_fraction, image_dir=None):
    """Save ``data_split.json`` and verify its fingerprint on repeated runs.

    Args:
        checkpoint_dir: Directory where the manifest is stored.
        csv_path: Path to the annotations CSV file.
        split: Mapping with ``train``, ``val`` and ``test`` image lists.
        seed: Split random seed.
        val_fraction: Validation fraction.
        test_fraction: Test fraction.
        image_dir: Optional image directory stored in the manifest.

    Returns:
        SHA-256 fingerprint of the split payload as a hexadecimal string.

    Raises:
        RuntimeError: If an existing manifest has a different fingerprint.
    """
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
        "description": "Severstal classification image-level stratified split; train.csv unchanged",
        "train_csv": str(Path(csv_path).resolve()),
        "split_fingerprint": fingerprint,
        **payload,
    }
    if image_dir is not None:
        manifest["image_dir"] = str(Path(image_dir).resolve())
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


def _print_split_summary(split_name, image_names, image_to_classes, class_ids):
    """Print per-class positive counts for a split.

    Args:
        split_name: Human-readable split name.
        image_names: Images in the split.
        image_to_classes: Mapping from image identifier to class set.
        class_ids: Class identifiers to report.

    Returns:
        Dictionary mapping each class identifier to the number of positive
        images in the split.
    """
    positives = {cid: 0 for cid in class_ids}
    with_defect = 0
    for name in image_names:
        cls_set = image_to_classes.get(name, set())
        if cls_set:
            with_defect += 1
            for cid in cls_set:
                if cid in positives:
                    positives[cid] += 1
    print(f"{split_name}: images={len(image_names)} | positive_images={with_defect} "
          f"| all-zero_images={len(image_names) - with_defect} | "
          + ", ".join(f"c{cid}={positives[cid]}" for cid in class_ids))
    return positives


def load_severstal_data(root, train_part=0.7, val_part=0.15, random_state=42,
                        checkpoint_dir=None):
    """Load image paths and multi-label targets and create a stratified split.

    Args:
        root: Directory containing ``train.csv`` and ``train_images/``.
        train_part: Fraction of all images assigned to train.
        val_part: Fraction of all images assigned to validation.
        random_state: Split random seed.
        checkpoint_dir: If provided, ``data_split.json`` is written or
            verified in this directory.

    Returns:
        Eleven-element tuple:
        ``(all_paths, all_labels, train_paths, train_labels, val_paths,
        val_labels, test_paths, test_labels, class_to_idx, num_classes,
        split_fingerprint)`` where ``split_fingerprint`` is a string or
        ``None`` when ``checkpoint_dir`` is not provided.

    Raises:
        FileNotFoundError: If the image directory is missing.
        RuntimeError: If no ``.jpg`` images are found.
        ValueError: If split fractions are invalid.
    """
    csv_path = os.path.join(root, 'train.csv')
    img_dir = os.path.join(root, 'train_images')

    image_to_classes = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row['ImageId']
            class_id = int(row['ClassId'])
            encoded_pixels = row['EncodedPixels']
            if encoded_pixels and encoded_pixels.strip():
                image_to_classes.setdefault(image_id, set()).add(class_id)

    all_classes = set()
    for classes in image_to_classes.values():
        all_classes.update(classes)
    class_list = sorted(all_classes)
    num_classes = len(class_list)
    class_to_idx = {cls: i for i, cls in enumerate(class_list)}

    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Папка с изображениями не найдена: {img_dir}")

    all_filenames = sorted(
        f for f in os.listdir(img_dir) if f.lower().endswith('.jpg')
    )
    if not all_filenames:
        raise RuntimeError(f"В {img_dir} нет .jpg изображений")

    image_to_idx = {
        name: {class_to_idx[c] for c in classes if c in class_to_idx}
        for name, classes in image_to_classes.items()
    }

    test_fraction = max(0.0, 1.0 - train_part - val_part)
    if val_part <= 0 or test_fraction <= 0 or train_part <= 0:
        raise ValueError("train_part, val_part и (1 - train_part - val_part) должны быть > 0")

    split = split_image_pool(
        all_filenames, image_to_idx, num_classes,
        seed=random_state, val_fraction=val_part, test_fraction=test_fraction,
    )

    split_fingerprint = None
    if checkpoint_dir is not None:
        split_fingerprint = save_split_manifest(
            checkpoint_dir, csv_path, split,
            seed=random_state, val_fraction=val_part, test_fraction=test_fraction,
            image_dir=img_dir,
        )
        _print_split_summary('TRAIN', split['train'], image_to_classes, class_list)
        _print_split_summary('VAL',   split['val'],   image_to_classes, class_list)
        _print_split_summary('TEST',  split['test'],  image_to_classes, class_list)

    def _build(filenames):
        """Build path and label lists for the given filenames.

        Args:
            filenames: Image filenames to process.

        Returns:
            Tuple ``(paths, labels)`` of image file paths and a float32
            multi-label matrix of shape ``(len(filenames), num_classes)``.
        """
        paths, labels = [], []
        for fname in filenames:
            img_path = os.path.join(img_dir, fname)
            classes = image_to_classes.get(fname, set())
            label_vec = [0] * num_classes
            for cls in classes:
                if cls in class_to_idx:
                    label_vec[class_to_idx[cls]] = 1
            paths.append(img_path)
            labels.append(label_vec)
        return paths, np.array(labels, dtype=np.float32)

    all_paths, all_labels = _build(all_filenames)
    train_paths, train_labels = _build(split['train'])
    val_paths, val_labels = _build(split['val'])
    test_paths, test_labels = _build(split['test'])

    return (all_paths, all_labels,
            train_paths, train_labels,
            val_paths, val_labels,
            test_paths, test_labels,
            class_to_idx, num_classes, split_fingerprint)


class SeverstalDataset(Dataset):
    """Dataset for multi-label Severstal defect classification."""

    def __init__(self, image_paths, labels, transform=None):
        """Initialize the dataset.

        Args:
            image_paths: Sequence of image file paths.
            labels: Multi-label targets, shape ``(N, num_classes)``.
            transform: Optional image transformation pipeline applied to a
                grayscale ``PIL.Image``.

        Returns:
            None.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """Return the number of samples in the dataset.

        Returns:
            Integer number of samples.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Return one transformed image and its multi-label target.

        Args:
            idx: Sample index.

        Returns:
            Tuple ``(image, label)`` where ``image`` is the transformed image
            tensor and ``label`` is a float32 tensor of class indicators.
        """
        img = Image.open(self.image_paths[idx]).convert('L')
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        if self.transform:
            img = self.transform(img)
        return img, label

    @staticmethod
    def collate_fn(batch):
        """Collate a list of samples into batched image and label tensors.

        Args:
            batch: Sequence of ``(image, label)`` pairs from ``__getitem__``.

        Returns:
            Tuple ``(images, labels)`` of stacked image and label tensors.
        """
        images, labels = zip(*batch)
        return torch.stack(images), torch.stack(labels)
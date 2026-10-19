"""Training and evaluation routines for multi-label classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class FocalLoss(nn.Module):
    """Focal loss for binary multi-label classification."""

    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        """Initialize the focal loss.

        Args:
            alpha: Optional balancing factor in ``[0, 1]``; when ``None`` no
                balancing is applied.
            gamma: Focusing parameter controlling the down-weighting of
                easy examples.
            reduction: Reduction mode, one of ``'mean'``, ``'sum'`` or
                ``'none'``.

        Returns:
            None.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """Compute the focal loss.

        Args:
            inputs: Predicted probabilities of shape ``(N, C)`` in ``[0, 1]``.
            targets: Binary targets of shape ``(N, C)``.

        Returns:
            Scalar loss tensor when ``reduction`` is ``'mean'`` or ``'sum'``,
            otherwise an element-wise loss tensor of shape ``(N, C)``.
        """
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = (1 - pt) ** self.gamma * BCE_loss
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            F_loss = alpha_t * F_loss
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


def compute_metrics(probs, targets, threshold=0.5, return_counts=False):
    """Compute accuracy and per-class precision, recall and F1.

    Args:
        probs: Predicted probabilities of shape ``(N, C)``.
        targets: Binary targets of shape ``(N, C)``.
        threshold: Decision threshold applied to ``probs``.
        return_counts: If True, return raw accuracy and per-class true
            positive, false positive and false negative counts instead of
            derived metrics.

    Returns:
        When ``return_counts`` is False, tuple ``(accuracy, precision,
        recall, f1)`` where ``accuracy`` is a scalar and the remaining
        entries are per-class tensors of shape ``(C,)``. When
        ``return_counts`` is True, tuple ``(accuracy, tp, fp, fn)`` with the
        same shapes.
    """
    preds = (probs > threshold).float()
    tp = (preds * targets).sum(dim=0)
    fp = (preds * (1 - targets)).sum(dim=0)
    fn = ((1 - preds) * targets).sum(dim=0)
    acc = (preds == targets).sum().item()
    if return_counts:
        return acc, tp, fp, fn
    eps = 1e-7
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    return acc, prec, rec, f1


def _as_device(device):
    """Normalize a device specifier to ``torch.device``.

    Args:
        device: ``torch.device`` instance or a string specifier.

    Returns:
        ``torch.device`` instance.
    """
    return device if isinstance(device, torch.device) else torch.device(device)


def _compute_loss_fp32(loss_fn, probs, target):
    """Evaluate the loss function in fp32 outside autocast.

    Args:
        loss_fn: Loss callable operating on float32 tensors.
        probs: Predicted probabilities from the model.
        target: Ground-truth targets.

    Returns:
        Scalar loss tensor computed in float32.
    """
    device_type = probs.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        return loss_fn(probs.float(), target.float())


def train_one_epoch(model, dataloader, loss_fn, optimizer, device, num_classes,
                    threshold=0.5, eps=1e-7, scaler=None, amp_dtype=None):
    """Train ``model`` for one epoch.

    Args:
        model: Model returning probabilities of shape ``(N, num_classes)``.
        dataloader: Iterable of ``(data, target)`` batches.
        loss_fn: Loss callable evaluated in float32.
        optimizer: Optimizer updating the model parameters.
        device: Device on which to run the epoch.
        num_classes: Number of classes.
        threshold: Decision threshold used for accuracy computation.
        eps: Numerical stability constant for metric division.
        scaler: Optional ``torch.amp.GradScaler`` used with float16 AMP.
        amp_dtype: Optional autocast dtype (``torch.float16`` or
            ``torch.bfloat16``); AMP is disabled when ``None``.

    Returns:
        Tuple ``(avg_loss, avg_acc, macro_prec, macro_rec, macro_f1,
        total_tp, total_fp, total_fn)`` where the first three scalars are
        Python floats, ``macro_prec``, ``macro_rec`` and ``macro_f1`` are
        per-class tensors of shape ``(num_classes,)``, and ``total_tp``,
        ``total_fp`` and ``total_fn`` are per-class count tensors of shape
        ``(num_classes,)``.
    """
    model.train()
    device = _as_device(device)
    use_amp = device.type == 'cuda' and amp_dtype is not None

    total_loss = 0.0
    total_loss_items = 0
    total_correct = 0
    total_items = 0
    total_tp = torch.zeros(num_classes, device=device)
    total_fp = torch.zeros(num_classes, device=device)
    total_fn = torch.zeros(num_classes, device=device)

    loop = tqdm(dataloader, desc='Training', leave=False)
    for data, target in loop:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            probs = model(data)

        loss = _compute_loss_fp32(loss_fn, probs, target)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = data.size(0)
        batch_items = batch_size * num_classes
        total_loss += loss.item() * batch_items
        total_loss_items += batch_items

        probs_f = probs.detach().float()
        acc, tp, fp, fn = compute_metrics(probs_f, target, threshold, return_counts=True)
        total_correct += acc
        total_items += batch_items
        total_tp += tp
        total_fp += fp
        total_fn += fn

        running_loss = total_loss / total_loss_items if total_loss_items else 0.0
        running_acc = total_correct / total_items if total_items else 0.0
        loop.set_postfix(loss=running_loss, acc=running_acc)

    avg_loss = total_loss / total_loss_items if total_loss_items else 0.0
    avg_acc = total_correct / total_items if total_items else 0.0

    macro_prec = total_tp / (total_tp + total_fp + eps)
    macro_rec = total_tp / (total_tp + total_fn + eps)
    macro_f1 = 2 * macro_prec * macro_rec / (macro_prec + macro_rec + eps)

    return avg_loss, avg_acc, macro_prec, macro_rec, macro_f1, total_tp, total_fp, total_fn


def evaluate(model, dataloader, loss_fn, device, num_classes, threshold=0.5,
             eps=1e-7, amp_dtype=None):
    """Evaluate ``model`` on the given dataloader.

    Args:
        model: Model returning probabilities of shape ``(N, num_classes)``.
        dataloader: Iterable of ``(data, target)`` batches.
        loss_fn: Loss callable evaluated in float32.
        device: Device on which to run the evaluation.
        num_classes: Number of classes.
        threshold: Decision threshold used for accuracy computation.
        eps: Numerical stability constant for metric division.
        amp_dtype: Optional autocast dtype (``torch.float16`` or
            ``torch.bfloat16``); AMP is disabled when ``None``.

    Returns:
        Tuple ``(avg_loss, avg_acc, macro_prec, macro_rec, macro_f1,
        total_tp, total_fp, total_fn)`` where ``avg_loss`` and ``avg_acc``
        are Python floats, ``macro_prec``, ``macro_rec`` and ``macro_f1``
        are per-class tensors of shape ``(num_classes,)``, and ``total_tp``,
        ``total_fp`` and ``total_fn`` are per-class count tensors of shape
        ``(num_classes,)``.
    """
    model.eval()
    device = _as_device(device)
    use_amp = device.type == 'cuda' and amp_dtype is not None

    total_loss = 0.0
    total_loss_items = 0
    total_correct = 0
    total_items = 0
    total_tp = torch.zeros(num_classes, device=device)
    total_fp = torch.zeros(num_classes, device=device)
    total_fn = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True).float()

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                probs = model(data)

            loss = _compute_loss_fp32(loss_fn, probs, target)

            batch_size = data.size(0)
            batch_items = batch_size * num_classes
            total_loss += loss.item() * batch_items
            total_loss_items += batch_items

            probs_f = probs.float()
            acc, tp, fp, fn = compute_metrics(probs_f, target, threshold, return_counts=True)
            total_correct += acc
            total_items += batch_items
            total_tp += tp
            total_fp += fp
            total_fn += fn

    avg_loss = total_loss / total_loss_items if total_loss_items else 0.0
    avg_acc = total_correct / total_items if total_items else 0.0

    macro_prec = total_tp / (total_tp + total_fp + eps)
    macro_rec = total_tp / (total_tp + total_fn + eps)
    macro_f1 = 2 * macro_prec * macro_rec / (macro_prec + macro_rec + eps)

    return avg_loss, avg_acc, macro_prec, macro_rec, macro_f1, total_tp, total_fp, total_fn
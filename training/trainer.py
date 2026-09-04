import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # веса классов (тензор)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
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
    preds = (probs > threshold).float()
    tp = (preds * targets).sum(dim=0)
    fp = (preds * (1 - targets)).sum(dim=0)
    fn = ((1 - preds) * targets).sum(dim=0)
    acc = (preds == targets).sum().item()
    if return_counts:
        return acc, tp, fp, fn
    eps = 1e-7
    prec = tp / (tp + fp + eps)
    rec  = tp / (tp + fn + eps)
    f1   = 2 * prec * rec / (prec + rec + eps)
    return acc, prec, rec, f1

def train_one_epoch(model, dataloader, loss_fn, optimizer, device, num_classes,
                    threshold=0.5, eps=1e-7):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_tp = torch.zeros(num_classes, device=device)
    total_fp = torch.zeros(num_classes, device=device)
    total_fn = torch.zeros(num_classes, device=device)
    total_samples = 0

    loop = tqdm(dataloader, desc='Training', leave=False)
    for data, target in loop:
        data, target = data.to(device), target.to(device).float()
        optimizer.zero_grad()
        probs = model(data)
        loss = loss_fn(probs, target)
        loss.backward()
        optimizer.step()

        batch_size = data.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size * num_classes

        acc, tp, fp, fn = compute_metrics(probs, target, threshold, return_counts=True)
        total_correct += acc
        total_tp += tp
        total_fp += fp
        total_fn += fn

        loop.set_postfix(loss=loss.item(), acc=total_correct / total_samples)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    macro_prec = total_tp / (total_tp + total_fp + eps)
    macro_rec = total_tp / (total_tp + total_fn + eps)
    macro_f1 = 2 * macro_prec * macro_rec / (macro_prec + macro_rec + eps)

    return avg_loss, avg_acc, macro_prec, macro_rec, macro_f1, total_tp, total_fp, total_fn

def evaluate(model, dataloader, loss_fn, device, num_classes, threshold=0.5, eps=1e-7):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tp = torch.zeros(num_classes, device=device)
    total_fp = torch.zeros(num_classes, device=device)
    total_fn = torch.zeros(num_classes, device=device)
    total_samples = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device).float()
            probs = model(data)
            loss = loss_fn(probs, target)

            batch_size = data.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size * num_classes

            acc, tp, fp, fn = compute_metrics(probs, target, threshold, return_counts=True)
            total_correct += acc
            total_tp += tp
            total_fp += fp
            total_fn += fn

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    macro_prec = total_tp / (total_tp + total_fp + eps)
    macro_rec = total_tp / (total_tp + total_fn + eps)
    macro_f1 = 2 * macro_prec * macro_rec / (macro_prec + macro_rec + eps)

    return avg_loss, avg_acc, macro_prec, macro_rec, macro_f1, total_tp, total_fp, total_fn
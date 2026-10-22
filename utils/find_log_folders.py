"""Discovery of training log folders matching specified criteria."""

import os
import re
from datetime import datetime

import torch
from config import Config


def find_log_folders(log_root, model_name=None, model_mode=None,
                     min_epochs=None, after_date=None):
    """Find log folders that match the given criteria.

    Args:
        log_root: Root directory containing per-run log folders.
        model_name: Optional model name filter (for example ``'clp_net'``,
            ``'resnet18'`` or ``'efficientnet'``).
        model_mode: Optional model mode filter (for example ``'classical'``
            or ``'quantum'``).
        min_epochs: Optional minimum number of completed epochs.
        after_date: Optional lower bound on folder creation time; accepts a
            ``datetime`` or a string in ``'YYYYMMDD'`` or
            ``'YYYYMMDD_HHMMSS'`` format.

    Returns:
        Sorted list of full paths to folders that satisfy all specified
        criteria.
    """
    if not os.path.exists(log_root):
        return []

    all_folders = []
    for item in os.listdir(log_root):
        folder_path = os.path.join(log_root, item)
        if os.path.isdir(folder_path):
            all_folders.append(folder_path)

    result = []

    for folder in all_folders:
        folder_name = os.path.basename(folder)
        match = re.match(r'(\d{8})(?:_(\d{6}))?', folder_name)
        if match:
            date_str = match.group(1)
            time_str = match.group(2)
            try:
                if time_str:
                    dt = datetime.strptime(f'{date_str}_{time_str}', '%Y%m%d_%H%M%S')
                else:
                    dt = datetime.strptime(date_str, '%Y%m%d')
                if after_date is not None:
                    if isinstance(after_date, str):
                        if '_' in after_date:
                            after_dt = datetime.strptime(after_date, '%Y%m%d_%H%M%S')
                        else:
                            after_dt = datetime.strptime(after_date, '%Y%m%d')
                    else:
                        after_dt = after_date
                    if dt < after_dt:
                        continue
            except ValueError:
                continue
        else:
            continue

        config_path = os.path.join(folder, 'config.json')
        if not os.path.exists(config_path):
            continue
        try:
            cfg = Config.from_json(config_path)
        except Exception:
            continue

        if model_name is not None and cfg.model_name != model_name:
            continue
        if model_mode is not None and cfg.model_mode != model_mode:
            continue

        checkpoint_path = os.path.join(folder, 'checkpoint_last.pth')
        epochs_completed = 0
        if os.path.exists(checkpoint_path):
            try:
                try:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
                except Exception:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                epoch = checkpoint.get('epoch', 0)
                epochs_completed = epoch + 1
            except Exception:
                pass
        else:
            log_file = os.path.join(folder, 'log.txt')
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                for line in reversed(lines):
                    match_epoch = re.search(r'Epoch\s+(\d+)/(\d+)', line)
                    if match_epoch:
                        epochs_completed = int(match_epoch.group(1))
                        break

        if min_epochs is not None and epochs_completed < min_epochs:
            continue

        result.append(folder)

    result.sort(key=lambda x: os.path.basename(x))
    return result
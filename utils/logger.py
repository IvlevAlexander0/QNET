"""Logging, checkpoint management and log-folder maintenance utilities."""

import os
import sys
import logging
import shutil
import re
from datetime import datetime

import torch

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False
    logging.warning("TensorBoard not installed. Install with: pip install tensorboard")

_logger = None
_run_dir = None
_writer = None


def get_tensorboard_writer(run_dir):
    """Return the global ``SummaryWriter`` instance, creating it if needed.

    Args:
        run_dir: Run directory in which the ``tensorboard`` subfolder is
            created.

    Returns:
        A ``SummaryWriter`` instance, or ``None`` when TensorBoard is not
        available or ``run_dir`` is ``None``.
    """
    global _writer
    if not TENSORBOARD_AVAILABLE:
        return None
    if _writer is None and run_dir is not None:
        tb_dir = os.path.join(run_dir, 'tensorboard')
        os.makedirs(tb_dir, exist_ok=True)
        _writer = SummaryWriter(log_dir=tb_dir)
    return _writer


def close_tensorboard_writer():
    """Close the global ``SummaryWriter`` if it was created.

    Returns:
        None.
    """
    global _writer
    if _writer is not None:
        _writer.close()
        _writer = None


def setup_logging(config, run_name=None, config_path='config.json', run_dir=None):
    """Configure logging and copy ``config.json`` into the run directory.

    Args:
        config: Configuration object providing ``enable_logging`` and
            ``log_dir``.
        run_name: Optional run name used as the run subfolder name; defaults
            to the current timestamp.
        config_path: Path to the configuration file copied into a newly
            created run directory.
        run_dir: Existing run directory to reuse; when ``None`` a new one is
            created under ``config.log_dir``.

    Returns:
        Tuple ``(run_dir, logger)`` where ``run_dir`` is the run directory
        path or ``None`` when logging is disabled, and ``logger`` is the
        configured ``logging.Logger``.
    """
    global _logger, _run_dir

    if not config.enable_logging:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        _suppress_noisy_libs()
        _logger = logging.getLogger('Severstal')
        _run_dir = None
        return _run_dir, _logger

    if run_dir is None:
        os.makedirs(config.log_dir, exist_ok=True)
        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(config.log_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        try:
            if os.path.exists(config_path):
                shutil.copy2(config_path, os.path.join(run_dir, 'config.json'))
        except Exception as e:
            logging.warning(f"Failed to copy config: {e}")
    else:
        if not os.path.exists(run_dir):
            os.makedirs(run_dir, exist_ok=True)

    log_file = os.path.join(run_dir, 'log.txt')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    _suppress_noisy_libs()

    _logger = logging.getLogger('Severstal')
    _run_dir = run_dir

    _logger.info(f"=== New run: {run_name if run_name else os.path.basename(run_dir)} ===")
    _logger.info(f"Configuration: {vars(config)}")
    _logger.info(f"Log directory: {run_dir}")
    return run_dir, _logger


def save_checkpoint(run_dir, model, optimizer, scheduler, epoch, best_val_acc,
                    best_val_f1, history, config, extra=None):
    """Save a training checkpoint.

    Args:
        run_dir: Run directory where ``checkpoint_last.pth`` is written.
        model: Model whose ``state_dict`` is stored.
        optimizer: Optimizer whose ``state_dict`` is stored.
        scheduler: Optional learning-rate scheduler whose ``state_dict`` is
            stored.
        epoch: Zero-based index of the last completed epoch.
        best_val_acc: Best validation accuracy observed so far.
        best_val_f1: Best validation macro F1 observed so far.
        history: Dictionary of per-epoch metric lists.
        config: Configuration object whose ``vars`` are stored in the
            checkpoint.
        extra: Optional dictionary of additional checkpoint fields such as
            RNG state, split fingerprint or AMP dtype.

    Returns:
        None.
    """
    checkpoint_path = os.path.join(run_dir, 'checkpoint_last.pth')
    payload = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_val_acc': best_val_acc,
        'best_val_f1': best_val_f1,
        'history': history,
        'config': vars(config),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, checkpoint_path)
    logger = get_logger()
    logger.info(f"Checkpoint saved to {checkpoint_path} (epoch {epoch+1})")


def load_checkpoint(run_dir, model, optimizer, scheduler):
    """Load a training checkpoint into the given objects.

    Args:
        run_dir: Run directory from which ``checkpoint_last.pth`` is loaded.
        model: Model whose weights are restored in place.
        optimizer: Optimizer whose state is restored in place.
        scheduler: Optional learning-rate scheduler whose state is restored
            in place.

    Returns:
        Tuple ``(next_epoch, best_val_acc, best_val_f1, history,
        raw_checkpoint)`` where ``next_epoch`` is the one-based index of the
        next epoch to run, ``history`` is the per-epoch metric dictionary and
        ``raw_checkpoint`` is the full loaded checkpoint dictionary.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    checkpoint_path = os.path.join(run_dir, 'checkpoint_last.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    best_val_acc = checkpoint.get('best_val_acc', 0.0)
    best_val_f1 = checkpoint.get('best_val_f1', 0.0)
    history = checkpoint.get('history', {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [],
    })
    logger = get_logger()
    logger.info(f"Checkpoint loaded from {checkpoint_path}, resuming from epoch {epoch+1}")
    return epoch + 1, best_val_acc, best_val_f1, history, checkpoint


def _suppress_noisy_libs():
    """Set verbose third-party loggers to a less noisy level.

    Returns:
        None.
    """
    noisy_libs = [
        'qiskit', 'qiskit_aer', 'qiskit_ibm_runtime', 'qiskit.primitives',
        'matplotlib', 'PIL', 'torch', 'numpy',
        'urllib3', 'requests', 'asyncio', 'tensorflow', 'qiskit_machine_learning'
    ]
    for lib in noisy_libs:
        logging.getLogger(lib).setLevel(logging.WARNING)
    for name in logging.root.manager.loggerDict:
        if name.startswith('qiskit'):
            logging.getLogger(name).setLevel(logging.WARNING)
        if name.startswith('qiskit_machine_learning'):
            logging.getLogger(name).setLevel(logging.ERROR)


def get_logger():
    """Return the project ``Severstal`` logger, creating a default one if needed.

    Returns:
        A ``logging.Logger`` instance.
    """
    global _logger
    if _logger is None:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        _suppress_noisy_libs()
        _logger = logging.getLogger('Severstal')
    return _logger


def get_run_dir():
    """Return the current run directory.

    Returns:
        Run directory path or ``None`` when logging is disabled.
    """
    global _run_dir
    return _run_dir


def get_completed_epochs(run_dir):
    """Return the number of completed epochs for a run directory.

    Args:
        run_dir: Run directory to inspect. The checkpoint is preferred; when
            it is absent, the last ``Epoch X/Y`` line in ``log.txt`` is used.

    Returns:
        Integer number of completed epochs, or ``0`` when nothing is found.
    """
    checkpoint_path = os.path.join(run_dir, 'checkpoint_last.pth')
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            epoch = checkpoint.get('epoch', -1)
            if epoch >= 0:
                return epoch + 1
        except Exception:
            pass

    log_file = os.path.join(run_dir, 'log.txt')
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        pattern = r'Epoch\s+(\d+)/(\d+)'
        for line in reversed(lines):
            match = re.search(pattern, line)
            if match:
                return int(match.group(1))
    return 0


def clean_log_folders(log_root, min_epochs=1, dry_run=False):
    """Remove log subfolders with too few completed epochs.

    A subfolder is removed when it has no ``log.txt`` or fewer than
    ``min_epochs`` completed epochs.

    Args:
        log_root: Root directory containing per-run log subfolders.
        min_epochs: Minimum number of completed epochs required to keep a
            folder.
        dry_run: If True, only report what would be removed without deleting.

    Returns:
        Tuple ``(removed, kept)`` with the counts of removed and kept folders.
    """
    if not os.path.exists(log_root):
        print(f"Папка {log_root} не существует.")
        return 0, 0

    removed = 0
    kept = 0

    for item in os.listdir(log_root):
        full_path = os.path.join(log_root, item)
        if not os.path.isdir(full_path):
            continue

        log_file = os.path.join(full_path, 'log.txt')
        if not os.path.exists(log_file):
            if dry_run:
                print(f"Будет удалено: {full_path} (нет log.txt)")
            else:
                shutil.rmtree(full_path)
                print(f"Удалено: {full_path} (нет log.txt)")
            removed += 1
            continue

        epochs = get_completed_epochs(full_path)
        if epochs < min_epochs:
            if dry_run:
                print(f"Будет удалено: {full_path} (эпох: {epochs} < {min_epochs})")
            else:
                shutil.rmtree(full_path)
                print(f"Удалено: {full_path} (эпох: {epochs} < {min_epochs})")
            removed += 1
        else:
            print(f"Оставлено: {full_path} (эпох: {epochs})")
            kept += 1

    return removed, kept
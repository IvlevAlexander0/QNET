import os
import sys
import logging
import shutil
from datetime import datetime
import torch

# Опциональный импорт TensorBoard
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
    """Возвращает глобальный экземпляр SummaryWriter, если TensorBoard доступен."""
    global _writer
    if not TENSORBOARD_AVAILABLE:
        return None
    if _writer is None and run_dir is not None:
        tb_dir = os.path.join(run_dir, 'tensorboard')
        os.makedirs(tb_dir, exist_ok=True)
        _writer = SummaryWriter(log_dir=tb_dir)
    return _writer


def close_tensorboard_writer():
    """Закрывает writer, если он был создан."""
    global _writer
    if _writer is not None:
        _writer.close()
        _writer = None

def setup_logging(config, run_name=None, config_path='config.json', run_dir=None):
    """Настраивает логирование и копирует config.json в папку запуска."""
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

    # Если run_dir не передан, создаём новую папку
    if run_dir is None:
        os.makedirs(config.log_dir, exist_ok=True)
        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(config.log_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        # Копируем конфиг только при первом создании папки
        try:
            if os.path.exists(config_path):
                shutil.copy2(config_path, os.path.join(run_dir, 'config.json'))
        except Exception as e:
            logging.warning(f"Failed to copy config: {e}")
    else:
        # Папка уже существует – не копируем конфиг, только логируем
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

def save_checkpoint(run_dir, model, optimizer, scheduler, epoch, best_val_acc, best_val_f1, history, config):
    """Сохраняет чекпоинт в папку логов."""
    checkpoint_path = os.path.join(run_dir, 'checkpoint_last.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_val_acc': best_val_acc,
        'best_val_f1': best_val_f1,
        'history': history,
        'config': vars(config)  # сохраняем конфиг для справки
    }, checkpoint_path)
    logger = get_logger()
    logger.info(f"Checkpoint saved to {checkpoint_path} (epoch {epoch+1})")

def load_checkpoint(run_dir, model, optimizer, scheduler):
    """Загружает чекпоинт, возвращает (epoch, best_val_acc, best_val_f1, history)."""
    checkpoint_path = os.path.join(run_dir, 'checkpoint_last.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    best_val_acc = checkpoint.get('best_val_acc', 0.0)
    best_val_f1 = checkpoint.get('best_val_f1', 0.0)
    history = checkpoint.get('history', {'train_loss': [], 'train_acc': [], 'train_f1': [],
                                         'val_loss': [], 'val_acc': [], 'val_f1': []})
    logger = get_logger()
    logger.info(f"Checkpoint loaded from {checkpoint_path}, resuming from epoch {epoch+1}")
    return epoch + 1, best_val_acc, best_val_f1, history  # возвращаем следующую эпоху

def _suppress_noisy_libs():
    noisy_libs = [
        'qiskit', 'qiskit_aer', 'qiskit_ibm_runtime', 'qiskit.primitives',
        'matplotlib', 'PIL', 'torch', 'numpy',
        'urllib3', 'requests', 'asyncio', 'tensorflow'
    ]
    for lib in noisy_libs:
        logging.getLogger(lib).setLevel(logging.WARNING)
    for name in logging.root.manager.loggerDict:
        if name.startswith('qiskit'):
            logging.getLogger(name).setLevel(logging.WARNING)

def get_logger():
    global _logger
    if _logger is None:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        _suppress_noisy_libs()
        _logger = logging.getLogger('Severstal')
    return _logger

def get_run_dir():
    global _run_dir
    return _run_dir
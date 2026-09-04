import os
import re
from datetime import datetime
import torch
from config import Config

def find_log_folders(log_root, model_name=None, model_mode=None, min_epochs=None, after_date=None):
    """
    Находит все папки с логами, соответствующие критериям.
    
    Args:
        log_root (str): корневая директория с логами (например, config.log_dir)
        model_name (str, optional): фильтр по имени модели ('clp_net', 'resnet18', 'efficientnet')
        model_mode (str, optional): фильтр по режиму ('classical', 'quantum')
        min_epochs (int, optional): минимальное количество завершённых эпох
        after_date (datetime or str, optional): дата, после которой папки должны быть созданы (можно str в формате 'YYYYMMDD' или 'YYYYMMDD_HHMMSS')
    
    Returns:
        list: список полных путей к папкам, удовлетворяющим условиям
    """
    if not os.path.exists(log_root):
        return []
    
    # Собираем все подпапки
    all_folders = []
    for item in os.listdir(log_root):
        folder_path = os.path.join(log_root, item)
        if os.path.isdir(folder_path):
            all_folders.append(folder_path)
    
    result = []
    
    for folder in all_folders:
        # 1. Проверка даты по имени папки
        folder_name = os.path.basename(folder)
        # Предполагаем формат 'YYYYMMDD_HHMMSS' или 'YYYYMMDD'
        match = re.match(r'(\d{8})(?:_(\d{6}))?', folder_name)
        if match:
            date_str = match.group(1)  # YYYYMMDD
            time_str = match.group(2)  # HHMMSS or None
            try:
                if time_str:
                    dt = datetime.strptime(f'{date_str}_{time_str}', '%Y%m%d_%H%M%S')
                else:
                    dt = datetime.strptime(date_str, '%Y%m%d')
                if after_date is not None:
                    if isinstance(after_date, str):
                        # Пытаемся разобрать
                        if '_' in after_date:
                            after_dt = datetime.strptime(after_date, '%Y%m%d_%H%M%S')
                        else:
                            after_dt = datetime.strptime(after_date, '%Y%m%d')
                    else:
                        after_dt = after_date
                    if dt < after_dt:
                        continue
            except ValueError:
                # Если не удалось распарсить, пропускаем эту папку
                continue
        else:
            # Если имя не соответствует формату даты, можно использовать время модификации
            # но для простоты пропускаем или используем mtime
            # Можно добавить fallback:
            # dt = datetime.fromtimestamp(os.path.getctime(folder))
            # if after_date and dt < after_date: continue
            # Но лучше полагаться на имя.
            # Пропускаем такие папки
            continue
        
        # 2. Загружаем config.json
        config_path = os.path.join(folder, 'config.json')
        if not os.path.exists(config_path):
            continue  # нет конфига – пропускаем
        try:
            cfg = Config.from_json(config_path)
        except:
            continue
        
        # Проверка модели
        if model_name is not None and cfg.model_name != model_name:
            continue
        if model_mode is not None and cfg.model_mode != model_mode:
            continue
        
        # 3. Определяем количество завершённых эпох
        # Пробуем из чекпоинта
        checkpoint_path = os.path.join(folder, 'checkpoint_last.pth')
        epochs_completed = 0
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                epoch = checkpoint.get('epoch', 0)
                # epoch - номер последней завершённой эпохи (индекс с 0)
                # В main при сохранении передаётся epoch (текущая эпоха), 
                # в save_checkpoint: torch.save({'epoch': epoch, ...}) 
                # В load_checkpoint возвращается epoch+1, т.е. следующая эпоха.
                # Значит в чекпоинте хранится индекс последней завершённой эпохи.
                # Поэтому количество пройденных эпох = epoch + 1 (если начали с 0)
                # Но если загружали с resume, то epoch может быть любым.
                # Для надёжности будем считать, что пройдено epoch+1 эпох.
                epochs_completed = epoch + 1
            except:
                pass
        else:
            # Если чекпоинта нет, пытаемся из log.txt
            log_file = os.path.join(folder, 'log.txt')
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                # Ищем последнюю строку с "Epoch X/Y"
                for line in reversed(lines):
                    # Пример: "Epoch 5/16 ..."
                    match_epoch = re.search(r'Epoch\s+(\d+)/(\d+)', line)
                    if match_epoch:
                        epochs_completed = int(match_epoch.group(1))  # номер текущей эпохи (1-indexed)
                        # Это означает, что пройдено epochs_completed эпох
                        break
        
        # Проверка min_epochs
        if min_epochs is not None and epochs_completed < min_epochs:
            continue
        
        # Все проверки пройдены – добавляем папку
        result.append(folder)
    
    # Сортируем по дате (по имени папки)
    result.sort(key=lambda x: os.path.basename(x))
    return result
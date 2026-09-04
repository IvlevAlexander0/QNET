import torch
import logging

def load_partial_weights(model, state_dict, prefixes, strict=False, logger=None):
    """
    Загружает в модель только те параметры, имена которых начинаются с одного из prefixes.

    Args:
        model (nn.Module): целевая модель.
        state_dict (dict): словарь загруженных весов.
        prefixes (str or list): один префикс или список префиксов.
        strict (bool): если True, требует, чтобы все ключи из prefixes присутствовали в model.
        logger: логгер для вывода информации.

    Returns:
        tuple: (loaded_keys, missing_keys, unexpected_keys)
    """
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    elif prefixes is None:
        prefixes = []

    # Фильтруем ключи, начинающиеся с любого из prefixes
    filtered_state = {}
    for key, value in state_dict.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                filtered_state[key] = value
                break

    # Логируем статистику
    if logger:
        logger.info(f"Attempting to load {len(filtered_state)} parameters with prefixes {prefixes}")

    # Загружаем отфильтрованный словарь
    missing, unexpected = model.load_state_dict(filtered_state, strict=False)

    # Определяем, какие ключи были загружены (пересечение filtered_state и model)
    model_keys = set(model.state_dict().keys())
    loaded_keys = set(filtered_state.keys()) & model_keys
    not_loaded = set(filtered_state.keys()) - model_keys

    if logger:
        logger.info(f"Loaded {len(loaded_keys)} parameters, {len(not_loaded)} parameters were not used")
        if missing:
            logger.warning(f"Missing keys: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected}")

    # Если strict=True и есть несовпадения, выбрасываем ошибку
    if strict:
        if missing:
            raise RuntimeError(f"Missing keys: {missing}")
        if unexpected:
            raise RuntimeError(f"Unexpected keys: {unexpected}")

    return loaded_keys, missing, unexpected
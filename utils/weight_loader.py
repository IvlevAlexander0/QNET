"""Selective loading of model weights by key prefix."""

import torch
import logging


def load_partial_weights(model, state_dict, prefixes, strict=False, logger=None):
    """Load only model parameters whose names start with one of ``prefixes``.

    Args:
        model: Target ``nn.Module`` into which weights are loaded.
        state_dict: Source state dictionary containing candidate parameters.
        prefixes: A single prefix string or a list of prefix strings used to
            select keys from ``state_dict``.
        strict: If True, raise when any selected key is missing from the
            model or any selected key is unexpected.
        logger: Optional ``logging.Logger`` used to report loading statistics.

    Returns:
        Tuple ``(loaded_keys, missing_keys, unexpected_keys)`` where
        ``loaded_keys`` is the set of keys that were actually transferred to
        the model, ``missing_keys`` is the list of keys reported missing by
        ``load_state_dict``, and ``unexpected_keys`` is the list of keys
        reported unexpected by ``load_state_dict``.

    Raises:
        RuntimeError: If ``strict`` is True and either ``missing_keys`` or
            ``unexpected_keys`` is non-empty.
    """
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    elif prefixes is None:
        prefixes = []

    filtered_state = {}
    for key, value in state_dict.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                filtered_state[key] = value
                break

    if logger:
        logger.info(f"Attempting to load {len(filtered_state)} parameters with prefixes {prefixes}")

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)

    model_keys = set(model.state_dict().keys())
    loaded_keys = set(filtered_state.keys()) & model_keys
    not_loaded = set(filtered_state.keys()) - model_keys

    if logger:
        logger.info(f"Loaded {len(loaded_keys)} parameters, {len(not_loaded)} parameters were not used")
        if missing:
            logger.warning(f"Missing keys: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected}")

    if strict:
        if missing:
            raise RuntimeError(f"Missing keys: {missing}")
        if unexpected:
            raise RuntimeError(f"Unexpected keys: {unexpected}")

    return loaded_keys, missing, unexpected
"""Registry for model creator factories."""

from typing import Callable, Dict, Tuple, Any

ModelCreator = Callable[[str, int, Tuple[int, int], Any], Tuple[Any, dict]]

_MODEL_REGISTRY: Dict[str, ModelCreator] = {}


def register_model(*names):
    """Register a model factory under one or more names.

    Args:
        *names: One or more registry names to associate with the creator.

    Returns:
        Decorator that registers the wrapped creator and returns it unchanged.

    Raises:
        ValueError: If any name is already registered to a different creator.
    """
    def decorator(fn):
        """Register ``fn`` under the captured names.

        Args:
            fn: Model creator callable to register.

        Returns:
            The same callable ``fn``.
        """
        for name in names:
            existing = _MODEL_REGISTRY.get(name)
            if existing is not None and existing is not fn:
                raise ValueError(f"Model '{name}' is already registered by {existing}")
            _MODEL_REGISTRY[name] = fn
        return fn
    return decorator


def get_model_creator(name):
    """Return the model creator registered under ``name``.

    Args:
        name: Registry name of the model creator.

    Returns:
        The registered model creator callable.

    Raises:
        KeyError: If no creator is registered under ``name``.
    """
    try:
        return _MODEL_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown model '{name}'. Available: {sorted(_MODEL_REGISTRY)}"
        ) from None


def available_models():
    """Return the names of all registered models.

    Returns:
        Sorted list of registered model names.
    """
    return sorted(_MODEL_REGISTRY)
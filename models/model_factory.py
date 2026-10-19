"""Model factory that discovers and instantiates registered model creators."""

import importlib
import pkgutil

from models.registry import get_model_creator


def _autodiscover(package_name="models"):
    """Import all ``*_model.py`` modules so their decorators run.

    Args:
        package_name: Name of the package to scan for model modules.

    Returns:
        None.
    """
    package = importlib.import_module(package_name)
    for _, modname, ispkg in pkgutil.iter_modules(package.__path__):
        if ispkg or not modname.endswith("_model"):
            continue
        importlib.import_module(f"{package_name}.{modname}")


_autodiscover()


def create_model(config, num_classes, input_shape):
    """Create a model instance using the creator registered under ``config.model_name``.

    Args:
        config: Configuration object providing at least ``model_name`` and
            ``model_mode``; additional attributes are forwarded to the
            creator.
        num_classes: Number of output classes.
        input_shape: Input spatial size as ``(height, width)``.

    Returns:
        Tuple ``(model, info)`` where ``model`` is an ``nn.Module`` ready for
        training or inference and ``info`` is a dictionary describing
        parameter groups, module prefixes and model metadata.

    Raises:
        KeyError: If no creator is registered under ``config.model_name``.
    """
    creator = get_model_creator(config.model_name)
    model, info = creator(config.model_mode, num_classes, input_shape, config)
    return model, info
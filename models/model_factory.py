# models/model_factory.py
from models.base_model import create_base_model
from models.resnet18_model import create_resnet18_model
from models.efficientnet_model import create_efficientnet_model

_MODEL_REGISTRY = {
    'clp_net': create_base_model,
    'resnet18': create_resnet18_model,
    'effnetv0': create_efficientnet_model,
}

def create_model(config, num_classes, input_shape):
    # input_shape = (height, width)
    creator = _MODEL_REGISTRY[config.model_name]
    model, info = creator(config.model_mode, num_classes, input_shape, config)
    return model, info

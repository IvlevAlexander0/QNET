"""ResNet18-based classical and hybrid quantum models."""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from models.quantum_core import build_quantum_layer
from models.registry import register_model

model_config = {"reps": 4}


class ResNet18FeatureExtractor(nn.Module):
    """ResNet18 backbone returning a 512-dimensional feature vector."""

    def __init__(self):
        """Initialize the backbone with pretrained ImageNet weights.

        Returns:
            None.
        """
        super().__init__()
        self.base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.base = nn.Sequential(*list(self.base.children())[:-1])

    def forward(self, x):
        """Compute backbone features for an input batch.

        Args:
            x: Input tensor of shape ``(N, 1, H, W)`` or ``(N, 3, H, W)``.
                Single-channel inputs are expanded to three channels.

        Returns:
            Feature tensor of shape ``(N, 512)``.
        """
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        features = self.base(x)
        return features.view(features.size(0), -1)


class ResNet(nn.Module):
    """ResNet18 backbone with a linear classification head."""

    def __init__(self, num_classes, config):
        """Initialize the classical ResNet model.

        Args:
            num_classes: Number of output classes.
            config: Configuration object (currently unused).

        Returns:
            None.
        """
        super().__init__()
        self.feature_extractor = ResNet18FeatureExtractor()
        self.projection = nn.Linear(512, num_classes)

    def forward(self, x):
        """Compute class logits for an input batch.

        Args:
            x: Input tensor of shape ``(N, 1, H, W)`` or ``(N, 3, H, W)``.

        Returns:
            Logit tensor of shape ``(N, num_classes)``.
        """
        features = self.feature_extractor(x)
        return self.projection(features)


class HybridQNet(nn.Module):
    """ResNet18 backbone with three parallel quantum layers."""

    def __init__(self, num_classes, config):
        """Initialize the hybrid quantum model.

        Args:
            num_classes: Number of output classes.
            config: Configuration object providing ``num_features``,
                ``encoding_type``, ``ansatz_type``, ``entanglement``,
                ``reps``, ``shots``, ``backend_type`` and
                ``transpile_before``.

        Returns:
            None.
        """
        super().__init__()
        self.feature_extractor = ResNet18FeatureExtractor()

        self.num_features = getattr(config, 'num_features', num_classes)
        self.projection = nn.Linear(512, self.num_features)
        self.BN_in = nn.BatchNorm1d(self.num_features)

        self.q_net1 = build_quantum_layer(
            self.num_features, num_classes, config, reps_default=model_config['reps']
        )
        self.q_net2 = build_quantum_layer(
            self.num_features, num_classes, config, reps_default=model_config['reps']
        )
        self.q_net3 = build_quantum_layer(
            self.num_features, num_classes, config, reps_default=model_config['reps']
        )

        q_out_size = self.q_net1.output_shape[0]
        self.BN_out = nn.BatchNorm1d(q_out_size * 3)
        self.final = nn.Linear(q_out_size * 3, num_classes)

    def forward(self, x):
        """Compute class logits for an input batch.

        Args:
            x: Input tensor of shape ``(N, 1, H, W)`` or ``(N, 3, H, W)``.

        Returns:
            Logit tensor of shape ``(N, num_classes)``.
        """
        features = self.feature_extractor(x)
        p_features = self.projection(features)
        projected_features = self.BN_in(p_features)

        qout1 = self.q_net1(projected_features)
        qout2 = self.q_net2(projected_features)
        qout3 = self.q_net3(projected_features)

        output = torch.cat((qout1, qout2, qout3), dim=1)
        BN_output = self.BN_out(output)
        return self.final(BN_output)


@register_model("resnet18")
def create_resnet18_model(mode, num_classes, input_shape, config):
    """Create a ResNet18-based model for the requested mode.

    Args:
        mode: Model mode, either ``'classical'`` or ``'quantum'``.
        num_classes: Number of output classes.
        input_shape: Input spatial size as ``(height, width)``; accepted for
            interface compatibility.
        config: Configuration object forwarded to the model constructor.

    Returns:
        Tuple ``(model, info)`` where ``model`` is an ``nn.Sequential``
        ending with a ``Sigmoid`` and ``info`` is a dictionary describing
        parameter groups and module prefixes.

    Raises:
        ValueError: If ``mode`` is not a supported mode.
    """
    if mode == 'classical':
        model = ResNet(num_classes, config)
        model = nn.Sequential(model, nn.Sigmoid())
        info = {
            'type': 'classical',
            'model_name': 'resnet18',
            'classical_params': list(model[0].parameters()),
            'quantum_params': [],
            'classical_prefix': '0.',
            'quantum_prefix': None,
            'base_model': model[0],
        }
        return model, info

    elif mode == 'quantum':
        hybrid_model = HybridQNet(num_classes, config)
        model = nn.Sequential(hybrid_model, nn.Sigmoid())

        classical_params = list(hybrid_model.feature_extractor.parameters()) + \
                           list(hybrid_model.projection.parameters())
        quantum_params = list(hybrid_model.q_net1.parameters()) + \
                         list(hybrid_model.q_net2.parameters()) + \
                         list(hybrid_model.q_net3.parameters())
        info = {
            'type': 'hybrid',
            'model_name': 'resnet18',
            'classical_params': classical_params,
            'quantum_params': quantum_params,
            'classical_prefix': ['feature_extractor.', 'projection.'],
            'quantum_prefix': 'q_net.',
            'base_model': hybrid_model,
        }
        return model, info

    else:
        raise ValueError(f"Unknown mode: {mode}")
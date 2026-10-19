import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.quantum_core import build_quantum_layer
from models.registry import register_model

model_config = {"reps": 4}


def findConv2dOutShape(hin, win, conv, pool=1):
    kernel_size = conv.kernel_size
    stride = conv.stride
    padding = conv.padding
    dilation = conv.dilation
    hout = np.floor((hin + 2*padding[0] - dilation[0]*(kernel_size[0]-1) - 1)/stride[0] + 1)
    wout = np.floor((win + 2*padding[1] - dilation[1]*(kernel_size[1]-1) - 1)/stride[1] + 1)
    if pool:
        hout /= pool
        wout /= pool
    return int(hout), int(wout)


class Clp_Net(nn.Module):
    def __init__(self, params):
        super(Clp_Net, self).__init__()
        Cin, Hin, Win = params["shape_in"]
        out_ = params["out"]
        num_classes = params["num_classes"]

        self.conv1 = nn.Conv2d(Cin, out_, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(out_)
        h, w = findConv2dOutShape(Hin, Win, self.conv1)

        self.conv2 = nn.Conv2d(out_, out_*2, kernel_size=3, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(out_*2)
        h, w = findConv2dOutShape(h, w, self.conv2)

        self.conv3 = nn.Conv2d(out_*2, out_*4, kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(out_*4)
        h, w = findConv2dOutShape(h, w, self.conv3)

        self.conv4 = nn.Conv2d(out_*4, out_*4, kernel_size=4, stride=2)
        self.bn4 = nn.BatchNorm2d(out_*4)
        h, w = findConv2dOutShape(h, w, self.conv4)

        self.pool_adaptive = nn.AdaptiveAvgPool2d((7, 7))

        self.fc1 = nn.Linear(7 * 7 * out_ * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, params.get("fc_out_size", num_classes))
        self.drop0 = nn.Dropout(0.05)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=1, padding=1)
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=1, padding=1)
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=1)
        x = self.bn3(x)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = self.bn4(x)

        x = self.pool_adaptive(x)

        x = torch.flatten(x, 1)
        x = self.drop0(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.normalize(x, p=2, dim=-1)
        return x

@register_model("clp_net")
def create_base_model(mode, num_classes, input_shape, config):
    height, width = input_shape
    if mode == 'classical':
        params = {"shape_in": (1, height, width), "num_classes": num_classes, "out": 6}
        clp = Clp_Net(params)
        model = nn.Sequential(clp, nn.Sigmoid())
        info = {
            'type': 'classical',
            'model_name': 'clp_net',
            'classical_params': list(clp.parameters()),
            'quantum_params': [],
            'classical_prefix': '0.',
            'quantum_prefix': None,
            'base_model': clp,
        }
        return model, info

    elif mode == 'quantum':
        num_features = 4
        params = {
            "shape_in": (1, height, width),
            "num_classes": num_classes,
            "out": 6,
            "fc_out_size": num_features,
        }
        clp = Clp_Net(params)

        # Для base_model по умолчанию — state encoding (совместимость).
        config.encoding_type = getattr(config, 'encoding_type', 'state')
        q_net = build_quantum_layer(
            num_features, num_classes, config, reps_default=model_config['reps']
        )

        model = nn.Sequential(clp, q_net, nn.Sigmoid())
        info = {
            'type': 'hybrid',
            'model_name': 'clp_net',
            'qnn': q_net.qnn,                       # это TorchConnector (для оптимизатора)
            'qnn_output_shape': q_net.output_shape, # <-- если нужно
            'classical_params': list(clp.parameters()),
            'quantum_params': list(q_net.parameters()),
            'classical_prefix': '0.',
            'quantum_prefix': '1.',
            'base_model': nn.Sequential(clp, q_net),
        }
        return model, info

    else:
        raise ValueError(f"Unknown mode: {mode}")
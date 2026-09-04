import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from models.quantum_core import create_qnn, Q_Net

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

        # В гибридном режиме мы будем использовать первые 4 выхода для квантовой части,
        # поэтому последний слой может выдавать 4 нейрона (если гибрид) или num_classes (классика).
        # Здесь мы делаем универсально: выходной размер будет задан через params["fc_out_size"].
        self.pool_adaptive = nn.AdaptiveAvgPool2d((7, 7))

        # Теперь fc1 всегда будет принимать 7*7*out_*4 нейронов
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


# --- Специфичные для CLP_Net квантовые функции ---
def create_2q_parameterized_state_preparation_circuit():
    thetas = ParameterVector('Ψ', 3)
    circuit = QuantumCircuit(2)
    circuit.ry(thetas[0], 0)
    circuit.x(0)
    circuit.cry(thetas[1], 0, 1)
    circuit.x(0)
    circuit.cry(thetas[2], 0, 1)
    return circuit


def compute_2q_state_preparation_angles(state_vector):
    """
    Преобразует вектор состояния (batch, 4) в углы для схемы подготовки состояния.
    Использует векторизованные операции и защиту от численных ошибок.
    """
    sv = torch.real(state_vector)
    single = sv.dim() == 1
    if single:
        sv = sv.unsqueeze(0)

    a, b, c, d = sv[:, 0], sv[:, 1], sv[:, 2], sv[:, 3]

    # Нормализация вектора с защитой от деления на ноль
    norm = torch.sqrt(a**2 + b**2 + c**2 + d**2)
    norm = torch.clamp(norm, min=1e-12)
    a = a / norm
    b = b / norm
    c = c / norm
    d = d / norm

    # Вычисление r1 = sqrt(a² + c²), r2 = sqrt(b² + d²)
    r1 = torch.sqrt(a**2 + c**2)
    r2 = torch.sqrt(b**2 + d**2)
    # r0 = sqrt(r1² + r2²) ≈ 1, но для устойчивости вычисляем явно
    r0 = torch.sqrt(r1**2 + r2**2)

    eps = 1e-12
    # Вычисление углов с clamp для защиты от выхода за [-1, 1]
    theta0 = 2 * torch.acos(torch.clamp(r1 / (r0 + eps), -1.0, 1.0))
    theta1 = 2 * torch.acos(torch.clamp(a / (r1 + eps), -1.0, 1.0))
    theta2 = 2 * torch.acos(torch.clamp(b / (r2 + eps), -1.0, 1.0))

    # Обнуление углов, где знаменатель равен нулю (избегаем NaN)
    theta1 = torch.where(r1 > eps, theta1, torch.zeros_like(theta1))
    theta2 = torch.where(r2 > eps, theta2, torch.zeros_like(theta2))
    theta0 = torch.where(r0 > eps, theta0, torch.zeros_like(theta0))

    result = torch.stack([theta0, theta1, theta2], dim=1)
    return result.squeeze(0) if single else result


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
        params = {"shape_in": (1, height, width), "num_classes": num_classes, "out": 6, "fc_out_size": 4}
        clp = Clp_Net(params)
        feature_map = create_2q_parameterized_state_preparation_circuit()
        ansatz = RealAmplitudes(2, reps=model_config['reps'], entanglement="circular")
        qnn = create_qnn(
            feature_map, ansatz, config.shots,
            backend_type=config.backend_type,
            octillion_token=config.octillion_token,
            octillion_chip=config.octillion_chip,
            transpile_before=config.transpile_before
        )
        q_net = Q_Net(qnn, num_classes, input_transform=compute_2q_state_preparation_angles)
        model = nn.Sequential(clp, q_net, nn.Sigmoid())
        info = {
            'type': 'hybrid',
            'model_name': 'clp_net',
            'qnn': qnn,
            'classical_params': list(clp.parameters()),
            'quantum_params': list(q_net.parameters()),
            'classical_prefix': '0.',
            'quantum_prefix': '1.',
            'base_model': nn.Sequential(clp, q_net),      # префикс для слоя Q_Net
        }
        return model, info

    else:
        raise ValueError(f"Unknown mode: {mode}")
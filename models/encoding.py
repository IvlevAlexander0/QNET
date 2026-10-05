# models/encoding.py
"""
Кодирование классических признаков в квантовое состояние.

Поддерживаемые типы:
  - 'angle' : угловое кодирование. num_qubits = num_features.
              RY(x_i) на кубите i. input_transform = None.
  - 'state' : амплитудное кодирование в состояние. num_features = 2^n.
              num_qubits = n. Используется input_transform для пересчёта
              вектора признаков в углы параметризованной схемы подготовки.

Функция create_encoding(encoding_type, num_features) возвращает:
    (feature_map: QuantumCircuit, input_transform: Callable | None, num_qubits: int)
"""
import math
import torch
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


# ---------------------------------------------------------------------------
# Угловое кодирование
# ---------------------------------------------------------------------------
def _create_angle_encoding(num_features: int):
    circuit = QuantumCircuit(num_features)
    pv = ParameterVector('x', num_features)
    for i in range(num_features):
        circuit.ry(pv[i], i)
    return circuit, None, num_features


# ---------------------------------------------------------------------------
# Кодирование в состояние (2 кубита, 4 признака)
# ---------------------------------------------------------------------------
def _create_2q_state_prep():
    thetas = ParameterVector('Ψ', 3)
    circuit = QuantumCircuit(2)
    circuit.ry(thetas[0], 0)
    circuit.x(0)
    circuit.cry(thetas[1], 0, 1)
    circuit.x(0)
    circuit.cry(thetas[2], 0, 1)

    def transform(state_vector):
        """(batch, 4) -> (batch, 3) углы схемы подготовки состояния."""
        sv = torch.real(state_vector)
        single = sv.dim() == 1
        if single:
            sv = sv.unsqueeze(0)

        a, b, c, d = sv[:, 0], sv[:, 1], sv[:, 2], sv[:, 3]

        norm = torch.sqrt(a**2 + b**2 + c**2 + d**2)
        norm = torch.clamp(norm, min=1e-6)
        a, b, c, d = a / norm, b / norm, c / norm, d / norm

        r1 = torch.sqrt(a**2 + c**2)   # амплитуда при q0 = 0 (индексы 0 и 2)
        r2 = torch.sqrt(b**2 + d**2)   # амплитуда при q0 = 1 (индексы 1 и 3)
        r0 = torch.sqrt(r1**2 + r2**2)

        # θ₀: Ry на q0, задаёт распределение между ветками q0=0 и q0=1
        eps = 1e-12
        theta0 = 2 * torch.acos(torch.clamp(r1 / (r0 + eps), -1.0, 1.0))
        theta1 = 2 * torch.acos(torch.clamp(a / (r1 + eps), -1.0, 1.0))
        theta2 = 2 * torch.acos(torch.clamp(b / (r2 + eps), -1.0, 1.0))

        theta1 = torch.where(r1 > eps, theta1, torch.zeros_like(theta1))
        theta2 = torch.where(r2 > eps, theta2, torch.zeros_like(theta2))
        theta0 = torch.where(r0 > eps, theta0, torch.zeros_like(theta0))

        result = torch.stack([theta0, theta1, theta2], dim=1)
        return result.squeeze(0) if single else result

    return circuit, transform


def _create_state_encoding(num_features: int):
    if num_features <= 0 or (num_features & (num_features - 1)) != 0:
        raise ValueError(
            f"State encoding requires num_features to be a power of 2, "
            f"got {num_features}"
        )
    num_qubits = int(math.log2(num_features))
    if num_qubits == 2:
        circuit, transform = _create_2q_state_prep()
        return circuit, transform, 2
    raise NotImplementedError(
        f"State encoding not implemented for {num_qubits} qubits "
        f"({num_features} features)"
    )


# ---------------------------------------------------------------------------
# Публичная фабрика
# ---------------------------------------------------------------------------
def create_encoding(encoding_type: str, num_features: int):
    """
    Returns:
        feature_map      : QuantumCircuit
        input_transform  : Callable | None
        num_qubits       : int
    """
    encoding_type = (encoding_type or 'angle').lower()
    if encoding_type == 'angle':
        return _create_angle_encoding(num_features)
    elif encoding_type == 'state':
        return _create_state_encoding(num_features)
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")
"""Encoding of classical features into quantum states.

Supported encoding types:
  - ``'angle'``: Angle encoding with ``num_qubits == num_features``.
    ``RY(x_i)`` is applied on qubit ``i``; ``input_transform`` is ``None``.
  - ``'state'``: Amplitude encoding into a state with
    ``num_features == 2 ** n``; ``num_qubits == n``. An ``input_transform``
    is used to convert the feature vector into the angles of the
    state-preparation circuit.
"""

import math

import torch
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _create_angle_encoding(num_features):
    """Create an angle-encoding feature map.

    Args:
        num_features: Number of features; also the number of qubits.

    Returns:
        Tuple ``(feature_map, input_transform, num_qubits)`` where
        ``feature_map`` is a ``QuantumCircuit`` with ``num_features``
        ``RY`` gates, ``input_transform`` is ``None`` and ``num_qubits`` is
        the integer ``num_features``.
    """
    circuit = QuantumCircuit(num_features)
    pv = ParameterVector('x', num_features)
    for i in range(num_features):
        circuit.ry(pv[i], i)
    return circuit, None, num_features


def _create_2q_state_prep():
    """Create a two-qubit state-preparation circuit and its angle transform.

    Returns:
        Tuple ``(circuit, transform)`` where ``circuit`` is a two-qubit
        ``QuantumCircuit`` parameterized by three angles, and ``transform``
        is a callable mapping a state vector of shape ``(batch, 4)`` or
        ``(4,)`` to state-preparation angles of shape ``(batch, 3)`` or
        ``(3,)`` respectively.
    """
    thetas = ParameterVector('Ψ', 3)
    circuit = QuantumCircuit(2)
    circuit.ry(thetas[0], 0)
    circuit.x(0)
    circuit.cry(thetas[1], 0, 1)
    circuit.x(0)
    circuit.cry(thetas[2], 0, 1)

    def transform(state_vector):
        """Convert a real state vector into state-preparation angles.

        Args:
            state_vector: Tensor of shape ``(batch, 4)`` or ``(4,)`` holding
                real amplitudes of a two-qubit state.

        Returns:
            Tensor of shape ``(batch, 3)`` or ``(3,)`` with the three
            rotation angles used by the state-preparation circuit.
        """
        sv = torch.real(state_vector)
        single = sv.dim() == 1
        if single:
            sv = sv.unsqueeze(0)

        a, b, c, d = sv[:, 0], sv[:, 1], sv[:, 2], sv[:, 3]

        norm = torch.sqrt(a ** 2 + b ** 2 + c ** 2 + d ** 2)
        norm = torch.clamp(norm, min=1e-6)
        a, b, c, d = a / norm, b / norm, c / norm, d / norm

        r1 = torch.sqrt(a ** 2 + c ** 2)
        r2 = torch.sqrt(b ** 2 + d ** 2)
        r0 = torch.sqrt(r1 ** 2 + r2 ** 2)

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


def _create_state_encoding(num_features):
    """Create a state-encoding feature map for ``2 ** n`` features.

    Args:
        num_features: Number of features; must be a power of two. Only
            ``num_features == 4`` (two qubits) is currently implemented.

    Returns:
        Tuple ``(feature_map, input_transform, num_qubits)`` where
        ``feature_map`` is a ``QuantumCircuit``, ``input_transform`` is a
        callable converting a state vector into preparation angles, and
        ``num_qubits`` is the integer number of qubits.

    Raises:
        ValueError: If ``num_features`` is not a positive power of two.
        NotImplementedError: If the required qubit count is not 2.
    """
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


def create_encoding(encoding_type, num_features):
    """Create a feature map circuit for the requested encoding type.

    Args:
        encoding_type: Encoding family, either ``'angle'`` or ``'state'``.
        num_features: Number of classical features to encode.

    Returns:
        Tuple ``(feature_map, input_transform, num_qubits)`` where
        ``feature_map`` is a ``QuantumCircuit``, ``input_transform`` is a
        callable or ``None`` and ``num_qubits`` is the integer number of
        qubits in the circuit.

    Raises:
        ValueError: If ``encoding_type`` is not a supported encoding family.
    """
    encoding_type = (encoding_type or 'angle').lower()
    if encoding_type == 'angle':
        return _create_angle_encoding(num_features)
    elif encoding_type == 'state':
        return _create_state_encoding(num_features)
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")
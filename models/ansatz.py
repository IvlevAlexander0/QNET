"""Variational ansatz circuits for quantum neural networks.

Supported ansatz types:
  - ``'ry'``: RealAmplitudes — alternating RY layers and CX entanglement.
  - ``'zz'``: RY layers interleaved with RZZ entanglement.

Supported entanglement types: ``'linear'``, ``'circular'``, ``'full'``.
"""

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes


def _entanglement_pairs(num_qubits, entanglement):
    """Build the list of qubit pairs used for entanglement.

    Args:
        num_qubits: Number of qubits in the circuit.
        entanglement: Entanglement topology, one of ``'linear'``,
            ``'circular'`` or ``'full'``.

    Returns:
        List of ``(i, j)`` integer tuples describing entangled qubit pairs.

    Raises:
        ValueError: If ``entanglement`` is not a supported topology.
    """
    entanglement = (entanglement or 'circular').lower()
    if entanglement == 'linear':
        return [(i, i + 1) for i in range(num_qubits - 1)]
    if entanglement == 'circular':
        pairs = [(i, i + 1) for i in range(num_qubits - 1)]
        if num_qubits > 2:
            pairs.append((num_qubits - 1, 0))
        return pairs
    if entanglement == 'full':
        return [(i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)]
    raise ValueError(f"Unknown entanglement type: {entanglement}")


def _create_ry_ansatz(num_qubits, reps, entanglement):
    """Create a RealAmplitudes (RY + CX) ansatz.

    Args:
        num_qubits: Number of qubits in the circuit.
        reps: Number of variational repetitions.
        entanglement: Entanglement topology, one of ``'linear'``,
            ``'circular'`` or ``'full'``.

    Returns:
        A ``QuantumCircuit`` implementing the RealAmplitudes ansatz.
    """
    return RealAmplitudes(num_qubits, reps=reps, entanglement=entanglement)


def _create_zz_ansatz(num_qubits, reps, entanglement):
    """Create an RY + RZZ variational ansatz.

    Args:
        num_qubits: Number of qubits in the circuit.
        reps: Number of variational repetitions.
        entanglement: Entanglement topology, one of ``'linear'``,
            ``'circular'`` or ``'full'``.

    Returns:
        A ``QuantumCircuit`` implementing the RY + RZZ ansatz.
    """
    circuit = QuantumCircuit(num_qubits)
    pairs = _entanglement_pairs(num_qubits, entanglement)

    for rep in range(reps):
        pv_ry = ParameterVector(f'ry_{rep}', num_qubits)
        for i in range(num_qubits):
            circuit.ry(pv_ry[i], i)

        if pairs:
            pv_zz = ParameterVector(f'zz_{rep}', len(pairs))
            for k, (i, j) in enumerate(pairs):
                circuit.rzz(pv_zz[k], i, j)

    return circuit


def create_ansatz(ansatz_type, num_qubits, reps, entanglement='circular'):
    """Create a parameterized variational ansatz circuit.

    Args:
        ansatz_type: Ansatz family, either ``'ry'`` or ``'zz'``.
        num_qubits: Number of qubits in the circuit.
        reps: Number of variational repetitions.
        entanglement: Entanglement topology, one of ``'linear'``,
            ``'circular'`` or ``'full'``.

    Returns:
        A parameterized ``QuantumCircuit`` implementing the requested ansatz.

    Raises:
        ValueError: If ``ansatz_type`` is not a supported ansatz family.
    """
    ansatz_type = (ansatz_type or 'ry').lower()
    if ansatz_type == 'ry':
        return _create_ry_ansatz(num_qubits, reps, entanglement)
    elif ansatz_type == 'zz':
        return _create_zz_ansatz(num_qubits, reps, entanglement)
    else:
        raise ValueError(f"Unknown ansatz type: {ansatz_type}")
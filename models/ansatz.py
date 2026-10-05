# models/ansatz.py
"""
Вариационные анзацы для QNN.

Поддерживаемые типы:
  - 'ry' : RealAmplitudes — чередование RY-слоёв и CX-запутывания.
  - 'zz' : RY-слои, чередующиеся с RZZ-запутыванием.

Типы запутывания: 'linear', 'circular', 'full'.

Функция create_ansatz(ansatz_type, num_qubits, reps, entanglement)
возвращает параметризованную QuantumCircuit.
"""
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes


def _entanglement_pairs(num_qubits: int, entanglement: str):
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


def _create_ry_ansatz(num_qubits: int, reps: int, entanglement: str):
    return RealAmplitudes(num_qubits, reps=reps, entanglement=entanglement)


def _create_zz_ansatz(num_qubits: int, reps: int, entanglement: str):
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


def create_ansatz(ansatz_type: str, num_qubits: int, reps: int,
                  entanglement: str = 'circular'):
    ansatz_type = (ansatz_type or 'ry').lower()
    if ansatz_type == 'ry':
        return _create_ry_ansatz(num_qubits, reps, entanglement)
    elif ansatz_type == 'zz':
        return _create_zz_ansatz(num_qubits, reps, entanglement)
    else:
        raise ValueError(f"Unknown ansatz type: {ansatz_type}")
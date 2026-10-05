import time
import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import BaseSamplerV2, PrimitiveResult, PubResult
from qiskit.primitives.containers import DataBin, BitArray
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import SamplerQNN

from models.encoding import create_encoding
from models.ansatz import create_ansatz

class NoisySamplerV2(BaseSamplerV2):
    def __init__(self, backend, shots=2048, transpile_before_bind=False, noise_model=None):
        super().__init__()

        # Запоминаем "настоящий" целевой бэкенд для транспиляции
        self._target_backend = backend

        # Для исполнения используем AerSimulator с шумом
        if isinstance(backend, AerSimulator):
            self._backend = backend
        else:
            self._backend = AerSimulator.from_backend(backend)

        if noise_model is not None:
            self._backend.set_options(noise_model=noise_model)

        self._shots = shots
        self._transpile_before_bind = transpile_before_bind
        self._transpiled_cache = {}

        # Pass manager целимся в ИСХОДНЫЙ бэкенд (у него корректный basis_gates),
        # а не в AerSimulator-обёртку.
        pm_target = backend  # <-- ключевой момент
        try:
            self._pm = generate_preset_pass_manager(
                optimization_level=3, backend=pm_target
            )
        except Exception:
            self._pm = None

    def _get_transpiled_circuit(self, circuit):
        cache_key = id(circuit)
        if cache_key not in self._transpiled_cache:
            transpiled = self._pm.run(circuit) if self._pm is not None else circuit
            self._transpiled_cache[cache_key] = transpiled
        return self._transpiled_cache[cache_key]
    
    def run(self, pubs, *, shots=None):
        """
        Публичный API сэмплера. Возвращает NoisySamplerJob,
        который лениво вычислит результат по .result().
        """
        return NoisySamplerJob(
            sampler=self,
            pubs=pubs,
            shots=shots if shots is not None else self._shots
        )

    def _run(self, pubs, shots):
        results = []

        for pub in pubs:
            circuit = pub[0]
            parameter_values = pub[1]

            if parameter_values is None:
                param_batch = np.empty((1, 0))
            else:
                param_batch = np.asarray(parameter_values)
                if param_batch.ndim == 1:
                    param_batch = param_batch.reshape(1, -1)
                elif param_batch.ndim != 2:
                    raise ValueError(
                        f"parameter_values has invalid shape {param_batch.ndim}"
                    )

            # Получаем транспилированную схему из кеша (если требуется)
            if self._transpile_before_bind:
                transpiled_circuit = self._get_transpiled_circuit(circuit)
            else:
                transpiled_circuit = None

            all_counts = []

            for params_row in param_batch:
                if self._transpile_before_bind:
                    base_circ = transpiled_circuit
                    if param_batch.size > 0:
                        param_dict = dict(zip(circuit.parameters, params_row))
                        bound_circuit = base_circ.assign_parameters(param_dict)
                    else:
                        bound_circuit = base_circ
                else:
                    if param_batch.size > 0:
                        param_dict = dict(zip(circuit.parameters, params_row))
                        bound_circuit = circuit.assign_parameters(param_dict)
                    else:
                        bound_circuit = circuit

                    if self._pm is not None:
                        bound_circuit = self._pm.run(bound_circuit)

                # Запуск на шумном бэкенде Qiskit Aer
                job = self._backend.run(bound_circuit, shots=shots)
                counts = job.result().get_counts()
                all_counts.append(counts)

            bitarray = BitArray.from_counts(
                all_counts, num_bits=circuit.num_clbits
            )
            data_bin = DataBin(meas=bitarray)
            pub_result = PubResult(data_bin, metadata={"shots": shots})
            results.append(pub_result)

        return PrimitiveResult(results)


class NoisySamplerJob:
    """Wrapper for noisy sampler job."""

    def __init__(self, sampler, pubs, shots):
        self._sampler = sampler
        self._pubs = pubs
        self._shots = shots
        self._result = None

    def result(self):
        if self._result is None:
            self._result = self._sampler._run(
                self._pubs, self._shots
            )
        return self._result


def create_qnn(
    feature_map,
    ansatz,
    shots,
    backend_type='noisy_simulator',
    backend=FakeSherbrooke(),
    noise_model=None,
    transpile_before=True,
):
    """
    Создаёт SamplerQNN на основе готовых схем feature_map и ansatz.

    Args:
        feature_map (QuantumCircuit): параметризованная схема кодирования входных данных.
        ansatz (QuantumCircuit): параметризованная вариационная схема.
        shots (int): количество измерений.
        backend_type (str): 'aer_simulator' | 'noisy_simulator' | 'custom'.
        backend (Backend | AerSimulator | None): конкретный бэкенд Qiskit.
        noise_model (NoiseModel | None): шумовая модель Aer (для noisy/custom).
        transpile_before (bool): транспилировать до привязки параметров
                                 (для NoisySamplerV2).

    Returns:
        SamplerQNN
    """
    num_qubits = feature_map.num_qubits
    qc = QuantumCircuit(num_qubits)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz.decompose(), inplace=True)

    if backend_type == 'aer_simulator':
        backend = AerSimulator()
        sampler = SamplerV2(mode=backend, options={"default_shots": shots})
        pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
        qc_isa = pm.run(qc)

    elif backend_type == 'noisy_simulator':
        # Шумный симулятор — используем NoisySamplerV2,
        # который сам делает транспиляцию под целевой backend
        backend = backend if backend is not None else AerSimulator()
        sampler = NoisySamplerV2(
            backend,
            shots=shots,
            transpile_before_bind=transpile_before,
            noise_model=noise_model,
        )
        qc_isa = qc  # транспиляция внутри сэмплера

    else:
        raise ValueError(f"Unknown backend_type: {backend_type}")

    qnn = SamplerQNN(
        circuit=qc_isa,
        sampler=sampler,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        input_gradients=True,
    )
    return qnn


class Q_Net(nn.Module):
    """
    Универсальный квантовый слой.
    Принимает qnn, функцию преобразования входа и количество классов.
    """
    def __init__(self, qnn, num_classes, input_transform=None, output_transform=None):
        super().__init__()
        # Запоминаем форму выхода ДО обёртки в TorchConnector
        self.output_shape = tuple(qnn.output_shape)
        self.qnn = TorchConnector(qnn)
        self.fc_out = nn.Linear(self.output_shape[0], num_classes)
        self.input_transform = input_transform
        self.output_transform = output_transform

    def forward(self, x):
        if self.input_transform is not None:
            x = self.input_transform(x)
        x = self.qnn(x)
        if self.output_transform is not None:
            x = self.fc_out(x)
        return x


def build_quantum_layer(num_features, num_classes, config, reps_default=4):
    """
    Собирает готовый квантовый слой Q_Net из конфига:
        config.encoding_type   : 'angle' | 'state'
        config.ansatz_type     : 'ry'    | 'zz'
        config.entanglement    : 'linear'|'circular'|'full'
        config.reps            : int
        config.shots, config.backend_type, config.transpile_before
    """
    encoding_type = getattr(config, 'encoding_type', 'angle')
    ansatz_type = getattr(config, 'ansatz_type', 'ry')
    entanglement = getattr(config, 'entanglement', 'circular')
    reps = getattr(config, 'reps', reps_default)

    feature_map, input_transform, num_qubits = create_encoding(
        encoding_type, num_features
    )
    ansatz = create_ansatz(ansatz_type, num_qubits, reps, entanglement)

    qnn = create_qnn(
        feature_map, ansatz, config.shots,
        backend_type=config.backend_type,
        transpile_before=config.transpile_before,
    )
    return Q_Net(qnn, num_classes, input_transform=input_transform)
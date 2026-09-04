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
# from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import SamplerQNN

try:
    from octillion.client import Client
except ImportError:
    Client = None

class OctillionSamplerV2(BaseSamplerV2):
    def __init__(self, backend, client=None, shots=2048, project="QNN_job", transpile_before_bind=False):
        super().__init__()
        self._backend = backend
        self._client = client
        self._shots = shots
        self._project = project
        self._transpile_before_bind = transpile_before_bind
        # Определяем бэкенд для транспиляции
        if hasattr(backend, 'basis_gates') and client is not None:
            self._backend_tr = client.local(backend.name) if hasattr(backend, 'name') else backend
        else:
            self._backend_tr = backend
        # Кеш транспилированных схем
        self._transpiled_cache = {}  # key: хеш схемы, value: транспилированная схема

    def _get_transpiled_circuit(self, circuit):
        """Возвращает транспилированную схему, используя кеш."""
        # Используем хеш от строкового представления (для простоты) или от параметров
        # Более надёжно: circuit.hash() (если доступно) или circuit.__hash__?
        # В Qiskit можно использовать circuit.qasm() для получения строки.
        # Но для производительности лучше использовать id(circuit), если схема не меняется.
        # Так как схема может быть изменена (например, добавлены параметры), используем комбинацию id и параметров.
        # Для простоты используем id(circuit), т.к. в нашем случае схема создаётся один раз.
        cache_key = id(circuit)
        if cache_key not in self._transpiled_cache:
            if hasattr(self._backend, 'basis_gates'):
                pm = generate_preset_pass_manager(optimization_level=3, backend=self._backend_tr)
                transpiled = pm.run(circuit)
            else:
                transpiled = circuit  # если нет базовых гейтов, оставляем как есть
            self._transpiled_cache[cache_key] = transpiled
        return self._transpiled_cache[cache_key]

    def _run(self, pubs, shots, project):
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
                elif param_batch.ndim == 2:
                    pass
                else:
                    raise ValueError(f"parameter_values has invalid shape {param_batch.ndim}")

            # Получаем транспилированную схему из кеша (если требуется)
            if self._transpile_before_bind:
                transpiled_circuit = self._get_transpiled_circuit(circuit)
            else:
                transpiled_circuit = None  # будет транспилироваться после привязки

            all_counts = []
            for params_row in param_batch:
                if self._transpile_before_bind:
                    # Используем кешированную транспилированную схему и привязываем параметры
                    base_circ = transpiled_circuit
                    if param_batch.size > 0:
                        param_dict = dict(zip(circuit.parameters, params_row))
                        bound_circuit = base_circ.assign_parameters(param_dict)
                    else:
                        bound_circuit = base_circ
                else:
                    # Привязываем параметры к оригинальной схеме, затем транспилируем
                    if param_batch.size > 0:
                        param_dict = dict(zip(circuit.parameters, params_row))
                        bound_circuit = circuit.assign_parameters(param_dict)
                    else:
                        bound_circuit = circuit
                    # Транспилируем (без кеша, т.к. схема меняется)
                    if hasattr(self._backend, 'basis_gates'):
                        pm = generate_preset_pass_manager(optimization_level=3, backend=self._backend_tr)
                        bound_circuit = pm.run(bound_circuit)

                # Выполняем на бэкенде
                if hasattr(self._backend, 'run') and callable(self._backend.run):
                    job = self._backend.run(bound_circuit, shots=shots, project=project)
                else:
                    raise RuntimeError("Backend does not support run()")

                counts = self._wait_for_result(job)
                all_counts.append(counts)

            bitarray = BitArray.from_counts(all_counts, num_bits=circuit.num_clbits)
            data_bin = DataBin(meas=bitarray)
            pub_result = PubResult(data_bin, metadata={"shots": shots})
            results.append(pub_result)

        return PrimitiveResult(results)


class OctillionSamplerJob:
    """Wrapper for Octillion job."""
    def __init__(self, sampler, pubs, shots, project):
        self._sampler = sampler
        self._pubs = pubs
        self._shots = shots
        self._project = project
        self._result = None

    def result(self):
        if self._result is None:
            self._result = self._sampler._run(self._pubs, self._shots, self._project)
        return self._result

    def status(self):
        return "COMPLETE"


def create_qnn(feature_map, ansatz, shots, backend_type='aer_simulator',
               octillion_token=None, octillion_chip=None, transpile_before=True):
    """
    Создаёт SamplerQNN на основе готовых схем feature_map и ansatz.

    Args:
        feature_map (QuantumCircuit): параметризованная схема для кодирования входных данных.
        ansatz (QuantumCircuit): параметризованная схема для вариационной части.
        shots (int): количество измерений.
        backend_type (str): тип бэкенда ('aer_simulator', 'fake_sherbrooke', 'octillion_remote', 'octillion_local').
        octillion_token (str): токен для Octillion (если нужен).
        octillion_chip (str): имя чипа Octillion.
        transpile_before (bool): транспилировать до привязки параметров.

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
    # elif backend_type == 'fake_sherbrooke':
    #     backend = FakeSherbrooke()
    #     sampler = SamplerV2(mode=backend, options={"default_shots": shots})
    #     pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    #     qc_isa = pm.run(qc)
    elif backend_type in ['octillion_remote', 'octillion_local']:
        if octillion_token is None:
            raise ValueError("Octillion token required")
        client = Client(octillion_token)
        if backend_type == 'octillion_remote':
            backend = client.remote(octillion_chip)
        else:
            backend = client.local(octillion_chip)
        sampler = OctillionSamplerV2(
            backend, client=client, shots=shots,
            project="QNN_training", transpile_before_bind=transpile_before
        )
        qc_isa = qc
    else:
        raise ValueError(f"Unknown backend_type: {backend_type}")

    qnn = SamplerQNN(
        circuit=qc_isa, sampler=sampler,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        input_gradients=True
    )
    return qnn


class Q_Net(nn.Module):
    """
    Универсальный квантовый слой.
    Принимает qnn, функцию преобразования входа и количество классов.
    """
    def __init__(self, qnn, num_classes, input_transform=None, output_transform=None):
        super().__init__()
        self.qnn = TorchConnector(qnn)
        self.fc_out = nn.Linear(qnn.output_shape[0], num_classes)
        self.input_transform = input_transform
        self.output_transform = output_transform

    def forward(self, x):
        if self.input_transform is not None:
            x = self.input_transform(x)
        x = self.qnn(x)
        if self.output_transform is not None:
            x = self.fc_out(x)
        return x
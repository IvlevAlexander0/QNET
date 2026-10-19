"""Quantum layer construction and noisy sampler primitives."""

import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit
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
    """SamplerV2 implementation executing circuits on a noisy Aer simulator."""

    def __init__(self, backend, shots=2048, transpile_before_bind=False,
                 noise_model=None):
        """Initialize the noisy sampler.

        Args:
            backend: Target backend used for transpilation; its noise model is
                used when running on Aer.
            shots: Default number of shots per circuit execution.
            transpile_before_bind: If True, transpile the unbound circuit
                once and reuse it across parameter rows.
            noise_model: Optional Aer noise model overriding the backend noise.

        Returns:
            None.
        """
        super().__init__()

        self._target_backend = backend

        if isinstance(backend, AerSimulator):
            self._backend = backend
        else:
            self._backend = AerSimulator.from_backend(backend)

        if noise_model is not None:
            self._backend.set_options(noise_model=noise_model)

        self._shots = shots
        self._transpile_before_bind = transpile_before_bind
        self._transpiled_cache = {}

        pm_target = backend
        try:
            self._pm = generate_preset_pass_manager(
                optimization_level=3, backend=pm_target
            )
        except Exception:
            self._pm = None

    def _get_transpiled_circuit(self, circuit):
        """Return a cached transpiled version of the given circuit.

        Args:
            circuit: Source ``QuantumCircuit`` to transpile.

        Returns:
            Transpiled ``QuantumCircuit`` (or the original circuit when no
            pass manager is available).
        """
        cache_key = id(circuit)
        if cache_key not in self._transpiled_cache:
            transpiled = self._pm.run(circuit) if self._pm is not None else circuit
            self._transpiled_cache[cache_key] = transpiled
        return self._transpiled_cache[cache_key]

    def run(self, pubs, *, shots=None):
        """Submit PUBs for execution and return a lazy job.

        Args:
            pubs: Iterable of PUBs, each ``(circuit, parameter_values)``.
            shots: Optional per-call shot override.

        Returns:
            ``NoisySamplerJob`` whose ``result()`` method produces a
            ``PrimitiveResult``.
        """
        return NoisySamplerJob(
            sampler=self,
            pubs=pubs,
            shots=shots if shots is not None else self._shots
        )

    def _run(self, pubs, shots):
        """Execute the PUBs synchronously and build a ``PrimitiveResult``.

        Args:
            pubs: Iterable of PUBs, each ``(circuit, parameter_values)``.
            shots: Number of shots per circuit execution.

        Returns:
            ``PrimitiveResult`` containing one ``PubResult`` per PUB, with
            measurement bit arrays in the ``meas`` data field.
        """
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
    """Lazy job wrapper returned by ``NoisySamplerV2.run``."""

    def __init__(self, sampler, pubs, shots):
        """Initialize the job.

        Args:
            sampler: The owning ``NoisySamplerV2`` instance.
            pubs: Iterable of PUBs to execute.
            shots: Number of shots per circuit execution.

        Returns:
            None.
        """
        self._sampler = sampler
        self._pubs = pubs
        self._shots = shots
        self._result = None

    def result(self):
        """Return the cached execution result, computing it on first call.

        Returns:
            ``PrimitiveResult`` produced by ``NoisySamplerV2._run``.
        """
        if self._result is None:
            self._result = self._sampler._run(
                self._pubs, self._shots
            )
        return self._result


def create_qnn(feature_map, ansatz, shots, backend_type='noisy_simulator',
               backend=FakeSherbrooke(), noise_model=None,
               transpile_before=True):
    """Build a ``SamplerQNN`` from a feature map and a variational ansatz.

    Args:
        feature_map: Parameterized ``QuantumCircuit`` encoding input data.
        ansatz: Parameterized ``QuantumCircuit`` holding trainable weights.
        shots: Number of measurement shots per execution.
        backend_type: Execution backend, one of ``'aer_simulator'`` or
            ``'noisy_simulator'``.
        backend: Qiskit backend instance (or ``AerSimulator``) used for the
            noisy execution path.
        noise_model: Optional Aer ``NoiseModel`` used by the noisy sampler.
        transpile_before: If True, transpile before binding parameters inside
            ``NoisySamplerV2``.

    Returns:
        ``SamplerQNN`` composed of the feature map followed by the decomposed
        ansatz.

    Raises:
        ValueError: If ``backend_type`` is not a supported execution backend.
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
        backend = backend if backend is not None else AerSimulator()
        sampler = NoisySamplerV2(
            backend,
            shots=shots,
            transpile_before_bind=transpile_before,
            noise_model=noise_model,
        )
        qc_isa = qc

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
    """Trainable quantum layer wrapping a ``SamplerQNN`` via ``TorchConnector``."""

    def __init__(self, qnn, num_classes, input_transform=None,
                 output_transform=None):
        """Initialize the quantum layer.

        Args:
            qnn: A ``SamplerQNN`` whose input and weight parameters are
                already configured.
            num_classes: Number of output classes.
            input_transform: Optional callable applied to the input tensor
                before it is passed to the QNN.
            output_transform: Optional flag; when not ``None`` a final linear
                projection ``fc_out`` is applied to the QNN output.

        Returns:
            None.
        """
        super().__init__()
        self.output_shape = tuple(qnn.output_shape)
        self.qnn = TorchConnector(qnn)
        self.fc_out = nn.Linear(self.output_shape[0], num_classes)
        self.input_transform = input_transform
        self.output_transform = output_transform

    def forward(self, x):
        """Compute the quantum layer output for an input batch.

        Args:
            x: Input tensor of shape ``(N, num_features)`` or
                ``(num_features,)``.

        Returns:
            Output tensor of shape ``(N, num_classes)`` when
            ``output_transform`` is set, otherwise the raw QNN output of
            shape ``(N, output_shape[0])``.
        """
        if self.input_transform is not None:
            x = self.input_transform(x)

        device_type = x.device.type if isinstance(x, torch.Tensor) else 'cpu'
        if isinstance(x, torch.Tensor) and x.dtype in (torch.float16, torch.bfloat16):
            x = x.float()
        with torch.autocast(device_type=device_type, enabled=False):
            x = self.qnn(x)

        if self.output_transform is not None:
            x = self.fc_out(x)
        return x


def build_quantum_layer(num_features, num_classes, config, reps_default=4):
    """Build a ``Q_Net`` from configuration attributes.

    Args:
        num_features: Number of classical features fed into the quantum
            circuit.
        num_classes: Number of output classes.
        config: Configuration object exposing ``encoding_type``,
            ``ansatz_type``, ``entanglement``, ``reps``, ``shots``,
            ``backend_type`` and ``transpile_before``.
        reps_default: Fallback number of variational repetitions when
            ``config.reps`` is not provided.

    Returns:
        ``Q_Net`` instance wrapping the constructed ``SamplerQNN`` and the
        appropriate input transform for the selected encoding.
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
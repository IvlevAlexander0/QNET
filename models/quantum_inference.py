"""Inference-only freezing of trained quantum layers."""

import copy

import numpy as np
import torch
import torch.nn as nn
from qiskit import qpy
from qiskit_machine_learning.neural_networks import SamplerQNN


class FrozenQNet(nn.Module):
    """Inference-only quantum layer with ansatz weights baked into the circuit."""

    def __init__(self, qc_bound, sampler, input_params, output_shape,
                 fc_out=None, input_transform=None, output_transform=None,
                 interpret=None):
        """Initialize the frozen quantum layer.

        Args:
            qc_bound: ``QuantumCircuit`` with the ansatz weights already
                assigned.
            sampler: Sampler primitive used to execute the circuit.
            input_params: List of input ``Parameter`` objects remaining in
                ``qc_bound``.
            output_shape: Output shape of the underlying ``SamplerQNN`` as a
                tuple.
            fc_out: Optional ``nn.Linear`` applied to the QNN output when
                ``output_transform`` is set.
            input_transform: Optional callable applied to the input tensor
                before it is passed to the QNN.
            output_transform: Optional flag; when not ``None`` ``fc_out`` is
                applied to the QNN output.
            interpret: Optional interpret function forwarded to
                ``SamplerQNN``.

        Returns:
            None.
        """
        super().__init__()
        self.output_shape = tuple(output_shape)
        self.fc_out = fc_out
        self.input_transform = input_transform
        self.output_transform = output_transform

        self._qc_bound = qc_bound
        self._sampler = sampler
        self._input_params = list(input_params)

        self._qnn = SamplerQNN(
            circuit=qc_bound,
            sampler=sampler,
            input_params=self._input_params,
            weight_params=[],
            interpret=interpret,
            output_shape=self.output_shape,
            input_gradients=False,
        )
        assert tuple(self._qnn.output_shape) == self.output_shape, (
            f"Frozen output shape mismatch: "
            f"{self._qnn.output_shape} vs {self.output_shape}"
        )

    @classmethod
    def from_trained(cls, trained_qnet, transpile=True):
        """Build a ``FrozenQNet`` from a trained ``Q_Net`` instance.

        Args:
            trained_qnet: Trained ``Q_Net`` whose weights should be baked in.
            transpile: If True, run a one-time transpilation of the bound
                circuit using the sampler pass manager (when available).

        Returns:
            ``FrozenQNet`` instance configured for inference.
        """
        qnn = trained_qnet.qnn.neural_network
        connector = trained_qnet.qnn

        weight_values = connector.weight.detach().cpu().numpy()
        weight_dict = dict(zip(qnn.weight_params, weight_values))
        qc_bound = qnn.circuit.assign_parameters(weight_dict)

        sampler = qnn.sampler
        if transpile:
            pm = getattr(sampler, "_pm", None)
            if pm is not None:
                qc_bound = pm.run(qc_bound)

        try:
            from models.quantum_core import NoisySamplerV2
            if isinstance(sampler, NoisySamplerV2):
                sampler = NoisySamplerV2(
                    sampler._target_backend,
                    shots=sampler._shots,
                    transpile_before_bind=False,
                )
                sampler._pm = None
        except ImportError:
            pass

        return cls(
            qc_bound=qc_bound,
            sampler=sampler,
            input_params=qnn.input_params,
            output_shape=trained_qnet.output_shape,
            fc_out=trained_qnet.fc_out,
            input_transform=trained_qnet.input_transform,
            output_transform=trained_qnet.output_transform,
            interpret=qnn.interpret,
        )

    def forward(self, x):
        """Compute the frozen quantum layer output for an input batch.

        Args:
            x: Input tensor of shape ``(N, num_features)`` or
                ``(num_features,)``, or an array-like convertible to a NumPy
                array.

        Returns:
            Output tensor of shape ``(N, num_classes)`` when
            ``output_transform`` is set, otherwise the raw QNN output of
            shape ``(N, output_shape[0])``, with dtype ``float32``.
        """
        if self.input_transform is not None:
            x = self.input_transform(x)

        device = x.device if isinstance(x, torch.Tensor) else torch.device("cpu")
        if isinstance(x, torch.Tensor) and x.dtype in (torch.float16, torch.bfloat16):
            x = x.float()

        x_np = (x.detach().cpu().numpy()
                if isinstance(x, torch.Tensor) else np.asarray(x, dtype=np.float32))
        if x_np.ndim == 1:
            x_np = x_np.reshape(1, -1)
        weights_np = np.empty((x_np.shape[0], 0), dtype=np.float64)

        with torch.autocast(device_type=device.type, enabled=False):
            out = self._qnn.forward(x_np, weights_np)

        out = torch.as_tensor(out, dtype=torch.float32, device=device)
        if self.output_transform is not None and self.fc_out is not None:
            out = self.fc_out(out)
        return out

    def save_circuit(self, path):
        """Serialize the bound quantum circuit to a QPY file.

        Args:
            path: Filesystem path where the QPY file is written.

        Returns:
            None.
        """
        with open(path, "wb") as f:
            qpy.dump(self._qc_bound, f)


def _iter_quantum_children(base, prefix):
    """Yield quantum child modules of ``base`` that should be frozen.

    Args:
        base: Parent ``nn.Module`` whose children are inspected.
        prefix: Optional name prefix used to select children (for example
            ``'q_net.'`` matches ``q_net1``, ``q_net2``, ``q_net3``). When
            falsy, any child exposing a ``qnn`` attribute is yielded.

    Yields:
        Tuples ``(name, module)`` where ``name`` is the child attribute name
        and ``module`` is a ``Q_Net``-like module with a ``.qnn`` attribute.
    """
    if not prefix:
        for name, module in base.named_children():
            if hasattr(module, "qnn") and not isinstance(module, FrozenQNet):
                yield name, module
        return

    child_prefix = prefix[:-1] if prefix.endswith(".") else prefix
    for name, module in base.named_children():
        if not name.startswith(child_prefix):
            continue
        if isinstance(module, FrozenQNet) or not hasattr(module, "qnn"):
            continue
        yield name, module


def freeze_for_inference(model, info=None, transpile=True, inplace=False):
    """Freeze quantum layers in a model for inference.

    Args:
        model: Model returned by ``create_model`` (typically an
            ``nn.Sequential`` ending with ``Sigmoid``) or the underlying base
            module.
        info: Optional metadata dictionary from ``create_model``; the
            ``quantum_prefix`` entry selects the quantum child modules.
        transpile: If True, perform a one-time transpilation of the frozen
            bound circuits.
        inplace: If True, modify ``model`` in place; otherwise operate on a
            deep copy.

    Returns:
        The model with quantum layers replaced by ``FrozenQNet`` instances.
        When at least one layer was frozen, the returned model is put in
        evaluation mode.
    """
    if not inplace:
        model = copy.deepcopy(model)
    quantum_prefix = info.get("quantum_prefix") if isinstance(info, dict) else None

    base = model[0] if isinstance(model, nn.Sequential) and len(model) > 0 else model

    frozen = 0
    for name, module in list(_iter_quantum_children(base, quantum_prefix)):
        setattr(base, name, FrozenQNet.from_trained(module, transpile=transpile))
        frozen += 1

    if frozen > 0:
        model.eval()
    return model
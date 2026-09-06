# models/efficientnet_model.py
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from models.quantum_core import create_qnn, Q_Net

model_config = {"reps": 4}  # можно вынести в конфиг

# ---------- Классический экстрактор признаков ----------
class EfficientNetFeatureExtractor(nn.Module):
    """EfficientNet-B0 без последнего классификатора, возвращает 1280-мерный вектор."""
    def __init__(self):
        super().__init__()
        # Загружаем предобученную модель
        self.base = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        # Удаляем классификатор (последние слои)
        self.base.classifier = nn.Identity()  # просто пропускаем

    def forward(self, x):
        # EfficientNet ожидает 3 канала. Если у нас 1 канал (например, ч/б), дублируем
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        features = self.base(x)  # форма [batch, 1280]
        return features


# ---------- Классическая модель (только классическая часть) ----------
class EfficientNetClassical(nn.Module):
    """Классическая модель: экстрактор признаков + проекция на num_classes."""
    def __init__(self, num_classes, config):
        super().__init__()
        self.feature_extractor = EfficientNetFeatureExtractor()
        self.projection = nn.Linear(1280, num_classes)
        self.feature_BN = nn.BatchNorm1d(1280)

    def forward(self, x):
        features = self.feature_extractor(x)   # [batch, 1280]
        #bn_features = self.feature_BN(features)
        output = self.projection(features)     # [batch, num_classes]
        return output


# ---------- Гибридная модель (EfficientNet + квантовые слои) ----------
class HybridQNet(nn.Module):
    """
    Гибридная модель:
    - классический экстрактор EfficientNet
    - проекция на 4 признака (для 4 кубитов)
    - три независимых квантовых слоя
    - объединение и финальный линейный слой
    """
    def __init__(self, num_classes, config):
        super().__init__()
        self.feature_extractor = EfficientNetFeatureExtractor()
        self.feature_BN = nn.BatchNorm1d(1280)
        self.projection = nn.Linear(1280, num_classes)          # num_qubits = 4
        self.BN_in = nn.BatchNorm1d(num_classes)

        # Создаём три независимых квантовых слоя
        self.q_net1 = self._create_qnet(num_classes, config)
        self.q_net2 = self._create_qnet(num_classes, config)
        self.q_net3 = self._create_qnet(num_classes, config)

        self.BN_out = nn.BatchNorm1d(2**num_classes * 3)
        self.final = nn.Linear(2**num_classes * 3, num_classes)

    def _create_qnet(self, num_classes, config):
        """Создаёт новый квантовый слой с независимыми параметрами."""
        num_qubits = num_classes
        # Собственная feature_map (кодирование входных данных)
        feature_map = QuantumCircuit(num_qubits)
        pv = ParameterVector(name=f"init_{id(feature_map)}", length=num_qubits)
        for i in range(num_qubits):
            feature_map.ry(pv[i], i)

        # Собственный ansatz (вариационная часть)
        ansatz = RealAmplitudes(num_qubits, reps=model_config['reps'], entanglement="circular")

        qnn = create_qnn(
            feature_map, ansatz, config.shots,
            backend_type=config.backend_type,
            octillion_token=config.octillion_token,
            octillion_chip=config.octillion_chip,
            transpile_before=config.transpile_before
        )
        return Q_Net(qnn, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)          # [batch, 1280]
        bn_features = self.feature_BN(features)
        p_features = self.projection(bn_features)        # [batch, 4]
        projected_features = self.BN_in(p_features)

        qout1 = self.q_net1(projected_features)
        qout2 = self.q_net2(projected_features)
        qout3 = self.q_net3(projected_features)

        output = torch.cat((qout1, qout2, qout3), dim=1)  # [batch, num_classes*3]
        BN_output = self.BN_out(output)
        return self.final(BN_output)


# ---------- Фабричная функция ----------
def create_efficientnet_model(mode, num_classes, input_shape, config):
    """
    Создаёт модель на основе EfficientNet-B0.
    mode: 'classical' или 'quantum' (гибрид)
    """
    if mode == 'classical':
        model = EfficientNetClassical(num_classes, config)
        model = nn.Sequential(model, nn.Sigmoid())
        info = {
            'type': 'classical',
            'model_name': 'efficientnet',
            'classical_params': list(model[0].parameters()),
            'quantum_params': [],
            'classical_prefix': '0.',
            'quantum_prefix': None,
            'base_model': model[0],
        }
        return model, info

    elif mode == 'quantum':
        hybrid_model = HybridQNet(num_classes, config)
        model = nn.Sequential(hybrid_model, nn.Sigmoid())
        # Извлекаем параметры классической и квантовой частей
        classical_params = list(hybrid_model.feature_extractor.parameters()) + \
                           list(hybrid_model.projection.parameters())
        quantum_params = list(hybrid_model.q_net1.parameters()) + \
                         list(hybrid_model.q_net2.parameters()) + \
                         list(hybrid_model.q_net3.parameters())
        info = {
            'type': 'hybrid',
            'model_name': 'efficientnet',
            'classical_params': classical_params,
            'quantum_params': quantum_params,
            'classical_prefix': ['feature_extractor.', 'projection.'],
            'quantum_prefix': 'q_net.',
            'base_model': hybrid_model
        }
        return model, info

    else:
        raise ValueError(f"Unknown mode: {mode}")
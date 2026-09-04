import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from models.quantum_core import create_qnn, Q_Net

model_config = {"reps": 4}

# --- Вспомогательные классы для ResNet18 ---

class ResNet18FeatureExtractor(nn.Module):
    """ResNet18 без последнего полносвязного слоя, возвращает 512-мерный вектор."""
    def __init__(self):
        super().__init__()
        # Загружаем предобученную модель
        self.base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Удаляем последний полносвязный слой (classifier)
        self.base = nn.Sequential(*list(self.base.children())[:-1])

    def forward(self, x):
        # ResNet18 ожидает 3 канала. Если у нас 1 канал (например, MNIST), дублируем его.
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        # Получаем признаки размерности [batch, 512, 1, 1]
        features = self.base(x)
        # Преобразуем в [batch, 512]
        return features.view(features.size(0), -1)
    

class ResNet(nn.Module):
    """
    классический экстрактор признаков (ResNet18)
    """
    def __init__(self, num_classes, config):
        super().__init__()
        self.feature_extractor = ResNet18FeatureExtractor()
        
        self.projection = nn.Linear(512, num_classes)

    def forward(self, x):
        # 1. Извлечение признаков классической частью
        features = self.feature_extractor(x)  # [batch, 512]
        
        # 2. Проекция признаков в размерность, совместимую с квантовым кодированием
        #    (В идеале здесь можно использовать PCA или другие методы уменьшения размерности)
        output = self.projection(features)  # [batch, num_qubits]

        return output


class HybridQNet(nn.Module):
    def __init__(self, num_classes, config):
        super().__init__()
        self.feature_extractor = ResNet18FeatureExtractor()
        self.projection = nn.Linear(512, 4)          # num_qubits = 4
        self.BN_in = nn.BatchNorm1d(4)

        # Создаём три независимых квантовых слоя
        self.q_net1 = self._create_qnet(num_classes, config)
        self.q_net2 = self._create_qnet(num_classes, config)
        self.q_net3 = self._create_qnet(num_classes, config)

        self.BN_out = nn.BatchNorm1d((2 ** num_classes) * 3)
        self.final = nn.Linear((2 ** num_classes) * 3, num_classes)

    def _create_qnet(self, num_classes, config):
        """Создаёт новый квантовый слой с независимыми параметрами."""
        num_qubits = 4
        # Собственная feature_map
        feature_map = QuantumCircuit(num_qubits)
        pv = ParameterVector(name=f"init_{id(feature_map)}", length=num_qubits)
        for i in range(num_qubits):
            feature_map.ry(pv[i], i)

        # Собственный ansatz
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
        features = self.feature_extractor(x)
        p_features = self.projection(features)
        projected_features = self.BN_in(p_features)

        qout1 = self.q_net1(projected_features)
        qout2 = self.q_net2(projected_features)
        qout3 = self.q_net3(projected_features)

        output = torch.cat((qout1, qout2, qout3), dim=1)   # учтём, что cat по умолчанию dim=0
        BN_output = self.BN_out(output)
        return self.final(BN_output)


# --- Фабричная функция ---

def create_resnet18_model(mode, num_classes, input_shape, config):
    if mode == 'classical':
        model = ResNet(num_classes, config)
        model = nn.Sequential(model, nn.Sigmoid())
        info = {
            'type': 'classical',
            'model_name': 'resnet18',
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
            'model_name': 'resnet18',
            'classical_params': classical_params,
            'quantum_params': quantum_params,
            'classical_prefix': ['feature_extractor.', 'projection.'],
            'quantum_prefix': 'q_net.',
            'base_model': hybrid_model
        }
        return model, info

    else:
        raise ValueError(f"Unknown mode: {mode}")
"""PyTorch Multilayer Perceptron with weight initialization and hook extraction."""

import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Dict, List

class VisionMLP(nn.Module):
    def __init__(self, input_dim: int = 784, hidden1: int = 64, hidden2: int = 32, output_dim: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2, output_dim)
        
        self.activations: Dict[str, torch.Tensor] = {}
        self._init_weights()
        self._register_hooks()

    def _init_weights(self):
        """Xavier initialization to ensure dynamic response to drawing inputs."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0.01)

    def _register_hooks(self):
        def get_hook(layer_name: str):
            def hook(module, input_tensor, output_tensor):
                self.activations[layer_name] = output_tensor.detach()
            return hook

        self.fc1.register_forward_hook(get_hook("layer1"))
        self.fc2.register_forward_hook(get_hook("layer2"))
        self.fc3.register_forward_hook(get_hook("layer3"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_layer_weights(self) -> List[torch.Tensor]:
        return [
            self.fc1.weight.detach(),
            self.fc2.weight.detach(),
            self.fc3.weight.detach()
        ]
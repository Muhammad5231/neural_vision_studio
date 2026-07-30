"""PyTorch Multilayer Perceptron with Weight Extraction for Inspector."""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from typing import Dict, List, Tuple

class VisionMLP(nn.Module):
    def __init__(self, input_dim: int = 784, hidden1: int = 64, hidden2: int = 32, output_dim: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2, output_dim)
        
        self.activations: Dict[str, torch.Tensor] = {}
        self._register_hooks()
        self.reload_weights()

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

    def reload_weights(self):
        """Reloads trained weights from disk if available."""
        weights_path = "mnist_weights.pt"
        if os.path.exists(weights_path):
            try:
                self.load_state_dict(torch.load(weights_path, weights_only=True))
            except Exception:
                pass

    def get_neuron_details(self, layer_idx: int, neuron_idx: int) -> Tuple[float, list[float], list[float]]:
        """Returns (bias, incoming_weights, outgoing_weights) for the Inspector."""
        with torch.no_grad():
            if layer_idx == 0:
                bias = 0.0
                in_w = []
                out_w = self.fc1.weight[:, neuron_idx].tolist() if neuron_idx < 784 else []
            elif layer_idx == 1:
                bias = float(self.fc1.bias[neuron_idx]) if neuron_idx < len(self.fc1.bias) else 0.0
                in_w = self.fc1.weight[neuron_idx, :5].tolist()
                out_w = self.fc2.weight[:, neuron_idx].tolist() if neuron_idx < self.fc2.weight.shape[1] else []
            elif layer_idx == 2:
                bias = float(self.fc2.bias[neuron_idx]) if neuron_idx < len(self.fc2.bias) else 0.0
                in_w = self.fc2.weight[neuron_idx, :5].tolist()
                out_w = self.fc3.weight[:, neuron_idx].tolist() if neuron_idx < self.fc3.weight.shape[1] else []
            else:
                bias = float(self.fc3.bias[neuron_idx]) if neuron_idx < len(self.fc3.bias) else 0.0
                in_w = self.fc3.weight[neuron_idx, :5].tolist()
                out_w = []

            return bias, in_w, out_w
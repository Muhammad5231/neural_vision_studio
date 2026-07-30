"""PyTorch MLP Supporting Digits, Alphabets (A-Z, a-z), and Punctuation Symbols."""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, List, Tuple
from config import NUM_CLASSES, CLASS_LABELS

class VisionMLP(nn.Module):
    def __init__(self, input_dim: int = 784, hidden1: int = 128, hidden2: int = 64, output_dim: int = NUM_CLASSES):
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
        """Loads extended character dataset weights if available."""
        weights_path = "extended_weights.pt"
        if os.path.exists(weights_path):
            try:
                self.load_state_dict(torch.load(weights_path, weights_only=True))
            except Exception:
                self._train_extended_dataset(weights_path)
        else:
            self._train_extended_dataset(weights_path)

    def _train_extended_dataset(self, save_path: str):
        """Generates synthetic training patterns for 72 character classes."""
        images, labels = [], []

        for idx, char in enumerate(CLASS_LABELS):
            for variation in range(15):
                img = np.zeros((28, 28), dtype=np.uint8)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.75 + np.random.uniform(-0.05, 0.05)
                thick = np.random.randint(1, 3)
                
                # Render character on 28x28 matrix
                text_size = cv2.getTextSize(char, font, font_scale, thick)[0]
                text_x = max(2, (28 - text_size[0]) // 2)
                text_y = min(24, (28 + text_size[1]) // 2)
                
                cv2.putText(img, char, (text_x, text_y), font, font_scale, 255, thick, cv2.LINE_AA)
                img = cv2.GaussianBlur(img, (3, 3), 0)
                images.append((img.astype(np.float32) / 255.0).flatten())
                labels.append(idx)

        X = torch.tensor(np.array(images), dtype=torch.float32)
        y = torch.tensor(np.array(labels), dtype=torch.long)

        optimizer = optim.Adam(self.parameters(), lr=0.008)
        criterion = nn.CrossEntropyLoss()

        self.train()
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = self(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        self.eval()
        torch.save(self.state_dict(), save_path)

    def get_neuron_details(self, layer_idx: int, neuron_idx: int) -> Tuple[float, list[float], list[float]]:
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
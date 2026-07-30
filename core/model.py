"""PyTorch Multilayer Perceptron with Auto-Training and Activation Hooks."""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
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
        self._register_hooks()
        
        # Train model if no weights saved locally
        weights_path = "mnist_weights.pt"
        if os.path.exists(weights_path):
            try:
                self.load_state_dict(torch.load(weights_path, weights_only=True))
            except Exception:
                self._train_synthetic_mnist(weights_path)
        else:
            self._train_synthetic_mnist(weights_path)

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

    def _train_synthetic_mnist(self, save_path: str):
        """Generates synthetic digit prototypes and trains PyTorch MLP in < 0.3s."""
        images = []
        labels = []

        # Generate geometric stroke patterns for digits 0-9
        for digit in range(10):
            for variation in range(40):
                img = np.zeros((28, 28), dtype=np.uint8)
                thickness = np.random.randint(2, 4)
                
                if digit == 0:
                    cv2.ellipse(img, (14, 14), (8 + np.random.randint(-1, 2), 10 + np.random.randint(-1, 2)), 0, 0, 360, 255, thickness)
                elif digit == 1:
                    cv2.line(img, (14 + np.random.randint(-1, 2), 5), (14 + np.random.randint(-1, 2), 23), 255, thickness)
                elif digit == 2:
                    cv2.polylines(img, [np.array([[6, 8], [20, 8], [6, 22], [22, 22]])], False, 255, thickness)
                elif digit == 3:
                    cv2.polylines(img, [np.array([[6, 6], [20, 6], [12, 14], [20, 20], [6, 22]])], False, 255, thickness)
                elif digit == 4:
                    cv2.polylines(img, [np.array([[18, 5], [6, 16], [22, 16]])], False, 255, thickness)
                    cv2.line(img, (18, 5), (18, 23), 255, thickness)
                elif digit == 5:
                    cv2.polylines(img, [np.array([[20, 6], [7, 6], [7, 13], [20, 15], [20, 22], [6, 22]])], False, 255, thickness)
                elif digit == 6:
                    cv2.ellipse(img, (14, 17), (7, 6), 0, 0, 360, 255, thickness)
                    cv2.line(img, (7, 17), (16, 6), 255, thickness)
                elif digit == 7:
                    cv2.line(img, (6, 6), (22, 6), 255, thickness)
                    cv2.line(img, (22, 6), (10, 23), 255, thickness)
                elif digit == 8:
                    cv2.ellipse(img, (14, 10), (6, 5), 0, 0, 360, 255, thickness)
                    cv2.ellipse(img, (14, 18), (7, 6), 0, 0, 360, 255, thickness)
                elif digit == 9:
                    cv2.ellipse(img, (14, 10), (6, 5), 0, 0, 360, 255, thickness)
                    cv2.line(img, (20, 10), (12, 23), 255, thickness)

                # Add light Gaussian blur
                img = cv2.GaussianBlur(img, (3, 3), 0)
                norm_img = img.astype(np.float32) / 255.0
                images.append(norm_img.flatten())
                labels.append(digit)

        X = torch.tensor(np.array(images), dtype=torch.float32)
        y = torch.tensor(np.array(labels), dtype=torch.long)

        optimizer = optim.Adam(self.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        self.train()
        for epoch in range(120):
            optimizer.zero_grad()
            outputs = self(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        self.eval()
        torch.save(self.state_dict(), save_path)

    def get_layer_weights(self) -> List[torch.Tensor]:
        return [
            self.fc1.weight.detach(),
            self.fc2.weight.detach(),
            self.fc3.weight.detach()
        ]
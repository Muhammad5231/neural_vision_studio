"""Main Layout Orchestrator connecting model, canvas, and neural scene."""

from PySide6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout
from PySide6.QtCore import QFile, QTextStream
import torch
import torch.nn.functional as F
import numpy as np

from config import APP_NAME, COLOR_BG_DARK
from core.model import VisionMLP
from ui.panels.left_canvas import LeftPanel
from ui.panels.right_control import RightPanel
from ui.panels.status_bar import StatusBar
from visualization.network_view import NetworkView

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1440, 900)
        
        self.model = VisionMLP()
        self.model.eval()

        self._init_ui()
        self._load_stylesheet()

    def _init_ui(self):
        central_widget = QWidget()
        central_widget.setStyleSheet(f"background-color: {COLOR_BG_DARK.name()};")
        self.setCentralWidget(central_widget)

        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(12, 12, 12, 0)
        root_layout.setSpacing(12)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(12)

        self.left_panel = LeftPanel()
        self.right_panel = RightPanel()
        self.network_view = NetworkView()
        self.status_bar = StatusBar()

        self.left_panel.canvas.canvas_updated.connect(self._handle_canvas_update)

        content_layout.addWidget(self.left_panel)
        content_layout.addWidget(self.network_view, stretch=1)
        content_layout.addWidget(self.right_panel)

        root_layout.addLayout(content_layout)
        root_layout.addWidget(self.status_bar)

    def _handle_canvas_update(self, image_data: np.ndarray):
        tensor_input = torch.from_numpy(image_data).float().unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(tensor_input)
            probs = F.softmax(output, dim=1).squeeze().tolist()

        self.right_panel.update_probabilities(probs)

        # Extract real activations
        l1 = torch.relu(self.model.activations["layer1"]).squeeze()
        l2 = torch.relu(self.model.activations["layer2"]).squeeze()
        l3 = F.softmax(self.model.activations["layer3"], dim=1).squeeze()

        target_activations = {}
        flat_input = image_data.flatten()
        
        for idx in range(16):
            target_activations[(0, idx)] = float(flat_input[idx * 49])
        for idx in range(12):
            target_activations[(1, idx)] = float(l1[idx % len(l1)])
        for idx in range(8):
            target_activations[(2, idx)] = float(l2[idx % len(l2)])
        for idx in range(10):
            target_activations[(3, idx)] = float(l3[idx])

        # Trigger energy particle signal animation
        self.network_view.animate_signal_flow(target_activations)

    def _load_stylesheet(self):
        file = QFile("assets/styles/dark_theme.qss")
        if file.open(QFile.OpenModeFlag.ReadOnly | QFile.OpenModeFlag.Text):
            stream = QTextStream(file)
            self.setStyleSheet(stream.readAll())
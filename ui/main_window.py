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
        self.resize(1400, 850)
        
        # Load Model
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

        # Panels
        self.left_panel = LeftPanel()
        self.right_panel = RightPanel()
        self.network_view = NetworkView()
        self.status_bar = StatusBar()

        # Connect Signals
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

        # Update Right Panel Probabilities
        self.right_panel.update_probabilities(probs)

        # Propagate Intermediate Layer Activations to Neural View
        l1 = torch.relu(self.model.activations["layer1"]).squeeze()
        l2 = torch.relu(self.model.activations["layer2"]).squeeze()
        l3 = F.softmax(self.model.activations["layer3"], dim=1).squeeze()

        for idx in range(16):
            self.network_view.nodes[(0, idx)].activation = float(image_data.flatten()[idx * 49])
        for idx in range(12):
            self.network_view.nodes[(1, idx)].activation = float(l1[idx % len(l1)])
        for idx in range(8):
            self.network_view.nodes[(2, idx)].activation = float(l2[idx % len(l2)])
        for idx in range(10):
            self.network_view.nodes[(3, idx)].activation = float(l3[idx])

    def _load_stylesheet(self):
        file = QFile("assets/styles/dark_theme.qss")
        if file.open(QFile.OpenModeFlag.ReadOnly | QFile.OpenModeFlag.Text):
            stream = QTextStream(file)
            self.setStyleSheet(stream.readAll())
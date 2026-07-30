"""Hardware-accelerated viewport handling zoom, pan, and graph display."""

from PySide6.QtWidgets import QGraphicsView, QGraphicsScene
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QWheelEvent
from visualization.items.neuron_item import NeuronItem
from visualization.items.synapse_item import SynapseItem

class NetworkView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        # Enable OpenGL acceleration
        self.setViewport(QOpenGLWidget())
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing |
            QPainter.RenderHint.SmoothPixmapTransform
        )
        
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setStyleSheet("background: transparent; border: none;")

        self.nodes = {}
        self.build_network_topology([16, 12, 8, 10])

    def build_network_topology(self, layers: list[int]):
        self.scene.clear()
        self.nodes.clear()
        
        x_spacing = 160
        y_spacing = 40

        # Construct Nodes
        for layer_idx, num_neurons in enumerate(layers):
            x = layer_idx * x_spacing
            y_offset = -((num_neurons - 1) * y_spacing) / 2.0

            for n_idx in range(num_neurons):
                y = y_offset + (n_idx * y_spacing)
                node = NeuronItem(layer_idx=layer_idx, neuron_idx=n_idx)
                node.setPos(x, y)
                self.scene.addItem(node)
                self.nodes[(layer_idx, n_idx)] = node

        # Construct Synapses between adjacent layers
        for layer_idx in range(len(layers) - 1):
            for i in range(layers[layer_idx]):
                p1 = self.nodes[(layer_idx, i)].pos()
                for j in range(layers[layer_idx + 1]):
                    p2 = self.nodes[(layer_idx + 1, j)].pos()
                    synapse = SynapseItem(p1.x(), p1.y(), p2.x(), p2.y())
                    self.scene.addItem(synapse)

    def wheelEvent(self, event: QWheelEvent):
        zoom_factor = 1.12 if event.angleDelta().y() > 0 else 1.0 / 1.12
        self.scale(zoom_factor, zoom_factor)
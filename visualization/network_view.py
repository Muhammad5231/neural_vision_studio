"""Hardware-accelerated Viewport with zoom/pan constraints and animated layer propagation."""

from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsTextItem
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import Qt, QSequentialAnimationGroup, QParallelAnimationGroup, QPropertyAnimation, QEasingCurve, QPointF
from PySide6.QtGui import QPainter, QWheelEvent, QFont, QColor

from visualization.items.neuron_item import NeuronItem
from visualization.items.synapse_item import SynapseItem
from visualization.items.particle_item import SignalParticle

class NetworkView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        # Hardware Acceleration & High-Quality Rendering
        self.setViewport(QOpenGLWidget())
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing |
            QPainter.RenderHint.SmoothPixmapTransform |
            QPainter.RenderHint.TextAntialiasing
        )
        
        # Zoom / Pan Anchors & Configuration
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setStyleSheet("background: transparent; border: none;")

        self.current_scale = 1.0
        self.nodes = {}
        self.synapses = []
        self.layers_config = [16, 12, 8, 10]
        
        self.build_network_topology(self.layers_config)

    def build_network_topology(self, layers: list[int]):
        self.scene.clear()
        self.nodes.clear()
        self.synapses.clear()
        
        x_spacing = 200
        y_spacing = 42
        layer_names = ["Input Layer", "Hidden Layer 1", "Hidden Layer 2", "Output Layer"]

        # 1. Build Nodes and Layer Labels
        for layer_idx, num_neurons in enumerate(layers):
            x = layer_idx * x_spacing
            y_offset = -((num_neurons - 1) * y_spacing) / 2.0

            # Add Layer Title Header
            lbl = QGraphicsTextItem(layer_names[layer_idx])
            lbl.setFont(QFont("Inter", 10, QFont.Weight.Bold))
            lbl.setDefaultTextColor(QColor("#06B6D4" if layer_idx < 2 else "#007AFF"))
            lbl.setPos(x - 45, y_offset - 45)
            self.scene.addItem(lbl)

            for n_idx in range(num_neurons):
                y = y_offset + (n_idx * y_spacing)
                node = NeuronItem(layer_idx=layer_idx, neuron_idx=n_idx)
                node.setPos(x, y)
                self.scene.addItem(node)
                self.nodes[(layer_idx, n_idx)] = node

        # 2. Build Synapse Connections
        for layer_idx in range(len(layers) - 1):
            for i in range(layers[layer_idx]):
                p1 = self.nodes[(layer_idx, i)].pos()
                for j in range(layers[layer_idx + 1]):
                    p2 = self.nodes[(layer_idx + 1, j)].pos()
                    synapse = SynapseItem(p1.x(), p1.y(), p2.x(), p2.y())
                    self.scene.addItem(synapse)
                    self.synapses.append((layer_idx, p1, p2))

        self.scene.setSceneRect(self.scene.itemsBoundingRect().adjusted(-50, -50, 50, 50))

    def animate_signal_flow(self, target_activations: dict[tuple[int, int], float]):
        """Sequentially animates energy particles down synapses and lights up neurons layer by layer."""
        main_seq = QSequentialAnimationGroup(self)

        for layer_idx in range(len(self.layers_config)):
            # Step A: Animate Neurons in current layer lighting up
            layer_neuron_group = QParallelAnimationGroup(self)
            for n_idx in range(self.layers_config[layer_idx]):
                node = self.nodes[(layer_idx, n_idx)]
                target_val = target_activations.get((layer_idx, n_idx), 0.0)
                
                anim = QPropertyAnimation(node, b"activation")
                anim.setDuration(220)
                anim.setStartValue(node.get_activation())
                anim.setEndValue(target_val)
                anim.setEasingCurve(QEasingCurve.Type.OutQuad)
                layer_neuron_group.addAnimation(anim)
            
            main_seq.addAnimation(layer_neuron_group)

            # Step B: Flow Particles down synapses to the next layer
            if layer_idx < len(self.layers_config) - 1:
                particle_group = QParallelAnimationGroup(self)
                
                # Sample connections to maintain 60 FPS
                layer_synapses = [s for s in self.synapses if s[0] == layer_idx]
                for _, p1, p2 in layer_synapses[::2]:
                    particle = SignalParticle(p1, p2)
                    self.scene.addItem(particle)
                    
                    p_anim = QPropertyAnimation(particle, b"progress")
                    p_anim.setDuration(180)
                    p_anim.setStartValue(0.0)
                    p_anim.setEndValue(1.0)
                    p_anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
                    
                    # Clean up particle from scene upon animation completion
                    p_anim.finished.connect(lambda p=particle: self.scene.removeItem(p))
                    particle_group.addAnimation(p_anim)

                main_seq.addAnimation(particle_group)

        main_seq.start()

    def wheelEvent(self, event: QWheelEvent):
        """Clamped smooth wheel zoom to prevent zoom dissection/disappearance."""
        zoom_in_factor = 1.12
        zoom_out_factor = 1.0 / zoom_in_factor

        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor

        new_scale = self.current_scale * zoom_factor
        
        # Clamp Zoom between 0.4x and 2.5x
        if 0.4 <= new_scale <= 2.5:
            self.current_scale = new_scale
            self.scale(zoom_factor, zoom_factor)
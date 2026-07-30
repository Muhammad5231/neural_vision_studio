"""Custom QGraphicsItem displaying glowing neuron, numeric activation, and math tooltips."""

from PySide6.QtWidgets import QGraphicsObject
from PySide6.QtCore import QRectF, Property, QPointF, Qt
from PySide6.QtGui import QPainter, QRadialGradient, QBrush, QPen, QColor, QFont
from config import COLOR_NEURON_OFF, COLOR_CYAN, COLOR_ELECTRIC_BLUE

class NeuronItem(QGraphicsObject):
    def __init__(self, layer_idx: int, neuron_idx: int, radius: float = 16.0):
        super().__init__()
        self.layer_idx = layer_idx
        self.neuron_idx = neuron_idx
        self.radius = radius
        self._activation: float = 0.0
        self.setAcceptHoverEvents(True)
        self._update_tooltip()

    def boundingRect(self) -> QRectF:
        margin = self.radius * 3.0
        return QRectF(-margin, -margin, margin * 2, margin * 2)

    def get_activation(self) -> float:
        return self._activation

    def set_activation(self, val: float):
        self._activation = max(0.0, min(1.0, val))
        self._update_tooltip()
        self.update()

    activation = Property(float, get_activation, set_activation)

    def _update_tooltip(self):
        layer_names = ["Input", "Hidden 1", "Hidden 2", "Output"]
        l_name = layer_names[self.layer_idx] if self.layer_idx < len(layer_names) else f"Layer {self.layer_idx}"
        func = "ReLU" if self.layer_idx in [1, 2] else ("Softmax" if self.layer_idx == 3 else "Input Pass")
        
        self.setToolTip(
            f"<b>{l_name} - Neuron [{self.neuron_idx}]</b><br/>"
            f"Activation (a): <b>{self._activation:.4f}</b><br/>"
            f"Function: <i>{func}</i><br/>"
            f"Math: a = {func}(∑ w_i·x_i + b)"
        )

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        active_color = COLOR_CYAN if self.layer_idx < 2 else COLOR_ELECTRIC_BLUE
        glow_factor = 1.0 + (self._activation * 0.9)
        
        # 1. Radial Glow Effect
        gradient = QRadialGradient(QPointF(0, 0), self.radius * glow_factor)
        
        inner_color = QColor(
            int(COLOR_NEURON_OFF.red() + (active_color.red() - COLOR_NEURON_OFF.red()) * self._activation),
            int(COLOR_NEURON_OFF.green() + (active_color.green() - COLOR_NEURON_OFF.green()) * self._activation),
            int(COLOR_NEURON_OFF.blue() + (active_color.blue() - COLOR_NEURON_OFF.blue()) * self._activation),
            255
        )
        outer_glow = QColor(active_color.red(), active_color.green(), active_color.blue(), int(180 * self._activation))
        
        gradient.setColorAt(0.0, inner_color)
        gradient.setColorAt(0.6, outer_glow)
        gradient.setColorAt(1.0, QColor(0, 0, 0, 0))

        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(0, 0), self.radius * 2.0, self.radius * 2.0)

        # 2. Outer Ring
        pen = QPen(active_color if self._activation > 0.05 else COLOR_NEURON_OFF, 2.0)
        painter.setPen(pen)
        painter.drawEllipse(QPointF(0, 0), self.radius, self.radius)

        # 3. Numeric Activation Text Display (Math Value)
        painter.setPen(QPen(QColor("#FFFFFF") if self._activation > 0.3 else QColor("#94A3B8")))
        font = QFont("JetBrains Mono", 8, QFont.Weight.Bold)
        painter.setFont(font)
        
        val_str = f"{self._activation:.2f}"
        painter.drawText(QRectF(-self.radius, -6, self.radius * 2, 12), Qt.AlignmentFlag.AlignCenter, val_str)
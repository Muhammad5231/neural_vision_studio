"""Custom QGraphicsItem representing a glowing visual neuron node."""

from PySide6.QtWidgets import QGraphicsObject
from PySide6.QtCore import QRectF, Property, QPointF, Qt
from PySide6.QtGui import QPainter, QRadialGradient, QBrush, QPen, QColor
from config import COLOR_NEURON_OFF, COLOR_CYAN, COLOR_ELECTRIC_BLUE

class NeuronItem(QGraphicsObject):
    def __init__(self, layer_idx: int, neuron_idx: int, radius: float = 12.0):
        super().__init__()
        self.layer_idx = layer_idx
        self.neuron_idx = neuron_idx
        self.radius = radius
        self._activation: float = 0.0

    def boundingRect(self) -> QRectF:
        margin = self.radius * 2.5
        return QRectF(-margin, -margin, margin * 2, margin * 2)

    def get_activation(self) -> float:
        return self._activation

    def set_activation(self, val: float):
        self._activation = max(0.0, min(1.0, val))
        self.update()

    # PySide6 safe property definition
    activation = Property(float, get_activation, set_activation)

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        active_color = COLOR_CYAN if self.layer_idx < 2 else COLOR_ELECTRIC_BLUE
        gradient = QRadialGradient(QPointF(0, 0), self.radius * (1.0 + self._activation * 0.8))
        
        inner_color = QColor(
            int(COLOR_NEURON_OFF.red() + (active_color.red() - COLOR_NEURON_OFF.red()) * self._activation),
            int(COLOR_NEURON_OFF.green() + (active_color.green() - COLOR_NEURON_OFF.green()) * self._activation),
            int(COLOR_NEURON_OFF.blue() + (active_color.blue() - COLOR_NEURON_OFF.blue()) * self._activation),
            255
        )
        outer_glow = QColor(active_color.red(), active_color.green(), active_color.blue(), int(180 * self._activation))
        
        gradient.setColorAt(0.0, inner_color)
        gradient.setColorAt(0.5, outer_glow)
        gradient.setColorAt(1.0, QColor(0, 0, 0, 0))

        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(0, 0), self.radius * 1.8, self.radius * 1.8)

        pen = QPen(active_color if self._activation > 0.1 else COLOR_NEURON_OFF, 2.0)
        painter.setPen(pen)
        painter.drawEllipse(QPointF(0, 0), self.radius, self.radius)
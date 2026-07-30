"""Custom QGraphicsItem displaying glowing neuron and click inspector triggers."""

from PySide6.QtWidgets import QGraphicsObject, QGraphicsSceneMouseEvent
from PySide6.QtCore import QRectF, Property, QPointF, Qt, Signal
from PySide6.QtGui import QPainter, QRadialGradient, QBrush, QPen, QColor, QFont
from config import COLOR_NEURON_OFF, COLOR_CYAN, COLOR_ELECTRIC_BLUE

class NeuronItem(QGraphicsObject):
    # Click signal emitted to open Neuron Inspector
    clicked = Signal(int, int, float)  # layer_idx, neuron_idx, activation

    def __init__(self, layer_idx: int, neuron_idx: int, radius: float = 16.0):
        super().__init__()
        self.layer_idx = layer_idx
        self.neuron_idx = neuron_idx
        self.radius = radius
        self._activation: float = 0.0
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def boundingRect(self) -> QRectF:
        margin = self.radius * 3.0
        return QRectF(-margin, -margin, margin * 2, margin * 2)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.layer_idx, self.neuron_idx, self._activation)
        super().mousePressEvent(event)

    def get_activation(self) -> float:
        return self._activation

    def set_activation(self, val: float):
        self._activation = max(0.0, min(1.0, val))
        self.update()

    activation = Property(float, get_activation, set_activation)

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        active_color = COLOR_CYAN if self.layer_idx < 2 else COLOR_ELECTRIC_BLUE
        glow_factor = 1.0 + (self._activation * 0.9)
        
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

        pen = QPen(active_color if self._activation > 0.05 else COLOR_NEURON_OFF, 2.0)
        painter.setPen(pen)
        painter.drawEllipse(QPointF(0, 0), self.radius, self.radius)

        painter.setPen(QPen(QColor("#FFFFFF") if self._activation > 0.3 else QColor("#94A3B8")))
        font = QFont("JetBrains Mono", 8, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(QRectF(-self.radius, -6, self.radius * 2, 12), Qt.AlignmentFlag.AlignCenter, f"{self._activation:.2f}")
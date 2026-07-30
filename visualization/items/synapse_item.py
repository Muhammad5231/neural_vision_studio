"""Animated synapse connection line between neurons."""

from PySide6.QtWidgets import QGraphicsLineItem
from PySide6.QtGui import QPen, QColor
from PySide6.QtCore import Qt

class SynapseItem(QGraphicsLineItem):
    def __init__(self, x1, y1, x2, y2, parent=None):
        super().__init__(x1, y1, x2, y2, parent)
        self.weight = 0.5
        self.setZValue(-1)  # Behind neurons
        self.update_pen()

    def update_pen(self):
        color = QColor(100, 150, 255, int(40 + self.weight * 120))
        pen = QPen(color, 1.0, Qt.PenStyle.SolidLine)
        self.setPen(pen)
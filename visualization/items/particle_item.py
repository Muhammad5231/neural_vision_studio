"""Signal energy particles moving down connections."""

from PySide6.QtWidgets import QGraphicsEllipseItem
from PySide6.QtGui import QBrush, QPen, QColor
from PySide6.QtCore import Qt
from config import COLOR_CYAN

class ParticleItem(QGraphicsEllipseItem):
    def __init__(self, parent=None):
        super().__init__(-3, -3, 6, 6, parent)
        self.setBrush(QBrush(COLOR_CYAN))
        self.setPen(QPen(Qt.PenStyle.NoPen))
        self.setZValue(1)
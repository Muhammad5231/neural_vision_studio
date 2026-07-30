"""Animated signal particles traveling along synaptic pathways."""

from PySide6.QtWidgets import QGraphicsEllipseItem
from PySide6.QtGui import QBrush, QPen
from PySide6.QtCore import Qt, QPointF, Property, QObject
from config import COLOR_CYAN

class ParticleItem(QObject, QGraphicsEllipseItem):
    def __init__(self, start_pos: QPointF, end_pos: QPointF, parent=None):
        QObject.__init__(self)
        QGraphicsEllipseItem.__init__(self, -4, -4, 8, 8, parent)
        
        self.start_pos = start_pos
        self.end_pos = end_pos
        self._progress = 0.0
        
        self.setBrush(QBrush(COLOR_CYAN))
        self.setPen(QPen(Qt.PenStyle.NoPen))
        self.setZValue(2)
        self.setPos(start_pos)

    def get_progress(self) -> float:
        return self._progress

    def set_progress(self, val: float):
        self._progress = val
        x = (1.0 - val) * self.start_pos.x() + val * self.end_pos.x()
        y = (1.0 - val) * self.start_pos.y() + val * self.end_pos.y()
        self.setPos(QPointF(x, y))

    progress = Property(float, get_progress, set_progress)

SignalParticle = ParticleItem
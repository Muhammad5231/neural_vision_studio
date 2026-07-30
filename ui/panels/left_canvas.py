"""Interactive PySide6 Drawing Canvas for digit recognition."""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel
from PySide6.QtGui import QPainter, QPen, QImage, QColor
from PySide6.QtCore import Qt, QPoint, Signal
import numpy as np
import cv2
from config import CANVAS_SIZE, MNIST_GRID_SIZE
from ui.components.glass_card import GlassCard

class DrawingCanvas(QWidget):
    canvas_updated = Signal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(CANVAS_SIZE, CANVAS_SIZE)
        # Fixed Enum: QImage.Format.Format_RGB32
        self.image = QImage(CANVAS_SIZE, CANVAS_SIZE, QImage.Format.Format_RGB32)
        self.image.fill(QColor(0, 0, 0))
        self.drawing = False
        self.last_point = QPoint()
        self.brush_size = 18

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            self.last_point = event.position().toPoint()

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.MouseButton.LeftButton) and self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(QColor(255, 255, 255), self.brush_size, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            painter.drawLine(self.last_point, event.position().toPoint())
            self.last_point = event.position().toPoint()
            self.update()
            self._emit_processed_image()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False

    def paintEvent(self, event):
        canvas_painter = QPainter(self)
        canvas_painter.drawImage(self.rect(), self.image, self.image.rect())

    def clear(self):
        self.image.fill(QColor(0, 0, 0))
        self.update()
        self._emit_processed_image()

    def _emit_processed_image(self):
        ptr = self.image.bits()
        arr = np.array(ptr).reshape(CANVAS_SIZE, CANVAS_SIZE, 4)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)
        resized = cv2.resize(gray, (MNIST_GRID_SIZE, MNIST_GRID_SIZE), interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0
        self.canvas_updated.emit(normalized)


class LeftPanel(GlassCard):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(320)
        
        lbl_title = QLabel("Input Canvas")
        lbl_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #F8FAFC;")
        
        self.canvas = DrawingCanvas()
        
        self.clear_btn = QPushButton("Clear Canvas")
        self.clear_btn.setObjectName("clearBtn")
        self.clear_btn.clicked.connect(self.canvas.clear)
        
        self.layout.addWidget(lbl_title)
        self.layout.addWidget(self.canvas, alignment=Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.clear_btn)
        self.layout.addStretch()
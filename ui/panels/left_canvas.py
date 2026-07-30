"""Interactive PySide6 Drawing Canvas with Stroke-End Debounced Prediction."""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PySide6.QtGui import QPainter, QPen, QImage, QColor
from PySide6.QtCore import Qt, QPoint, Signal, QTimer
import numpy as np
import cv2
from config import CANVAS_SIZE, MNIST_GRID_SIZE
from ui.components.glass_card import GlassCard

class DrawingCanvas(QWidget):
    canvas_updated = Signal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(CANVAS_SIZE, CANVAS_SIZE)
        self.image = QImage(CANVAS_SIZE, CANVAS_SIZE, QImage.Format.Format_RGBA8888)
        self.image.fill(QColor(0, 0, 0))
        self.drawing = False
        self.last_point = QPoint()
        self.brush_size = 20

        # Debounce timer: wait 350ms after last mouse movement before predicting
        self.debounce_timer = QTimer(self)
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.setInterval(350)
        self.debounce_timer.timeout.connect(self._emit_processed_image)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            self.last_point = event.position().toPoint()
            self.debounce_timer.stop()

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.MouseButton.LeftButton) and self.drawing:
            painter = QPainter(self.image)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(QPen(
                QColor(255, 255, 255),
                self.brush_size,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin
            ))
            painter.drawLine(self.last_point, event.position().toPoint())
            painter.end()
            self.last_point = event.position().toPoint()
            self.update()
            # Start debounce timer instead of predicting mid-stroke
            self.debounce_timer.start()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False
            # Immediately trigger prediction on pen release
            self.debounce_timer.start(50)

    def paintEvent(self, event):
        canvas_painter = QPainter(self)
        canvas_painter.drawImage(self.rect(), self.image, self.image.rect())

    def clear(self):
        self.debounce_timer.stop()
        self.image.fill(QColor(0, 0, 0))
        self.update()
        self._emit_processed_image()

    def trigger_predict_now(self):
        self.debounce_timer.stop()
        self._emit_processed_image()

    def _emit_processed_image(self):
        width = self.image.width()
        height = self.image.height()
        ptr = self.image.bits()
        
        arr = np.array(ptr).reshape((height, width, 4)).copy()
        gray = cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)
        resized = cv2.resize(gray, (MNIST_GRID_SIZE, MNIST_GRID_SIZE), interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0
        self.canvas_updated.emit(normalized)


class LeftPanel(GlassCard):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(320)
        
        lbl_title = QLabel("Input Drawing Canvas")
        lbl_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #F8FAFC;")
        
        self.canvas = DrawingCanvas()
        
        btn_layout = QHBoxLayout()
        self.predict_btn = QPushButton("Predict")
        self.predict_btn.clicked.connect(self.canvas.trigger_predict_now)
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setObjectName("clearBtn")
        self.clear_btn.clicked.connect(self.canvas.clear)
        
        btn_layout.addWidget(self.predict_btn)
        btn_layout.addWidget(self.clear_btn)
        
        self.layout.addWidget(lbl_title)
        self.layout.addWidget(self.canvas, alignment=Qt.AlignmentFlag.AlignCenter)
        self.layout.addLayout(btn_layout)
        self.layout.addStretch()
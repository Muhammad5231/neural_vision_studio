"""Interactive Canvas with Center-of-Mass MNIST Formatting."""

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
        self.brush_size = 24

        self.debounce_timer = QTimer(self)
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.setInterval(250)
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
            self.debounce_timer.start()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False
            self.debounce_timer.start(30)

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

    def _preprocess_mnist(self, gray_img: np.ndarray) -> np.ndarray:
        """Standard MNIST Center-of-Mass normalization pipeline."""
        if np.max(gray_img) < 15:
            return np.zeros((MNIST_GRID_SIZE, MNIST_GRID_SIZE), dtype=np.float32)

        # 1. Bounding box cropping
        coords = cv2.findNonZero(gray_img)
        x, y, w, h = cv2.boundingRect(coords)
        cropped = gray_img[y:y+h, x:x+w]

        # 2. Aspect ratio scaling into 20x20 box
        max_dim = max(w, h)
        scale = 20.0 / max_dim
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 3. Canvas placement
        padded = np.zeros((28, 28), dtype=np.uint8)
        start_x = (28 - new_w) // 2
        start_y = (28 - new_h) // 2
        padded[start_y:start_y+new_h, start_x:start_x+new_w] = resized

        # 4. Center of mass alignment
        M = cv2.moments(padded)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            shift_x = int(np.round(14 - cx))
            shift_y = int(np.round(14 - cy))
            M_translation = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            padded = cv2.warpAffine(padded, M_translation, (28, 28))

        return padded.astype(np.float32) / 255.0

    def _emit_processed_image(self):
        width = self.image.width()
        height = self.image.height()
        ptr = self.image.bits()
        
        arr = np.array(ptr).reshape((height, width, 4)).copy()
        gray = cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)
        processed = self._preprocess_mnist(gray)
        self.canvas_updated.emit(processed)


class LeftPanel(GlassCard):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(320)
        
        lbl_title = QLabel("Input Drawing Canvas")
        lbl_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #F8FAFC;")
        
        self.canvas = DrawingCanvas()
        
        btn_layout = QHBoxLayout()
        self.predict_btn = QPushButton("Predict Digit")
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
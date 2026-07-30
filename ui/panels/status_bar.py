"""Bottom status bar for hardware and epoch metrics."""

from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel
from config import COLOR_TEXT_SECONDARY

class StatusBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(32)
        self.setStyleSheet("background-color: #0B0E14; border-top: 1px solid rgba(255, 255, 255, 0.05);")
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)
        
        self.lbl_fps = QLabel("FPS: 60")
        self.lbl_status = QLabel("System Ready")
        self.lbl_epoch = QLabel("Epoch: 0/10 | Loss: -- | Acc: --%")
        
        for lbl in [self.lbl_fps, self.lbl_status, self.lbl_epoch]:
            lbl.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY.name()}; font-size: 11px;")
            
        layout.addWidget(self.lbl_fps)
        layout.addWidget(self.lbl_status)
        layout.addStretch()
        layout.addWidget(self.lbl_epoch)
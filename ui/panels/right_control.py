"""Right analytics panel displaying predictions and confidence metrics."""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar
from PySide6.QtCore import Qt
from ui.components.glass_card import GlassCard
from config import COLOR_TEXT_PRIMARY, COLOR_TEXT_SECONDARY

class RightPanel(GlassCard):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(320)
        
        lbl_title = QLabel("Prediction Analysis")
        lbl_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #F8FAFC;")
        
        self.lbl_prediction = QLabel("-")
        self.lbl_prediction.setStyleSheet("font-size: 64px; font-weight: bold; color: #007AFF;")
        self.lbl_prediction.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.bars = []
        bars_layout = QVBoxLayout()
        for i in range(10):
            row_layout = QVBoxLayout()
            lbl = QLabel(f"Digit {i}")
            lbl.setStyleSheet("font-size: 11px; color: #94A3B8;")
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFixedHeight(8)
            row_layout.addWidget(lbl)
            row_layout.addWidget(bar)
            bars_layout.addLayout(row_layout)
            self.bars.append(bar)
            
        self.layout.addWidget(lbl_title)
        self.layout.addWidget(self.lbl_prediction)
        self.layout.addLayout(bars_layout)
        self.layout.addStretch()

    def update_probabilities(self, probs: list[float]):
        best_digit = max(range(len(probs)), key=lambda i: probs[i])
        self.lbl_prediction.setText(str(best_digit))
        
        for i, prob in enumerate(probs):
            self.bars[i].setValue(int(prob * 100))
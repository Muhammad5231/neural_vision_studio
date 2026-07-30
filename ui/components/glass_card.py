"""Reusable glassmorphism card container."""

from PySide6.QtWidgets import QFrame, QVBoxLayout
from config import COLOR_CARD_BG

class GlassCard(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            GlassCard {{
                background-color: rgba({COLOR_CARD_BG.red()}, {COLOR_CARD_BG.green()}, {COLOR_CARD_BG.blue()}, 0.8);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 12px;
            }}
        """)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(16, 16, 16, 16)
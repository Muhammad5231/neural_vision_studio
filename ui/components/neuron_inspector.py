"""Sleek Glassmorphic Inspector Popup for Detailed Neuron Parameters."""

from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QColor
from config import COLOR_BG_DARK, COLOR_CYAN, COLOR_ELECTRIC_BLUE

class NeuronInspectorDialog(QDialog):
    def __init__(self, layer_idx: int, neuron_idx: int, activation: float, bias: float, 
                 in_weights: list[float], out_weights: list[float], parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Neuron Inspector - Layer {layer_idx}, Node #{neuron_idx}")
        self.setFixedSize(420, 480)
        self.setStyleSheet(f"""
            QDialog {{
                background-color: #0B0E14;
                border: 2px solid #007AFF;
                border-radius: 14px;
            }}
            QLabel {{ color: #F8FAFC; font-family: 'Inter'; }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        layer_names = ["Input Layer", "Hidden Layer 1", "Hidden Layer 2", "Output Layer"]
        l_name = layer_names[layer_idx] if layer_idx < len(layer_names) else f"Layer {layer_idx}"
        func = "ReLU" if layer_idx in [1, 2] else ("Softmax" if layer_idx == 3 else "Pass-through")

        # Header Title
        lbl_header = QLabel(f"<b>{l_name}</b> — Neuron #{neuron_idx}")
        lbl_header.setFont(QFont("Inter", 14))
        lbl_header.setStyleSheet("color: #06B6D4;")

        # Metrics Card
        card = QFrame()
        card.setStyleSheet("background: rgba(18, 22, 31, 0.9); border-radius: 10px; padding: 10px;")
        card_layout = QVBoxLayout(card)

        lbl_act = QLabel(f"Activation Value ($a$): <b>{activation:.4f}</b>")
        lbl_act.setFont(QFont("JetBrains Mono", 11))
        
        lbl_bias = QLabel(f"Neuron Bias ($b$): <b>{bias:.4f}</b>")
        lbl_bias.setFont(QFont("JetBrains Mono", 10))
        
        lbl_func = QLabel(f"Activation Function: <i>{func}</i>")
        lbl_func.setStyleSheet("color: #94A3B8;")

        card_layout.addWidget(lbl_act)
        card_layout.addWidget(lbl_bias)
        card_layout.addWidget(lbl_func)

        # Top Incoming Weights List
        lbl_in_title = QLabel("Top Incoming Weights ($w_{in}$):")
        lbl_in_title.setFont(QFont("Inter", 10, QFont.Weight.Bold))
        in_str = ", ".join([f"{w:+.3f}" for w in in_weights[:5]]) if in_weights else "None (Input Layer)"
        lbl_in_val = QLabel(in_str)
        lbl_in_val.setStyleSheet("color: #06B6D4; font-family: 'JetBrains Mono'; font-size: 11px;")

        # Top Outgoing Weights List
        lbl_out_title = QLabel("Top Outgoing Weights ($w_{out}$):")
        lbl_out_title.setFont(QFont("Inter", 10, QFont.Weight.Bold))
        out_str = ", ".join([f"{w:+.3f}" for w in out_weights[:5]]) if out_weights else "None (Output Layer)"
        lbl_out_val = QLabel(out_str)
        lbl_out_val.setStyleSheet("color: #007AFF; font-family: 'JetBrains Mono'; font-size: 11px;")

        # Math Formula Box
        lbl_math = QLabel(f"<b>Math:</b> $a = \\text{{{func}}}(\\sum w_i \\cdot x_i + {bias:.2f})$")
        lbl_math.setStyleSheet("color: #94A3B8; font-size: 11px; background: #1E293B; padding: 8px; border-radius: 6px;")

        btn_close = QPushButton("Close Inspector")
        btn_close.setStyleSheet("background-color: #007AFF; color: white; border: none; padding: 10px; border-radius: 8px; font-weight: bold;")
        btn_close.clicked.connect(self.accept)

        layout.addWidget(lbl_header)
        layout.addWidget(card)
        layout.addSpacing(10)
        layout.addWidget(lbl_in_title)
        layout.addWidget(lbl_in_val)
        layout.addSpacing(10)
        layout.addWidget(lbl_out_title)
        layout.addWidget(lbl_out_val)
        layout.addSpacing(10)
        layout.addWidget(lbl_math)
        layout.addStretch()
        layout.addWidget(btn_close)
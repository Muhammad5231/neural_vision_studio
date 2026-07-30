"""Right Analytics Panel with Dynamic Character Class Ranking."""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar, QTabWidget, QHBoxLayout
from PySide6.QtCore import Qt
from ui.components.glass_card import GlassCard
from config import CLASS_LABELS

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MetricsChartCanvas(FigureCanvas):
    def __init__(self, parent=None, width=3, height=2, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#12161F')
        super().__init__(self.fig)
        self.setParent(parent)

        self.ax_loss = self.fig.add_subplot(211)
        self.ax_acc = self.fig.add_subplot(212)
        self._format_axes()

    def _format_axes(self):
        for ax in [self.ax_loss, self.ax_acc]:
            ax.set_facecolor('#0B0E14')
            ax.tick_params(colors='#94A3B8', labelsize=7)
            for spine in ax.spines.values():
                spine.set_color('#334155')

        self.ax_loss.set_title("Training Loss", color="#06B6D4", fontsize=9, pad=2)
        self.ax_acc.set_title("Accuracy (%)", color="#007AFF", fontsize=9, pad=2)
        self.fig.tight_layout(pad=1.2)


class RightPanel(GlassCard):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(340)
        
        lbl_title = QLabel("Prediction Analytics")
        lbl_title.setStyleSheet("font-size: 15px; font-weight: bold; color: #F8FAFC;")
        
        self.lbl_prediction = QLabel("-")
        self.lbl_prediction.setStyleSheet("font-size: 54px; font-weight: bold; color: #007AFF;")
        self.lbl_prediction.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #334155; border-radius: 8px; background: #0B0E14; }
            QTabBar::tab { background: #1E293B; color: #94A3B8; padding: 6px 12px; border-radius: 4px; }
            QTabBar::tab:selected { background: #007AFF; color: #FFFFFF; font-weight: bold; }
        """)

        # Dynamic Top 8 Prediction Rows
        prob_widget = QWidget()
        prob_layout = QVBoxLayout(prob_widget)
        self.prob_rows = []

        for i in range(8):
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 2, 0, 2)

            lbl_char = QLabel(f"#{i+1}")
            lbl_char.setFixedWidth(50)
            lbl_char.setStyleSheet("font-size: 11px; font-weight: bold; color: #06B6D4;")

            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFixedHeight(8)

            row_layout.addWidget(lbl_char)
            row_layout.addWidget(bar)
            prob_layout.addWidget(row_widget)

            self.prob_rows.append((lbl_char, bar))

        graph_widget = QWidget()
        graph_layout = QVBoxLayout(graph_widget)
        self.chart_canvas = MetricsChartCanvas(graph_widget)
        graph_layout.addWidget(self.chart_canvas)

        self.tabs.addTab(prob_widget, "Top Predictions")
        self.tabs.addTab(graph_widget, "Loss/Accuracy")

        self.layout.addWidget(lbl_title)
        self.layout.addWidget(self.lbl_prediction)
        self.layout.addWidget(self.tabs)

    def update_probabilities(self, probs: list[float]):
        # Sort top probabilities descending
        indexed_probs = list(enumerate(probs))
        sorted_probs = sorted(indexed_probs, key=lambda x: x[1], reverse=True)

        top_class_idx, top_prob = sorted_probs[0]
        top_char = CLASS_LABELS[top_class_idx] if top_class_idx < len(CLASS_LABELS) else "?"
        self.lbl_prediction.setText(top_char)

        # Update top 8 ranked bars
        for rank_idx in range(min(8, len(sorted_probs))):
            c_idx, prob_val = sorted_probs[rank_idx]
            char_str = CLASS_LABELS[c_idx] if c_idx < len(CLASS_LABELS) else "?"
            
            lbl, bar = self.prob_rows[rank_idx]
            lbl.setText(f"'{char_str}'")
            bar.setValue(int(prob_val * 100))
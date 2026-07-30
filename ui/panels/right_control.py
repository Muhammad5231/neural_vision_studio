"""Right Analytics Panel with Probability Distributions and Real-Time Loss/Accuracy Graphs."""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar, QTabWidget
from PySide6.QtCore import Qt
from ui.components.glass_card import GlassCard

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

    def update_metrics(self, epochs: list[int], loss_vals: list[float], acc_vals: list[float]):
        self.ax_loss.clear()
        self.ax_acc.clear()
        self._format_axes()

        if epochs:
            self.ax_loss.plot(epochs, loss_vals, color="#06B6D4", linewidth=2, marker='o', markersize=3)
            self.ax_acc.plot(epochs, acc_vals, color="#007AFF", linewidth=2, marker='o', markersize=3)

        self.draw()


class RightPanel(GlassCard):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(340)
        
        lbl_title = QLabel("Prediction & Network Analytics")
        lbl_title.setStyleSheet("font-size: 15px; font-weight: bold; color: #F8FAFC;")
        
        self.lbl_prediction = QLabel("-")
        self.lbl_prediction.setStyleSheet("font-size: 54px; font-weight: bold; color: #007AFF;")
        self.lbl_prediction.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Tabs for Probability Bars vs Loss/Accuracy Graphs
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #334155; border-radius: 8px; background: #0B0E14; }
            QTabBar::tab { background: #1E293B; color: #94A3B8; padding: 6px 12px; border-radius: 4px; }
            QTabBar::tab:selected { background: #007AFF; color: #FFFFFF; font-weight: bold; }
        """)

        # Tab 1: Output Probabilities
        prob_widget = QWidget()
        prob_layout = QVBoxLayout(prob_widget)
        self.bars = []
        for i in range(10):
            row = QVBoxLayout()
            lbl = QLabel(f"Digit {i}")
            lbl.setStyleSheet("font-size: 10px; color: #94A3B8;")
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFixedHeight(6)
            row.addWidget(lbl)
            row.addWidget(bar)
            prob_layout.addLayout(row)
            self.bars.append(bar)

        # Tab 2: Loss & Accuracy Graphs
        graph_widget = QWidget()
        graph_layout = QVBoxLayout(graph_widget)
        self.chart_canvas = MetricsChartCanvas(graph_widget)
        graph_layout.addWidget(self.chart_canvas)

        self.tabs.addTab(prob_widget, "Probabilities")
        self.tabs.addTab(graph_widget, "Loss/Accuracy")

        self.layout.addWidget(lbl_title)
        self.layout.addWidget(self.lbl_prediction)
        self.layout.addWidget(self.tabs)

        # Initialize mock training metrics plot
        self.chart_canvas.update_metrics(
            epochs=[1, 2, 3, 4, 5],
            loss_vals=[2.1, 1.4, 0.8, 0.4, 0.15],
            acc_vals=[32.0, 58.0, 78.5, 89.0, 96.2]
        )

    def update_probabilities(self, probs: list[float]):
        best_digit = max(range(len(probs)), key=lambda i: probs[i])
        self.lbl_prediction.setText(str(best_digit))
        
        for i, prob in enumerate(probs):
            self.bars[i].setValue(int(prob * 100))
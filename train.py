"""Neural Vision Studio - Standalone Interactive Training Application."""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QSlider, QProgressBar)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QColor

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from core.model import VisionMLP
from config import APP_NAME, COLOR_BG_DARK, COLOR_CARD_BG, COLOR_ELECTRIC_BLUE, COLOR_CYAN

class TrainingThread(QThread):
    epoch_progress = Signal(int, float, float)  # epoch, loss, accuracy
    training_finished = Signal()

    def __init__(self, model: VisionMLP, lr: float = 0.005, epochs: int = 10, parent=None):
        super().__init__(parent)
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.is_running = True

    def run(self):
        # Generate synthetic MNIST dataset for interactive training
        X, y = self._generate_dataset()
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(1, self.epochs + 1):
            if not self.is_running:
                break

            optimizer.zero_grad()
            outputs = self.model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            # Calculate Accuracy
            preds = torch.argmax(outputs, dim=1)
            acc = float((preds == y).float().mean() * 100.0)

            self.epoch_progress.emit(epoch, float(loss.item()), acc)
            self.msleep(180)  # Visual pause for real-time inspection

        self.model.eval()
        torch.save(self.model.state_dict(), "mnist_weights.pt")
        self.training_finished.emit()

    def stop(self):
        self.is_running = False

    def _generate_dataset(self):
        images, labels = [], []
        for digit in range(10):
            for _ in range(60):
                img = np.zeros((28, 28), dtype=np.uint8)
                thick = np.random.randint(2, 4)
                if digit == 0:
                    cv2.ellipse(img, (14, 14), (8, 10), 0, 0, 360, 255, thick)
                elif digit == 1:
                    cv2.line(img, (14, 5), (14, 23), 255, thick)
                elif digit == 2:
                    cv2.polylines(img, [np.array([[6, 8], [20, 8], [6, 22], [22, 22]])], False, 255, thick)
                elif digit == 3:
                    cv2.polylines(img, [np.array([[6, 6], [20, 6], [12, 14], [20, 20], [6, 22]])], False, 255, thick)
                elif digit == 4:
                    cv2.polylines(img, [np.array([[18, 5], [6, 16], [22, 16]])], False, 255, thick)
                    cv2.line(img, (18, 5), (18, 23), 255, thick)
                elif digit == 5:
                    cv2.polylines(img, [np.array([[20, 6], [7, 6], [7, 13], [20, 15], [20, 22], [6, 22]])], False, 255, thick)
                elif digit == 6:
                    cv2.ellipse(img, (14, 17), (7, 6), 0, 0, 360, 255, thick)
                    cv2.line(img, (7, 17), (16, 6), 255, thick)
                elif digit == 7:
                    cv2.line(img, (6, 6), (22, 6), 255, thick)
                    cv2.line(img, (22, 6), (10, 23), 255, thick)
                elif digit == 8:
                    cv2.ellipse(img, (14, 10), (6, 5), 0, 0, 360, 255, thick)
                    cv2.ellipse(img, (14, 18), (7, 6), 0, 0, 360, 255, thick)
                elif digit == 9:
                    cv2.ellipse(img, (14, 10), (6, 5), 0, 0, 360, 255, thick)
                    cv2.line(img, (20, 10), (12, 23), 255, thick)

                img = cv2.GaussianBlur(img, (3, 3), 0)
                images.append((img.astype(np.float32) / 255.0).flatten())
                labels.append(digit)

        return torch.tensor(np.array(images), dtype=torch.float32), torch.tensor(np.array(labels), dtype=torch.long)


class TrainingPlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(figsize=(6, 4), dpi=100, facecolor='#12161F')
        super().__init__(fig)
        self.ax_loss = fig.add_subplot(211)
        self.ax_acc = fig.add_subplot(212)
        self._format_axes()

    def _format_axes(self):
        for ax in [self.ax_loss, self.ax_acc]:
            ax.set_facecolor('#0B0E14')
            ax.tick_params(colors='#94A3B8', labelsize=8)
            for spine in ax.spines.values():
                spine.set_color('#334155')

        self.ax_loss.set_title("Training Loss (CrossEntropy)", color="#06B6D4", fontsize=10)
        self.ax_acc.set_title("Accuracy (%)", color="#007AFF", fontsize=10)
        self.figure.tight_layout(pad=1.5)

    def plot_data(self, epochs, losses, accs):
        self.ax_loss.clear()
        self.ax_acc.clear()
        self._format_axes()

        if epochs:
            self.ax_loss.plot(epochs, losses, color="#06B6D4", linewidth=2, marker='o', markersize=4)
            self.ax_acc.plot(epochs, accs, color="#007AFF", linewidth=2, marker='o', markersize=4)

        self.draw()


class TrainingStudioWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} - Interactive Training Studio")
        self.resize(900, 650)

        self.model = VisionMLP()
        self.epochs_data = []
        self.loss_data = []
        self.acc_data = []
        self.trainer = None

        self._init_ui()

    def _init_ui(self):
        central_widget = QWidget()
        central_widget.setStyleSheet(f"background-color: {COLOR_BG_DARK.name()}; color: #F8FAFC;")
        self.setCentralWidget(central_widget)

        layout = QHBoxLayout(central_widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # Left Control Panel
        left_card = QWidget()
        left_card.setStyleSheet("background: rgba(18, 22, 31, 0.9); border: 1px solid #334155; border-radius: 12px;")
        left_card.setFixedWidth(300)
        left_layout = QVBoxLayout(left_card)

        lbl_title = QLabel("Training Controls")
        lbl_title.setFont(QFont("Inter", 14, QFont.Weight.Bold))
        lbl_title.setStyleSheet("color: #007AFF;")

        self.lbl_status = QLabel("Status: Idle")
        self.lbl_status.setStyleSheet("color: #94A3B8; font-size: 12px;")

        # Epoch Counter
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 20)
        self.progress_bar.setValue(0)

        # Learning Rate Slider
        lbl_lr = QLabel("Learning Rate: 0.005")
        self.lr_slider = QSlider(Qt.Orientation.Horizontal)
        self.lr_slider.setRange(1, 20)
        self.lr_slider.setValue(5)
        self.lr_slider.valueChanged.connect(lambda v: lbl_lr.setText(f"Learning Rate: {v/1000.0:.3f}"))

        # Buttons
        self.btn_start = QPushButton("Start Live Training")
        self.btn_start.setStyleSheet("background-color: #007AFF; color: white; padding: 12px; border-radius: 8px; font-weight: bold;")
        self.btn_start.clicked.connect(self.start_training)

        self.btn_stop = QPushButton("Stop Training")
        self.btn_stop.setStyleSheet("background-color: #334155; color: white; padding: 10px; border-radius: 8px;")
        self.btn_stop.clicked.connect(self.stop_training)

        left_layout.addWidget(lbl_title)
        left_layout.addWidget(self.lbl_status)
        left_layout.addWidget(QLabel("Epoch Progress:"))
        left_layout.addWidget(self.progress_bar)
        left_layout.addWidget(lbl_lr)
        left_layout.addWidget(self.lr_slider)
        left_layout.addSpacing(20)
        left_layout.addWidget(self.btn_start)
        left_layout.addWidget(self.btn_stop)
        left_layout.addStretch()

        # Right Chart Display
        right_card = QWidget()
        right_card.setStyleSheet("background: rgba(18, 22, 31, 0.9); border: 1px solid #334155; border-radius: 12px;")
        right_layout = QVBoxLayout(right_card)

        self.chart = TrainingPlotCanvas()
        right_layout.addWidget(self.chart)

        layout.addWidget(left_card)
        layout.addWidget(right_card, stretch=1)

    def start_training(self):
        self.epochs_data.clear()
        self.loss_data.clear()
        self.acc_data.clear()
        
        lr = self.lr_slider.value() / 1000.0
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Status: Training in progress...")
        self.btn_start.setEnabled(False)

        self.trainer = TrainingThread(self.model, lr=lr, epochs=20)
        self.trainer.epoch_progress.connect(self.on_epoch_update)
        self.trainer.training_finished.connect(self.on_training_finished)
        self.trainer.start()

    def stop_training(self):
        if self.trainer:
            self.trainer.stop()
            self.lbl_status.setText("Status: Stopped by user.")
            self.btn_start.setEnabled(True)

    def on_epoch_update(self, epoch: int, loss: float, acc: float):
        self.epochs_data.append(epoch)
        self.loss_data.append(loss)
        self.acc_data.append(acc)

        self.progress_bar.setValue(epoch)
        self.lbl_status.setText(f"Epoch {epoch}/20 | Loss: {loss:.4f} | Acc: {acc:.1f}%")
        self.chart.plot_data(self.epochs_data, self.loss_data, self.acc_data)

    def on_training_finished(self):
        self.lbl_status.setText("Status: Completed & Saved to mnist_weights.pt!")
        self.btn_start.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TrainingStudioWindow()
    window.show()
    sys.exit(app.exec())
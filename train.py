"""Neural Vision Studio - Interactive Visual Trainer for Digits, Alphabets, & Symbols."""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import cv2

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                 QHBoxLayout, QLabel, QPushButton, QSlider, QProgressBar,
                                 QFrame, QListWidget, QListWidgetItem)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QColor, QImage, QPixmap

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from core.model import VisionMLP
from config import APP_NAME, COLOR_BG_DARK, CLASS_LABELS, NUM_CLASSES
from visualization.network_view import NetworkView


class TrainingWorker(QThread):
    sample_processed = Signal(np.ndarray, str, str, float, float, dict)
    epoch_completed = Signal(int, float, float)
    training_finished = Signal()

    def __init__(self, model: VisionMLP, lr: float = 0.006, epochs: int = 5, delay_ms: int = 80, parent=None):
        super().__init__(parent)
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.delay_ms = delay_ms
        self.is_running = True

    def run(self):
        images, labels = self._generate_extended_dataset()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        total_samples = len(images)
        
        for epoch in range(1, self.epochs + 1):
            if not self.is_running:
                break

            self.model.train()
            running_loss = 0.0
            correct_count = 0
            indices = np.random.permutation(total_samples)

            for step_idx, idx in enumerate(indices):
                if not self.is_running:
                    break

                img_vec = images[idx:idx+1]
                target_idx = labels[idx:idx+1]

                optimizer.zero_grad()
                logits = self.model(img_vec)
                loss = criterion(logits, target_idx)
                loss.backward()
                optimizer.step()

                probs = F.softmax(logits, dim=1).squeeze()
                pred_idx = int(torch.argmax(probs).item())
                target_val = int(target_idx.item())

                running_loss += loss.item()
                if pred_idx == target_val:
                    correct_count += 1

                current_loss = running_loss / (step_idx + 1)
                current_acc = (correct_count / (step_idx + 1)) * 100.0

                l1 = torch.relu(self.model.activations["layer1"]).squeeze()
                l2 = torch.relu(self.model.activations["layer2"]).squeeze()
                
                img_2d = img_vec.squeeze().view(28, 28).numpy()
                target_activations = {}
                flat_input = img_2d.flatten()
                
                for i in range(16):
                    target_activations[(0, i)] = float(flat_input[i * 49])
                for i in range(12):
                    target_activations[(1, i)] = float(l1[i % len(l1)])
                for i in range(8):
                    target_activations[(2, i)] = float(l2[i % len(l2)])
                for i in range(12):
                    target_activations[(3, i)] = float(probs[i % len(probs)])

                target_char = CLASS_LABELS[target_val] if target_val < len(CLASS_LABELS) else "?"
                pred_char = CLASS_LABELS[pred_idx] if pred_idx < len(CLASS_LABELS) else "?"

                self.sample_processed.emit(img_2d, target_char, pred_char, current_loss, current_acc, target_activations)
                self.msleep(self.delay_ms)

            epoch_loss = running_loss / total_samples
            epoch_acc = (correct_count / total_samples) * 100.0
            self.epoch_completed.emit(epoch, epoch_loss, epoch_acc)

        self.model.eval()
        torch.save(self.model.state_dict(), "extended_weights.pt")
        self.training_finished.emit()

    def set_delay(self, ms: int):
        self.delay_ms = ms

    def stop(self):
        self.is_running = False

    def _generate_extended_dataset(self):
        images, labels = [], []
        for idx, char in enumerate(CLASS_LABELS):
            for _ in range(5):  # 5 variations per class
                img = np.zeros((28, 28), dtype=np.uint8)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.75
                thick = np.random.randint(1, 3)
                
                text_size = cv2.getTextSize(char, font, font_scale, thick)[0]
                text_x = max(2, (28 - text_size[0]) // 2)
                text_y = min(24, (28 + text_size[1]) // 2)
                
                cv2.putText(img, char, (text_x, text_y), font, font_scale, 255, thick, cv2.LINE_AA)
                img = cv2.GaussianBlur(img, (3, 3), 0)
                images.append(torch.tensor((img.astype(np.float32) / 255.0).flatten()))
                labels.append(idx)

        return torch.stack(images), torch.tensor(labels, dtype=torch.long)


class TrainingPlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(figsize=(4, 3), dpi=100, facecolor='#12161F')
        super().__init__(fig)
        self.ax_loss = fig.add_subplot(211)
        self.ax_acc = fig.add_subplot(212)
        self._format_axes()

    def _format_axes(self):
        for ax in [self.ax_loss, self.ax_acc]:
            ax.set_facecolor('#0B0E14')
            ax.tick_params(colors='#94A3B8', labelsize=7)
            for spine in ax.spines.values():
                spine.set_color('#334155')

        self.ax_loss.set_title("Live Loss", color="#06B6D4", fontsize=9, pad=2)
        self.ax_acc.set_title("Accuracy (%)", color="#007AFF", fontsize=9, pad=2)
        self.figure.tight_layout(pad=1.2)

    def update_charts(self, steps, losses, accs):
        self.ax_loss.clear()
        self.ax_acc.clear()
        self._format_axes()

        if steps:
            self.ax_loss.plot(steps, losses, color="#06B6D4", linewidth=1.5)
            self.ax_acc.plot(steps, accs, color="#007AFF", linewidth=1.5)

        self.draw()


class LiveInputDisplay(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            LiveInputDisplay {
                background: rgba(18, 22, 31, 0.9);
                border: 1px solid #334155;
                border-radius: 12px;
            }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        lbl_header = QLabel("Auto-Feeding Character")
        lbl_header.setFont(QFont("Inter", 11, QFont.Weight.Bold))
        lbl_header.setStyleSheet("color: #06B6D4;")

        self.lbl_image = QLabel()
        self.lbl_image.setFixedSize(160, 160)
        self.lbl_image.setStyleSheet("background-color: #000000; border-radius: 8px;")
        self.lbl_image.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.lbl_target = QLabel("Target: -")
        self.lbl_target.setFont(QFont("JetBrains Mono", 10, QFont.Weight.Bold))
        self.lbl_target.setStyleSheet("color: #F8FAFC;")

        self.lbl_pred = QLabel("Predicted: -")
        self.lbl_pred.setFont(QFont("JetBrains Mono", 10, QFont.Weight.Bold))
        self.lbl_pred.setStyleSheet("color: #007AFF;")

        self.lbl_match = QLabel("STATUS: WAITING")
        self.lbl_match.setFont(QFont("Inter", 10, QFont.Weight.Bold))
        self.lbl_match.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_match.setStyleSheet("background: #1E293B; color: #94A3B8; padding: 6px; border-radius: 6px;")

        layout.addWidget(lbl_header)
        layout.addWidget(self.lbl_image, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_target)
        layout.addWidget(self.lbl_pred)
        layout.addWidget(self.lbl_match)

    def update_sample(self, img_2d: np.ndarray, target: str, pred: str):
        scaled_img = (img_2d * 255.0).astype(np.uint8)
        colored_img = cv2.cvtColor(scaled_img, cv2.COLOR_GRAY2RGB)
        h, w, c = colored_img.shape
        qimg = QImage(colored_img.data, w, h, w * c, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(160, 160, Qt.AspectRatioMode.KeepAspectRatio)
        
        self.lbl_image.setPixmap(pixmap)
        self.lbl_target.setText(f"Target: '{target}'")
        self.lbl_pred.setText(f"Predicted: '{pred}'")

        if target == pred:
            self.lbl_match.setText("MATCH ✅ (WIN)")
            self.lbl_match.setStyleSheet("background: rgba(6, 182, 212, 0.2); color: #06B6D4; padding: 6px; border-radius: 6px; border: 1px solid #06B6D4;")
        else:
            self.lbl_match.setText("MISMATCH ❌ (LOSS)")
            self.lbl_match.setStyleSheet("background: rgba(239, 68, 68, 0.2); color: #EF4444; padding: 6px; border-radius: 6px; border: 1px solid #EF4444;")


class VisualTrainingStudioWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} — Extended Vocabulary Trainer")
        self.resize(1400, 850)

        self.model = VisionMLP()
        self.worker = None

        self.step_history = []
        self.loss_history = []
        self.acc_history = []
        self.step_counter = 0

        self._init_ui()

    def _init_ui(self):
        central_widget = QWidget()
        central_widget.setStyleSheet(f"background-color: {COLOR_BG_DARK.name()}; color: #F8FAFC;")
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        # Left Panel
        left_panel = QWidget()
        left_panel.setFixedWidth(280)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.input_display = LiveInputDisplay()

        controls_card = QFrame()
        controls_card.setStyleSheet("background: rgba(18, 22, 31, 0.9); border: 1px solid #334155; border-radius: 12px; padding: 10px;")
        controls_layout = QVBoxLayout(controls_card)

        lbl_ctrl_title = QLabel("Vocabulary Controls")
        lbl_ctrl_title.setFont(QFont("Inter", 11, QFont.Weight.Bold))
        lbl_ctrl_title.setStyleSheet("color: #007AFF;")

        self.lbl_speed = QLabel("Step Speed: 80 ms")
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(20, 300)
        self.speed_slider.setValue(80)
        self.speed_slider.valueChanged.connect(self._on_speed_changed)

        self.btn_start = QPushButton("Start Vocabulary Training")
        self.btn_start.setStyleSheet("background-color: #007AFF; color: white; padding: 10px; border-radius: 8px; font-weight: bold;")
        self.btn_start.clicked.connect(self.start_training)

        self.btn_stop = QPushButton("Pause / Stop")
        self.btn_stop.setStyleSheet("background-color: #334155; color: white; padding: 8px; border-radius: 8px;")
        self.btn_stop.clicked.connect(self.stop_training)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 5)
        self.progress_bar.setValue(0)

        controls_layout.addWidget(lbl_ctrl_title)
        controls_layout.addWidget(self.lbl_speed)
        controls_layout.addWidget(self.speed_slider)
        controls_layout.addSpacing(10)
        controls_layout.addWidget(self.btn_start)
        controls_layout.addWidget(self.btn_stop)
        controls_layout.addWidget(QLabel("Epoch Progress:"))
        controls_layout.addWidget(self.progress_bar)

        left_layout.addWidget(self.input_display)
        left_layout.addWidget(controls_card)
        left_layout.addStretch()

        # Center Panel
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)

        self.network_view = NetworkView()
        center_layout.addWidget(self.network_view)

        # Right Panel
        right_panel = QWidget()
        right_panel.setFixedWidth(320)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.chart_canvas = TrainingPlotCanvas()

        stream_card = QFrame()
        stream_card.setStyleSheet("background: rgba(18, 22, 31, 0.9); border: 1px solid #334155; border-radius: 12px; padding: 8px;")
        stream_layout = QVBoxLayout(stream_card)

        lbl_stream = QLabel("Character Log Stream")
        lbl_stream.setFont(QFont("Inter", 10, QFont.Weight.Bold))
        lbl_stream.setStyleSheet("color: #06B6D4;")

        self.log_list = QListWidget()
        self.log_list.setStyleSheet("""
            QListWidget {
                background-color: #0B0E14;
                border: 1px solid #1E293B;
                border-radius: 6px;
                color: #F8FAFC;
                font-family: 'JetBrains Mono';
                font-size: 10px;
            }
        """)

        stream_layout.addWidget(lbl_stream)
        stream_layout.addWidget(self.log_list)

        right_layout.addWidget(self.chart_canvas)
        right_layout.addWidget(stream_card, stretch=1)

        main_layout.addWidget(left_panel)
        main_layout.addWidget(center_panel, stretch=1)
        main_layout.addWidget(right_panel)

    def _on_speed_changed(self, val: int):
        self.lbl_speed.setText(f"Step Speed: {val} ms")
        if self.worker:
            self.worker.set_delay(val)

    def start_training(self):
        self.step_history.clear()
        self.loss_history.clear()
        self.acc_history.clear()
        self.log_list.clear()
        self.step_counter = 0

        self.btn_start.setEnabled(False)
        delay = self.speed_slider.value()

        self.worker = TrainingWorker(self.model, lr=0.006, epochs=5, delay_ms=delay)
        self.worker.sample_processed.connect(self._on_sample_step)
        self.worker.epoch_completed.connect(self._on_epoch_complete)
        self.worker.training_finished.connect(self._on_training_finished)
        self.worker.start()

    def stop_training(self):
        if self.worker:
            self.worker.stop()
            self.btn_start.setEnabled(True)

    def _on_sample_step(self, img_2d: np.ndarray, target: str, pred: str, loss: float, acc: float, activations: dict):
        self.step_counter += 1
        
        self.input_display.update_sample(img_2d, target, pred)
        self.network_view.animate_signal_flow(activations)

        self.step_history.append(self.step_counter)
        self.loss_history.append(loss)
        self.acc_history.append(acc)

        if len(self.step_history) % 2 == 0:
            self.chart_canvas.update_charts(self.step_history, self.loss_history, self.acc_history)

        match_str = "MATCH" if target == pred else "LOSS"
        item = QListWidgetItem(f"[{match_str}] Target:'{target}' | Pred:'{pred}' | Acc:{acc:.1f}%")
        if target == pred:
            item.setForeground(QColor("#06B6D4"))
        else:
            item.setForeground(QColor("#EF4444"))
        
        self.log_list.addItem(item)
        self.log_list.scrollToBottom()

    def _on_epoch_complete(self, epoch: int, loss: float, acc: float):
        self.progress_bar.setValue(epoch)

    def _on_training_finished(self):
        self.btn_start.setEnabled(True)
        item = QListWidgetItem("=== VOCABULARY WEIGHTS SAVED ===")
        item.setForeground(QColor("#007AFF"))
        self.log_list.addItem(item)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VisualTrainingStudioWindow()
    window.show()
    sys.exit(app.exec())
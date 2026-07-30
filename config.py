"""Global configuration parameters and theme palette for Neural Vision Studio."""

from PySide6.QtGui import QColor, QFont

APP_NAME = "Neural Vision Studio"
APP_VERSION = "1.0.0"

# Target FPS and Rendering Parameters
TARGET_FPS = 60
FRAME_INTERVAL_MS = int(1000 / TARGET_FPS)

# Dark Modern Color Palette
COLOR_BG_DARK = QColor("#0B0E14")         # Space / Deep Charcoal
COLOR_CARD_BG = QColor(18, 22, 31, 200)   # Glassmorphic gray
COLOR_ELECTRIC_BLUE = QColor("#007AFF")   # Primary Accent
COLOR_PURPLE = QColor("#A855F7")          # Secondary Accent
COLOR_CYAN = QColor("#06B6D4")            # Highlighting / Flow Accent
COLOR_NEURON_OFF = QColor("#1E293B")      # Inactive state

# Text Colors
COLOR_TEXT_PRIMARY = QColor("#F8FAFC")
COLOR_TEXT_SECONDARY = QColor("#94A3B8")

# Canvas Parameters
CANVAS_SIZE = 280
MNIST_GRID_SIZE = 28

# Fonts
FONT_PRIMARY = QFont("Inter", 10)
FONT_HEADING = QFont("Inter", 12, QFont.Weight.Bold)
FONT_DISPLAY = QFont("JetBrains Mono", 16, QFont.Weight.Bold)
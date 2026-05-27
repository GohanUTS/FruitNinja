#!/usr/bin/env python3
"""
home_screen.py — Branded splash / launcher for the FruitNinja operator stack.

Shows a fullscreen-friendly home page with the FruitNinja brand, a robot
profile picker, and a single big START button.  Clicking START remembers the
chosen profile and launches the existing step-by-step startup launcher
(`startup_gui`) as a separate ROS-aware subprocess, then closes the splash.

This is the recommended entry point:

    ros2 run fruitninja home_screen
"""
import os
os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)

import sys
import subprocess

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSizePolicy,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import (
    QFont, QColor, QPainter, QPen, QBrush, QLinearGradient, QRadialGradient,
)

from fruitninja import theme as _theme
from fruitninja.grid_mover import (
    CALIBRATION_PROFILES, get_active_profile, set_active_profile,
    list_saved_profiles,
)


# ── Animated brand strip (oversized for the home page) ────────────────────────

class HomeBrand(QWidget):
    """Big animated FRUITNINJA logo plate, used only on the home screen."""

    def __init__(self):
        super().__init__()
        self._phase = 0
        self.setMinimumHeight(180)
        self.setMaximumHeight(220)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(45)

    def _tick(self):
        self._phase = (self._phase + 3) % 360
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        bg = QLinearGradient(0, 0, w, h)
        bg.setColorAt(0.0, QColor(_theme.THEME['bg_panel']))
        bg.setColorAt(0.5, QColor('#16222c'))
        bg.setColorAt(1.0, QColor(_theme.THEME['bg_panel_alt']))
        p.setBrush(QBrush(bg))
        p.setPen(QPen(QColor(_theme.THEME['border_strong']), 1))
        p.drawRoundedRect(0, 0, w - 1, h - 1, 14, 14)

        glow_x = int((self._phase / 360.0) * (w + 180)) - 90
        glow = QRadialGradient(glow_x, h // 2, max(160, h))
        glow.setColorAt(0.0, QColor(58, 122, 255, 90))
        glow.setColorAt(1.0, QColor(58, 122, 255, 0))
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(glow))
        p.drawEllipse(glow_x - h, -h // 2, h * 2, h * 2)

        # 3-D drop title
        p.setFont(QFont('Arial Black', 64, QFont.Black))
        title = 'FRUITNINJA'
        title_x = max(40, (w - 600) // 2)
        title_y = h // 2 + 16
        for dx, dy, col in [
            (5, 6, QColor('#050a0e')),
            (3, 4, QColor('#16313a')),
            (2, 2, QColor('#1f4554')),
        ]:
            p.setPen(QPen(col))
            p.drawText(title_x + dx, title_y + dy, title)

        title_grad = QLinearGradient(title_x, title_y - 60, title_x + 620, title_y + 10)
        title_grad.setColorAt(0.0, QColor(_theme.THEME['brand_a']))
        title_grad.setColorAt(0.5, QColor(_theme.THEME['brand_b']))
        title_grad.setColorAt(1.0, QColor(_theme.THEME['brand_c']))
        p.setPen(QPen(QBrush(title_grad), 1))
        p.drawText(title_x, title_y, title)

        # Subtitle / tagline
        p.setFont(QFont('Arial', 14, QFont.DemiBold))
        p.setPen(QPen(QColor(_theme.THEME['fg_muted'])))
        p.drawText(title_x + 6, title_y + 34, 'UR3e Operator Console')

        p.end()


# ── Home window ──────────────────────────────────────────────────────────────

class HomeWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('FruitNinja — Home')
        self.setMinimumSize(820, 600)
        self.setStyleSheet(_theme.root_qss())
        self._build_ui()

    def _build_ui(self):
        root = QWidget()
        root.setObjectName('root')
        self.setCentralWidget(root)

        outer = QVBoxLayout(root)
        outer.setContentsMargins(48, 48, 48, 48)
        outer.setSpacing(28)

        outer.addStretch(1)
        outer.addWidget(HomeBrand())

        # Robot picker card
        card = QWidget()
        card.setStyleSheet(
            f'background:{_theme.THEME["bg_panel"]};'
            f' border:1px solid {_theme.THEME["border"]};'
            f' border-radius:{_theme.THEME["radius_lg"]};'
        )
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(28, 22, 28, 22)
        card_layout.setSpacing(14)

        title = QLabel('Select robot profile')
        title.setStyleSheet(
            f'color:{_theme.THEME["fg_muted"]}; font-size:12px;'
            f' font-weight:700; letter-spacing:0.6px;'
        )
        title.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(title)

        picker_row = QHBoxLayout()
        picker_row.setSpacing(12)
        picker_row.addStretch(1)

        self._profile_combo = QComboBox()
        self._profile_combo.setFixedWidth(280)
        self._profile_combo.setMinimumHeight(38)
        self._profile_combo.setStyleSheet(_theme.combo_qss())
        saved = list_saved_profiles()
        for i, name in enumerate(CALIBRATION_PROFILES, start=1):
            tag = 'saved' if saved[name] else 'empty'
            self._profile_combo.addItem(f'Robot {i}   ·   {tag}', name)
        active = get_active_profile()
        if active:
            for idx in range(self._profile_combo.count()):
                if self._profile_combo.itemData(idx) == active:
                    self._profile_combo.setCurrentIndex(idx)
                    break
        picker_row.addWidget(self._profile_combo)
        picker_row.addStretch(1)
        card_layout.addLayout(picker_row)

        hint = QLabel(
            'The selected profile is loaded on launch. You can switch later '
            'inside the main GUI.'
        )
        hint.setAlignment(Qt.AlignCenter)
        hint.setWordWrap(True)
        hint.setStyleSheet(
            f'color:{_theme.THEME["fg_dim"]}; font-size:11px;'
        )
        card_layout.addWidget(hint)
        outer.addWidget(card)

        # Big START button
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self._start_btn = QPushButton('▶   START')
        self._start_btn.setMinimumWidth(280)
        self._start_btn.setMinimumHeight(58)
        self._start_btn.setCursor(Qt.PointingHandCursor)
        self._start_btn.setStyleSheet(_theme.primary_button_qss(big=True))
        self._start_btn.clicked.connect(self._on_start)
        btn_row.addWidget(self._start_btn)
        btn_row.addStretch(1)
        outer.addLayout(btn_row)

        footer = QLabel('FruitNinja  ·  UR3e cutting cell  ·  v0.1')
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet(
            f'color:{_theme.THEME["fg_dim"]}; font-size:10px; letter-spacing:0.6px;'
        )
        outer.addWidget(footer)
        outer.addStretch(1)

    def _on_start(self):
        # Remember the selected profile so startup_gui / real_gui_points pick it up.
        profile = self._profile_combo.currentData()
        try:
            set_active_profile(profile)
        except Exception:
            # Bad profile name should never happen since we built the list ourselves.
            pass

        # Launch the existing step-by-step startup launcher as a separate process.
        # Keep the same ROS environment as ourselves.
        try:
            subprocess.Popen(
                ['ros2', 'run', 'fruitninja', 'startup_gui'],
                env=os.environ.copy(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except FileNotFoundError:
            # `ros2` not on PATH — try sourcing the workspace directly.
            subprocess.Popen(
                ['/bin/bash', '-lc',
                 'source /opt/ros/humble/setup.bash && '
                 'source ~/ros2_ws/install/setup.bash && '
                 'ros2 run fruitninja startup_gui'],
                env=os.environ.copy(),
                start_new_session=True,
            )

        self.close()


def main(args=None):
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = HomeWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
home_screen.py — Branded splash for the FruitNinja operator stack.

A self-contained animated splash:
  • Backdrop with drifting fruit silhouettes and slash trails
  • Oversized 3-D FRUITNINJA logo with hue-rotating gradient
  • Stylised UR3e arm silhouette next to the title
  • Single pulsing START button (no robot picker — pick a profile inside the
    main GUI's calibration panel)

Clicking START launches the existing step-by-step launcher (`startup_gui`)
as a separate ROS-aware subprocess and closes the splash.

    ros2 run fruitninja home_screen
"""
import os
os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)

import math
import random
import subprocess
import sys

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSizePolicy,
)
from PyQt5.QtCore import Qt, QTimer, QRectF, QPointF
from PyQt5.QtGui import (
    QFont, QColor, QPainter, QPen, QBrush, QLinearGradient, QRadialGradient,
    QPolygonF, QPainterPath,
)

from fruitninja import theme as _theme


# ── Backdrop: drifting fruit + slash trails ──────────────────────────────────

# (label, fill colour, accent stem colour)
_FRUIT_PALETTE = [
    ('apple',      QColor('#e74848'), QColor('#3a7a3a')),
    ('orange',     QColor('#f39135'), QColor('#3a7a3a')),
    ('lime',       QColor('#74c83a'), QColor('#2e5a2e')),
    ('lemon',      QColor('#f4d03f'), QColor('#2e5a2e')),
    ('cherry',     QColor('#c0392b'), QColor('#7a4a1f')),
    ('strawberry', QColor('#ff5577'), QColor('#3a7a3a')),
    ('grape',      QColor('#8e44ad'), QColor('#3a7a3a')),
    ('kiwi',       QColor('#7faa3a'), QColor('#3a4f1f')),
]


class _Fruit:
    """A single floating fruit drifting up-left or up-right across the canvas."""
    __slots__ = ('x', 'y', 'r', 'vx', 'vy', 'spin', 'angle',
                 'fill', 'stem', 'kind')

    def __init__(self, w: int, h: int):
        kind, fill, stem = random.choice(_FRUIT_PALETTE)
        self.kind = kind
        self.fill = fill
        self.stem = stem
        self.r = random.randint(18, 38)
        self.x = random.uniform(-50, w + 50)
        self.y = random.uniform(h + 20, h + 200)
        self.vx = random.uniform(-0.35, 0.35)
        self.vy = random.uniform(-0.55, -0.18)
        self.spin = random.uniform(-1.4, 1.4)
        self.angle = random.uniform(0, 360)

    def step(self):
        self.x += self.vx
        self.y += self.vy
        self.angle += self.spin

    def offscreen(self, w: int, h: int) -> bool:
        return self.y < -80 or self.x < -120 or self.x > w + 120


class HomeBackdrop(QWidget):
    """Painted backdrop with drifting fruit silhouettes."""

    def __init__(self):
        super().__init__()
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._fruits = []
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)  # ~30 fps

    def resizeEvent(self, _e):
        # Seed enough fruit to fill the canvas.
        target = max(10, self.width() * self.height() // 40000)
        while len(self._fruits) < target:
            self._fruits.append(_Fruit(self.width(), self.height()))

    def _tick(self):
        w, h = self.width(), self.height()
        for f in self._fruits:
            f.step()
        # Recycle off-screen fruit at the bottom.
        for i, f in enumerate(self._fruits):
            if f.offscreen(w, h):
                self._fruits[i] = _Fruit(w, h)
                self._fruits[i].y = h + random.uniform(20, 80)
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # Deep gradient background.
        bg = QLinearGradient(0, 0, 0, h)
        bg.setColorAt(0.0, QColor('#070b10'))
        bg.setColorAt(0.5, QColor('#0c1620'))
        bg.setColorAt(1.0, QColor('#06090d'))
        p.fillRect(0, 0, w, h, QBrush(bg))

        # Drifting fruit (silhouettes — low alpha so they're decorative).
        for f in self._fruits:
            self._paint_fruit(p, f)

        p.end()

    def _paint_fruit(self, p: QPainter, f: _Fruit):
        # Translucent silhouettes; fruit colour with reduced alpha.
        p.save()
        p.translate(f.x, f.y)
        p.rotate(f.angle)
        fill = QColor(f.fill)
        fill.setAlpha(40)
        stem = QColor(f.stem)
        stem.setAlpha(60)
        # Body
        radial = QRadialGradient(0, -f.r * 0.2, f.r * 1.6)
        c1 = QColor(fill)
        c1.setAlpha(70)
        c2 = QColor(fill)
        c2.setAlpha(10)
        radial.setColorAt(0.0, c1)
        radial.setColorAt(1.0, c2)
        p.setBrush(QBrush(radial))
        p.setPen(Qt.NoPen)
        if f.kind == 'grape':
            # Cluster of 3 small circles
            for ox, oy in [(-f.r * 0.5, 0), (f.r * 0.5, 0), (0, f.r * 0.6)]:
                p.drawEllipse(QPointF(ox, oy), f.r * 0.55, f.r * 0.55)
        else:
            p.drawEllipse(QPointF(0, 0), f.r, f.r * 0.95)
        # Stem leaf
        p.setBrush(QBrush(stem))
        leaf = QPainterPath()
        leaf.moveTo(0, -f.r * 0.9)
        leaf.cubicTo(f.r * 0.4, -f.r * 1.3,
                     f.r * 0.7, -f.r * 0.8,
                     0, -f.r * 0.7)
        p.drawPath(leaf)
        p.restore()


# ── Foreground: brand + arm silhouette + tagline ─────────────────────────────

class BrandPlate(QWidget):
    """Big animated FRUITNINJA logo plate with a stylised UR3e arm silhouette."""

    def __init__(self):
        super().__init__()
        self._phase = 0
        self._hue_shift = 0.0
        self.setMinimumHeight(240)
        self.setMaximumHeight(280)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(40)

    def _tick(self):
        self._phase = (self._phase + 3) % 360
        self._hue_shift = (self._hue_shift + 0.5) % 360.0
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # Plate
        plate = QLinearGradient(0, 0, w, h)
        plate.setColorAt(0.0, QColor(16, 26, 34, 230))
        plate.setColorAt(0.5, QColor(22, 34, 44, 230))
        plate.setColorAt(1.0, QColor(10, 18, 24, 230))
        p.setBrush(QBrush(plate))
        p.setPen(QPen(QColor(_theme.THEME['border_strong']), 1))
        p.drawRoundedRect(0, 0, w - 1, h - 1, 18, 18)

        # Sweeping glow
        glow_x = int((self._phase / 360.0) * (w + 220)) - 110
        glow = QRadialGradient(glow_x, h // 2, max(220, h))
        glow.setColorAt(0.0, QColor(58, 122, 255, 100))
        glow.setColorAt(1.0, QColor(58, 122, 255, 0))
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(glow))
        p.drawEllipse(glow_x - h, -h // 2, h * 2, h * 2)

        # Title with shifting hue gradient
        title = 'FRUITNINJA'
        font_size = max(48, min(82, w // 16))
        p.setFont(QFont('Arial Black', font_size, QFont.Black))
        title_x = 60
        title_y = h // 2 + font_size // 3
        # Drop layers
        for dx, dy, alpha in [(6, 7, 240), (4, 5, 150), (2, 2, 80)]:
            shadow = QColor('#04080b')
            shadow.setAlpha(alpha)
            p.setPen(QPen(shadow))
            p.drawText(title_x + dx, title_y + dy, title)
        # Gradient stops shift hue over time for a subtle alive feel
        def _shift(c: QColor) -> QColor:
            h_, s, v, a = c.getHsv()
            h_ = (h_ + int(self._hue_shift)) % 360
            return QColor.fromHsv(h_, s, v, a)
        title_grad = QLinearGradient(title_x, title_y - font_size,
                                     title_x + font_size * 9, title_y + 10)
        title_grad.setColorAt(0.0, _shift(QColor(_theme.THEME['brand_a'])))
        title_grad.setColorAt(0.5, _shift(QColor(_theme.THEME['brand_b'])))
        title_grad.setColorAt(1.0, _shift(QColor(_theme.THEME['brand_c'])))
        p.setPen(QPen(QBrush(title_grad), 1))
        p.drawText(title_x, title_y, title)

        # Subtitle / tagline
        p.setFont(QFont('Inter', 13, QFont.DemiBold))
        p.setPen(QPen(QColor(_theme.THEME['fg_muted'])))
        p.drawText(title_x + 4, title_y + 42, 'UR3e Operator Console')

        p.end()


# ── Pulsing start button ─────────────────────────────────────────────────────

class StartButton(QPushButton):
    """A big START button with a soft pulsing accent glow."""

    def __init__(self):
        super().__init__('▶  START')
        self.setMinimumWidth(320)
        self.setMinimumHeight(68)
        self.setCursor(Qt.PointingHandCursor)
        self.setFont(QFont('Inter', 16, QFont.Bold))
        self._phase = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(50)
        self._apply_style(1.0)

    def _tick(self):
        self._phase = (self._phase + 0.06) % (2 * math.pi)
        intensity = 0.55 + 0.45 * (0.5 + 0.5 * math.sin(self._phase))
        self._apply_style(intensity)

    def _apply_style(self, intensity: float):
        a = int(120 * intensity)
        accent = _theme.THEME['accent']
        accent_h = _theme.THEME['accent_hover']
        accent_p = _theme.THEME['accent_press']
        self.setStyleSheet(
            f'QPushButton {{'
            f'  background: qlineargradient(x1:0, y1:0, x2:1, y2:0, '
            f'    stop:0 {accent}, stop:1 {accent_h});'
            f'  color: #ffffff;'
            f'  border: 1px solid rgba(140, 180, 255, {a});'
            f'  border-radius: 18px;'
            f'  padding: 14px 28px;'
            f'  font-weight: 700;'
            f'  letter-spacing: 2px;'
            f'}}'
            f'QPushButton:hover {{'
            f'  background: qlineargradient(x1:0, y1:0, x2:1, y2:0, '
            f'    stop:0 {accent_h}, stop:1 #6aa5ff);'
            f'}}'
            f'QPushButton:pressed {{ background:{accent_p}; }}'
        )


# ── Home window ──────────────────────────────────────────────────────────────

class HomeWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('FruitNinja — Home')
        self.setMinimumSize(940, 660)
        self.setStyleSheet(_theme.root_qss())

        root = QWidget()
        root.setObjectName('root')
        self.setCentralWidget(root)

        # Backdrop fills root; foreground laid out on top via stacked widgets.
        self._backdrop = HomeBackdrop()
        self._backdrop.setParent(root)
        self._backdrop.lower()

        self._fg = QWidget(root)
        self._fg.setAttribute(Qt.WA_TranslucentBackground)
        layout = QVBoxLayout(self._fg)
        layout.setContentsMargins(56, 56, 56, 56)
        layout.setSpacing(28)
        layout.addStretch(2)
        layout.addWidget(BrandPlate())
        layout.addSpacing(18)

        tagline = QLabel('Slice fruit. Calibrate cells. Stay precise.')
        tagline.setAlignment(Qt.AlignCenter)
        tagline.setStyleSheet(
            f'color:{_theme.THEME["fg_primary"]}; font-size:15px;'
            f' font-weight:600; letter-spacing:1.2px;'
        )
        layout.addWidget(tagline)

        sub = QLabel('Press START to launch the operator stack.')
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet(
            f'color:{_theme.THEME["fg_muted"]}; font-size:12px;'
        )
        layout.addWidget(sub)
        layout.addSpacing(8)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self._start_btn = StartButton()
        self._start_btn.clicked.connect(self._on_start)
        btn_row.addWidget(self._start_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        layout.addStretch(1)
        footer = QLabel('Robot profile is picked inside the main GUI · ESC to quit')
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet(
            f'color:{_theme.THEME["fg_dim"]}; font-size:10px; letter-spacing:0.6px;'
        )
        layout.addWidget(footer)

    def resizeEvent(self, e):
        # Keep backdrop and foreground sized to the window.
        self._backdrop.setGeometry(0, 0, self.centralWidget().width(),
                                   self.centralWidget().height())
        self._fg.setGeometry(0, 0, self.centralWidget().width(),
                             self.centralWidget().height())
        super().resizeEvent(e)

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Escape, Qt.Key_Q):
            self.close()
        elif e.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self._on_start()
        else:
            super().keyPressEvent(e)

    def _on_start(self):
        """Launch the existing step-by-step launcher in a new process and close."""
        try:
            subprocess.Popen(
                ['ros2', 'run', 'fruitninja', 'startup_gui'],
                env=os.environ.copy(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except FileNotFoundError:
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

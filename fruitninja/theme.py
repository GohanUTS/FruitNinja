#!/usr/bin/env python3
"""
FruitNinja shared UI theme.

All operator GUIs (home_screen, startup_gui, real_gui_points) import THEME
and the qss helpers from here so the look stays consistent and tweaking a
colour only takes one edit.
"""

THEME = {
    # Backgrounds
    'bg_root':        '#0a0e13',
    'bg_panel':       '#101a22',
    'bg_panel_alt':   '#0f1820',
    'bg_input':       '#14222b',
    'bg_hover':       '#1b2c38',

    # Borders
    'border':         '#2f4654',
    'border_strong':  '#3a5466',
    'border_focus':   '#3a7aff',

    # Foreground / text
    'fg_primary':     '#e6edf3',
    'fg_muted':       '#8b9aa8',
    'fg_dim':         '#5b6f7d',

    # Accent / status
    'accent':         '#3a7aff',
    'accent_hover':   '#5491ff',
    'accent_press':   '#2864e0',
    'success':        '#00cc88',
    'success_hover':  '#00e69a',
    'warning':        '#e0a000',
    'danger':         '#ff4444',
    'danger_hover':   '#ff6464',

    # Brand gradient stops
    'brand_a':        '#ff4d6d',
    'brand_b':        '#ffae00',
    'brand_c':        '#3ad29f',

    # Geometry
    'radius':         '8px',
    'radius_sm':      '5px',
    'radius_lg':      '14px',
    'pad':            '10px',
    'pad_sm':         '6px',
    'pad_lg':         '16px',
}


def root_qss() -> str:
    """Top-level QSS applied to every QMainWindow."""
    return (
        f"QMainWindow, QWidget#root {{ background:{THEME['bg_root']};"
        f" color:{THEME['fg_primary']}; font-family:'Inter','Segoe UI','Helvetica Neue',sans-serif; }}"
        f" QLabel {{ color:{THEME['fg_primary']}; }}"
        f" QToolTip {{ background:{THEME['bg_panel']}; color:{THEME['fg_primary']};"
        f" border:1px solid {THEME['border']}; padding:4px 8px; border-radius:{THEME['radius_sm']}; }}"
    )


def panel_qss() -> str:
    """QGroupBox styled as a flat panel card."""
    return (
        f"QGroupBox {{"
        f" background:{THEME['bg_panel']};"
        f" color:{THEME['fg_primary']};"
        f" border:1px solid {THEME['border']};"
        f" border-radius:{THEME['radius']};"
        f" margin-top:14px; padding:10px;"
        f" font-weight:600; font-size:13px;"
        f"}}"
        f"QGroupBox::title {{"
        f" subcontrol-origin:margin; subcontrol-position:top left;"
        f" left:12px; padding:0 6px;"
        f" color:{THEME['fg_muted']};"
        f" letter-spacing:0.6px; text-transform:uppercase; font-size:11px;"
        f"}}"
    )


def _btn_base(bg: str, hover: str, press: str, fg: str = None,
              radius: str = None, padding: str = '9px 14px') -> str:
    fg = fg or '#ffffff'
    radius = radius or THEME['radius']
    return (
        f"QPushButton {{"
        f" background:{bg}; color:{fg};"
        f" border:1px solid {bg}; border-radius:{radius};"
        f" padding:{padding}; font-weight:600; font-size:13px;"
        f"}}"
        f"QPushButton:hover {{ background:{hover}; border-color:{hover}; }}"
        f"QPushButton:pressed {{ background:{press}; border-color:{press}; }}"
        f"QPushButton:disabled {{"
        f" background:{THEME['bg_input']}; color:{THEME['fg_dim']};"
        f" border-color:{THEME['border']};"
        f"}}"
    )


def primary_button_qss(big: bool = False) -> str:
    pad = '14px 28px' if big else '9px 14px'
    return _btn_base(THEME['accent'], THEME['accent_hover'],
                     THEME['accent_press'], padding=pad)


def success_button_qss() -> str:
    return _btn_base(THEME['success'], THEME['success_hover'],
                     '#00b378', fg='#001a13')


def warning_button_qss() -> str:
    return _btn_base(THEME['warning'], '#ffc340', '#b87f00', fg='#1a1200')


def danger_button_qss() -> str:
    return _btn_base(THEME['danger'], THEME['danger_hover'], '#cc2626')


def ghost_button_qss() -> str:
    """Subtle button on dark panels — used for low-priority actions."""
    return (
        f"QPushButton {{"
        f" background:{THEME['bg_input']}; color:{THEME['fg_primary']};"
        f" border:1px solid {THEME['border']}; border-radius:{THEME['radius']};"
        f" padding:9px 14px; font-weight:600; font-size:13px;"
        f"}}"
        f"QPushButton:hover {{ background:{THEME['bg_hover']}; border-color:{THEME['border_strong']}; }}"
        f"QPushButton:pressed {{ background:{THEME['bg_panel_alt']}; }}"
        f"QPushButton:disabled {{ color:{THEME['fg_dim']}; border-color:{THEME['border']}; }}"
    )


def cell_button_qss(checked: bool = False) -> str:
    """Compact grid-cell toggle button used by the cell selector."""
    if checked:
        bg = THEME['accent']
        border = THEME['accent']
        fg = '#ffffff'
    else:
        bg = THEME['bg_input']
        border = THEME['border']
        fg = THEME['fg_primary']
    return (
        f"QPushButton {{"
        f" background:{bg}; color:{fg};"
        f" border:1px solid {border}; border-radius:{THEME['radius_sm']};"
        f" padding:6px 0; font-weight:600; font-size:12px;"
        f"}}"
        f"QPushButton:hover {{ background:{THEME['bg_hover']}; }}"
        f"QPushButton:disabled {{"
        f" background:{THEME['bg_panel_alt']}; color:{THEME['fg_dim']};"
        f" border-color:{THEME['border']};"
        f"}}"
    )


def status_label_qss(colour: str = None) -> str:
    colour = colour or THEME['fg_muted']
    return (
        f"background:{THEME['bg_panel_alt']}; color:{colour};"
        f" font-size:13px; font-weight:600;"
        f" padding:9px; border:1px solid {THEME['border']};"
        f" border-radius:{THEME['radius']};"
    )


def log_widget_qss() -> str:
    return (
        f"QPlainTextEdit, QTextEdit {{"
        f" background:{THEME['bg_panel_alt']}; color:{THEME['fg_primary']};"
        f" border:1px solid {THEME['border']}; border-radius:{THEME['radius']};"
        f" padding:8px; font-family:'JetBrains Mono','Fira Code','Menlo',monospace;"
        f" font-size:12px;"
        f"}}"
    )


def combo_qss() -> str:
    return (
        f"QComboBox {{"
        f" background:{THEME['bg_input']}; color:{THEME['fg_primary']};"
        f" border:1px solid {THEME['border']}; border-radius:{THEME['radius_sm']};"
        f" padding:6px 10px; font-size:13px;"
        f"}}"
        f"QComboBox:hover {{ border-color:{THEME['border_strong']}; }}"
        f"QComboBox::drop-down {{ border:none; width:18px; }}"
        f"QComboBox QAbstractItemView {{"
        f" background:{THEME['bg_panel']}; color:{THEME['fg_primary']};"
        f" border:1px solid {THEME['border']}; selection-background-color:{THEME['accent']};"
        f"}}"
    )


def tab_qss() -> str:
    return (
        f"QTabWidget::pane {{"
        f" border:1px solid {THEME['border']}; border-radius:{THEME['radius']};"
        f" background:{THEME['bg_panel']}; top:-1px;"
        f"}}"
        f"QTabBar::tab {{"
        f" background:{THEME['bg_panel_alt']}; color:{THEME['fg_muted']};"
        f" padding:8px 16px; margin-right:2px;"
        f" border:1px solid {THEME['border']}; border-bottom:none;"
        f" border-top-left-radius:{THEME['radius_sm']}; border-top-right-radius:{THEME['radius_sm']};"
        f" font-size:12px; font-weight:600;"
        f"}}"
        f"QTabBar::tab:selected {{"
        f" background:{THEME['bg_panel']}; color:{THEME['fg_primary']};"
        f" border-bottom:1px solid {THEME['bg_panel']};"
        f"}}"
        f"QTabBar::tab:hover {{ color:{THEME['fg_primary']}; }}"
    )


def status_dot(state: str) -> str:
    """Unicode dot label for step status badges."""
    return {
        'idle':    '◌',
        'running': '●',
        'done':    '✓',
        'error':   '✕',
    }.get(state, '◌')


def status_colour(state: str) -> str:
    return {
        'idle':    THEME['fg_dim'],
        'running': THEME['warning'],
        'done':    THEME['success'],
        'error':   THEME['danger'],
    }.get(state, THEME['fg_dim'])

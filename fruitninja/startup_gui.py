#!/usr/bin/env python3
import os
os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)

import sys
import subprocess
import threading

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTextEdit, QGroupBox, QLineEdit, QTabWidget,
)
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QProcess
from PyQt5.QtGui import QFont, QColor, QTextCursor


# ── Step definitions ──────────────────────────────────────────────────────────

DEFAULT_ROBOT_IP = '192.168.0.194'

SOURCE = (
    'source /opt/ros/humble/setup.bash && '
    'source ~/ros2_ws/install/setup.bash && '
)


def make_steps(robot_ip: str) -> list:
    return [
        {
            'label':   'Step 1 — UR Driver',
            'desc':    f'Launch the UR3e robot driver  (robot_ip: {robot_ip})',
            'cmd':     SOURCE + (
                'ros2 launch ur_robot_driver ur_control.launch.py '
                f'ur_type:=ur3e robot_ip:={robot_ip} '
                'launch_rviz:=false '
                'initial_joint_controller:=scaled_joint_trajectory_controller'
            ),
            'oneshot': False,
            'note':    'Wait until "Robot ready to receive control commands" appears in the log.',
        },
        {
            'label':   'Step 2 — MoveIt',
            'desc':    'Launch MoveIt motion planning + RViz',
            'cmd':     SOURCE + (
                'ros2 launch ur_moveit_config ur_moveit.launch.py '
                'ur_type:=ur3e '
                'description_package:=fruitninja '
                'description_file:=ur3e_workcell.urdf.xacro '
                'launch_servo:=false '
                'launch_rviz:=true'
            ),
            'oneshot': False,
            'note':    'Wait for RViz to open and the robot model to appear.',
        },
        {
            'label':   'Step 3 — Planning Scene',
            'desc':    'Publish the planning scene (workcell / collision objects)',
            'cmd':     SOURCE + 'ros2 run fruitninja planning_scene',
            'oneshot': False,
            'note':    '',
        },
        {
            'label':   'Step 4 — Main GUI',
            'desc':    'Open the grid-control GUI for cutting',
            'cmd':     SOURCE + 'ros2 run fruitninja real_gui_points',
            'oneshot': False,
            'note':    '',
        },
        {
            'label':   'Step 5 — Fix Controller',
            'desc':    'Switch to scaled_joint_trajectory_controller (run once after Step 1)',
            'cmd':     SOURCE + (
                "ros2 service call /controller_manager/switch_controller "
                "controller_manager_msgs/srv/SwitchController "
                "\"{activate_controllers: ['scaled_joint_trajectory_controller'], "
                "deactivate_controllers: ['joint_trajectory_controller'], strictness: 2}\""
            ),
            'oneshot': True,
            'note':    'One-shot command — completes on its own.',
        },
    ]


# ── Status colours ────────────────────────────────────────────────────────────

STATUS_IDLE    = ('● Idle',    '#666666')
STATUS_RUNNING = ('● Running', '#00cc88')
STATUS_DONE    = ('✔ Done',    '#00ddff')
STATUS_FAILED  = ('✖ Failed',  '#ff4444')
STATUS_STOPPED = ('■ Stopped', '#e0a000')

TAB_STYLE = """
QTabWidget::pane {
    border: 1px solid #444;
    background: #0a0a0a;
}
QTabBar::tab {
    background: #2a2a2a;
    color: #aaa;
    padding: 6px 14px;
    border: 1px solid #444;
    border-bottom: none;
    font-size: 11px;
}
QTabBar::tab:selected {
    background: #1e1e1e;
    color: white;
    font-weight: bold;
}
QTabBar::tab:hover {
    background: #3a3a3a;
}
"""


# ── Step row widget ───────────────────────────────────────────────────────────

class StepRow(QObject):
    """Manages one step: QProcess, status label, start/stop buttons, output widget."""

    status_signal = pyqtSignal(str, str)   # text, colour

    def __init__(self, step: dict, on_output):
        super().__init__()
        self._step     = step
        self._on_output = on_output   # callable(label, text)

        # ── QProcess ──────────────────────────────────────────────────────────
        self._proc = QProcess()
        self._proc.setProcessChannelMode(QProcess.MergedChannels)
        self._proc.readyReadStandardOutput.connect(self._read_output)
        self._proc.finished.connect(self._on_finished)

        # ── output tab widget ─────────────────────────────────────────────────
        self.output_widget = QTextEdit()
        self.output_widget.setReadOnly(True)
        self.output_widget.setStyleSheet(
            'background:#0a0a0a; color:#00ee00;'
            'font-family:monospace; font-size:11px;'
        )

        # ── row container ─────────────────────────────────────────────────────
        self.container = QWidget()
        layout = QHBoxLayout(self.container)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(10)

        label_col = QVBoxLayout()
        label_col.setSpacing(1)

        title = QLabel(step['label'])
        title.setStyleSheet('color:white; font-size:13px; font-weight:bold;')
        label_col.addWidget(title)

        desc = QLabel(step['desc'])
        desc.setStyleSheet('color:#aaa; font-size:11px;')
        label_col.addWidget(desc)

        if step.get('note'):
            note = QLabel(step['note'])
            note.setStyleSheet('color:#888; font-size:10px; font-style:italic;')
            label_col.addWidget(note)

        layout.addLayout(label_col, stretch=1)

        self._status_lbl = QLabel(STATUS_IDLE[0])
        self._status_lbl.setFixedWidth(90)
        self._status_lbl.setAlignment(Qt.AlignCenter)
        self._apply_status(*STATUS_IDLE)
        layout.addWidget(self._status_lbl)

        self._btn_start = QPushButton('▶  Start')
        self._btn_start.setFixedWidth(90)
        self._btn_start.setStyleSheet(self._btn_style('#1a5c1a'))
        self._btn_start.clicked.connect(self.start)
        layout.addWidget(self._btn_start)

        self._btn_stop = QPushButton('■  Stop')
        self._btn_stop.setFixedWidth(90)
        self._btn_stop.setEnabled(False)
        self._btn_stop.setStyleSheet(self._btn_style('#5a1a1a'))
        self._btn_stop.clicked.connect(self.stop)
        layout.addWidget(self._btn_stop)

        self.status_signal.connect(self._apply_status)

    # ── process control ───────────────────────────────────────────────────────

    def start(self):
        if self._proc.state() != QProcess.NotRunning:
            return

        self.output_widget.clear()
        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(not self._step['oneshot'])
        self._apply_status(*STATUS_RUNNING)
        self._write_output(f'Starting: {self._step["label"]}\n')

        self._proc.start('/bin/bash', ['-c', self._step['cmd']])

    def stop(self):
        if self._proc.state() != QProcess.NotRunning:
            self._write_output('Stopping process…\n')
            self._proc.terminate()
            self._apply_status(*STATUS_STOPPED)

    def is_running(self) -> bool:
        return self._proc.state() != QProcess.NotRunning

    def kill(self):
        if self._proc.state() != QProcess.NotRunning:
            self._proc.kill()

    # ── QProcess slots ────────────────────────────────────────────────────────

    def _read_output(self):
        raw = self._proc.readAllStandardOutput().data().decode(errors='replace')
        self._write_output(raw)

    def _on_finished(self, exit_code, exit_status):
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        if exit_status == QProcess.CrashExit or exit_code not in (0, 15, -15):
            self._apply_status(*STATUS_FAILED)
            self._write_output(f'\nProcess exited with code {exit_code}\n')
        elif self._step['oneshot']:
            self._apply_status(*STATUS_DONE)
            self._write_output('\nCompleted.\n')
        else:
            # Only update to STOPPED if not already marked stopped by user
            if self._status_lbl.text() not in (STATUS_STOPPED[0],):
                self._apply_status(*STATUS_STOPPED)
            self._write_output('\nProcess ended.\n')

    # ── helpers ───────────────────────────────────────────────────────────────

    def _write_output(self, text: str):
        self.output_widget.moveCursor(QTextCursor.End)
        self.output_widget.insertPlainText(text)
        self.output_widget.moveCursor(QTextCursor.End)
        self._on_output(self._step['label'], text)

    @staticmethod
    def _btn_style(colour):
        return (
            f'QPushButton{{background:{colour};color:white;border-radius:5px;'
            f'padding:7px 10px;font-size:12px;font-weight:bold;}}'
            f'QPushButton:hover{{background:{colour}cc;}}'
            f'QPushButton:disabled{{background:#2a2a2a;color:#555;}}'
        )

    def _apply_status(self, text: str, colour: str):
        self._status_lbl.setText(text)
        self._status_lbl.setStyleSheet(
            f'color:{colour}; font-size:12px; font-weight:bold;'
            f'background:#111; border-radius:4px; padding:4px 6px;'
        )


# ── Main window ───────────────────────────────────────────────────────────────

class StartupWindow(QMainWindow):
    _log_sig = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle('FruitNinja — Startup Launcher')
        self.setMinimumSize(820, 700)
        self.setStyleSheet('background:#1e1e1e; color:white;')

        self._step_rows    = []
        self._steps_layout = None
        self._tab_widget   = None
        self._build_ui()

    def _build_ui(self):
        root_w = QWidget()
        self.setCentralWidget(root_w)
        root = QVBoxLayout(root_w)
        root.setSpacing(8)
        root.setContentsMargins(14, 14, 14, 14)

        # Title
        title = QLabel('UR3e Startup Sequence')
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            'color:white; font-size:18px; font-weight:bold;'
            'padding:8px; border-bottom:1px solid #444;'
        )
        root.addWidget(title)

        subtitle = QLabel('Run each step in order. Steps 1 and 2 must be fully ready before continuing.')
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet('color:#888; font-size:11px; padding-bottom:4px;')
        root.addWidget(subtitle)

        # ── Robot IP config ───────────────────────────────────────────────────
        ip_group = QGroupBox('Robot IP Address')
        ip_group.setStyleSheet(self._group_style())
        ip_layout = QHBoxLayout(ip_group)
        ip_layout.setContentsMargins(10, 8, 10, 8)

        ip_label = QLabel('IP:')
        ip_label.setStyleSheet('color:#ccc; font-size:13px;')
        ip_label.setFixedWidth(24)
        ip_layout.addWidget(ip_label)

        self._ip_input = QLineEdit(DEFAULT_ROBOT_IP)
        self._ip_input.setFixedWidth(160)
        self._ip_input.setFont(QFont('monospace', 13))
        self._ip_input.setStyleSheet(
            'background:#2a2a3a; color:#00ddff;'
            'border:1px solid #555; border-radius:4px;'
            'padding:5px 10px; font-size:13px;'
        )
        self._ip_input.setPlaceholderText('e.g. 192.168.0.194')
        ip_layout.addWidget(self._ip_input)

        apply_btn = QPushButton('Apply')
        apply_btn.setFixedWidth(80)
        apply_btn.setStyleSheet(
            'QPushButton{background:#1a3a5c;color:white;border-radius:4px;'
            'padding:6px 10px;font-size:12px;font-weight:bold;}'
            'QPushButton:hover{background:#2a5a8ccc;}'
        )
        apply_btn.setToolTip('Rebuild step commands with the new IP.\nStop running processes first.')
        apply_btn.clicked.connect(self._apply_ip)
        ip_layout.addWidget(apply_btn)

        ping_btn = QPushButton('Ping')
        ping_btn.setFixedWidth(70)
        ping_btn.setStyleSheet(
            'QPushButton{background:#2a2a3a;color:#ccc;border:1px solid #555;border-radius:4px;'
            'padding:6px 10px;font-size:12px;font-weight:bold;}'
            'QPushButton:hover{background:#3a3a5acc;}'
        )
        ping_btn.setToolTip('Ping the robot IP to check connectivity.')
        ping_btn.clicked.connect(self._ping_ip)
        ip_layout.addWidget(ping_btn)

        self._ip_status = QLabel('')
        self._ip_status.setStyleSheet('color:#888; font-size:11px; padding-left:8px;')
        ip_layout.addWidget(self._ip_status)
        ip_layout.addStretch()

        root.addWidget(ip_group)

        # ── Steps group ───────────────────────────────────────────────────────
        self._steps_group = QGroupBox('Steps')
        self._steps_group.setStyleSheet(self._group_style())
        self._steps_layout = QVBoxLayout(self._steps_group)
        self._steps_layout.setSpacing(4)
        root.addWidget(self._steps_group)

        # Stop-all button
        stop_all_btn = QPushButton('■  Stop All Running Processes')
        stop_all_btn.setStyleSheet(
            'QPushButton{background:#5a1a1a;color:white;border-radius:6px;'
            'padding:10px;font-size:13px;font-weight:bold;}'
            'QPushButton:hover{background:#7a2a2acc;}'
        )
        stop_all_btn.clicked.connect(self._stop_all)
        root.addWidget(stop_all_btn)

        # ── Output tabs ───────────────────────────────────────────────────────
        self._tab_widget = QTabWidget()
        self._tab_widget.setStyleSheet(TAB_STYLE)
        self._tab_widget.setMinimumHeight(220)
        root.addWidget(self._tab_widget, stretch=1)

        # Populate steps + tabs with default IP
        self._populate_steps(DEFAULT_ROBOT_IP)

    # ── IP helpers ────────────────────────────────────────────────────────────

    def _apply_ip(self):
        ip = self._ip_input.text().strip()
        if not ip:
            self._set_ip_status('Enter a valid IP.', '#ff4444')
            return
        if any(row.is_running() for row in self._step_rows):
            self._set_ip_status('Stop all processes first.', '#e0a000')
            return
        self._populate_steps(ip)
        self._set_ip_status(f'Applied — using {ip}', '#00cc88')

    def _ping_ip(self):
        ip = self._ip_input.text().strip()
        if not ip:
            self._set_ip_status('Enter an IP to ping.', '#ff4444')
            return
        self._set_ip_status(f'Pinging {ip}…', '#aaa')
        threading.Thread(target=self._do_ping, args=(ip,), daemon=True).start()

    def _do_ping(self, ip: str):
        try:
            result = subprocess.run(
                ['ping', '-c', '3', '-W', '2', ip],
                capture_output=True, text=True,
            )
            # Emit ping output to the first tab
            for line in result.stdout.splitlines():
                self._append_to_tab('Step 1 — UR Driver', f'[Ping] {line}\n')
            if result.returncode == 0:
                rtt = ''
                for line in result.stdout.splitlines():
                    if 'rtt' in line or 'round-trip' in line:
                        rtt = line.split('=')[-1].strip()
                        break
                msg   = f'Reachable  {("— " + rtt) if rtt else ""}'.strip()
                color = '#00cc88'
            else:
                msg   = f'{ip} unreachable'
                color = '#ff4444'
        except Exception as e:
            msg   = f'Ping error: {e}'
            color = '#ff4444'
        self._set_ip_status(msg, color)

    def _set_ip_status(self, text: str, colour: str):
        self._ip_status.setText(text)
        self._ip_status.setStyleSheet(
            f'color:{colour}; font-size:11px; padding-left:8px;'
        )

    # ── Step population ───────────────────────────────────────────────────────

    def _populate_steps(self, robot_ip: str):
        # Kill + clear old rows
        for row in self._step_rows:
            row.kill()
            row.container.setParent(None)
        self._step_rows.clear()

        while self._steps_layout.count():
            item = self._steps_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._tab_widget.clear()

        steps = make_steps(robot_ip)
        for i, step in enumerate(steps):
            row = StepRow(step, self._append_to_tab)
            self._step_rows.append(row)
            self._steps_layout.addWidget(row.container)

            if i < len(steps) - 1:
                sep = QWidget()
                sep.setFixedHeight(1)
                sep.setStyleSheet('background:#333;')
                self._steps_layout.addWidget(sep)

            # Add a tab for this step's output
            self._tab_widget.addTab(row.output_widget, step['label'])

    def _append_to_tab(self, label: str, text: str):
        """Called from StepRow (main thread via QProcess signals) to write to a tab."""
        for i in range(self._tab_widget.count()):
            if self._tab_widget.tabText(i) == label:
                widget = self._tab_widget.widget(i)
                widget.moveCursor(QTextCursor.End)
                widget.insertPlainText(text)
                widget.moveCursor(QTextCursor.End)
                break

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _group_style():
        return (
            'QGroupBox{color:white;font-weight:bold;'
            'border:1px solid #444;border-radius:4px;margin-top:8px;}'
            'QGroupBox::title{subcontrol-origin:margin;left:8px;}'
        )

    def _stop_all(self):
        for row in self._step_rows:
            row.stop()

    def closeEvent(self, event):
        for row in self._step_rows:
            row.kill()
        event.accept()


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = StartupWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

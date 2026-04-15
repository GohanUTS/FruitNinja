#!/usr/bin/env python3
import os
os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)

import sys
import subprocess
import threading

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTextEdit, QGroupBox, QLineEdit,
)
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QFont


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


# ── Step row widget ───────────────────────────────────────────────────────────

class StepRow(QObject):
    """Manages one step: process, status label, start/stop buttons."""

    log_signal    = pyqtSignal(str)
    status_signal = pyqtSignal(str, str)   # text, colour

    def __init__(self, step: dict, parent_log_signal):
        super().__init__()
        self._step    = step
        self._proc    = None
        self._thread  = None

        # Forward log lines to the main window's log
        self.log_signal.connect(parent_log_signal)

        # ── widgets ───────────────────────────────────────────────────────────
        self.container = QWidget()
        layout = QHBoxLayout(self.container)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(10)

        # Label column
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

        # Status dot
        self._status_lbl = QLabel(STATUS_IDLE[0])
        self._status_lbl.setFixedWidth(90)
        self._status_lbl.setAlignment(Qt.AlignCenter)
        self._apply_status(*STATUS_IDLE)
        layout.addWidget(self._status_lbl)

        # Start button
        self._btn_start = QPushButton('▶  Start')
        self._btn_start.setFixedWidth(90)
        self._btn_start.setStyleSheet(self._btn_style('#1a5c1a'))
        self._btn_start.clicked.connect(self.start)
        layout.addWidget(self._btn_start)

        # Stop button
        self._btn_stop = QPushButton('■  Stop')
        self._btn_stop.setFixedWidth(90)
        self._btn_stop.setEnabled(False)
        self._btn_stop.setStyleSheet(self._btn_style('#5a1a1a'))
        self._btn_stop.clicked.connect(self.stop)
        layout.addWidget(self._btn_stop)

        # Connect status signal
        self.status_signal.connect(self._apply_status)

    # ── process control ───────────────────────────────────────────────────────

    def start(self):
        if self._proc and self._proc.poll() is None:
            return   # already running

        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(not self._step['oneshot'])
        self.status_signal.emit(*STATUS_RUNNING)
        self.log_signal.emit(f'[{self._step["label"]}] Starting in new terminal…')

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            label = self._step['label']
            cmd   = self._step['cmd']

            # Wrap the command so the terminal stays open on failure
            # (for one-shot steps the window closes automatically on success)
            if self._step['oneshot']:
                inner = f'bash -c {repr(cmd)}'
            else:
                inner = (
                    f'bash -c {repr(cmd + "; echo; echo --- Process ended, press Enter to close ---; read")}'
                )

            # Try gnome-terminal first, fall back to xterm
            terminal_cmd = (
                f'gnome-terminal --title={repr(label)} -- bash -c {repr(inner)} '
                f'|| xterm -title {repr(label)} -e bash -c {repr(inner)}'
            )

            self._proc = subprocess.Popen(
                terminal_cmd,
                shell=True,
                executable='/bin/bash',
            )
            self._proc.wait()   # waits for the terminal process itself to exit
            rc = self._proc.returncode

            if rc == 0:
                if self._step['oneshot']:
                    self.status_signal.emit(*STATUS_DONE)
                    self.log_signal.emit(f'[{label}] Terminal closed.')
                else:
                    self.status_signal.emit(*STATUS_STOPPED)
                    self.log_signal.emit(f'[{label}] Terminal closed.')
            else:
                self.status_signal.emit(*STATUS_FAILED)
                self.log_signal.emit(f'[{label}] Terminal exited with code {rc}')
        except Exception as e:
            self.status_signal.emit(*STATUS_FAILED)
            self.log_signal.emit(f'[{self._step["label"]}] Error: {e}')
        finally:
            self._reset_buttons()

    def stop(self):
        if self._proc and self._proc.poll() is None:
            self.log_signal.emit(f'[{self._step["label"]}] Closing terminal…')
            self._proc.terminate()
            self.status_signal.emit(*STATUS_STOPPED)

    def _reset_buttons(self):
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)

    # ── style helpers ─────────────────────────────────────────────────────────

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
        self.setMinimumSize(760, 640)
        self.setStyleSheet('background:#1e1e1e; color:white;')

        self._log_sig.connect(self._append_log)
        self._step_rows = []
        self._steps_layout = None   # filled in _build_ui
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
        ip_group.setStyleSheet(
            'QGroupBox{color:white;font-weight:bold;'
            'border:1px solid #444;border-radius:4px;margin-top:8px;}'
            'QGroupBox::title{subcontrol-origin:margin;left:8px;}'
        )
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
        self._steps_group.setStyleSheet(
            'QGroupBox{color:white;font-weight:bold;'
            'border:1px solid #444;border-radius:4px;margin-top:8px;}'
            'QGroupBox::title{subcontrol-origin:margin;left:8px;}'
        )
        self._steps_layout = QVBoxLayout(self._steps_group)
        self._steps_layout.setSpacing(4)
        self._populate_steps(DEFAULT_ROBOT_IP)
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

        # Log area
        log_group = QGroupBox('Output Log')
        log_group.setStyleSheet(
            'QGroupBox{color:white;font-weight:bold;'
            'border:1px solid #444;border-radius:4px;margin-top:8px;}'
            'QGroupBox::title{subcontrol-origin:margin;left:8px;}'
        )
        log_layout = QVBoxLayout(log_group)

        self._log_widget = QTextEdit()
        self._log_widget.setReadOnly(True)
        self._log_widget.setStyleSheet(
            'background:#0a0a0a; color:#00ee00;'
            'font-family:monospace; font-size:11px;'
        )
        log_layout.addWidget(self._log_widget)
        root.addWidget(log_group)

    # ── IP helpers ────────────────────────────────────────────────────────────

    def _apply_ip(self):
        ip = self._ip_input.text().strip()
        if not ip:
            self._ip_status.setText('Enter a valid IP.')
            self._ip_status.setStyleSheet('color:#ff4444; font-size:11px; padding-left:8px;')
            return

        any_running = any(
            row._proc and row._proc.poll() is None
            for row in self._step_rows
        )
        if any_running:
            self._ip_status.setText('Stop all processes first.')
            self._ip_status.setStyleSheet('color:#e0a000; font-size:11px; padding-left:8px;')
            return

        self._populate_steps(ip)
        self._ip_status.setText(f'Applied — using {ip}')
        self._ip_status.setStyleSheet('color:#00cc88; font-size:11px; padding-left:8px;')
        self._append_log(f'[Config] Robot IP set to {ip}')

    def _ping_ip(self):
        ip = self._ip_input.text().strip()
        if not ip:
            self._ip_status.setText('Enter an IP to ping.')
            self._ip_status.setStyleSheet('color:#ff4444; font-size:11px; padding-left:8px;')
            return
        self._ip_status.setText(f'Pinging {ip}…')
        self._ip_status.setStyleSheet('color:#aaa; font-size:11px; padding-left:8px;')
        self._append_log(f'[Ping] Pinging {ip}…')
        threading.Thread(target=self._do_ping, args=(ip,), daemon=True).start()

    def _do_ping(self, ip: str):
        try:
            result = subprocess.run(
                ['ping', '-c', '3', '-W', '2', ip],
                capture_output=True, text=True,
            )
            for line in result.stdout.splitlines():
                self._log_sig.emit(f'[Ping] {line}')
            if result.returncode == 0:
                # Extract round-trip time from the summary line
                rtt = ''
                for line in result.stdout.splitlines():
                    if 'rtt' in line or 'round-trip' in line:
                        rtt = line.split('=')[-1].strip()
                        break
                msg   = f'Reachable  {("— " + rtt) if rtt else ""}'.strip()
                style = 'color:#00cc88; font-size:11px; padding-left:8px;'
            else:
                msg   = f'{ip} unreachable'
                style = 'color:#ff4444; font-size:11px; padding-left:8px;'
        except Exception as e:
            msg   = f'Ping error: {e}'
            style = 'color:#ff4444; font-size:11px; padding-left:8px;'

        # Update status label from the main thread
        self._ip_status.setText(msg)
        self._ip_status.setStyleSheet(style)

    def _populate_steps(self, robot_ip: str):
        # Clear existing rows
        for row in self._step_rows:
            row.container.setParent(None)
        self._step_rows.clear()

        # Remove all widgets from the layout cleanly
        while self._steps_layout.count():
            item = self._steps_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        steps = make_steps(robot_ip)
        for i, step in enumerate(steps):
            row = StepRow(step, self._log_sig)
            self._step_rows.append(row)
            self._steps_layout.addWidget(row.container)

            if i < len(steps) - 1:
                sep = QWidget()
                sep.setFixedHeight(1)
                sep.setStyleSheet('background:#333;')
                self._steps_layout.addWidget(sep)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _append_log(self, text: str):
        self._log_widget.append(text)

    def _stop_all(self):
        for row in self._step_rows:
            row.stop()

    def closeEvent(self, event):
        self._stop_all()
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

"""
PySide6 GUI Dashboard for AtomCode2API.

Provides:
- Connection info panel (baseURL, API Key, models) with copy buttons
- Scrollable log viewer (auto-trims old lines)
"""

import logging
import sys

from PySide6.QtCore import Qt, QObject, QTimer, Signal, Slot
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

try:
    from config import settings
except ModuleNotFoundError:
    from atomcode2api.config import settings  # noqa: F811

# ── Constants ──────────────────────────────────────────────────────────

MAX_LOG_LINES = 5000
"""Number of log lines kept in the viewer before old lines are trimmed."""

LOG_FLUSH_INTERVAL_MS = 300
"""Milliseconds between flushing the log buffer to the text widget."""

# ── Stylesheet ────────────────────────────────────────────────────────

STYLESHEET = """
QMainWindow {
    background-color: #1e1e2e;
}
QLabel {
    color: #cdd6f4;
    font-size: 13px;
}
QGroupBox {
    font-size: 14px;
    font-weight: bold;
    color: #89b4fa;
    border: 1px solid #45475a;
    border-radius: 8px;
    margin-top: 14px;
    padding: 12px 8px 8px;
    background-color: #181825;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}
QPushButton {
    background-color: #45475a;
    color: #cdd6f4;
    border: 1px solid #585b70;
    border-radius: 5px;
    padding: 4px 14px;
    font-size: 12px;
}
QPushButton:hover {
    background-color: #585b70;
    border-color: #89b4fa;
}
QPushButton:pressed {
    background-color: #313244;
}
QPlainTextEdit {
    background-color: #11111b;
    color: #a6adc8;
    border: 1px solid #45475a;
    border-radius: 6px;
    font-family: "Cascadia Code", "Consolas", "Courier New", monospace;
    font-size: 12px;
    padding: 6px;
    selection-background-color: #45475a;
}
"""


# ── Log signal ────────────────────────────────────────────────────────

class LogSignal(QObject):
    """Signal emitted when a new log line arrives from any thread."""

    new_log = Signal(str)


log_signal = LogSignal()


# ── Qt Log Handler ────────────────────────────────────────────────────

class QtLogHandler(logging.Handler):
    """A logging.Handler that emits log records as Qt signals.

    Attach it to any logger (or the root logger) to push logs into the
    GUI text viewer in a thread-safe manner.
    """

    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
        self.setFormatter(logging.Formatter(
            "%(asctime)s  %(levelname)-8s %(message)s",
            datefmt="%H:%M:%S",
        ))

    def emit(self, record):
        try:
            msg = self.format(record)
            log_signal.new_log.emit(msg)
        except Exception:
            self.handleError(record)


# ── Dashboard (main window) ───────────────────────────────────────────

class Dashboard(QMainWindow):
    """Main application window for AtomCode2API Dashboard."""

    # The maximum number of blocks (≈lines) kept in the log viewer.
    _log_max_blocks = MAX_LOG_LINES

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AtomCode2API  Dashboard")
        self.setMinimumSize(780, 620)
        self.resize(900, 700)

        # ── Log buffer (avoids hammering QPlainTextEdit on every line) ─
        self._log_buffer: list[str] = []

        # ── Central widget ─────────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(10)

        # ── Sections ───────────────────────────────────────────────
        layout.addWidget(self._build_info_section())
        layout.addWidget(self._build_log_section(), stretch=1)

        # ── Log handler ────────────────────────────────────────────
        self._setup_logging()

        # ── Log flush timer (batched writes to text widget) ─────────
        self._flush_timer = QTimer(self)
        self._flush_timer.timeout.connect(self._flush_log_buffer)
        self._flush_timer.start(LOG_FLUSH_INTERVAL_MS)

    # ── UI builders ────────────────────────────────────────────────

    def _build_info_section(self) -> QGroupBox:
        group = QGroupBox("连接信息")
        grid = QGridLayout(group)
        grid.setVerticalSpacing(8)
        grid.setHorizontalSpacing(10)

        # Build a readonly value label.
        def value_label(text: str) -> QLabel:
            lbl = QLabel(text)
            lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            lbl.setStyleSheet(
                "color: #a6e3a1; background: #181825; "
                "border: 1px solid #313244; border-radius: 4px; "
                "padding: 3px 8px; font-family: Consolas, monospace;"
            )
            return lbl

        # Build a copy button.
        def copy_btn(copy_text: str) -> QPushButton:
            btn = QPushButton("复制")
            btn.setFixedWidth(56)
            btn.clicked.connect(lambda: self._copy_to_clipboard(copy_text, btn))
            return btn

        # ── Row 0: Base URL ────────────────────────────────────────
        base_url = f"http://127.0.0.1:{settings.port}/v1"
        grid.addWidget(QLabel("Base URL:"), 0, 0)
        grid.addWidget(value_label(base_url), 0, 1)
        grid.addWidget(copy_btn(base_url), 0, 2)

        # ── Row 1: API Key ─────────────────────────────────────────
        api_key = settings.api_key or "(未设置)"
        grid.addWidget(QLabel("API Key:"), 1, 0)
        grid.addWidget(value_label(api_key), 1, 1)
        grid.addWidget(copy_btn(settings.api_key or ""), 1, 2)

        # ── Row 2: Model 1 ─────────────────────────────────────────
        model_1 = "deepseek-v4-flash"
        grid.addWidget(QLabel("Model 1:"), 2, 0)
        grid.addWidget(value_label(model_1), 2, 1)
        grid.addWidget(copy_btn(model_1), 2, 2)

        # ── Row 3: Model 2 ─────────────────────────────────────────
        model_2 = "Qwen/Qwen3-VL-8B-Instruct"
        grid.addWidget(QLabel("Model 2:"), 3, 0)
        grid.addWidget(value_label(model_2), 3, 1)
        grid.addWidget(copy_btn(model_2), 3, 2)

        return group

    def _build_log_section(self) -> QGroupBox:
        group = QGroupBox("运行日志")
        layout = QVBoxLayout(group)

        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setLineWrapMode(QPlainTextEdit.NoWrap)
        layout.addWidget(self._log_view)

        return group

    # ── Logging setup ──────────────────────────────────────────────

    def _setup_logging(self):
        """Attach QtLogHandler to the root logger and connect signal."""
        handler = QtLogHandler()
        handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)

        # Ensure uvicorn logs also flow through our handler.
        logging.getLogger("uvicorn").setLevel(logging.INFO)
        logging.getLogger("uvicorn.access").setLevel(logging.INFO)
        logging.getLogger("uvicorn.error").setLevel(logging.INFO)

        log_signal.new_log.connect(self._buffer_log)

    @Slot(str)
    def _buffer_log(self, msg: str):
        """Buffer a log line (avoids hammering QPlainTextEdit per line)."""
        self._log_buffer.append(msg)

    def _flush_log_buffer(self):
        """Flush the accumulated log buffer into the text widget.

        Called by _flush_timer every LOG_FLUSH_INTERVAL_MS ms,
        so the GUI only updates ~3 times/sec instead of on every log line.
        """
        if not self._log_buffer:
            return

        view = self._log_view
        lines = self._log_buffer
        self._log_buffer = []

        # ── Trim if at limit ────────────────────────────────────────
        if view.blockCount() >= self._log_max_blocks:
            # Join buffered text, trim oldest blocks in one go.
            text = "\n".join(lines)
            cursor = view.textCursor()
            cursor.movePosition(QTextCursor.Start)
            # Remove ~1/3 of the oldest blocks each time we hit the limit.
            remove_n = self._log_max_blocks // 3
            for _ in range(remove_n):
                cursor.movePosition(QTextCursor.Down, QTextCursor.KeepAnchor)
            cursor.removeSelectedText()
            cursor.deleteChar()  # trailing newline
            cursor.movePosition(QTextCursor.End)
            view.setTextCursor(cursor)

        # ── Batch-append all buffered lines ─────────────────────────
        view.setUpdatesEnabled(False)
        for line in lines:
            view.appendPlainText(line)
        view.setUpdatesEnabled(True)

        # ── Auto-scroll ─────────────────────────────────────────────
        scrollbar = view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    # ── Clipboard helper ───────────────────────────────────────────

    def _copy_to_clipboard(self, text: str, btn: QPushButton):
        """Copy *text* to the system clipboard and briefly flash the button."""
        if not text:
            return
        QApplication.clipboard().setText(text)
        original = btn.text()
        btn.setText("✓ 已复制")
        btn.setEnabled(False)
        QTimer.singleShot(1200, lambda: self._reset_btn(btn, original))

    def _reset_btn(self, btn: QPushButton, original: str):
        btn.setText(original)
        btn.setEnabled(True)


# ── Entry point ───────────────────────────────────────────────────────

def main():
    """Create the QApplication and show the Dashboard."""
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    window = Dashboard()
    window.show()
    sys.exit(app.exec())

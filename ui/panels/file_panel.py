"""Panel operasi file untuk UI."""

from PyQt6.QtWidgets import QPushButton, QLabel, QVBoxLayout, QWidget, QMessageBox
from PyQt6.QtCore import Qt
from ui.widgets.collapsible_box import CollapsibleBox


class FilePanel(CollapsibleBox):
    """Panel untuk operasi file (open, new session)."""

    def __init__(self, main_window):
        super().__init__("File Operations")
        self.main_window = main_window
        self.init_ui()

    def init_ui(self):
        """Inisialisasi UI panel file."""
        self.btn_open = QPushButton("Open Raster File")
        self.btn_open.clicked.connect(self.main_window.open_file)
        self.addWidget(self.btn_open)

        self.btn_new_session = QPushButton("New Session")
        self.btn_new_session.clicked.connect(self.main_window.new_session)
        self.btn_new_session.setVisible(False)
        self.addWidget(self.btn_new_session)

        self.label_file = QLabel("No file loaded")
        self.label_file.setWordWrap(True)
        self.addWidget(self.label_file)

    def update_file_label(self, text):
        """Update label informasi file."""
        self.label_file.setText(text)

    def set_new_session_visible(self, visible):
        """Set visibility tombol new session."""
        self.btn_new_session.setVisible(visible)

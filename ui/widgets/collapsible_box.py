"""Widget collapsible box untuk UI."""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QToolButton, QFormLayout, QHBoxLayout
from PyQt6.QtCore import Qt


class CollapsibleBox(QWidget):
    """Widget box yang bisa di-collapse/expand."""

    def __init__(self, title="", parent=None, nested=False):
        super().__init__(parent)
        self.nested = nested

        self.toggle_button = QToolButton()
        self.toggle_button.setText(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(True)

        if nested:
            self.toggle_button.setStyleSheet("""
                QToolButton {
                    border: none;
                    font-weight: normal;
                    text-align: left;
                    padding: 3px 5px 3px 3px;
                    background-color: #2a2a2a;
                    font-size: 10px;
                    spacing: 4px;
                }
                QToolButton:hover {
                    background-color: #3a3a3a;
                }
            """)
        else:
            self.toggle_button.setStyleSheet("""
                QToolButton {
                    border: none;
                    font-weight: bold;
                    text-align: left;
                    padding: 5px 5px 5px 5px;
                    background-color: #3a3a3a;
                    spacing: 6px;
                }
                QToolButton:hover {
                    background-color: #4a4a4a;
                }
            """)

        self.toggle_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.toggle_button.setArrowType(Qt.ArrowType.DownArrow)
        self.toggle_button.clicked.connect(self.toggle)

        self.content_area = QWidget()
        self.content_layout = QVBoxLayout()
        self.content_layout.setContentsMargins(5, 5, 5, 5)
        self.content_area.setLayout(self.content_layout)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(self.toggle_button)
        main_layout.addWidget(self.content_area)

        self.setLayout(main_layout)

    def toggle(self):
        """Toggle visibility dari content area."""
        checked = self.toggle_button.isChecked()
        arrow_type = Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        self.toggle_button.setArrowType(arrow_type)
        self.content_area.setVisible(checked)

    def setContentLayout(self, layout):
        """Set layout untuk content area."""
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if isinstance(layout, (QVBoxLayout, QHBoxLayout, QFormLayout)):
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    self.content_layout.addWidget(item.widget())
                elif item.layout():
                    self.content_layout.addLayout(item.layout())

    def addWidget(self, widget):
        """Tambahkan widget ke content area."""
        self.content_layout.addWidget(widget)

    def addLayout(self, layout):
        """Tambahkan layout ke content area."""
        self.content_layout.addLayout(layout)

"""Panel ekspor data training."""

import logging
from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QCheckBox, QWidget, QFormLayout, QSizePolicy
)
from PyQt6.QtCore import Qt
from ui.widgets.collapsible_box import CollapsibleBox


class ExportPanel(CollapsibleBox):
    """Panel untuk mengekspor data training."""
    def __init__(self, main_window):
        super().__init__("Training Data Export", nested=True)
        self.main_window = main_window
        self.logger = logging.getLogger(__name__)
        self.init_ui()

    def init_ui(self):
        """Inisialisasi UI panel ekspor."""

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Output directory selection
        output_dir_label = QLabel("Output Directory:")
        layout.addWidget(output_dir_label)

        output_dir_layout = QHBoxLayout()
        self.label_export_dir = QLabel("Not selected")
        self.label_export_dir.setWordWrap(True)
        self.label_export_dir.setStyleSheet("QLabel { color: gray; font-style: italic; }")
        output_dir_layout.addWidget(self.label_export_dir, 1)

        self.btn_browse_export_dir = QPushButton("Browse")
        self.btn_browse_export_dir.setMaximumWidth(100)
        self.btn_browse_export_dir.setMinimumHeight(26)
        self.btn_browse_export_dir.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_browse_export_dir.clicked.connect(self.main_window.browse_export_directory)
        output_dir_layout.addWidget(self.btn_browse_export_dir)
        layout.addLayout(output_dir_layout)

        # Export options
        self.check_export_tiles = QCheckBox("Export image tiles")
        self.check_export_tiles.setChecked(True)
        self.check_export_tiles.stateChanged.connect(self.main_window.update_export_info_labels)
        layout.addWidget(self.check_export_tiles)

        self.check_export_mask = QCheckBox("Export segmentation mask for layer")
        self.check_export_mask.setChecked(False)
        layout.addWidget(self.check_export_mask)

        # RGB/Grayscale format selection
        self.check_export_grayscale = QCheckBox("Export as grayscale (convert RGB to grayscale)")
        self.check_export_grayscale.setChecked(False)
        layout.addWidget(self.check_export_grayscale)

        # Parameters info (read-only, from processing parameters)
        params_info_label = QLabel("Selected in 'Processing Parameters':")
        layout.addWidget(params_info_label)
        
        params_info_layout = QFormLayout()
        params_info_layout.setContentsMargins(20, 0, 0, 5)

        self.label_export_overlap = QLabel("10 %")
        self.label_export_overlap.setStyleSheet("QLabel { color: #666; font-size: 10px; }")
        params_info_layout.addRow("Tiles Overlap:", self.label_export_overlap)

        self.label_export_tile_size = QLabel("640 px")
        self.label_export_tile_size.setStyleSheet("QLabel { color: #666; font-size: 10px; }")
        params_info_layout.addRow("Tile Size:", self.label_export_tile_size)

        self.label_export_resolution = QLabel("10 cm/px")
        self.label_export_resolution.setStyleSheet("QLabel { color: #666; font-size: 10px; }")
        params_info_layout.addRow("Resolution:", self.label_export_resolution)

        layout.addLayout(params_info_layout)

        # Export button
        self.btn_export_training_data = QPushButton("Export Training Data")
        self.btn_export_training_data.setMinimumHeight(30)
        self.btn_export_training_data.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_export_training_data.clicked.connect(self.main_window.export_training_data)
        self.btn_export_training_data.setEnabled(False)
        layout.addWidget(self.btn_export_training_data)

        self.setContentLayout(layout)

    def update_info_labels(self, overlap, tile_size, resolution):
        self.label_export_overlap.setText(overlap)
        self.label_export_tile_size.setText(tile_size)
        self.label_export_resolution.setText(resolution)

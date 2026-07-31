"""Centroid detection panel."""

import logging
from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QWidget, QFormLayout, QSizePolicy, QSpinBox
)
from PyQt6.QtCore import Qt
from ui.widgets.collapsible_box import CollapsibleBox


class CentroidPanel(CollapsibleBox):
    """Panel for centroid conversion and management."""
    def __init__(self, main_window):
        super().__init__("Centroid Detector")
        self.main_window = main_window
        self.logger = logging.getLogger(__name__)
        self.init_ui()

    def init_ui(self):
        """Inisialisasi UI panel centroid."""

        centroid_layout = QVBoxLayout()
        centroid_layout.setContentsMargins(6, 6, 6, 6)
        centroid_layout.setSpacing(10)

        # 1. Info label
        info_label = QLabel("Convert bounding boxes to centroid points for easier post-processing and counting.")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("QLabel { color: #888; font-size: 10px; padding: 4px; }")
        centroid_layout.addWidget(info_label)

        # 2. Convert button
        self.btn_convert_to_centroids = QPushButton("Convert to Centroids")
        self.btn_convert_to_centroids.setEnabled(False)
        self.btn_convert_to_centroids.setMinimumHeight(32)
        self.btn_convert_to_centroids.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_convert_to_centroids.clicked.connect(self.main_window.convert_to_centroids)
        centroid_layout.addWidget(self.btn_convert_to_centroids)

        # 3. Centroid count info
        self.label_centroid_count = QLabel("Centroids: 0")
        self.label_centroid_count.setStyleSheet("QLabel { color: #ccc; font-size: 11px; padding: 4px; }")
        centroid_layout.addWidget(self.label_centroid_count)

        # 4. Edit mode buttons
        edit_label = QLabel("Manual Edit Mode:")
        edit_label.setStyleSheet("QLabel { color: #aaa; font-size: 10px; margin-top: 8px; }")
        centroid_layout.addWidget(edit_label)

        edit_buttons_layout = QHBoxLayout()
        edit_buttons_layout.setSpacing(6)

        self.btn_add_centroid = QPushButton("Add Point")
        self.btn_add_centroid.setEnabled(False)
        self.btn_add_centroid.setCheckable(True)
        self.btn_add_centroid.setMinimumHeight(28)
        self.btn_add_centroid.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_add_centroid.clicked.connect(self.main_window.toggle_add_centroid_mode)
        edit_buttons_layout.addWidget(self.btn_add_centroid)

        self.btn_delete_centroid = QPushButton("Delete Point")
        self.btn_delete_centroid.setEnabled(False)
        self.btn_delete_centroid.setCheckable(True)
        self.btn_delete_centroid.setMinimumHeight(28)
        self.btn_delete_centroid.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_delete_centroid.clicked.connect(self.main_window.toggle_delete_centroid_mode)
        edit_buttons_layout.addWidget(self.btn_delete_centroid)

        centroid_layout.addLayout(edit_buttons_layout)

        # 5. Save to shapefile button
        self.btn_save_centroids = QPushButton("Save to Shapefile")
        self.btn_save_centroids.setEnabled(False)
        self.btn_save_centroids.setMinimumHeight(32)
        self.btn_save_centroids.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_save_centroids.clicked.connect(self.main_window.save_centroids_to_shapefile)
        centroid_layout.addWidget(self.btn_save_centroids)

        # 6. Point Styling
        style_label = QLabel("Point Styling:")
        style_label.setStyleSheet("QLabel { color: #aaa; font-size: 10px; margin-top: 8px; }")
        centroid_layout.addWidget(style_label)

        style_form = QFormLayout()
        style_form.setContentsMargins(4, 4, 4, 4)
        style_form.setSpacing(6)

        self.spin_point_radius = QSpinBox()
        self.spin_point_radius.setRange(1, 50)
        self.spin_point_radius.setValue(4)
        style_form.addRow("Point Radius:", self.spin_point_radius)

        # Color picker button
        self.btn_centroid_color = QPushButton()
        self.btn_centroid_color.setFixedSize(50, 22)
        self.btn_centroid_color.setStyleSheet("background-color: #ff0000; border: 1px solid #555;")
        self.btn_centroid_color.clicked.connect(self.main_window.pick_centroid_color)
        style_form.addRow("Point Color:", self.btn_centroid_color)

        centroid_layout.addLayout(style_form)

        # 7. Summary Statistics
        stats_label = QLabel("Summary Statistics:")
        stats_label.setStyleSheet("QLabel { color: #aaa; font-size: 10px; margin-top: 8px; }")
        centroid_layout.addWidget(stats_label)

        self.label_stats_total = QLabel("Total Centroids: 0")
        self.label_stats_total.setStyleSheet("QLabel { color: #ccc; font-size: 10px; padding-left: 4px; }")
        centroid_layout.addWidget(self.label_stats_total)

        self.label_stats_avg = QLabel("Avg Size: -")
        self.label_stats_avg.setStyleSheet("QLabel { color: #ccc; font-size: 10px; padding-left: 4px; }")
        centroid_layout.addWidget(self.label_stats_avg)

        self.label_stats_density = QLabel("Estimated Density: -")
        self.label_stats_density.setStyleSheet("QLabel { color: #ccc; font-size: 10px; padding-left: 4px; }")
        centroid_layout.addWidget(self.label_stats_density)

        self.setContentLayout(centroid_layout)

    def update_stats(self, total, avg, density):
        """Update statistik centroid."""
        self.label_stats_total.setText(f"Total Centroids: {total}")
        self.label_stats_avg.setText(f"Avg Size: {avg}")
        self.label_stats_density.setText(f"Estimated Density: {density}")
        self.label_centroid_count.setText(f"Centroids: {total}")

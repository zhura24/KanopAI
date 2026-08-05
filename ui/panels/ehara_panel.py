"""Panel for the eHara feature: raster pixel value extraction per band,
plus optional NDVI/GNDVI/SR calculation and N/P/K/Mg leaf nutrient
prediction (via a PCA + Linear Regression calibration trained from a
user-provided historical Excel dataset)."""

import logging
from pathlib import Path

from PyQt6.QtWidgets import (
    QVBoxLayout, QPushButton, QLabel, QFormLayout, QDoubleSpinBox,
    QSpinBox, QFileDialog, QHBoxLayout
)
from PyQt6.QtCore import Qt
from ui.widgets.collapsible_box import CollapsibleBox


class EHaraPanel(CollapsibleBox):
    """Panel for extracting raster pixel values (mean per band) around the
    center point of the bounding box/detection result currently shown, and
    optionally predicting leaf nutrient content (N/P/K/Mg)."""

    def __init__(self, main_window):
        super().__init__("eHara - Pixel Extraction")
        self.main_window = main_window
        self.logger = logging.getLogger(__name__)
        self.training_data_path = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        info_label = QLabel(
            "Extract the average pixel value of each band around the center "
            "point of the bounding box/detection result currently shown."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("QLabel { color: #888; font-size: 10px; padding: 4px; }")
        layout.addWidget(info_label)

        form = QFormLayout()
        form.setContentsMargins(4, 4, 4, 4)
        form.setSpacing(6)

        self.spin_ehara_radius = QDoubleSpinBox()
        self.spin_ehara_radius.setRange(0.1, 100.0)
        self.spin_ehara_radius.setDecimals(1)
        self.spin_ehara_radius.setSingleStep(0.5)
        self.spin_ehara_radius.setValue(2.0)
        self.spin_ehara_radius.setSuffix(" m")
        form.addRow("Buffer Radius:", self.spin_ehara_radius)

        layout.addLayout(form)

        # --- NDVI/GNDVI/SR band index selection -------------------------
        band_label = QLabel(
            "Band index used for NDVI/GNDVI/SR formulas:\n"
            "NDVI = (b3-b1)/(b3+b1)   GNDVI = (b3-b2)/(b3+b2)   SR = b3/b1"
        )
        band_label.setWordWrap(True)
        band_label.setStyleSheet("QLabel { color: #888; font-size: 10px; padding: 4px; }")
        layout.addWidget(band_label)

        band_form = QFormLayout()
        band_form.setContentsMargins(4, 4, 4, 4)
        band_form.setSpacing(6)

        self.spin_ehara_band1 = QSpinBox()
        self.spin_ehara_band1.setRange(1, 999)
        self.spin_ehara_band1.setValue(1)
        band_form.addRow("Band 1 index:", self.spin_ehara_band1)

        self.spin_ehara_band2 = QSpinBox()
        self.spin_ehara_band2.setRange(1, 999)
        self.spin_ehara_band2.setValue(2)
        band_form.addRow("Band 2 index:", self.spin_ehara_band2)

        self.spin_ehara_band3 = QSpinBox()
        self.spin_ehara_band3.setRange(1, 999)
        self.spin_ehara_band3.setValue(3)
        band_form.addRow("Band 3 index:", self.spin_ehara_band3)

        layout.addLayout(band_form)

        # --- Training data (for N/P/K/Mg prediction) ---------------------
        training_label = QLabel(
            "Optional: load a historical Excel dataset (columns: ID, X, Y, "
            "N, P, K, Mg, band1, band2, band3, NDVI, GNDVI, SR) to also "
            "predict leaf nutrient content (N/P/K/Mg) for each point. If no "
            "training data is loaded, only band values + NDVI/GNDVI/SR are "
            "extracted."
        )
        training_label.setWordWrap(True)
        training_label.setStyleSheet("QLabel { color: #888; font-size: 10px; padding: 4px; }")
        layout.addWidget(training_label)

        training_row = QHBoxLayout()
        self.btn_load_training_data = QPushButton("Load Training Data (.xlsx)...")
        self.btn_load_training_data.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_load_training_data.clicked.connect(self._browse_training_data)
        training_row.addWidget(self.btn_load_training_data)

        self.btn_clear_training_data = QPushButton("Clear")
        self.btn_clear_training_data.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_clear_training_data.clicked.connect(self._clear_training_data)
        self.btn_clear_training_data.setEnabled(False)
        training_row.addWidget(self.btn_clear_training_data)
        layout.addLayout(training_row)

        self.lbl_training_data = QLabel("Training data: (none — nutrient prediction disabled)")
        self.lbl_training_data.setWordWrap(True)
        self.lbl_training_data.setStyleSheet("QLabel { color: #888; font-size: 10px; padding: 4px; }")
        layout.addWidget(self.lbl_training_data)

        self.btn_run_ehara = QPushButton("Extract Pixels (eHara)")
        self.btn_run_ehara.setMinimumHeight(32)
        self.btn_run_ehara.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_run_ehara.clicked.connect(self.main_window.run_ehara_extraction)
        layout.addWidget(self.btn_run_ehara)

        self.setContentLayout(layout)

    def _browse_training_data(self):
        path, _ = QFileDialog.getOpenFileName(
            self.main_window, "Select Training Data (.xlsx)", "", "Excel Files (*.xlsx)"
        )
        if not path:
            return
        self.training_data_path = path
        self.lbl_training_data.setText(f"Training data: {Path(path).name}")
        self.btn_clear_training_data.setEnabled(True)

    def _clear_training_data(self):
        self.training_data_path = None
        self.lbl_training_data.setText("Training data: (none — nutrient prediction disabled)")
        self.btn_clear_training_data.setEnabled(False)

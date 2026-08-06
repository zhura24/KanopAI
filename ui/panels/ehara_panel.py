"""Panel for the eHara feature: raster pixel value extraction per band,
plus optional NDVI/GNDVI/SR calculation and N/P/K/Mg leaf nutrient
prediction (via a PCA + Linear Regression calibration trained from a
user-provided historical Excel dataset)."""

import logging
from pathlib import Path

from PyQt6.QtWidgets import (
    QVBoxLayout, QPushButton, QLabel, QFormLayout, QDoubleSpinBox,
    QSpinBox, QFileDialog, QHBoxLayout, QFrame, QWidget
)
from PyQt6.QtCore import Qt
from ui.widgets.collapsible_box import CollapsibleBox


# --- Shared toggle button style ---
_TOGGLE_ON_STYLE = (
    "QPushButton {"
    "  background-color: #16a34a;"
    "  color: white;"
    "  font-size: 10px;"
    "  font-weight: bold;"
    "  border-radius: 3px;"
    "  padding: 3px 10px;"
    "}"
    "QPushButton:hover { background-color: #15803d; }"
)
_TOGGLE_OFF_STYLE = (
    "QPushButton {"
    "  background-color: #374151;"
    "  color: #9ca3af;"
    "  font-size: 10px;"
    "  font-weight: bold;"
    "  border-radius: 3px;"
    "  padding: 3px 10px;"
    "}"
    "QPushButton:hover { background-color: #4b5563; }"
)
_SECTION_ENABLED_BORDER = "QFrame { border: 1px solid #16a34a; border-radius: 4px; background: #0d1f14; }"
_SECTION_DISABLED_BORDER = "QFrame { border: 1px solid #374151; border-radius: 4px; background: #111827; }"


class EHaraPanel(CollapsibleBox):
    """Panel for extracting raster pixel values (mean per band) around the
    center point of the bounding box/detection result currently shown, and
    optionally predicting leaf nutrient content (N/P/K/Mg).

    Features two enable/disable toggles:
    - **NDVI/GNDVI/SR**: when disabled the band index controls are hidden and
      no spectral indices are computed — the Excel output only has band mean
      columns.
    - **Load Training (N/P/K/Mg)**: when disabled the training data loader is
      hidden and no nutrient prediction is run.
    """

    def __init__(self, main_window):
        super().__init__("eHara - Pixel Extraction")
        self.main_window = main_window
        self.logger = logging.getLogger(__name__)
        self.training_data_path = None

        # Toggleable feature states
        self._ndvi_enabled: bool = True
        self._training_enabled: bool = True

        self.init_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

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

        # ================================================================
        # NDVI/GNDVI/SR Section (toggleable)
        # ================================================================
        self._ndvi_frame = QFrame()
        self._ndvi_frame.setStyleSheet(_SECTION_ENABLED_BORDER)
        ndvi_outer = QVBoxLayout(self._ndvi_frame)
        ndvi_outer.setContentsMargins(6, 6, 6, 6)
        ndvi_outer.setSpacing(6)

        # Header row: label + toggle button
        ndvi_header_row = QHBoxLayout()
        ndvi_title = QLabel("📊 Spectral Indices (NDVI / GNDVI / SR)")
        ndvi_title.setStyleSheet("QLabel { color: #a3e635; font-weight: bold; font-size: 10px; }")
        ndvi_header_row.addWidget(ndvi_title)
        ndvi_header_row.addStretch()

        self.btn_toggle_ndvi = QPushButton("✔ Enabled")
        self.btn_toggle_ndvi.setCheckable(True)
        self.btn_toggle_ndvi.setChecked(True)
        self.btn_toggle_ndvi.setFixedHeight(22)
        self.btn_toggle_ndvi.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_toggle_ndvi.setStyleSheet(_TOGGLE_ON_STYLE)
        self.btn_toggle_ndvi.setToolTip(
            "Toggle ON: NDVI, GNDVI, and SR columns will be computed and\n"
            "included in the Excel output.\n\n"
            "Toggle OFF: only band mean values are extracted — no spectral\n"
            "index columns are added to the output."
        )
        self.btn_toggle_ndvi.clicked.connect(self._toggle_ndvi_section)
        ndvi_header_row.addWidget(self.btn_toggle_ndvi)

        ndvi_outer.addLayout(ndvi_header_row)

        # Collapsible body — band index spinboxes
        self._ndvi_body = QWidget()
        ndvi_body_layout = QVBoxLayout(self._ndvi_body)
        ndvi_body_layout.setContentsMargins(0, 0, 0, 0)
        ndvi_body_layout.setSpacing(4)

        band_label = QLabel(
            "Band index used for NDVI/GNDVI/SR formulas:\n"
            "NDVI = (b3-b1)/(b3+b1)   GNDVI = (b3-b2)/(b3+b2)   SR = b3/b1"
        )
        band_label.setWordWrap(True)
        band_label.setStyleSheet("QLabel { color: #888; font-size: 10px; }")
        ndvi_body_layout.addWidget(band_label)

        band_form = QFormLayout()
        band_form.setContentsMargins(4, 2, 4, 2)
        band_form.setSpacing(5)

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

        ndvi_body_layout.addLayout(band_form)
        ndvi_outer.addWidget(self._ndvi_body)

        layout.addWidget(self._ndvi_frame)

        # ================================================================
        # Load Training Section (toggleable)
        # ================================================================
        self._training_frame = QFrame()
        self._training_frame.setStyleSheet(_SECTION_ENABLED_BORDER)
        training_outer = QVBoxLayout(self._training_frame)
        training_outer.setContentsMargins(6, 6, 6, 6)
        training_outer.setSpacing(6)

        # Header row: label + toggle button
        training_header_row = QHBoxLayout()
        training_title = QLabel("🌿 Nutrient Prediction (N/P/K/Mg)")
        training_title.setStyleSheet("QLabel { color: #38bdf8; font-weight: bold; font-size: 10px; }")
        training_header_row.addWidget(training_title)
        training_header_row.addStretch()

        self.btn_toggle_training = QPushButton("✔ Enabled")
        self.btn_toggle_training.setCheckable(True)
        self.btn_toggle_training.setChecked(True)
        self.btn_toggle_training.setFixedHeight(22)
        self.btn_toggle_training.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_toggle_training.setStyleSheet(_TOGGLE_ON_STYLE)
        self.btn_toggle_training.setToolTip(
            "Toggle ON: load a training Excel dataset (N/P/K/Mg ground truth) to\n"
            "fit a PCA + Linear Regression model and predict nutrient content.\n\n"
            "Toggle OFF: training data loader is hidden and nutrient columns\n"
            "are NOT added to the Excel output."
        )
        self.btn_toggle_training.clicked.connect(self._toggle_training_section)
        training_header_row.addWidget(self.btn_toggle_training)

        training_outer.addLayout(training_header_row)

        # Collapsible body — training data loader
        self._training_body = QWidget()
        training_body_layout = QVBoxLayout(self._training_body)
        training_body_layout.setContentsMargins(0, 0, 0, 0)
        training_body_layout.setSpacing(4)

        training_desc = QLabel(
            "Optional: load a historical Excel dataset (columns: ID, X, Y, "
            "N, P, K, Mg, band1, band2, band3, NDVI, GNDVI, SR) to also "
            "predict leaf nutrient content (N/P/K/Mg) for each point."
        )
        training_desc.setWordWrap(True)
        training_desc.setStyleSheet("QLabel { color: #888; font-size: 10px; }")
        training_body_layout.addWidget(training_desc)

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
        training_body_layout.addLayout(training_row)

        self.lbl_training_data = QLabel("Training data: (none — nutrient prediction disabled)")
        self.lbl_training_data.setWordWrap(True)
        self.lbl_training_data.setStyleSheet("QLabel { color: #888; font-size: 10px; }")
        training_body_layout.addWidget(self.lbl_training_data)

        training_outer.addWidget(self._training_body)

        layout.addWidget(self._training_frame)

        # ================================================================
        # Extract button
        # ================================================================
        self.btn_run_ehara = QPushButton("Extract Pixels (eHara)")
        self.btn_run_ehara.setMinimumHeight(32)
        self.btn_run_ehara.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_run_ehara.setStyleSheet(
            "QPushButton {"
            "  background-color: #0f766e;"
            "  color: white;"
            "  font-weight: bold;"
            "  font-size: 12px;"
            "  border: none;"
            "  border-radius: 4px;"
            "}"
            "QPushButton:hover { background-color: #0d9488; }"
        )
        self.btn_run_ehara.clicked.connect(self.main_window.run_ehara_extraction)
        layout.addWidget(self.btn_run_ehara)

        self.setContentLayout(layout)

    # ------------------------------------------------------------------
    # Toggle handlers
    # ------------------------------------------------------------------

    def _toggle_ndvi_section(self):
        """Enable or disable the NDVI/GNDVI/SR computation section."""
        self._ndvi_enabled = self.btn_toggle_ndvi.isChecked()
        if self._ndvi_enabled:
            self.btn_toggle_ndvi.setText("✔ Enabled")
            self.btn_toggle_ndvi.setStyleSheet(_TOGGLE_ON_STYLE)
            self._ndvi_frame.setStyleSheet(_SECTION_ENABLED_BORDER)
            self._ndvi_body.setVisible(True)
        else:
            self.btn_toggle_ndvi.setText("✘ Disabled")
            self.btn_toggle_ndvi.setStyleSheet(_TOGGLE_OFF_STYLE)
            self._ndvi_frame.setStyleSheet(_SECTION_DISABLED_BORDER)
            self._ndvi_body.setVisible(False)

    def _toggle_training_section(self):
        """Enable or disable the Load Training / nutrient prediction section."""
        self._training_enabled = self.btn_toggle_training.isChecked()
        if self._training_enabled:
            self.btn_toggle_training.setText("✔ Enabled")
            self.btn_toggle_training.setStyleSheet(_TOGGLE_ON_STYLE)
            self._training_frame.setStyleSheet(_SECTION_ENABLED_BORDER)
            self._training_body.setVisible(True)
        else:
            self.btn_toggle_training.setText("✘ Disabled")
            self.btn_toggle_training.setStyleSheet(_TOGGLE_OFF_STYLE)
            self._training_frame.setStyleSheet(_SECTION_DISABLED_BORDER)
            self._training_body.setVisible(False)

    # ------------------------------------------------------------------
    # Training data helpers
    # ------------------------------------------------------------------

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

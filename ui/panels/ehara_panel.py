"""Panel for the eHara feature: raster pixel value extraction per band."""

import logging
from PyQt6.QtWidgets import (
    QVBoxLayout, QPushButton, QLabel, QFormLayout, QDoubleSpinBox
)
from PyQt6.QtCore import Qt
from ui.widgets.collapsible_box import CollapsibleBox


class EHaraPanel(CollapsibleBox):
    """Panel for extracting raster pixel values (mean per band) around the
    center point of the bounding box / detection result currently shown."""

    def __init__(self, main_window):
        super().__init__("eHara - Pixel Extraction")
        self.main_window = main_window
        self.logger = logging.getLogger(__name__)
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

        self.btn_run_ehara = QPushButton("Extract Pixels (eHara)")
        self.btn_run_ehara.setMinimumHeight(32)
        self.btn_run_ehara.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_run_ehara.clicked.connect(self.main_window.run_ehara_extraction)
        layout.addWidget(self.btn_run_ehara)

        self.setContentLayout(layout)

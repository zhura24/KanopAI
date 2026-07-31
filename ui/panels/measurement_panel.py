"""Panel tool pengukuran."""

import logging
from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QCheckBox, QComboBox, QWidget
)
from PyQt6.QtCore import Qt
from ui.widgets.collapsible_box import CollapsibleBox


class MeasurementPanel(CollapsibleBox):
    """Panel untuk tool pengukuran jarak."""
    def __init__(self, main_window):
        super().__init__("Measurement Tools")
        self.main_window = main_window
        self.logger = logging.getLogger(__name__)
        self.init_ui()

    def init_ui(self):
        """Inisialisasi UI panel measurement."""

        # Measurement mode checkbox
        self.check_measurement_mode = QCheckBox("Enable Measurement Mode")
        self.check_measurement_mode.setChecked(False)
        self.check_measurement_mode.stateChanged.connect(self.main_window.toggle_measurement_mode)
        self.check_measurement_mode.setEnabled(False)
        self.addWidget(self.check_measurement_mode)

        # Unit selector
        unit_layout = QHBoxLayout()
        unit_label = QLabel("Unit:")
        unit_layout.addWidget(unit_label)

        self.combo_unit = QComboBox()
        self.combo_unit.addItems(["Meters (m)", "Centimeters (cm)", "Millimeters (mm)",
                                   "Kilometers (km)", "Feet (ft)", "Yards (yd)",
                                   "Miles (mi)", "Inches (in)", "Degrees (°)"])
        self.combo_unit.setCurrentIndex(0)  # Default to Meters
        self.combo_unit.setEnabled(False)
        self.combo_unit.currentIndexChanged.connect(self.main_window.on_unit_changed)
        unit_layout.addWidget(self.combo_unit)

        self.addLayout(unit_layout)

        # Last measurement result label
        self.label_measurement = QLabel("No measurements yet")
        self.label_measurement.setWordWrap(True)
        self.label_measurement.setStyleSheet("QLabel { color: #888; font-size: 10px; padding: 8px; background-color: #2b2b2b; border-radius: 4px; }")
        self.addWidget(self.label_measurement)

        # Clear measurements button
        self.btn_clear_measurements = QPushButton("Clear All Measurements")
        self.btn_clear_measurements.clicked.connect(self.main_window.clear_measurements)
        self.btn_clear_measurements.setEnabled(False)
        self.addWidget(self.btn_clear_measurements)

    def update_measurement_result(self, text):
        self.label_measurement.setText(text)

    def set_tools_enabled(self, enabled):
        self.check_measurement_mode.setEnabled(enabled)
        self.combo_unit.setEnabled(enabled)
        
    def set_clear_button_enabled(self, enabled):
        self.btn_clear_measurements.setEnabled(enabled)

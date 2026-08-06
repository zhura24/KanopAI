"""Handler untuk tool pengukuran jarak."""

from PyQt6.QtCore import Qt
import logging


class MeasurementHandler:
    """Handler untuk mode pengukuran dan hasil pengukuran."""
    
    def __init__(self, main_window):
        self.main_window = main_window
        self.logger = logging.getLogger(__name__)
        self.last_measurement_info = None
    
    def toggle_measurement_mode(self, state):
        enabled = state == Qt.CheckState.Checked.value
        self.main_window.viewer.enable_measurement_mode(enabled)
        
        if enabled:
            self.logger.info("Measurement mode enabled")
            self.main_window.label_measurement.setText("Click two points to measure")
        else:
            self.logger.info("Measurement mode disabled")
            self.main_window.label_measurement.setText("No measurements yet")
    
    def clear_measurements(self):
        self.main_window.viewer.clear_measurements()
        self.last_measurement_info = None
        self.main_window.label_measurement.setText("All measurements cleared")
        self.main_window.label_measurement.setStyleSheet(
            "QLabel { color: #888; font-size: 10px; padding: 8px; "
            "background-color: #2b2b2b; border-radius: 4px; }"
        )
        self.logger.info("All measurements cleared")
    
    
    def on_measurement_finished(self, measurement_info):
        self.last_measurement_info = measurement_info
        self.display_measurement_result(measurement_info)
        
        mode_name = measurement_info.get('measurement_mode', 'cartesian').title()
        if measurement_info['calibrated'] or measurement_info['georeferenced']:
            self.logger.info(
                f"Measurement completed | "
                f"Distance: {measurement_info['meters']:.2f}m | "
                f"Mode: {mode_name} | "
                f"Georeferenced: {measurement_info['georeferenced']}"
            )
        else:
            self.logger.info(
                f"Measurement completed | "
                f"Pixels: {measurement_info['distance_pixels']:.0f} | "
                f"Status: Not calibrated"
            )

    
    def on_unit_changed(self, index):
        unit_map = {
            0: 'meters',
            1: 'centimeters',
            2: 'millimeters',
            3: 'kilometers',
            4: 'feet',
            5: 'yards',
            6: 'miles',
            7: 'inches',
            8: 'degrees'
        }
        unit = unit_map.get(index, 'meters')
        self.main_window.viewer.measurement_manager.set_unit(unit)
        self.logger.info(f"Measurement unit changed to {unit}")
        
        if self.last_measurement_info:
            self.display_measurement_result(self.last_measurement_info)
    
    def display_measurement_result(self, measurement_info):
        unit_map = {
            0: ('meters', 'm', 2),
            1: ('centimeters', 'cm', 1),
            2: ('millimeters', 'mm', 0),
            3: ('kilometers', 'km', 4),
            4: ('feet', 'ft', 2),
            5: ('yards', 'yd', 2),
            6: ('miles', 'mi', 4),
            7: ('inches', 'in', 1),
            8: ('degrees', '°', 6)
        }
        selected_idx = self.main_window.combo_unit.currentIndex()
        unit_key, unit_symbol, decimals = unit_map.get(selected_idx, ('meters', 'm', 2))
        
        base_style = (
            "QLabel { color: #888; font-size: 10px; padding: 8px; "
            "background-color: #2b2b2b; border-radius: 4px; }"
        )
        
        if measurement_info['calibrated'] or measurement_info['georeferenced']:
            mode_name = measurement_info.get('measurement_mode', 'cartesian').title()
            
            if measurement_info[unit_key] is not None:
                distance_value = measurement_info[unit_key]
                result_text = f"Distance: {distance_value:.{decimals}f} {unit_symbol}\n"
                result_text += f"Mode: {mode_name}\n"
                result_text += f"Pixels: {measurement_info['distance_pixels']:.0f} px"
            
            self.main_window.label_measurement.setText(result_text)
            self.main_window.label_measurement.setStyleSheet(base_style)
        else:
            result_text = "Physical measurements unavailable\n"
            result_text += f"Pixels: {measurement_info['distance_pixels']:.0f} px\n"
            result_text += "Please load a georeferenced TIFF file"
            
            self.main_window.label_measurement.setText(result_text)
            self.main_window.label_measurement.setStyleSheet(base_style)

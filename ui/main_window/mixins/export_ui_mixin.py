"""
Export UI Mixin for MainWindow
Handles export-related UI controls and updates
"""

class ExportUIMixin:
    """Mixin for handling export UI controls"""

    def on_overlap_mode_changed(self, checked):
        """Handle overlap mode change between percentage and distance"""
        if checked:  # Percentage mode
            self.spin_overlap_percent.setEnabled(True)
            self.spin_overlap_percent.setVisible(True)
            self.spin_overlap_dx.setEnabled(False)
            self.spin_overlap_dx.setVisible(False)
            self.logger.debug("Overlap mode: Percentage")
        else:  # Distance mode
            self.spin_overlap_percent.setEnabled(False)
            self.spin_overlap_percent.setVisible(False)
            self.spin_overlap_dx.setEnabled(True)
            self.spin_overlap_dx.setVisible(True)
            self.logger.debug("Overlap mode: Distance (dx)")

        # Update export info labels when mode changes
        self.update_export_info_labels()

    def update_export_info_labels(self):
        """Update export info labels when processing parameters change"""
        # Update overlap label
        if self.radio_overlap_percent.isChecked():
            overlap_val = self.spin_overlap_percent.value()
            self.label_export_overlap.setText(f"{overlap_val} %")
        else:
            overlap_val = self.spin_overlap_dx.value()
            self.label_export_overlap.setText(f"{overlap_val} px")

        # Update tile size label
        tile_size = self.spin_tile_size.value()
        self.label_export_tile_size.setText(f"{tile_size} px")

        # Update resolution label
        resolution = self.spin_resolution.value()
        self.label_export_resolution.setText(f"{resolution} cm/px")

    def _update_export_button_state(self):
        """Update export training data button state based on prerequisites"""
        # Enable export button only if:
        # 1. Export directory is selected
        # 2. Data is loaded
        has_export_dir = hasattr(self, 'export_directory') and self.export_directory is not None
        has_data = hasattr(self, 'current_data') and self.current_data is not None

        can_export = has_export_dir and has_data

        if hasattr(self, 'btn_export_training_data'):
            self.btn_export_training_data.setEnabled(can_export)

            def _data_desc(data):
                """Return a human-readable description of current_data."""
                if data is None:
                    return "N/A"
                if hasattr(data, 'get_metadata'):  # RasterLoader
                    m = data.get_metadata()
                    return f"{m.get('width','?')}x{m.get('height','?')} bands={m.get('bands','?')}"
                if hasattr(data, 'shape'):  # legacy numpy array
                    return str(data.shape)
                return str(type(data))

            if can_export:
                self.logger.info(f"[OK] Export button ENABLED | Directory: {self.export_directory} | Data: {_data_desc(self.current_data)}")
            else:
                reasons = []
                if not has_export_dir:
                    reasons.append("[x] No export directory selected")
                else:
                    reasons.append(f"[v] Export directory: {self.export_directory}")

                if not has_data:
                    reasons.append("[x] No raster data loaded")
                else:
                    reasons.append(f"[v] Data loaded: {_data_desc(self.current_data)}")

                self.logger.info(f"Export button DISABLED | {' | '.join(reasons)}")

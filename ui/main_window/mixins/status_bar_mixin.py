"""
Status Bar and UI Utility Mixin for MainWindow
Handles status bar updates and UI utility methods
"""


class StatusBarMixin:
    """Mixin for status bar updates and UI utility methods"""

    def _detect_device(self):
        """Detect and display available computing device (GPU/CPU)"""
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()

            if 'CUDAExecutionProvider' in providers:
                device_text = "Device: GPU (CUDA)"
                self.logger.info("CUDA GPU detected and available")
            elif 'DmlExecutionProvider' in providers:
                device_text = "Device: GPU (DirectML)"
                self.logger.info("DirectML GPU detected and available")
            elif 'CPUExecutionProvider' in providers:
                device_text = "Device: CPU"
                self.logger.info("Using CPU for inference")
            else:
                device_text = "Device: Unknown"
                self.logger.warning("No known execution provider found")

            self.label_device.setText(device_text)
        except Exception as e:
            self.logger.error(f"Failed to detect device: {e}")
            self.label_device.setText("Device: CPU")

    def _update_crs_label(self, metadata):
        """Update CRS label in footer (QGIS style)"""
        try:
            if metadata and 'crs' in metadata and metadata['crs'] is not None:
                crs = metadata['crs']

                # Try to get EPSG code
                try:
                    from pyproj import CRS
                    pyproj_crs = CRS.from_wkt(crs.to_wkt())

                    # Check if it has an EPSG code
                    if pyproj_crs.to_epsg():
                        crs_text = f"EPSG:{pyproj_crs.to_epsg()}"
                        self.logger.info(f"CRS detected: {crs_text}")
                    else:
                        # No EPSG, show CRS name (compact)
                        crs_name = pyproj_crs.name if pyproj_crs.name else "Custom"
                        crs_text = crs_name[:20]  # Limit to 20 chars (more compact)
                        self.logger.info(f"CRS detected: {crs_text}")
                except:
                    # Fallback: show basic CRS string
                    crs_text = str(crs)[:20] if str(crs) else "Unknown"

                self.label_crs.setText(crs_text)  # No "CRS:" prefix for compactness
            else:
                self.label_crs.setText("EPSG: -")
                self.logger.info("No CRS information in raster file")
        except Exception as e:
            self.logger.error(f"Failed to update CRS label: {e}")
            self.label_crs.setText("CRS: Error")

    def _update_layer_info_panel(self):
        """Update the layer info panel via DisplayPanel"""
        if hasattr(self, 'display_panel'):
            self.display_panel.update_layer_info(self._get_active_layer())
        else:
            self.logger.warning("display_panel not found, skipping _update_layer_info_panel")

    def _update_display_dimensions(self):
        """Update Display label with current viewport dimensions"""
        active_layer = self._get_active_layer()
        if not active_layer:
            self.label_display.setText("Display: -")
            return

        metadata = active_layer.get('metadata', {})
        width = metadata.get('width', 0)
        height = metadata.get('height', 0)

        if hasattr(self, 'viewer') and self.viewer:
            try:
                # Get visible rect in scene coordinates
                visible_rect = self.viewer.mapToScene(self.viewer.viewport().rect()).boundingRect()
                display_width = int(visible_rect.width())
                display_height = int(visible_rect.height())
                self.label_display.setText(f"Display: {display_width}x{display_height}")
            except Exception as e:
                # Fallback to full image size
                self.logger.debug(f"Failed to get viewport dimensions: {e}")
                self.label_display.setText(f"Display: {width}x{height}")
        else:
            self.label_display.setText(f"Display: {width}x{height}")

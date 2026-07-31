"""
Layer UI Update Mixin for MainWindow
Handles UI updates based on active layer changes
"""
from PyQt6.QtCore import Qt, QRectF


class LayerUIMixin:
    """Mixin for layer-specific UI updates"""

    def _update_ui_for_active_layer(self):
        """Update all UI elements (footer, detector info) based on active layer"""
        active_layer = self._get_active_layer()
        if not active_layer:
            # No active layer - reset UI
            self.label_size.setText("Size: -")
            self.label_bands.setText("Bands: -")
            self.label_detection.setText("Detection: -")
            self.label_display.setText("Display: -")
            if hasattr(self, 'label_image_bands_info'):
                self.label_image_bands_info.setText("Image Input Bands: -")
            return

        metadata = active_layer.get('metadata', {})

        # === 1. UPDATE FOOTER LABELS ===
        # Size (full raster dimensions)
        width = metadata.get('width', 0)
        height = metadata.get('height', 0)
        self.label_size.setText(f"Size: {width}x{height}")

        # Bands (use 'bands' key, not 'count')
        num_bands = metadata.get('bands', 0)
        self.label_bands.setText(f"Bands: {num_bands}")

        # Detection status
        det = active_layer.get('detections')
        has_detections = det is not None
        if has_detections:
            # 'detections' is an InferenceResult object (has .boxes), not a plain list
            num_detections = len(det.boxes) if hasattr(det, 'boxes') else len(det)
        else:
            num_detections = 0
        self.label_detection.setText(f"Detection: {num_detections} objects" if num_detections > 0 else "Detection: None")

        # Display status (current viewport dimensions)
        self._update_display_dimensions()

        # CRS
        crs = metadata.get('crs')
        if crs and hasattr(self, 'label_crs'):
            epsg_code = crs.to_epsg() if hasattr(crs, 'to_epsg') else None
            if epsg_code:
                self.label_crs.setText(f"EPSG: {epsg_code}")
            else:
                self.label_crs.setText(f"CRS: {str(crs)[:15]}")

        # === 2. UPDATE DETECTOR SECTION - IMAGE INPUT BANDS ===
        if hasattr(self, 'label_image_bands_info'):
            band_names = []
            if num_bands == 1:
                band_names = ["Grayscale"]
            elif num_bands == 2:
                band_names = ["Band 1", "Band 2"]
            elif num_bands == 3:
                band_names = ["Red", "Green", "Blue"]
            elif num_bands == 4:
                band_names = ["Red", "Green", "Blue", "NIR/Alpha"]
            else:
                band_names = [f"Band {i+1}" for i in range(num_bands)]

            bands_text = ", ".join(band_names)
            self.label_image_bands_info.setText(f"Image Input Bands: {bands_text} ({num_bands} bands)")

        # === 3. UPDATE INPUT CHANNEL MAPPING COMBOBOXES ===
        # Update all channel mapping comboboxes with current image bands
        if hasattr(self, 'channel_mapping_combos') and self.channel_mapping_combos:
            try:
                # Block signals to prevent triggering mapping updates
                for combo in self.channel_mapping_combos:
                    combo.blockSignals(True)

                # Clear and repopulate with new band options with descriptive labels
                for combo in self.channel_mapping_combos:
                    combo.clear()
                    if num_bands == 0:
                        combo.addItem("No image loaded")
                        combo.setEnabled(False)
                    else:
                        # Add bands with descriptive labels based on image type
                        for i in range(num_bands):
                            band_label = self._get_band_label(i, num_bands)
                            combo.addItem(band_label, i)  # Store band index as data
                        combo.setEnabled(True)

                        # Auto-select appropriate default mapping
                        combo_index = self.channel_mapping_combos.index(combo)
                        if combo_index < num_bands:
                            combo.setCurrentIndex(combo_index)
                        else:
                            combo.setCurrentIndex(0)

                # Unblock signals
                for combo in self.channel_mapping_combos:
                    combo.blockSignals(False)

                self.logger.info(f"Input channel mapping updated: {num_bands} bands available")
            except Exception as e:
                self.logger.error(f"Error updating channel mapping: {e}")

        # Recreate channel mapping widgets if model is loaded and bands changed
        # This ensures widgets match the current layer's band count
        if hasattr(self, 'detector_model_session') and self.detector_model_session:
            try:
                self._create_channel_mapping_widgets()
            except Exception as e:
                self.logger.debug(f"Could not recreate channel mapping widgets: {e}")

        # Update mapping preview
        self._update_mapping_preview()

        self.logger.info(f"UI updated for active layer: {active_layer['name']} | Bands: {num_bands} | Detections: {num_detections}")

    def _auto_zoom_to_vector_extent(self, layer, reference_transform, vector_crs, reference_crs):
        """Auto-zoom viewer to show entire vector layer extent"""
        try:
            metadata = layer.get('metadata', {})
            bounds = metadata.get('bounds', {})

            if not bounds:
                self.logger.warning("[VECTOR] No bounds available for auto-zoom")
                return

            minx = bounds.get('minx')
            miny = bounds.get('miny')
            maxx = bounds.get('maxx')
            maxy = bounds.get('maxy')

            if None in [minx, miny, maxx, maxy]:
                self.logger.warning("[VECTOR] Incomplete bounds for auto-zoom")
                return

            self.logger.info(f"[VECTOR] Auto-zoom to extent: X=[{minx:.2f}, {maxx:.2f}], Y=[{miny:.2f}, {maxy:.2f}]")

            # Setup CRS transformer if needed
            transformer = None
            if vector_crs and reference_crs and vector_crs != reference_crs:
                try:
                    from pyproj import CRS, Transformer
                    from_crs = CRS(vector_crs) if not isinstance(vector_crs, str) else CRS.from_user_input(vector_crs)
                    to_crs = CRS(reference_crs) if not isinstance(reference_crs, str) else CRS.from_user_input(reference_crs)
                    transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
                except Exception as e:
                    self.logger.warning(f"[VECTOR] Could not create transformer for zoom: {e}")

            # Convert bounds to pixel coordinates
            # Get corner coordinates in pixel space
            px_min, py_min = self.viewer._geo_to_pixel(minx, miny, reference_transform, transformer)
            px_max, py_max = self.viewer._geo_to_pixel(maxx, maxy, reference_transform, transformer)

            if None in [px_min, py_min, px_max, py_max]:
                self.logger.warning("[VECTOR] Could not convert bounds to pixel coordinates")
                return

            # Create rect (note: y coordinates are inverted in pixel space)
            # In image coordinates, y increases downward
            left = min(px_min, px_max)
            top = min(py_min, py_max)
            width = abs(px_max - px_min)
            height = abs(py_max - py_min)

            rect = QRectF(left, top, width, height)

            # Add padding (10% margin)
            margin = 0.1
            rect = rect.adjusted(
                -width * margin,
                -height * margin,
                width * margin,
                height * margin
            )

            self.logger.info(f"[VECTOR] Zoom rect (pixels): ({left:.1f}, {top:.1f}, {width:.1f}x{height:.1f})")

            # Fit view to rect
            self.viewer.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

            self.logger.info("[VECTOR] Auto-zoom completed")

        except Exception as e:
            self.logger.error(f"[VECTOR] Error in auto-zoom: {e}", exc_info=True)

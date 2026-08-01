"""
Layer Management Mixin for MainWindow
Handles layer CRUD operations, visibility, and tools state
"""
from typing import Optional, Any
from PyQt6.QtWidgets import QMessageBox, QGraphicsPixmapItem


class LayerManagementMixin:
    """Mixin for layer management operations (add/remove/switch/configure)"""

    def remove_active_layer(self) -> None:
        """Remove the currently active layer"""
        if not self.active_layer_id:
            QMessageBox.warning(self, "No Active Layer", "No layer selected to remove")
            return

        # Find active layer
        layer = self._get_layer_by_id(self.active_layer_id)
        if not layer:
            return

        # Confirm removal
        reply = QMessageBox.question(
            self, "Remove Layer?",
            f"Are you sure you want to remove this layer?\n\n{layer['name']}\n\n"
            f"This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.No:
            return

        # Remove existing graphics from scene
        try:
            if layer.get('layer_type') == 'vector' and layer.get('vector_items'):
                self.viewer.clear_vector_items(layer['vector_items'])
                layer['vector_items'] = []
            elif layer['pixmap_item'] and layer['pixmap_item'].scene():
                layer['pixmap_item'].scene().removeItem(layer['pixmap_item'])
        except Exception:
            pass

        # Remove from list
        self.raster_layers.remove(layer)

        removed_loader = layer.get('loader')

        # Clear any active inference overlay state before switching or removing active layer
        if hasattr(self, 'inference_overlay_handler') and self.inference_overlay_handler:
            try:
                self.inference_overlay_handler.clear_overlay()
            except Exception as e:
                self.logger.warning(f"Failed to clear inference overlay while removing active layer: {e}")

        # Set new active layer (first remaining layer, if any)
        if self.raster_layers:
            self._set_active_layer(self.raster_layers[0]['id'])
            self._update_viewer_for_active_layer()
        else:
            self.active_layer_id = None
            # Clear viewer - no more layers
            if hasattr(self, 'viewer') and self.viewer:
                try:
                    # Cancel any pending tile loads
                    if hasattr(self.viewer, 'async_tile_loader'):
                        self.viewer.async_tile_loader.cancel_all_pending()

                    # Clear all scene items and overlays
                    scene = self.viewer.scene
                    if scene:
                        scene.clear()
                        self.logger.info("Cleared viewer scene when removing last layer")

                    # Clear viewer-managed overlays
                    if hasattr(self.viewer, 'clear_overlay'):
                        self.viewer.clear_overlay()

                    # Reset tile manager
                    self.viewer.set_tile_manager(None)

                    # Disable full resolution mode
                    self.viewer.enable_full_resolution(False)

                except Exception as e:
                    self.logger.error(f"Error clearing viewer: {e}", exc_info=True)

            # Clear backward compatibility reference
            self.raster_loader = None

        if removed_loader is not None and removed_loader not in [
            current.get('loader') for current in self.raster_layers
        ]:
            try:
                removed_loader.close()
            except Exception as e:
                self.logger.warning(f"Failed to close removed raster: {e}")

        # Refresh UI
        self._refresh_layer_list_ui()

        # Update buttons
        if not self.raster_layers:
            self.btn_remove_layer.setEnabled(False)
            self.btn_clear_layers.setEnabled(False)

        self.logger.info(f"Layer removed: {layer['name']}")

    def clear_all_layers(self) -> None:
        """Remove all layers"""
        if not self.raster_layers:
            return

        # Confirm
        reply = QMessageBox.question(
            self, "Clear All Layers?",
            f"Are you sure you want to remove ALL {len(self.raster_layers)} layer(s)?\n\n"
            f"This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.No:
            return

        # Remove all graphics items from scene
        for layer in self.raster_layers:
            try:
                if layer.get('layer_type') == 'vector' and layer.get('vector_items'):
                    self.viewer.clear_vector_items(layer['vector_items'])
                    layer['vector_items'] = []
                elif layer.get('pixmap_item') and layer['pixmap_item'].scene():
                    layer['pixmap_item'].scene().removeItem(layer['pixmap_item'])
            except Exception:
                pass

        loaders_to_close = [layer.get('loader') for layer in self.raster_layers]

        # Clear list
        self.raster_layers.clear()
        self.active_layer_id = None
        self.layer_counter = 0  # Reset layer counter

        # Clear viewer - remove all tiles and reset
        if hasattr(self, 'viewer') and self.viewer:
            try:
                # Cancel any pending tile loads
                if hasattr(self.viewer, 'async_tile_loader'):
                    self.viewer.async_tile_loader.cancel_all_pending()

                # Clear all scene items and overlays
                scene = self.viewer.scene
                if scene:
                    scene.clear()
                    self.logger.info("Cleared viewer scene when clearing all layers")

                # Clear viewer-managed overlays
                if hasattr(self.viewer, 'clear_overlay'):
                    self.viewer.clear_overlay()

                # Reset tile manager
                self.viewer.set_tile_manager(None)

                # Disable full resolution mode
                self.viewer.enable_full_resolution(False)

            except Exception as e:
                self.logger.error(f"Error clearing viewer: {e}", exc_info=True)

        # Clear backward compatibility reference
        self.raster_loader = None

        for loader in loaders_to_close:
            if loader is not None:
                try:
                    loader.close()
                except Exception as e:
                    self.logger.warning(f"Failed to close raster during clear: {e}")

        # Clear inference overlay items if any
        if hasattr(self, 'inference_overlay_handler') and self.inference_overlay_handler:
            try:
                self.inference_overlay_handler.clear_overlay()
            except Exception as e:
                self.logger.warning(f"Failed to clear inference overlay while clearing all layers: {e}")

        # Disable tools when no layers remain
        self._disable_tools()

        # Reset UI labels
        if hasattr(self, 'file_panel'):
            self.file_panel.update_file_label("No file loaded")
        elif hasattr(self, 'label_file'):
            self.label_file.setText("No file loaded")
        self.label_display.setText("Display: --")
        self.label_size.setText("Size: --")
        self.label_bands.setText("Bands: --")
        self.label_crs.setText("CRS: --")

        # Refresh UI
        self._refresh_layer_list_ui()

        # Update buttons
        self.btn_remove_layer.setEnabled(False)
        self.btn_clear_layers.setEnabled(False)

        self.logger.info("All layers cleared")

    def _set_active_layer(self, layer_id):
        """Set the active layer"""
        # Deactivate all layers
        for layer in self.raster_layers:
            layer['is_active'] = False

        # Activate specified layer
        layer = self._get_layer_by_id(layer_id)
        if layer:
            layer['is_active'] = True
            self.active_layer_id = layer_id

            # Debug: Log layer data before sync
            has_detections = layer.get('detections') is not None
            has_polygons = len(layer.get('polygons', [])) > 0
            has_centroids = len(layer.get('centroids', [])) > 0
            self.logger.info(
                f"[SET ACTIVE] Switching to layer: {layer['name']} | "
                f"Detections: {has_detections} | Polygons: {has_polygons} | Centroids: {has_centroids}"
            )

            # Update legacy raster_loader for backward compatibility
            self.raster_loader = layer['loader']

            # Load full raster data for processing (detector, measurements, etc.)
            # This is done lazily - only when needed by features
            self._update_current_data_for_active_layer()

            self.logger.info(f"Active layer set to: {layer['name']}")

    def _get_layer_by_id(self, layer_id):
        """Get layer by ID"""
        for layer in self.raster_layers:
            if layer['id'] == layer_id:
                return layer
        return None

    def _toggle_layer_visibility(self, layer_id):
        """Toggle layer visibility - hide/show layer in viewer"""
        layer = self._get_layer_by_id(layer_id)
        if not layer:
            return

        # Toggle visibility state
        layer['visible'] = not layer['visible']
        is_visible = layer['visible']

        self.logger.info(f"[VISIBILITY] Layer {layer_id} visibility toggled: {is_visible}")

        # If this is the active layer, we need to show/hide in viewer
        if layer['is_active']:
            if hasattr(self, 'viewer') and self.viewer:
                try:
                    if layer.get('layer_type') == 'vector':
                        self.viewer.set_vector_visibility(layer.get('vector_items', []), is_visible)
                    else:
                        # Only toggle the raster image itself. Inference overlays
                        # (bounding boxes) are intentionally left untouched here so
                        # that hiding/showing the raster never hides or drops
                        # existing detection results - they stay in sync with the
                        # active layer and simply keep rendering on top.
                        self.viewer.show_raster(is_visible)
                    self.logger.info(f"[VISIBILITY] Active layer viewer visibility set to: {is_visible}")
                except Exception as e:
                    self.logger.error(f"Error toggling active layer visibility: {e}")
        else:
            # For non-active layers, update pixmap_item if it exists
            if layer.get('layer_type') == 'vector':
                if layer.get('vector_items'):
                    self.viewer.set_vector_visibility(layer['vector_items'], is_visible)
                    self.logger.info(f"[VISIBILITY] Non-active vector layer items visibility: {is_visible}")
            elif layer['pixmap_item']:
                layer['pixmap_item'].setVisible(is_visible)
                self.logger.info(f"[VISIBILITY] Non-active layer pixmap visibility: {is_visible}")

        # Refresh UI to update button icon
        self._refresh_layer_list_ui()

    def toggle_layer_visibility(self, layer_id, visible):
        """Set layer visibility to a specific state (True/False).
        
        Args:
            layer_id: ID of the layer to modify
            visible: True to show layer, False to hide layer
        """
        layer = self._get_layer_by_id(layer_id)
        if not layer:
            self.logger.warning(f"Layer {layer_id} not found")
            return

        # Set visibility state
        layer['visible'] = visible

        self.logger.info(f"[VISIBILITY] Layer {layer_id} visibility set to: {visible}")

        # If this is the active layer, update viewer
        if layer['is_active']:
            if hasattr(self, 'viewer') and self.viewer:
                try:
                    if layer.get('layer_type') == 'vector':
                        self.viewer.set_vector_visibility(layer.get('vector_items', []), visible)
                    else:
                        self.viewer.show_raster(visible)
                    self.logger.info(f"[VISIBILITY] Active layer viewer visibility set to: {visible}")
                except Exception as e:
                    self.logger.error(f"Error setting active layer visibility: {e}")
        else:
            if layer.get('layer_type') == 'vector':
                if layer.get('vector_items'):
                    self.viewer.set_vector_visibility(layer['vector_items'], visible)
                    self.logger.info(f"[VISIBILITY] Non-active vector layer items visibility: {visible}")
            elif layer['pixmap_item']:
                layer['pixmap_item'].setVisible(visible)
                self.logger.info(f"[VISIBILITY] Non-active layer pixmap visibility: {visible}")

        # Refresh UI to update button icon
        self._refresh_layer_list_ui()

    def _delete_layer(self, layer_id):
        """Delete a specific layer"""
        layer = self._get_layer_by_id(layer_id)
        if not layer:
            return

        # If this is the active layer, use remove_active_layer method
        if layer['is_active']:
            self.remove_active_layer()
        else:
            try:
                if layer.get('layer_type') == 'vector' and layer.get('vector_items'):
                    self.viewer.clear_vector_items(layer['vector_items'])
                    layer['vector_items'] = []
                elif layer['pixmap_item'] and layer['pixmap_item'].scene():
                    layer['pixmap_item'].scene().removeItem(layer['pixmap_item'])
            except Exception:
                pass

            # Remove from list
            self.raster_layers.remove(layer)

            # Refresh UI
            self._refresh_layer_list_ui()

            # Update buttons
            if not self.raster_layers:
                self.btn_remove_layer.setEnabled(False)
                self.btn_clear_layers.setEnabled(False)

            self.logger.info(f"Layer deleted: {layer['name']}")

    def _enable_tools_for_layer(self):
        """Enable tools after a layer is loaded"""
        # Reset detection state
        self._reset_detection_state()

        # Enable measurement tools
        self.combo_input_layer.setEnabled(True)
        self.check_measurement_mode.setEnabled(True)
        self.combo_unit.setEnabled(True)
        self.btn_clear_measurements.setEnabled(True)

        # Enable zoom controls
        try:
            if hasattr(self, 'btn_zoom_in'):
                self.btn_zoom_in.setEnabled(True)
            if hasattr(self, 'btn_zoom_out'):
                self.btn_zoom_out.setEnabled(True)
            if hasattr(self, 'btn_reset_view'):
                self.btn_reset_view.setEnabled(True)
        except Exception as e:
            self.logger.debug(f"Failed to enable zoom controls: {e}")

        # Enable inference button if a multispectral model (.pt + band_stats.json) is loaded.
        # (Diganti dari cek self.onnx_engine.session -- alur ONNX sudah tidak dipakai.)
        model_ready = bool(getattr(self, 'detector_model_path', None)) and bool(getattr(self, 'band_stats_path', None))
        if model_ready:
            self.btn_run_inference.setEnabled(True)
            # Update channel mapping widgets
            try:
                self._create_channel_mapping_widgets()
                self.logger.info("Channel mapping widgets updated for layer")
            except Exception as e:
                self.logger.warning(f"Failed to update channel mapping widgets: {e}")
        else:
            self.btn_run_inference.setEnabled(False)

        # Give focus to viewer for immediate scroll wheel zoom
        self.viewer.setFocus()

        self.logger.info("Tools enabled for active layer")

    def _disable_tools(self):
        """Disable tools when no layer is loaded"""
        # Disable measurement tools
        self.combo_input_layer.setEnabled(False)
        self.check_measurement_mode.setEnabled(False)
        self.combo_unit.setEnabled(False)
        self.btn_clear_measurements.setEnabled(False)

        # Disable zoom controls
        try:
            if hasattr(self, 'btn_zoom_in'):
                self.btn_zoom_in.setEnabled(False)
            if hasattr(self, 'btn_zoom_out'):
                self.btn_zoom_out.setEnabled(False)
            if hasattr(self, 'btn_reset_view'):
                self.btn_reset_view.setEnabled(False)
        except Exception as e:
            self.logger.debug(f"Failed to disable zoom controls: {e}")

        # Disable inference button
        self.btn_run_inference.setEnabled(False)

        self.logger.info("Tools disabled (no active layer)")

    def new_session(self):
        """Start a new session - clear all layers and reset application state"""
        # Check if there are any layers to clear
        if not self.raster_layers:
            QMessageBox.information(
                self, "No Active Session",
                "There is no active session to clear."
            )
            return

        # Confirmation dialog
        reply = QMessageBox.question(
            self, "New Session?",
            f"Start a new session?\n\n"
            f"This will clear all {len(self.raster_layers)} layer(s), "
            f"polygons, measurements, and detection results.\n\n"
            f"This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.No:
            return

        self.logger.info("Starting new session - clearing all data")

        # Clear all layers (this already handles viewer clearing and tool disabling)
        # Remove all pixmap items from scene
        for layer in self.raster_layers:
            if layer['pixmap_item'] and layer['pixmap_item'].scene():
                layer['pixmap_item'].scene().removeItem(layer['pixmap_item'])
            if layer.get('vector_items'):
                try:
                    self.viewer.clear_vector_items(layer['vector_items'])
                except Exception:
                    pass
                layer['vector_items'] = []

        # Clear list
        self.raster_layers.clear()
        self.active_layer_id = None
        self.layer_counter = 0  # Reset layer counter

        # Clear viewer - remove all tiles and reset
        if hasattr(self, 'viewer') and self.viewer:
            try:
                # Cancel any pending tile loads
                if hasattr(self.viewer, 'async_tile_loader'):
                    self.viewer.async_tile_loader.cancel_all_pending()

                # Clear all scene items to remove overlays, vector items, and raster pixmaps
                scene = self.viewer.scene
                if scene:
                    scene.clear()

                # Reset tile manager
                self.viewer.set_tile_manager(None)

                # Disable full resolution mode
                self.viewer.enable_full_resolution(False)

                # Clear measurements
                if hasattr(self.viewer, 'clear_measurements'):
                    self.viewer.clear_measurements()

                # Clear polygons
                if hasattr(self.viewer, 'clear_polygon'):
                    self.viewer.clear_polygon()

            except Exception as e:
                self.logger.error(f"Error clearing viewer: {e}", exc_info=True)

        # Clear backward compatibility reference
        self.raster_loader = None

        # Clear all drawn polygons
        self.drawn_polygons.clear()
        self.polygon_counter = 0

        # Clear inference overlay items if any
        if hasattr(self, 'inference_overlay_handler') and self.inference_overlay_handler:
            try:
                self.inference_overlay_handler.clear_overlay()
            except Exception as e:
                self.logger.warning(f"Failed to clear inference overlay while clearing all layers: {e}")

        # Clear inference overlay items if any
        if hasattr(self, 'inference_overlay_handler') and self.inference_overlay_handler:
            try:
                self.inference_overlay_handler.clear_overlay()
            except Exception as e:
                self.logger.warning(f"Failed to clear inference overlay during new session: {e}")

        # Reset detection state
        self._reset_detection_state()

        # Disable tools when no layers remain
        self._disable_tools()

        # Reset UI labels
        if hasattr(self, 'file_panel'):
            self.file_panel.update_file_label("No file loaded")
        elif hasattr(self, 'label_file'):
            self.label_file.setText("No file loaded")
        self.label_display.setText("Display: --")
        self.label_size.setText("Size: --")
        self.label_bands.setText("Bands: --")
        self.label_crs.setText("CRS: --")

        # Hide New Session button
        if hasattr(self, 'file_panel'):
            self.file_panel.set_new_session_visible(False)
        elif hasattr(self, 'btn_new_session'):
             self.btn_new_session.setVisible(False)

        # Refresh UI
        self._refresh_layer_list_ui()
        self._refresh_polygon_list_ui()

        # Update buttons
        self.btn_remove_layer.setEnabled(False)
        self.btn_clear_layers.setEnabled(False)

        self.logger.info("New session started - all data cleared")

        QMessageBox.information(
            self, "New Session Started",
            "New session started successfully.\n\n"
            "All layers, polygons, and measurements have been cleared."
        )

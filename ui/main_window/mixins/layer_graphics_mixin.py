"""
Layer Graphics Synchronization Mixin for MainWindow
Handles graphics synchronization when switching layers
"""


class LayerGraphicsMixin:
    """Mixin for layer graphics redrawing and clearing operations"""

    def _on_polygon_drawing_toggled(self, checked):
        """Toggle ALL polygons visibility"""
        # Toggle visibility for all drawn polygons
        for polygon in self.drawn_polygons:
            polygon['visible'] = checked

            # Update graphics items visibility
            items = polygon['items']
            for item in items.get('vertex_items', []):
                if item:
                    item.setVisible(checked)
            for item in items.get('line_items', []):
                if item:
                    item.setVisible(checked)
            if items.get('closing_line'):
                items['closing_line'].setVisible(checked)
            if items.get('filled_item'):
                items['filled_item'].setVisible(checked)

        # Also toggle current drawing polygon in viewer
        if hasattr(self, 'viewer') and self.viewer:
            self.viewer.set_polygon_visibility(checked)

        self.logger.info(f"All polygons visibility set to {checked}")

    def _redraw_detection_overlay(self, layer_detections):
        """Redraw detection bounding boxes for the active layer.

        This is called when switching to a layer that already has detection results.

        Args:
            layer_detections: Detection results dict with 'detections' list
        """
        if not layer_detections:
            return

        try:
            # Extract detections list from result
            detections = layer_detections
            if isinstance(detections, dict):
                detections = detections.get('detections', [])

            if not detections:
                self.logger.info("[LAYER SYNC] No detections to redraw")
                return

            # Redraw detection overlay using existing method
            self.logger.info(f"[LAYER SYNC] Redrawing {len(detections)} detection boxes")
            self._display_detections_as_overlay(detections)

            # Update detection checkbox - IMPORTANT: Block signals to prevent toggle handler
            if hasattr(self, 'chk_detector_overlay'):
                self.chk_detector_overlay.blockSignals(True)  # Block to prevent triggering toggle handler
                self.chk_detector_overlay.setChecked(True)
                self.chk_detector_overlay.setEnabled(True)
                self.chk_detector_overlay.setText(f"Detections ({len(detections)})")
                self.chk_detector_overlay.blockSignals(False)  # Unblock

        except Exception as e:
            self.logger.error(f"Error redrawing detection overlay: {e}", exc_info=True)

    def _clear_detection_overlay(self):
        """Clear detection bounding boxes from the viewer.

        This is called when switching to a layer that doesn't have detection results.
        """
        try:
            if hasattr(self, 'viewer') and self.viewer:
                # Clear overlay tiles (detection boxes)
                self.viewer.set_overlay_tiles([])
                self.logger.info("[LAYER SYNC] Cleared detection overlay")

            # Update detection checkbox - IMPORTANT: Block signals to prevent toggle handler
            if hasattr(self, 'chk_detector_overlay'):
                self.chk_detector_overlay.blockSignals(True)  # Block to prevent triggering toggle handler
                self.chk_detector_overlay.setChecked(False)
                self.chk_detector_overlay.setEnabled(False)
                self.chk_detector_overlay.setText("Detections")
                self.chk_detector_overlay.blockSignals(False)  # Unblock

        except Exception as e:
            self.logger.error(f"Error clearing detection overlay: {e}", exc_info=True)

    def _redraw_layer_specific_graphics(self):
        """Redraw polygons and centroids for the active layer.

        NOTE: For now, we rely on the existing polygon items stored in each polygon's 'items' dict.
        When switching layers, polygons and centroids will be redrawn by the existing rendering logic.
        This method primarily handles clearing old graphics.
        """
        # Clear ALL graphics from scene when switching layers
        # This includes tile items which will be reloaded by the viewer
        self._clear_all_non_tile_graphics()

        # Redraw polygons using existing items
        for polygon in self.drawn_polygons:
            if polygon.get('items'):
                # Re-add polygon items to scene
                self._readd_polygon_items(polygon)

        # Redraw centroids using existing rendering method
        if self.centroid_points:
            self._render_centroids()

    def _clear_all_non_tile_graphics(self):
        """Clear all non-tile graphics (polygons, centroids, measurements) from scene."""
        if not hasattr(self, 'viewer') or not self.viewer:
            return

        scene = self.viewer.scene
        if not scene:
            return

        # Don't remove anything here - let the specific clear methods handle it
        # Graphics will be managed by their respective features

    def _readd_polygon_items(self, polygon):
        """Re-add polygon graphics items to the scene."""
        # For now, polygon items are managed by the polygon drawing system
        # When viewer scene is cleared and reloaded, polygons need to be redrawn
        # This is handled by the existing polygon rendering logic
        pass

    def _clear_polygon_graphics(self):
        """Clear all polygon graphics from the scene."""
        if not hasattr(self, 'viewer') or not self.viewer:
            return

        scene = self.viewer.scene
        if not scene:
            return

        # Remove polygon graphics items
        items_to_remove = []
        for item in scene.items():
            # Check if item has a custom data flag indicating it's a polygon
            if hasattr(item, 'data') and item.data(0) == 'polygon':
                items_to_remove.append(item)

        for item in items_to_remove:
            scene.removeItem(item)

    def _clear_centroid_graphics(self):
        """Clear all centroid graphics from the scene."""
        if not hasattr(self, 'viewer') or not self.viewer:
            return

        scene = self.viewer.scene
        if not scene:
            return

        # Remove centroid graphics items
        items_to_remove = []
        for item in scene.items():
            # Check if item has a custom data flag indicating it's a centroid
            if hasattr(item, 'data') and item.data(0) == 'centroid':
                items_to_remove.append(item)

        for item in items_to_remove:
            scene.removeItem(item)

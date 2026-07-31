"""
Centroid UI Handler Mixin for MainWindow
Handles all centroid-related UI event handlers and display utilities
"""
from PyQt6.QtWidgets import QGraphicsView
from PyQt6.QtCore import Qt


class CentroidUIHandlersMixin:
    """Mixin for centroid UI event handlers, visibility toggles, and display utilities"""

    def _on_scene_clicked(self, scene_x, scene_y):
        """Handle scene click for centroid add/delete mode"""
        if self.centroid_edit_mode == 'add':
            self.add_centroid_at_click(scene_x, scene_y)
        elif self.centroid_edit_mode == 'delete':
            self.delete_centroid_at_click(scene_x, scene_y)

    def _clear_centroid_rendering(self):
        """Clear centroid visual elements from viewer"""
        if not hasattr(self, 'centroid_items'):
            return

        for item in self.centroid_items:
            try:
                self.viewer.scene.removeItem(item)
            except Exception as e:
                self.logger.debug(f"Failed to remove centroid item from scene: {e}")

        self.centroid_items.clear()

    def _update_centroid_ui(self):
        """Update centroid-related UI elements with canopy statistics"""
        count = len(self.centroid_points)

        # Calculate canopy statistics if available using user preference
        canopy_stats_text = ""
        if count > 0:
            canopy_avg = self._get_canopy_avg_text(self.centroid_points)
            if canopy_avg:
                canopy_stats_text = f" | {canopy_avg}"

        # Update count label in Centroid Detector section
        if hasattr(self, 'label_centroid_count'):
            self.label_centroid_count.setText(f"Centroids: {count}{canopy_stats_text}")

        # Update checkbox in Display Options
        if hasattr(self, 'chk_centroid_layer'):
            self.chk_centroid_layer.setText(f"Centroids ({count})")
            self.chk_centroid_layer.setEnabled(count > 0)
            if count > 0:
                self.chk_centroid_layer.setChecked(True)

        # Enable/disable buttons
        has_centroids = count > 0

        if hasattr(self, 'btn_add_centroid'):
            # Add button enabled when we have loaded raster (whether or not we have centroids)
            self.btn_add_centroid.setEnabled(self.raster_loader.dataset is not None)

        if hasattr(self, 'btn_delete_centroid'):
            self.btn_delete_centroid.setEnabled(has_centroids)

        if hasattr(self, 'btn_save_centroids'):
            self.btn_save_centroids.setEnabled(has_centroids)

    def _on_centroid_layer_toggled(self, state):
        """Handle toggling of centroid point visibility (only the point markers)"""
        visible = bool(state)

        for item in self.centroid_items:
            try:
                # Only toggle items tagged as centroid points (not canopy circles or labels)
                if hasattr(item, '_is_centroid') and item._is_centroid:
                    item.setVisible(visible)
            except Exception as e:
                self.logger.debug(f"Failed to toggle centroid point visibility: {e}")

        self.logger.info(f"Centroid points visibility: {visible}")

    def _on_canopy_layer_toggled(self, state):
        """Handle toggling of canopy circles visibility (only the circle outlines)"""
        visible = bool(state)

        for item in self.centroid_items:
            try:
                # Only toggle items tagged as canopy circles (not centroid points or labels)
                if hasattr(item, '_is_centroid_canopy') and item._is_centroid_canopy:
                    item.setVisible(visible)
            except Exception as e:
                self.logger.debug(f"Failed to toggle canopy circle visibility: {e}")

        self.logger.info(f"Canopy circles visibility: {visible}")

    def _on_canopy_labels_toggled(self, state):
        """Handle toggling of canopy measurement labels visibility (only the text labels)"""
        visible = bool(state)

        count = 0
        for item in self.centroid_items:
            try:
                # Only toggle items tagged as centroid labels (not centroid points or canopy circles)
                if hasattr(item, '_is_centroid_label') and item._is_centroid_label:
                    item.setVisible(visible)
                    count += 1
            except Exception as e:
                self.logger.error(f"Error toggling label visibility: {e}")

        self.logger.info(f"Canopy measurement labels visibility: {visible} (toggled {count} labels)")

    def toggle_add_centroid_mode(self, checked):
        """Toggle add centroid mode"""
        if checked:
            # Enable add mode
            self.centroid_edit_mode = 'add'
            self.viewer.setCursor(Qt.CursorShape.CrossCursor)

            # Uncheck delete button
            if hasattr(self, 'btn_delete_centroid'):
                self.btn_delete_centroid.blockSignals(True)
                self.btn_delete_centroid.setChecked(False)
                self.btn_delete_centroid.blockSignals(False)

            # Disable panning/measurement
            try:
                self.viewer.setDragMode(QGraphicsView.DragMode.NoDrag)
            except Exception as e:
                self.logger.debug(f"Failed to set viewer drag mode: {e}")

            self.logger.info("Add centroid mode: ENABLED")
        else:
            # Disable add mode
            self.centroid_edit_mode = None
            self.viewer.setCursor(Qt.CursorShape.ArrowCursor)

            # Restore panning
            try:
                self.viewer.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            except Exception as e:
                self.logger.debug(f"Failed to restore viewer drag mode: {e}")

            self.logger.info("Add centroid mode: DISABLED")

    def toggle_delete_centroid_mode(self, checked):
        """Toggle delete centroid mode"""
        if checked:
            # Enable delete mode
            self.centroid_edit_mode = 'delete'
            self.viewer.setCursor(Qt.CursorShape.PointingHandCursor)

            # Uncheck add button
            if hasattr(self, 'btn_add_centroid'):
                self.btn_add_centroid.blockSignals(True)
                self.btn_add_centroid.setChecked(False)
                self.btn_add_centroid.blockSignals(False)

            # Disable panning
            try:
                self.viewer.setDragMode(QGraphicsView.DragMode.NoDrag)
            except Exception as e:
                self.logger.debug(f"Failed to set viewer drag mode: {e}")

            self.logger.info("Delete centroid mode: ENABLED")
        else:
            # Disable delete mode
            self.centroid_edit_mode = None
            self.viewer.setCursor(Qt.CursorShape.ArrowCursor)

            # Restore panning
            try:
                self.viewer.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            except Exception as e:
                self.logger.debug(f"Failed to restore viewer drag mode: {e}")

            self.logger.info("Delete centroid mode: DISABLED")

    def _on_canopy_circle_toggled(self, state):
        """Handle canopy circle visibility toggle"""
        # Re-render centroids to show/hide canopy circles
        self._render_centroids()

        is_visible = self.chk_show_canopy_circles.isChecked()
        self.logger.info(f"Canopy circles visibility: {'ON' if is_visible else 'OFF'}")

    def _on_measurement_labels_toggled(self, state):
        """Handle measurement labels visibility toggle"""
        # Re-render centroids to show/hide measurement labels
        self._render_centroids()

        is_visible = self.chk_show_measurement_labels.isChecked()
        self.logger.info(f"Measurement labels visibility: {'ON' if is_visible else 'OFF'}")

    def _on_canopy_display_changed(self, checked):
        """Handle canopy display preference change"""
        if not checked:
            return  # Ignore uncheck events

        # Re-render centroids to update measurement labels immediately
        self._render_centroids()

        # Update all UI elements to reflect new preference
        self._update_centroid_ui()
        self._update_layer_info_panel()

        # Log preference
        if self.radio_show_radius.isChecked():
            pref = "Radius only"
        elif self.radio_show_diameter.isChecked():
            pref = "Diameter only"
        else:
            pref = "Both (Radius & Diameter)"

        self.logger.info(f"Canopy display preference changed to: {pref}")

    def _get_canopy_display_text(self, radius_m):
        """Get canopy measurement text based on user preference.

        Args:
            radius_m: Canopy radius in meters

        Returns:
            Formatted string based on user preference (e.g., "R: 3.75m" or "D: 7.50m" or "R: 3.75m | D: 7.50m")
        """
        diameter_m = radius_m * 2

        if hasattr(self, 'radio_show_radius') and self.radio_show_radius.isChecked():
            return f"R: {radius_m:.2f}m"
        elif hasattr(self, 'radio_show_both') and self.radio_show_both.isChecked():
            return f"R: {radius_m:.2f}m | D: {diameter_m:.2f}m"
        else:  # Default: diameter only
            return f"D: {diameter_m:.2f}m"

    def _get_canopy_avg_text(self, centroids):
        """Get average canopy measurement text based on user preference.

        Args:
            centroids: List of centroid dicts with 'radius_m' key

        Returns:
            Formatted string with average measurement (e.g., "Avg: D: 7.50m")
        """
        if not centroids or len(centroids) == 0:
            return ""

        # Check if centroids have canopy data
        first_pt = centroids[0]
        if 'radius_m' not in first_pt:
            return ""

        avg_radius = sum(pt['radius_m'] for pt in centroids) / len(centroids)
        avg_diameter = avg_radius * 2

        if hasattr(self, 'radio_show_radius') and self.radio_show_radius.isChecked():
            return f"Avg R: {avg_radius:.2f}m"
        elif hasattr(self, 'radio_show_both') and self.radio_show_both.isChecked():
            return f"Avg R: {avg_radius:.2f}m | D: {avg_diameter:.2f}m"
        else:  # Default: diameter only
            return f"Avg D: {avg_diameter:.2f}m"

"""
Display Controls Mixin for MainWindow
Handles display overlay toggles (detections, tiles, labels)
"""

from typing import Any
from PyQt6.QtCore import Qt


class DisplayControlsMixin:
    """Mixin for handling display overlay controls"""

    def _on_detector_overlay_toggled(self, state: Any) -> None:
        """Handle toggling of the Detections layer from the Display Options.

        Shows or hides detection overlay items (bounding boxes + labels) while keeping
        tile preview overlays independent. Filters by _overlay_type='detection'.
        """
        visible = False
        try:
            visible = (state == Qt.CheckState.Checked.value) or bool(state)
        except Exception as e:
            self.logger.debug(f"Exception converting state, using bool: {e}")
            visible = bool(state)

        self.logger.info(f"[DETECTION_TOGGLE] Detection overlay toggle: visible={visible}")

        try:
            # Get all overlay items from viewer
            items = getattr(self.viewer, 'overlay_items', [])

            if not items:
                self.logger.debug("[DETECTION_TOGGLE] No overlay items to toggle")
                return

            # Filter and toggle only detection overlays (not tile previews)
            detection_count = 0
            tile_count = 0
            other_count = 0

            for item in items:
                try:
                    # Check overlay type tag
                    otype = getattr(item, '_overlay_type', None)
                    if otype == 'detection':
                        item.setVisible(visible)
                        detection_count += 1
                    elif otype == 'tile':
                        tile_count += 1
                    else:
                        other_count += 1
                except Exception as e:
                    self.logger.warning(f"[DETECTION_TOGGLE] Failed to toggle overlay item visibility: {e}")
                    continue

            action = "shown" if visible else "hidden"
            self.logger.info(
                f"[DETECTION_TOGGLE] Detection layer {action}: {detection_count} detection items "
                f"(boxes + labels), {tile_count} tile overlays unchanged, {other_count} other"
            )

            # Update scene and viewport
            try:
                self.viewer.scene().update()
                self.viewer.viewport().update()
            except Exception as e:
                self.logger.debug(f"[DETECTION_TOGGLE] Failed to update scene/viewport: {e}")

        except Exception as e:
            self.logger.error(f"[DETECTION_TOGGLE] Error toggling detection overlay: {e}", exc_info=True)

    def _on_tile_preview_toggled(self, state: Any) -> None:
        """Handle toggling of the Tile Preview overlays independently of detections.

        Tile preview overlays are created by the Tile Preview dialog and are tagged
        with _overlay_type == 'tile' on the QGraphicsRectItem. This handler will
        iterate current overlay items and show/hide only those whose type is 'tile',
        leaving detection overlays untouched.
        """
        visible = False
        try:
            visible = (state == Qt.CheckState.Checked.value) or bool(state)
        except Exception as e:
            self.logger.debug(f"Exception converting state, using bool: {e}")
            visible = bool(state)

        self.logger.info(f"[TILE_TOGGLE] Checkbox state changed, visible={visible}")

        try:
            items = getattr(self.viewer, 'overlay_items', [])
            tile_count = 0
            detection_count = 0
            other_count = 0
            total_items = len(items)

            self.logger.info(f"[TILE_TOGGLE] Processing {total_items} total overlay items")

            for it in items:
                try:
                    otype = getattr(it, '_overlay_type', None)
                    self.logger.debug(f"[TILE_TOGGLE] Item type: {otype}, visible before: {it.isVisible()}")

                    if otype == 'tile':
                        it.setVisible(visible)
                        tile_count += 1
                        self.logger.debug(f"[TILE_TOGGLE] Set tile overlay visible={visible}")
                    elif otype == 'detection':
                        detection_count += 1
                    else:
                        other_count += 1
                except Exception as e:
                    self.logger.debug(f"[TILE_TOGGLE] Error toggling tile item: {e}")

            status = "shown" if visible else "hidden"
            self.logger.info(f"[TILE_TOGGLE] Tile preview {status}: {tile_count} tile overlays, {detection_count} detection overlays, {other_count} other overlays (total {total_items})")

            # Update both scene and viewport to reflect changes
            try:
                self.viewer.scene().update()
                self.viewer.viewport().update()
                self.logger.debug(f"[TILE_TOGGLE] Scene and viewport updated")
            except Exception as e:
                self.logger.debug(f"[TILE_TOGGLE] Error updating scene/viewport: {e}")

        except Exception as e:
            self.logger.error(f"[TILE_TOGGLE] Failed toggling tile preview overlays: {e}", exc_info=True)

    def _on_detection_labels_toggled(self, state: Any) -> None:
        """Handle toggling of Detection Labels independently of detection boxes.

        Detection labels are text items showing ID numbers on detection boxes.
        They are tagged with _is_label=True for independent toggling.
        """
        visible = False
        try:
            visible = (state == Qt.CheckState.Checked.value) or bool(state)
        except Exception as e:
            self.logger.debug(f"Exception converting state, using bool: {e}")
            visible = bool(state)

        self.logger.info(f"[LABEL_TOGGLE] Detection labels toggle: visible={visible}")

        try:
            items = getattr(self.viewer, 'overlay_items', [])
            label_count = 0
            total_items = len(items)

            self.logger.info(f"[LABEL_TOGGLE] Processing {total_items} total overlay items")

            for it in items:
                try:
                    # Check if this is a label item (text or text background)
                    is_label = getattr(it, '_is_label', False)
                    if is_label:
                        it.setVisible(visible)
                        label_count += 1
                        self.logger.debug(f"[LABEL_TOGGLE] Set label visible={visible}")
                except Exception as e:
                    self.logger.debug(f"[LABEL_TOGGLE] Error toggling label item: {e}")

            status = "shown" if visible else "hidden"
            self.logger.info(f"[LABEL_TOGGLE] Detection labels {status}: {label_count} label items affected")

            # Update scene and viewport
            try:
                self.viewer.scene().update()
                self.viewer.viewport().update()
                self.logger.debug(f"[LABEL_TOGGLE] Scene and viewport updated")
            except Exception as e:
                self.logger.debug(f"[LABEL_TOGGLE] Error updating scene/viewport: {e}")

        except Exception as e:
            self.logger.error(f"[LABEL_TOGGLE] Failed toggling detection labels: {e}", exc_info=True)

    def _on_detection_class_toggled(self, cls: int, state: Any) -> None:
        """Show/hide overlay items belonging to a particular detection class."""
        visible = False
        try:
            visible = (state == Qt.CheckState.Checked.value) or bool(state)
        except Exception as e:
            self.logger.debug(f"Exception converting state, using bool: {e}")
            visible = bool(state)

        try:
            for item in getattr(self.viewer, 'overlay_items', []):
                try:
                    item_cls = getattr(item, '_det_class', None)
                    if item_cls == cls:
                        item.setVisible(visible and self.chk_detector_overlay.isChecked())
                except Exception as e:
                    self.logger.debug(f"Failed to toggle item for class {cls}: {e}")
        except Exception as e:
            self.logger.error(f"Failed toggling class {cls}: {e}", exc_info=True)

    def _on_class_filter_changed(self, class_id: int, state: Any) -> None:
        """Handle class filter checkbox change"""
        pass

    def _on_object_filter_changed(self, object_id: int, state: Any) -> None:
        """Handle object filter checkbox change"""
        pass

    def _on_layer_checkbox_changed(self, state: Any) -> None:
        """Handle show/hide of layers from Display Options"""
        try:
            visible = (state == Qt.CheckState.Checked.value)
        except Exception as e:
            self.logger.debug(f"Exception converting state, using bool: {e}")
            visible = bool(state)

        try:
            if hasattr(self, 'viewer') and self.viewer is not None:
                try:
                    self.viewer.show_raster(visible)
                except Exception as e:
                    self.logger.warning(f"show_raster() failed, trying pixmap_item fallback: {e}")
                    try:
                        if hasattr(self.viewer, 'pixmap_item') and self.viewer.pixmap_item is not None:
                            self.viewer.pixmap_item.setVisible(visible)
                    except Exception as e2:
                        self.logger.error(f"Failed to set pixmap_item visibility: {e2}")

                self.logger.info(f"Layer visibility changed: raster visible={visible}")
        except Exception as e:
            self.logger.error(f"Failed to toggle raster visibility: {e}", exc_info=True)

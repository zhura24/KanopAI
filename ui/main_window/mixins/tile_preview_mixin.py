"""
Tile Preview Mixin for MainWindow
Handles tile preview dialog with overlay visualization
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QListWidget, QDialogButtonBox, 
    QListWidgetItem, QApplication
)
from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QColor, QPen, QBrush, QPolygonF
from PyQt6.QtWidgets import QGraphicsPolygonItem


class TilePreviewMixin:
    """Mixin for handling tile preview dialog and visualization"""

    def show_tile_preview_dialog(self, tiles):
        """Show a modal preview dialog listing prepared tiles and allow overlaying them.

        Returns True if the user chooses to proceed, False if cancelled.
        """
        dlg = QDialog(self)
        dlg.setWindowTitle("Tile Preview")
        v = QVBoxLayout(dlg)

        count = len(tiles)

        # Check if this is polygon mode
        is_polygon_mode = hasattr(self, '_last_inference_mode') and self._last_inference_mode == 'polygon'

        if is_polygon_mode:
            info = QLabel(
                f"Prepared tiles: {count}\n"
                f"Tile size (first): {tiles[0].get('w', '?')}x{tiles[0].get('h','?')} px\n"
                f"Mode: Polygon filtering (tiles intersecting polygon only)\n"
                f"Polygons: {len(self._last_inference_polygons) if self._last_inference_polygons else 0}"
            )
        else:
            info = QLabel(f"Prepared tiles: {count} \nTile size (first): {tiles[0].get('w', '?')}x{tiles[0].get('h','?')} px")

        v.addWidget(info)

        list_w = QListWidget()
        list_w.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        # Show all tiles in the list (user requested no 200-item cap)
        for i, t in enumerate(tiles):
            try:
                txt = f"{i+1}: x={t.get('x')} y={t.get('y')} w={t.get('w')} h={t.get('h')}"
            except Exception as e:
                self.logger.debug(f"Failed to format tile {i} as dict, trying tuple: {e}")
                try:
                    x, y, w, h = t
                    txt = f"{i+1}: x={x} y={y} w={w} h={h}"
                except Exception as e2:
                    self.logger.debug(f"Failed to format tile {i} as tuple: {e2}")
                    txt = f"{i+1}: (tile)"
            item = QListWidgetItem(txt)
            list_w.addItem(item)

        v.addWidget(list_w, 1)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Proceed")
        v.addWidget(buttons)

        # Wire overlay toggle
        def on_toggle_overlay(state):
            enabled = bool(state)
            if enabled:
                try:
                    # Convert to simple rect list with 'type': 'tile' tag
                    rects = []
                    for t in tiles:
                        if isinstance(t, dict):
                            rects.append({
                                'x': t.get('x', 0),
                                'y': t.get('y', 0),
                                'w': t.get('w', 0),
                                'h': t.get('h', 0),
                                'type': 'tile'  # Tag as tile preview overlay
                            })
                        else:
                            rects.append({
                                'x': t[0],
                                'y': t[1],
                                'w': t[2],
                                'h': t[3],
                                'type': 'tile'  # Tag as tile preview overlay
                            })

                    # Use orange color for tile preview (different from purple detections)
                    self.viewer.set_overlay_tiles(
                        rects,
                        outline_color=QColor(255, 165, 0),      # Orange
                        fill_color=QColor(255, 165, 0, 60),     # Semi-transparent orange
                        outline_width=2
                    )

                    # If polygon mode, also draw polygon outline on viewer for reference
                    if is_polygon_mode and hasattr(self, '_last_inference_polygons') and self._last_inference_polygons:
                        try:
                            # Draw each polygon
                            for poly in self._last_inference_polygons:
                                # Convert to QPolygonF
                                qpoly = QPolygonF([QPointF(x, y) for x, y in poly])

                                # Create polygon item
                                poly_item = QGraphicsPolygonItem(qpoly)
                                poly_item.setPen(QPen(QColor(0, 255, 255), 3))  # Cyan outline, thicker
                                poly_item.setBrush(QBrush(QColor(0, 255, 255, 30)))  # Semi-transparent cyan fill
                                poly_item.setZValue(999)  # Draw on top

                                # Tag for easy removal
                                poly_item.setData(0, 'polygon_preview')

                                # Add to scene
                                self.viewer.scene.addItem(poly_item)

                            self.logger.info("Added polygon overlay to tile preview")
                        except Exception as e:
                            self.logger.warning(f"Failed to add polygon overlay: {e}")

                    # Auto-check tile preview checkbox in Display Options
                    try:
                        if hasattr(self, 'chk_tile_preview'):
                            # Block signals to avoid triggering _on_tile_preview_toggled recursively
                            self.chk_tile_preview.blockSignals(True)
                            self.chk_tile_preview.setChecked(True)
                            self.chk_tile_preview.blockSignals(False)
                            self.logger.info("Tile preview checkbox auto-checked")
                    except Exception as e:
                        self.logger.debug(f"Failed to auto-check tile preview: {e}")

                except Exception as e:
                    self.logger.error(f"Failed to set tile preview overlay: {e}")
            else:
                try:
                    # Clear only tile overlays, keep detection overlays
                    self.viewer.clear_overlay(overlay_type='tile')

                    # Clear polygon preview overlays
                    if is_polygon_mode:
                        items_to_remove = []
                        for item in self.viewer.scene.items():
                            if item.data(0) == 'polygon_preview':
                                items_to_remove.append(item)
                        for item in items_to_remove:
                            self.viewer.scene.removeItem(item)
                        self.logger.info(f"Removed {len(items_to_remove)} polygon preview items")

                    # Uncheck tile preview checkbox
                    try:
                        if hasattr(self, 'chk_tile_preview'):
                            # Block signals to avoid triggering _on_tile_preview_toggled recursively
                            self.chk_tile_preview.blockSignals(True)
                            self.chk_tile_preview.setChecked(False)
                            self.chk_tile_preview.blockSignals(False)
                            self.logger.info("Tile preview checkbox auto-unchecked")
                    except Exception as e:
                        self.logger.debug(f"Failed to auto-uncheck tile preview: {e}")
                except Exception as e:
                    self.logger.error(f"Failed to clear tile preview overlay: {e}")

        # Apply overlay automatically when dialog opens
        try:
            on_toggle_overlay(True)
            self.logger.info(f"Tile preview dialog opened - showing {len(tiles)} tile overlays (orange)")
        except Exception as e:
            self.logger.error(f"Failed to show tile overlay on dialog open: {e}")

        def on_accept():
            # Keep tile overlays visible when user proceeds
            self.logger.info("Tile preview dialog accepted - keeping tile overlays visible")
            dlg.accept()

        def on_reject():
            # Keep tile overlays visible even when cancelled (user can manually toggle via Display Options)
            self.logger.info("Tile preview dialog cancelled - tile overlays remain visible (toggle via Display Options)")
            dlg.reject()

        buttons.accepted.connect(on_accept)
        buttons.rejected.connect(on_reject)
        res = dlg.exec()
        return res == QDialog.DialogCode.Accepted

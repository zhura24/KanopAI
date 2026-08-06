"""Handler untuk operasi polygon drawing."""

import logging


class PolygonHandler:
    """Handler untuk mode gambar polygon dan operasi terkait."""
    
    def __init__(self, main_window):
        self.main_window = main_window
        self.logger = logging.getLogger(__name__)
    
    def toggle_draw_polygon_mode(self):
        """Toggle mode gambar polygon on/off."""
        from PyQt6.QtWidgets import QMessageBox
        
        # Check if raster is loaded
        if not hasattr(self.main_window, 'viewer') or not self.main_window.viewer:
            QMessageBox.warning(self.main_window, "No Viewer", "Viewer not initialized")
            return

        # Check if raster file is loaded
        if not self.main_window.raster_loader or not self.main_window.raster_loader.get_metadata():
            QMessageBox.warning(
                self.main_window, "No Raster Loaded",
                "Please load a raster file before drawing polygons.\n\n"
                "Go to File Operations → Open Raster File"
            )
            return

        if self.main_window.polygon_drawing_mode:
            # Cancel drawing mode
            self.main_window.polygon_drawing_mode = False
            self.main_window.viewer.set_polygon_drawing_mode(False)

            # Clear any partial polygon in viewer
            if hasattr(self.main_window.viewer, 'polygon_vertices') and len(self.main_window.viewer.polygon_vertices) > 0:
                # User was in middle of drawing, ask if they want to save partial polygon
                if len(self.main_window.viewer.polygon_vertices) >= 3:
                    reply = QMessageBox.question(
                        self.main_window, "Save Partial Polygon?",
                        f"You have drawn {len(self.main_window.viewer.polygon_vertices)} vertices.\n\n"
                        "Do you want to finish and save this polygon?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.No
                    )
                    if reply == QMessageBox.StandardButton.Yes:
                        # Call finish_polygon_drawing if it exists
                        if hasattr(self.main_window, 'finish_polygon_drawing'):
                            self.main_window.finish_polygon_drawing()
                        return

                # Clear partial polygon
                self.main_window.viewer.clear_polygon()

            if hasattr(self.main_window, 'polygon_panel'):
                self.main_window.polygon_panel.set_drawing_buttons_state(False)
                if self.main_window.drawn_polygons:
                    self.main_window.polygon_panel.update_status(f"Status: {len(self.main_window.drawn_polygons)} polygon(s) drawn")
                else:
                    self.main_window.polygon_panel.update_status("Status: Drawing cancelled")

            self.logger.info("Polygon drawing mode cancelled")
        else:
            # Start drawing mode (multi-polygon support - no need to clear existing)
            self.main_window.polygon_drawing_mode = True
            self.main_window.viewer.set_polygon_drawing_mode(True)

            if hasattr(self.main_window, 'polygon_panel'):
                self.main_window.polygon_panel.set_drawing_buttons_state(True)
                self.main_window.polygon_panel.update_status("Status: Drawing... (left-click: add vertex, right-click/double-click: finish)")
            self.logger.info("Polygon drawing mode started")

    
    def finish_polygon_drawing(self):
        """Finalize the current polygon and persist it on the active layer."""
        from PyQt6.QtWidgets import QMessageBox
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QColor, QBrush, QPen, QPolygonF

        viewer = getattr(self.main_window, 'viewer', None)
        if not viewer:
            self.logger.warning("Cannot finish polygon: viewer not initialized")
            return None

        if len(getattr(viewer, 'polygon_vertices', [])) < 3:
            QMessageBox.warning(
                self.main_window,
                "Incomplete Polygon",
                "Draw at least 3 vertices before finishing the polygon.",
            )
            return None

        # Finish the in-progress polygon when called from the panel button.
        if getattr(viewer, 'polygon_filled_item', None) is None:
            if getattr(viewer, 'polygon_closing_line', None):
                viewer.scene.removeItem(viewer.polygon_closing_line)
                viewer.polygon_closing_line = None

            first_pos = viewer.polygon_vertices[0]
            last_pos = viewer.polygon_vertices[-1]
            line_item = viewer.scene.addLine(
                last_pos.x(), last_pos.y(),
                first_pos.x(), first_pos.y(),
                QPen(self.main_window.polygon_line_color, self.main_window.polygon_line_width)
            )
            viewer.polygon_line_items.append(line_item)

            polygon_shape = QPolygonF(viewer.polygon_vertices)
            viewer.polygon_filled_item = viewer.scene.addPolygon(
                polygon_shape,
                QPen(self.main_window.polygon_line_color, self.main_window.polygon_line_width),
                QBrush(QColor(255, 0, 0, 50))
            )

        polygon_data = viewer.get_drawn_polygon_data()
        if not polygon_data:
            self.logger.warning("Cannot finish polygon: viewer returned no polygon data")
            return None

        polygon_id = self.main_window.polygon_counter + 1
        color = self.main_window.polygon_colors[
            self.main_window.polygon_counter % len(self.main_window.polygon_colors)
        ]

        items = {
            'vertex_items': list(viewer.polygon_vertex_items),
            'line_items': list(viewer.polygon_line_items),
            'closing_line': viewer.polygon_closing_line,
            'filled_item': viewer.polygon_filled_item,
        }

        for item in items['vertex_items']:
            if item:
                item.setData(0, 'polygon')
        for item in items['line_items']:
            if item:
                item.setData(0, 'polygon')
        if items['closing_line']:
            items['closing_line'].setData(0, 'polygon')
        if items['filled_item']:
            items['filled_item'].setData(0, 'polygon')

        polygon = {
            'id': polygon_id,
            'name': f"Polygon {polygon_id}",
            'pixel_coords': polygon_data['pixel_coords'],
            'geo_coords': polygon_data.get('geo_coords', []),
            'area_m2': polygon_data.get('area_m2', 0) or 0,
            'color': color,
            'visible': True,
            'items': items,
        }

        self.main_window.drawn_polygons.append(polygon)
        self.main_window.polygon_counter = polygon_id

        if hasattr(self.main_window, '_update_polygon_colors'):
            self.main_window._update_polygon_colors(polygon)

        # Make sure the area label is generated
        if hasattr(self.main_window, 'updateAreaLabel'):
            self.main_window.updateAreaLabel(polygon)

        # Inject polygon data into vertex items for editability
        for i, v_item in enumerate(items['vertex_items']):
            if hasattr(v_item, 'polygon_data'):
                v_item.polygon_data = polygon
            if v_item:
                v_item.polygon_id = polygon_id
                v_item.vertex_idx = i

        # Detach temporary viewer state without removing the persisted scene items.
        viewer.polygon_vertices = []
        viewer.polygon_vertex_items = []
        viewer.polygon_line_items = []
        viewer.polygon_closing_line = None
        viewer.polygon_filled_item = None
        viewer.polygon_drawing_mode = False
        viewer.setCursor(Qt.CursorShape.ArrowCursor)
        self.main_window.polygon_drawing_mode = False

        if hasattr(self.main_window, 'polygon_panel'):
            self.main_window.polygon_panel.set_drawing_buttons_state(False)
            self.main_window.polygon_panel.update_status(
                f"Status: {len(self.main_window.drawn_polygons)} polygon(s) drawn"
            )
            self.main_window.polygon_panel.set_action_buttons_enabled(True)

        if hasattr(self.main_window, '_refresh_polygon_list_ui'):
            self.main_window._refresh_polygon_list_ui()
        if hasattr(self.main_window, '_update_layer_info_panel'):
            self.main_window._update_layer_info_panel()
        if hasattr(self.main_window, '_update_export_button_state'):
            self.main_window._update_export_button_state()

        self.logger.info(
            "Polygon %s saved with %s vertices",
            polygon_id,
            len(polygon['pixel_coords'])
        )
        return polygon

    def clear_drawn_polygon(self):
        """Clear all saved polygons and the current in-progress polygon."""
        from PyQt6.QtCore import Qt

        for polygon in list(getattr(self.main_window, 'drawn_polygons', [])):
            items = polygon.get('items', {})
            for item in items.get('vertex_items', []):
                if item and item.scene():
                    item.scene().removeItem(item)
            for item in items.get('line_items', []):
                if item and item.scene():
                    item.scene().removeItem(item)
            if items.get('closing_line') and items['closing_line'].scene():
                items['closing_line'].scene().removeItem(items['closing_line'])
            if items.get('filled_item') and items['filled_item'].scene():
                items['filled_item'].scene().removeItem(items['filled_item'])
            if items.get('area_label') and items['area_label'].scene():
                items['area_label'].scene().removeItem(items['area_label'])

        self.main_window.drawn_polygons.clear()
        self.main_window.selected_polygon_ids.clear()

        viewer = getattr(self.main_window, 'viewer', None)
        if viewer:
            viewer.clear_polygon()
            viewer.polygon_drawing_mode = False
            viewer.setCursor(Qt.CursorShape.ArrowCursor)

        self.main_window.polygon_drawing_mode = False

        if hasattr(self.main_window, 'polygon_panel'):
            self.main_window.polygon_panel.set_drawing_buttons_state(False)
            self.main_window.polygon_panel.update_status("Status: No polygon")
            self.main_window.polygon_panel.set_action_buttons_enabled(False)

        if hasattr(self.main_window, '_refresh_polygon_list_ui'):
            self.main_window._refresh_polygon_list_ui()
        if hasattr(self.main_window, '_update_layer_info_panel'):
            self.main_window._update_layer_info_panel()
        if hasattr(self.main_window, '_update_export_button_state'):
            self.main_window._update_export_button_state()

        self.logger.info("All polygons cleared")
    
    
    def save_polygon_to_file(self):
        """Save all drawn polygons to file (GeoJSON or Shapefile).
        
        Opens file dialog for user to choose format and location.
        Exports all drawn polygons with their properties.
        """
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        
        if not self.main_window.drawn_polygons:
            QMessageBox.warning(self.main_window, "No Polygons", "No polygons to export")
            return

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self.main_window,
            f"Export {len(self.main_window.drawn_polygons)} Polygon(s)",
            "",
            "Shapefile (*.shp);;GeoJSON (*.geojson)"
        )

        if not file_path:
            return

        try:
            if selected_filter == "Shapefile (*.shp)" or file_path.lower().endswith('.shp'):
                self._save_polygons_shapefile(file_path)
            elif selected_filter == "GeoJSON (*.geojson)" or file_path.lower().endswith('.geojson'):
                self._save_polygons_geojson(file_path)

            QMessageBox.information(
                self.main_window,
                "Success",
                f"{len(self.main_window.drawn_polygons)} polygon(s) exported to:\n{file_path}"
            )
            self.logger.info(f"{len(self.main_window.drawn_polygons)} polygon(s) exported to: {file_path}")
        except Exception as e:
            QMessageBox.critical(
                self.main_window,
                "Error",
                f"Failed to export polygons:\n{e}"
            )
            self.logger.error(f"Failed to export polygons: {e}", exc_info=True)
    
    def _validate_and_repair_ring(self, ring, polygon_id=None):
        """Ensure a polygon ring is topologically valid before export.

        A self-intersecting ring (e.g. from a stray double-click vertex, or
        from the user's own crossed lines) is treated as valid by pyshp/json
        writers - they'll happily write the coordinates to disk - but QGIS
        (via GEOS) considers it an invalid geometry and silently renders
        nothing, even though the layer's extent/bbox is still correct.

        This repairs invalid rings with the standard buffer(0) technique and
        falls back to the original ring if shapely isn't available or the
        repair itself fails, so export never hard-fails because of this check.
        """
        try:
            from shapely.geometry import Polygon
            from shapely.validation import explain_validity

            poly = Polygon(ring)
            if poly.is_valid and not poly.is_empty:
                return ring

            reason = explain_validity(poly)
            self.logger.warning(
                f"Polygon {polygon_id} is topologically invalid ({reason}) - "
                f"this is why it may show an extent but no shape in QGIS. Attempting auto-repair."
            )

            repaired = poly.buffer(0)
            if repaired.is_empty:
                self.logger.warning(f"Polygon {polygon_id} could not be repaired (result is empty); exporting original coordinates.")
                return ring

            # buffer(0) can turn a bowtie into a MultiPolygon; keep the largest part
            if repaired.geom_type == 'MultiPolygon':
                repaired = max(repaired.geoms, key=lambda g: g.area)

            repaired_ring = list(repaired.exterior.coords)
            self.logger.info(f"Polygon {polygon_id} auto-repaired for export (self-intersection removed).")
            return repaired_ring
        except ImportError:
            self.logger.debug("shapely not available - skipping geometry validity check")
            return ring
        except Exception as e:
            self.logger.debug(f"Geometry validity check failed for polygon {polygon_id}, exporting original coordinates: {e}")
            return ring

    def _save_polygons_geojson(self, file_path):
        """Save all polygons as GeoJSON.
        
        Args:
            file_path: Path to save the GeoJSON file
        """
        import json
        from pyproj import CRS, Transformer

        active_layer = self.main_window._get_active_layer() if hasattr(self.main_window, '_get_active_layer') else None
        metadata = active_layer.get('metadata', {}) if active_layer else {}
        raster_crs = metadata.get('crs')
        to_wgs84 = None
        if raster_crs is not None:
            try:
                to_wgs84 = Transformer.from_crs(
                    CRS.from_user_input(raster_crs),
                    CRS.from_epsg(4326),
                    always_xy=True,
                )
            except Exception as e:
                self.logger.warning(f"Could not create GeoJSON CRS transformer: {e}")
        
        features = []
        for polygon in self.main_window.drawn_polygons:
            geo_coords = polygon.get('geo_coords', [])
            if not geo_coords:
                self.logger.warning(f"Polygon {polygon['id']} has no geographic coordinates, skipping")
                continue
            
            # GeoJSON coordinates must be [longitude, latitude] in WGS84.
            coordinates = []
            for coord in geo_coords:
                x, y = float(coord[0]), float(coord[1])
                if to_wgs84 is not None:
                    x, y = to_wgs84.transform(x, y)
                coordinates.append([float(x), float(y)])
            coordinates.append(coordinates[0])  # Close the ring

            # Guard against self-intersecting geometry (e.g. double-click
            # artifact) that would otherwise show an extent but no shape in QGIS
            coordinates = self._validate_and_repair_ring(coordinates, polygon['id'])
            if coordinates[0] != coordinates[-1]:
                coordinates.append(coordinates[0])

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coordinates]
                },
                "properties": {
                    "id": polygon['id'],
                    "name": polygon['name'],
                    "area_m2": polygon.get('area_m2', 0),
                    "color": polygon['color'].name()
                }
            }
            features.append(feature)
        
        geojson = {
            "type": "FeatureCollection",
            "features": features
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, indent=2)
    
    def _save_polygons_shapefile(self, file_path):
        """Save all polygons as Shapefile.
        
        Args:
            file_path: Path to save the shapefile
        """
        try:
            import shapefile
        except ImportError:
            raise ImportError("pyshp library required for shapefile export. Install: pip install pyshp")

        def _ensure_clockwise(points):
            """ESRI shapefile spec requires exterior rings to be wound clockwise.
            pyshp does NOT reorder coordinates automatically - a counter-clockwise
            ring gets interpreted as a hole (no exterior), which is why QGIS shows
            the layer in the panel but renders nothing. This computes the signed
            area and reverses the ring if it is counter-clockwise."""
            signed_area = 0.0
            n = len(points)
            for i in range(n):
                x1, y1 = points[i]
                x2, y2 = points[(i + 1) % n]
                signed_area += (x1 * y2) - (x2 * y1)
            # Positive signed area (standard y-up convention) = counter-clockwise
            if signed_area > 0:
                return list(reversed(points))
            return points

        w = shapefile.Writer(file_path)
        w.field('id', 'N', 10)
        w.field('name', 'C', 50)
        w.field('area_m2', 'F', 20, 10)
        w.field('color', 'C', 10)

        # Add all polygons
        for polygon in self.main_window.drawn_polygons:
            geo_coords = polygon.get('geo_coords', [])
            if not geo_coords:
                self.logger.warning(f"Polygon {polygon['id']} has no geographic coordinates, skipping")
                continue

            pts = [[float(coord[0]), float(coord[1])] for coord in geo_coords]
            # Ensure closed ring for QGIS shapefile specification
            if len(pts) > 0 and (pts[0][0] != pts[-1][0] or pts[0][1] != pts[-1][1]):
                pts.append(pts[0])

            # Guard against self-intersecting geometry (e.g. double-click
            # artifact) that would otherwise show an extent but no shape in QGIS
            ring = pts[:-1] if pts[0] == pts[-1] else pts
            ring = self._validate_and_repair_ring(ring, polygon['id'])

            # Enforce clockwise winding so QGIS renders the exterior ring
            # instead of treating it as an empty hole
            ring = _ensure_clockwise(ring)
            if ring[0] != ring[-1]:
                ring.append(ring[0])

            # Add polygon geometry
            w.poly([ring])
            w.record(
                polygon['id'],
                polygon['name'],
                polygon.get('area_m2', 0),
                polygon['color'].name()
            )

        w.close()


        # Write .prj file for QGIS compatibility
        prj_text = None
        if hasattr(self.main_window, '_get_active_layer'):
            active_layer = self.main_window._get_active_layer()
            if active_layer and active_layer.get('metadata') and active_layer['metadata'].get('crs'):
                crs = active_layer['metadata']['crs']
                if hasattr(crs, 'to_wkt'):
                    try:
                        prj_text = crs.to_wkt()
                    except Exception:
                        pass
        if not prj_text:
            prj_text = (
                'GEOGCS["WGS 84",'
                'DATUM["WGS_1984",'
                'SPHEROID["WGS 84",6378137,298.257223563]],'
                'PRIMEM["Greenwich",0],'
                'UNIT["degree",0.0174532925199433]]'
            )
        try:
            from pathlib import Path
            Path(file_path).with_suffix('.prj').write_text(prj_text, encoding='utf-8')
        except Exception as prj_err:
            self.logger.warning(f"Failed to write .prj file for exported polygon shapefile: {prj_err}")
    
    def select_all_polygons(self):
        self.logger.info("PolygonHandler.select_all_polygons() - delegating")
        if hasattr(self.main_window, 'select_all_polygons'):
            return self.main_window.select_all_polygons()
    
    def deselect_all_polygons(self):
        self.logger.info("PolygonHandler.deselect_all_polygons() - delegating")
        if hasattr(self.main_window, 'deselect_all_polygons'):
            return self.main_window.deselect_all_polygons()
    
    def delete_selected_polygons(self):
        self.logger.info("PolygonHandler.delete_selected_polygons() - delegating")
        if hasattr(self.main_window, 'delete_selected_polygons'):
            return self.main_window.delete_selected_polygons()

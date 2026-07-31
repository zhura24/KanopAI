"""Polygon Operations Mixin

Handles polygon drawing, management, and export operations.
"""

from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
import logging


class PolygonMixin:
    """Mixin for polygon drawing and management operations.
    
    Pure delegation pattern - ALL logic in PolygonHandler.
    This mixin exists only for method exposure and backward compatibility.
    """
    
    # Polygon UI update methods - keep in mixin for UI rendering
    def _update_polygon_colors(self, polygon):
        """Update the colors of polygon graphics items.
        
        Args:
            polygon: Polygon dict containing 'items' and 'color'
        """
        from PyQt6.QtGui import QBrush, QPen
        
        items = polygon['items']
        color = polygon['color']
        
        # Update filled polygon color (semi-transparent)
        if items.get('filled_item'):
            filled_color = QColor(color)
            filled_color.setAlpha(50)  # Semi-transparent
            brush = QBrush(filled_color)
            items['filled_item'].setBrush(brush)
            
            # Update pen for outline
            pen = QPen(color, self.polygon_line_width)
            items['filled_item'].setPen(pen)
        
        # Update line colors
        for item in items.get('line_items', []):
            if item:
                pen = QPen(color, self.polygon_line_width)
                item.setPen(pen)
        
        # Update closing line
        if items.get('closing_line'):
            pen = QPen(color, self.polygon_line_width)
            pen.setStyle(Qt.PenStyle.DashLine)
            items['closing_line'].setPen(pen)
    
    def _refresh_polygon_list_ui(self):
        """Refresh the polygon list UI via PolygonPanel."""
        if hasattr(self, 'polygon_panel'):
            self.polygon_panel.refresh_polygon_list(self.drawn_polygons, self.selected_polygon_ids)
            return
    
    def _on_polygon_selection_changed(self, polygon_id, state):
        """Handle polygon selection checkbox change.
        
        Args:
            polygon_id: ID of the polygon
            state: Qt.CheckState value
        """
        if state == Qt.CheckState.Checked.value:
            self.selected_polygon_ids.add(polygon_id)
        else:
            self.selected_polygon_ids.discard(polygon_id)
        
        # Update delete button state
        if hasattr(self, 'btn_delete_selected_polygons'):
            self.btn_delete_selected_polygons.setEnabled(len(self.selected_polygon_ids) > 0)
        
        self.logger.info(f"Polygon {polygon_id} {'selected' if state == Qt.CheckState.Checked.value else 'deselected'} for inference")
    
    def _toggle_polygon_visibility(self, polygon_id):
        """Toggle visibility of a specific polygon.
        
        Args:
            polygon_id: ID of the polygon to toggle
        """
        for polygon in self.drawn_polygons:
            if polygon['id'] == polygon_id:
                polygon['visible'] = not polygon['visible']
                
                # Update graphics items visibility
                items = polygon['items']
                visible = polygon['visible']
                
                for item in items.get('vertex_items', []):
                    if item:
                        item.setVisible(visible)
                for item in items.get('line_items', []):
                    if item:
                        item.setVisible(visible)
                if items.get('closing_line'):
                    items['closing_line'].setVisible(visible)
                if items.get('filled_item'):
                    items['filled_item'].setVisible(visible)
                
                self.logger.info(f"Polygon {polygon_id} visibility: {visible}")
                break
        
        # Refresh UI to update button icon
        self._refresh_polygon_list_ui()
    
    def _delete_single_polygon(self, polygon_id):
        """Delete a single polygon.
        
        Args:
            polygon_id: ID of the polygon to delete
        """
        # Find polygon
        polygon = None
        for p in self.drawn_polygons:
            if p['id'] == polygon_id:
                polygon = p
                break
        
        if not polygon:
            return
        
        # Confirm deletion
        reply = QMessageBox.question(
            self, "Delete Polygon?",
            f"Are you sure you want to delete {polygon['name']}?\n\n"
            f"This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.No:
            return
        
        # Remove graphics items from scene
        items = polygon['items']
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
        
        # Remove from lists
        self.drawn_polygons.remove(polygon)
        self.selected_polygon_ids.discard(polygon_id)
        
        # Update UI
        self._refresh_polygon_list_ui()
        if hasattr(self, 'polygon_panel'):
            self.polygon_panel.update_status(f"Status: {len(self.drawn_polygons)} polygon(s) drawn")
        
        # Update layer info panel in real-time
        if hasattr(self, '_update_layer_info_panel'):
            self._update_layer_info_panel()
        
        self.logger.info(f"Polygon {polygon_id} deleted")
    
    def select_all_polygons(self):
        """Select all polygons for inference."""
        for polygon in self.drawn_polygons:
            self.selected_polygon_ids.add(polygon['id'])
        self._refresh_polygon_list_ui()
        self.logger.info("All polygons selected")
    
    def deselect_all_polygons(self):
        """Deselect all polygons."""
        self.selected_polygon_ids.clear()
        self._refresh_polygon_list_ui()
        self.logger.info("All polygons deselected")
    
    def delete_selected_polygons(self):
        """Delete all selected polygons."""
        if not self.selected_polygon_ids:
            return
        
        # Confirm deletion
        reply = QMessageBox.question(
            self, "Delete Selected Polygons?",
            f"Are you sure you want to delete {len(self.selected_polygon_ids)} selected polygon(s)?\n\n"
            f"This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.No:
            return
        
        # Delete each selected polygon
        polygons_to_delete = [p for p in self.drawn_polygons if p['id'] in self.selected_polygon_ids]
        
        for polygon in polygons_to_delete:
            # Remove graphics items
            items = polygon['items']
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
            
            # Remove from list
            self.drawn_polygons.remove(polygon)
        
        # Clear selection
        self.selected_polygon_ids.clear()
        
        # Update UI
        self._refresh_polygon_list_ui()
        if hasattr(self, 'polygon_panel'):
            self.polygon_panel.update_status(f"Status: {len(self.drawn_polygons)} polygon(s) drawn")
        
        # Update layer info panel
        if hasattr(self, '_update_layer_info_panel'):
            self._update_layer_info_panel()
        
        self.logger.info(f"{len(polygons_to_delete)} polygon(s) deleted")
    
    def save_polygon_to_file(self):
        """Export polygons to file - delegated to handler."""
        return self.polygon_handler.save_polygon_to_file()
    
    def _save_polygons_geojson(self, file_path):
        """Helper for GeoJSON export - delegated to handler."""
        return self.polygon_handler._save_polygons_geojson(file_path)
    
    def _save_polygons_shapefile(self, file_path):
        """Helper for Shapefile export - delegated to handler."""
        return self.polygon_handler._save_polygons_shapefile(file_path)
    
    # Delegation methods to PolygonHandler
    def finish_polygon_drawing(self):
        """Delegate to polygon_handler."""
        if hasattr(self, 'polygon_handler'):
            return self.polygon_handler.finish_polygon_drawing()
    
    def save_polygons_to_file(self, file_path, format='geojson'):
        """Delegate to polygon_handler."""
        if format == 'geojson':
            return self._save_polygons_geojson(file_path)
        elif format == 'shapefile':
            return self._save_polygons_shapefile(file_path)
    
    def export_polygons(self):
        """Delegate to export_handler."""
        if hasattr(self, 'export_handler'):
            return self.export_handler.export_polygons()
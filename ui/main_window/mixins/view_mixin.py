"""View operations mixin for zoom, pan, and coordinate display."""

from typing import Optional
import logging


class ViewMixin:
    """Mixin for view operations - zoom, pan, coordinates."""
    
    def zoom_in(self) -> None:
        """Zoom in - delegated to view handler."""
        return self.view_handler.zoom_in()
    
    def zoom_out(self) -> None:
        """Zoom out - delegated to view handler."""
        return self.view_handler.zoom_out()
    
    def reset_view(self) -> None:
        """Reset view to default - delegated to view handler."""
        return self.view_handler.reset_view()
    
    def update_zoom_label(self) -> None:
        """Update zoom label - delegated to view handler."""
        return self.view_handler.update_zoom_label()
    
    def update_scale_label(self) -> None:
        """Update scale display in footer - QGIS style."""
        try:
            if not hasattr(self, 'viewer') or not hasattr(self, 'label_scale'):
                return
            
            zoom = getattr(self.viewer, 'zoom_factor', 1.0)
            
            active_layer = None
            if hasattr(self, 'layers') and hasattr(self, '_active_layer_index'):
                if 0 <= self._active_layer_index < len(self.layers):
                    active_layer = self.layers[self._active_layer_index]
            
            if not active_layer or 'metadata' not in active_layer:
                self.label_scale.setText("Scale: -")
                return
            
            metadata = active_layer['metadata']
            pixel_size = metadata.get('pixel_size_x', None)
            
            if pixel_size is None or pixel_size <= 0:
                self.label_scale.setText("Scale: -")
                return
            
            meters_per_pixel = abs(pixel_size) / zoom
            scale_ratio = int(meters_per_pixel / 0.0002645833)
            
            if scale_ratio < 1:
                scale_ratio = 1
            
            self.label_scale.setText(f"Scale: 1:{scale_ratio:,}")
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.debug(f"Failed to update scale label: {e}")
    
    def update_coordinates(self, pixel_x: float, pixel_y: float, 
                          geo_x: Optional[float], geo_y: Optional[float]) -> None:
        """Update coordinate display in footer."""
        try:
            if geo_x is not None and geo_y is not None:
                active_layer = None
                if hasattr(self, 'layers') and hasattr(self, '_active_layer_index'):
                    if 0 <= self._active_layer_index < len(self.layers):
                        active_layer = self.layers[self._active_layer_index]
                
                if active_layer and 'metadata' in active_layer:
                    metadata = active_layer['metadata']
                    
                    if metadata.get('is_geographic', False):
                        self.label_coordinates.setText(f"Lon: {geo_x:.6f} | Lat: {geo_y:.6f}")
                    else:
                        self.label_coordinates.setText(f"X: {geo_x:.2f} | Y: {geo_y:.2f}")
                else:
                    self.label_coordinates.setText(f"X: {geo_x:.2f} | Y: {geo_y:.2f}")
            else:
                self.label_coordinates.setText("Lon: - | Lat: -")
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.debug(f"Failed to update coordinates: {e}")

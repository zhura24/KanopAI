"""Handler untuk operasi view (zoom, pan)."""

from typing import Any
import logging


class ViewHandler:
    """Handler untuk kontrol zoom dan view."""
    
    def __init__(self, main_window: Any) -> None:
        self.main_window = main_window
        self.logger = logging.getLogger(__name__)
    
    def zoom_in(self) -> None:
        self.main_window.viewer.zoom_in()
        self.update_zoom_label()
        self.logger.debug("Zoom in button clicked")
    
    def zoom_out(self) -> None:
        self.main_window.viewer.zoom_out()
        self.update_zoom_label()
        self.logger.debug("Zoom out button clicked")
    
    def reset_view(self) -> None:
        self.main_window.viewer.reset_zoom()
        self.update_zoom_label()
        self.logger.debug("Reset view button clicked")
    
    def update_zoom_label(self) -> None:
        zoom = self.main_window.viewer.zoom_factor
        self.logger.info(f"[ZOOM] update_zoom_label called, zoom: {zoom:.2f}x")
        if zoom >= 1.0:
            self.main_window.label_zoom.setText(f"Zoom: {zoom:.1f}x")
        else:
            self.main_window.label_zoom.setText(f"Zoom: {zoom:.2f}x")
        
        self.logger.info("[ZOOM] Calling update_scale_label from update_zoom_label")
        self.update_scale_label()
    
    def update_scale_label(self) -> None:
        if hasattr(self.main_window, 'update_scale_label'):
            return self.main_window.update_scale_label()

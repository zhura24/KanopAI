"""Signals and Helpers Mixin

Handles signal connections and common helper methods.
"""

from typing import Optional
from PyQt6.QtWidgets import QMessageBox, QApplication
from PyQt6.QtCore import QEvent
import logging


class SignalsMixin:
    """Mixin for signal connections and helper methods."""
    
    def _connect_signals(self) -> None:
        """Connect all Qt signals to their handlers."""
        # Raster loader signals
        self.raster_loader.error_occurred.connect(self.show_error)
        
        # Viewer signals
        self.viewer.viewport_changed.connect(self.update_zoom_label)
        self.viewer.viewport_changed.connect(self.update_scale_label)
        self.viewer.measurement_finished.connect(self.on_measurement_finished)
        self.viewer.mouse_moved.connect(self.update_coordinates)
        
        # Centroid edit mode signals
        self.viewer.scene_clicked.connect(self._on_scene_clicked)
        
        # Polygon drawing signals
        self.viewer.polygon_finish_requested.connect(self.finish_polygon_drawing)
        
        # Install global event filter for wheel diagnostics
        try:
            app = QApplication.instance()
            if app is not None:
                app.installEventFilter(self)
                self.logger.debug("Global event filter installed on QApplication")
        except Exception as e:
            self.logger.warning(f"Failed to install global event filter: {e}")
    
    def show_error_detailed(self, message: str, details: Optional[str] = None) -> None:
        """Show a critical error message with optional detailed text.
        
        Args:
            message: Main error message
            details: Optional detailed error text (traceback)
        """
        try:
            self.logger.error(message)
            if details:
                self.logger.debug(details)
            
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Critical)
            msg.setWindowTitle("Error")
            msg.setText(str(message))
            if details:
                try:
                    msg.setDetailedText(str(details))
                except Exception as e:
                    self.logger.debug(f"Failed to set detailed text: {e}")
            msg.exec()
        except Exception as e:
            self.logger.error(f"Failed to show detailed error dialog: {e}")
            try:
                QMessageBox.critical(self, "Error", str(message))
            except Exception as e2:
                self.logger.error(f"Failed to show simple error dialog: {e2}")
    
    def show_error(self, message: str) -> None:
        """Backwards-compatible wrapper for show_error_detailed.
        
        Args:
            message: Error message to show
        """
        self.show_error_detailed(message)

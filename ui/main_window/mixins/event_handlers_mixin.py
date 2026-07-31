"""
Event Handlers Mixin for MainWindow

This mixin handles Qt event overrides and filters:
- showEvent: Set focus to viewer on window show
- eventFilter: Global wheel event handling and forwarding
- closeEvent: Clean shutdown with confirmation

Extracted to improve MainWindow modularity - focused on Qt event lifecycle management.
"""

from PyQt6.QtCore import QEvent
from PyQt6.QtWidgets import QApplication, QMessageBox


class EventHandlersMixin:
    """Handles Qt event lifecycle and filtering"""

    def showEvent(self, event):
        """Override showEvent to set focus to viewer when window is shown"""
        super().showEvent(event)
        # Set focus to viewer so wheel zoom works immediately
        self.viewer.setFocus()

    def eventFilter(self, obj, event):
        """Global event filter to log wheel events and their target object for debugging."""
        try:
            if event.type() == QEvent.Type.Wheel:
                try:
                    a = event.angleDelta().y()
                except Exception as e:
                    self.logger.debug(f"Failed to get wheel angle delta: {e}")
                    a = 0
                try:
                    p = event.pixelDelta().y()
                except Exception as e:
                    self.logger.debug(f"Failed to get wheel pixel delta: {e}")
                    p = 0

                # Widget under global cursor position (best-effort)
                try:
                    gp = event.globalPosition().toPoint()
                    widget_at = QApplication.widgetAt(gp)
                except Exception as e:
                    self.logger.debug(f"Failed to get widget at cursor position: {e}")
                    widget_at = None

                self.logger.debug(
                    f"[GLOBAL WHEEL] target={obj.__class__.__name__} | angleDelta={a} | pixelDelta={p} | widgetAtCursor={widget_at.__class__.__name__ if widget_at else None}"
                )

                # Fallback: if the wheel event occurred over the viewer area but
                # some other widget swallowed it, forward it to the viewer so
                # zoom still works. This is a best-effort attempt to avoid
                # focus-stealing or overlay widgets preventing wheel zoom.
                try:
                    gp = event.globalPosition().toPoint()
                    # Map global point into the viewer's viewport coordinates
                    if self.viewer is not None:
                        try:
                            vp_pt = self.viewer.mapFromGlobal(gp)
                            vw = self.viewer.viewport().width()
                            vh = self.viewer.viewport().height()
                            if 0 <= vp_pt.x() <= vw and 0 <= vp_pt.y() <= vh:
                                # Log at INFO so the existing run (INFO level) will show
                                self.logger.info("[GLOBAL WHEEL] Forwarding wheel event to RasterViewer (fallback)")
                                try:
                                    # Deliver event by calling the viewer's wheelEvent
                                    # directly. Some platforms/widgets will not accept
                                    # a generic sendEvent for wheel; calling the handler
                                    # ensures our viewer logic runs.
                                    try:
                                        self.viewer.wheelEvent(event)
                                        return True
                                    except Exception as e:
                                        self.logger.debug(f"Failed to call wheelEvent directly, trying postEvent: {e}")
                                        # Fallback to posting the event to the viewport
                                        QApplication.postEvent(self.viewer.viewport(), event)
                                        return True
                                except Exception as e:
                                    self.logger.debug(f"Failed to forward wheel event to viewer: {e}")
                        except Exception as e:
                            self.logger.debug(f"Failed to check if event over viewer area: {e}")
                except Exception as e:
                    self.logger.debug(f"Failed to get global position for wheel forwarding: {e}")

        except Exception as e:
            self.logger.error(f"Error in eventFilter: {e}", exc_info=True)

        return super().eventFilter(obj, event)

    def closeEvent(self, event):
        """Handle application close event with cleanup"""
        # Confirm exit with the user
        reply = QMessageBox.question(
            self,
            "Exit Application",
            "Are you sure you want to exit?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Stop any running workers
            try:
                if self.detection_worker is not None and self.detection_worker.isRunning():
                    self.detection_worker.quit()
                    self.detection_worker.wait()
            except Exception as e:
                self.logger.warning(f"Failed to stop detection worker: {e}")

            # Close raster_loader if available
            try:
                if self.raster_loader and hasattr(self.raster_loader, 'close'):
                    self.raster_loader.close()
            except Exception as e:
                self.logger.warning(f"Failed to close raster loader: {e}")

            event.accept()
        else:
            event.ignore()

from typing import Optional, Any, Tuple, List
# pyrefly: ignore [missing-import]
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsItem, QGraphicsRectItem, QGraphicsTextItem
from PyQt6.QtWidgets import QToolTip
from typing import Optional, Any, Tuple, List
# pyrefly: ignore [missing-import]
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsItem, QGraphicsRectItem, QGraphicsTextItem, QGraphicsEllipseItem
from PyQt6.QtWidgets import QToolTip
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal, QTimer, QEvent
from PyQt6.QtGui import QPixmap, QImage, QPainter, QWheelEvent, QCursor, QBrush, QColor, QPen, QFont, QTransform
import numpy as np
from numpy.typing import NDArray
from core.tile_manager import TileManager
from core.tile_loader import AsyncTileLoader
from ui.measurement_tool import MeasurementManager
from utils.logger_config import get_logger

class PolygonVertexItem(QGraphicsEllipseItem):
    def __init__(self, scene_pos, size, pen, brush, viewer):
        half_size = size / 2
        # Set Rect to center at origin so pos() is exactly the scene pos
        super().__init__(-half_size, -half_size, size, size)
        self.setPos(scene_pos)
        self.setPen(pen)
        self.setBrush(brush)
        self.viewer = viewer
        
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, False)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setZValue(101) # Above lines
        
        self.polygon_data = None
        
    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            if self.polygon_data is not None and hasattr(self.viewer, 'refreshPolygonGeometry'):
                self.viewer.refreshPolygonGeometry(self.polygon_data)
        return super().itemChange(change, value)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        # When drag is finished, update the UI sidebar
        window = self.viewer.window()
        if hasattr(window, '_refresh_polygon_list_ui'):
            window._refresh_polygon_list_ui()


class RasterViewer(QGraphicsView):
    viewport_changed = pyqtSignal()
    measurement_finished = pyqtSignal(dict)  # Emits measurement info when finished
    mouse_moved = pyqtSignal(int, int, float, float)  # pixel_x, pixel_y, geo_x, geo_y
    scene_clicked = pyqtSignal(float, float)  # Emits scene position when clicked (for centroid add/delete)
    polygon_finish_requested = pyqtSignal()  # Emits when polygon drawing is finished

    def __init__(self, parent: Optional[Any] = None) -> None:
        super().__init__(parent)
        self.logger = get_logger(__name__)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        # Set initial scene rect so zoom works even without image
        self.scene.setSceneRect(-1000, -1000, 2000, 2000)

        # Enable mouse tracking to get mouse move events without clicking
        self.setMouseTracking(True)

        # Accept wheel events for zooming
        self.viewport().setMouseTracking(True)

        # CRITICAL: Set focus policy to accept wheel events
        # StrongFocus accepts focus by clicking AND programmatically
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.viewport().setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Enable mouse tracking for auto-focus
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        # IMPORTANT: Use NoDrag so wheel events work properly
        # Panning will be handled by Middle Mouse Button
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Install event filter on viewport to intercept wheel events (QGIS style)
        self.viewport().installEventFilter(self)

        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.MinimalViewportUpdate)
        self.setCacheMode(QGraphicsView.CacheModeFlag.CacheBackground)

        self.pixmap_item = None
        # Raster visibility flag: if False, newly set images will be hidden until enabled
        self._raster_visible = True
        self.zoom_factor = 1.0
        self.min_zoom = 0.05
        self.max_zoom = 500.0  # Much higher max zoom for detailed inspection
        self.pixel_zoom_threshold = 5.0

        self._is_panning = False
        self._pan_start_pos = None

        # Overlay subsystem removed per user request
        self.tile_manager = None
        self.use_full_resolution = False
        self.current_display_data = None

        self.async_tile_loader = AsyncTileLoader(self)
        self.async_tile_loader.tiles_ready.connect(self._on_tiles_ready)

        self.viewport_update_timer = QTimer()
        self.viewport_update_timer.setSingleShot(True)
        self.viewport_update_timer.timeout.connect(self._delayed_tile_load)
        self.debounce_delay = 100  # Standard debounce

        # User activity tracking: while user is actively zooming/panning we
        # avoid triggering heavy immediate tile loads to keep the UI responsive.
        self.user_activity_timer = QTimer()
        self.user_activity_timer.setSingleShot(True)
        self.user_activity_timer.timeout.connect(self._on_user_activity_stopped)
        self.user_activity_debounce = 350  # ms to wait after last interaction
        self._user_active = False

        self.zoom_timer = QTimer()
        self.zoom_timer.setSingleShot(True)
        self.zoom_timer.timeout.connect(self._on_zoom_finished)
        self.zoom_debounce = 200  # Long delay: allows smooth continuous zoom without reloading

        self.pan_timer = QTimer()
        self.pan_timer.setSingleShot(True)
        self.pan_timer.timeout.connect(self._on_pan_finished)
        self.pan_debounce = 50  # Quick panning

        self.tile_pixmap_items = {}
        self.current_tile_keys = set()
        self.current_render_level = 0
        # Overlay items (rectangles used for tile previews / detections)
        self.overlay_items = []
        self._overlay_visible = False
        self.global_statistics = None  # Cache global statistics
        self.use_color_normalization = True

        self.is_zooming = False
        self.last_viewport_rect = None
        self.viewport_move_threshold = 50  # Reduced from 100px to 50px
        self.last_pan_load_time = 0  # Track last time we loaded during pan

        # Timer to continuously check viewport during ScrollHandDrag panning
        self.pan_check_timer = QTimer()
        self.pan_check_timer.setInterval(100)  # Check every 100ms during panning
        self.pan_check_timer.timeout.connect(self._on_pan_check_timeout)

        # Measurement tool
        self.measurement_manager = MeasurementManager(self.scene)
        self.measurement_mode = False
        self._is_measuring = False
        self._measurement_first_point_set = False  # Track if first point is set in click mode

        # Polygon drawing
        self.polygon_drawing_mode = False
        self.polygon_vertices = []  # List of QPointF (scene coordinates)
        self.polygon_vertex_items = []  # QGraphicsEllipseItem for vertices
        self.polygon_line_items = []  # QGraphicsLineItem for edges
        self.polygon_closing_line = None  # Temporary line from last to first vertex
        self.polygon_filled_item = None  # Final filled polygon
        self.polygon_vertex_color = QColor(255, 255, 0)  # Yellow
        self.polygon_vertex_outline_color = QColor(255, 0, 0)  # Red
        self.polygon_line_color = QColor(255, 0, 0)  # Red
        self.polygon_vertex_size = 30
        self.polygon_line_width = 5

        # Blocker item used to completely hide raster image when layer is toggled off
        self._raster_blocker = None

    def set_image(self, numpy_array, update_overlay=True):
        if numpy_array is None:
            return

        self.current_display_data = numpy_array

        try:
            # Log array shape for debugging
            if hasattr(self, 'logger'):
                if len(numpy_array.shape) == 3:
                    num_bands, height, width = numpy_array.shape
                    self.logger.info(f"[VIEWER] Setting image: {num_bands} bands, {width}x{height} pixels")
                else:
                    height, width = numpy_array.shape
                    self.logger.info(f"[VIEWER] Setting image: Single band, {width}x{height} pixels")

            if len(numpy_array.shape) == 3:
                if numpy_array.shape[0] >= 3:
                    # Multi-band image (RGB or more).
                    # Per product spec there is NO manual/auto band selector:
                    # display always uses the literal band 1 -> R, band 2 -> G,
                    # band 3 -> B order, identical to the full-resolution tile
                    # pipeline (core/tile_loader.py) and the RGB overview
                    # pre-calculated in main_window_impl.py. This keeps the
                    # overview and the streamed-in tiles visually identical.
                    num_bands = numpy_array.shape[0]
                    if hasattr(self, 'logger') and num_bands > 3:
                        self.logger.info("[VIEWER] Using bands 1-3 for RGB display")

                    r = self._normalize_band(numpy_array[0], band_idx=0)
                    g = self._normalize_band(numpy_array[1], band_idx=1)
                    b = self._normalize_band(numpy_array[2], band_idx=2)

                    height, width = r.shape
                    rgb_array = np.zeros((height, width, 3), dtype=np.uint8)
                    rgb_array[:, :, 0] = r
                    rgb_array[:, :, 1] = g
                    rgb_array[:, :, 2] = b

                    qimage = QImage(rgb_array.data, width, height, width * 3, QImage.Format.Format_RGB888)
                else:
                    # 1 or 2 band image - display as grayscale.
                    # For 2 bands, average both instead of discarding the
                    # second one, so it actually contributes to the preview.
                    if numpy_array.shape[0] == 2:
                        if hasattr(self, 'logger'):
                            self.logger.info("[VIEWER] 2-band raster: averaging band 1+2 for grayscale display")
                        b1 = numpy_array[0].astype(np.float32)
                        b2 = numpy_array[1].astype(np.float32)
                        band = self._normalize_band((b1 + b2) / 2.0, band_idx=0)
                    else:
                        if hasattr(self, 'logger'):
                            self.logger.info("[VIEWER] Using band 1 for grayscale display")
                        band = self._normalize_band(numpy_array[0], band_idx=0)
                    height, width = band.shape
                    qimage = QImage(band.data, width, height, width, QImage.Format.Format_Grayscale8)
            else:
                # 2D array - single band grayscale
                band = self._normalize_band(numpy_array, band_idx=0)
                height, width = band.shape
                qimage = QImage(band.data, width, height, width, QImage.Format.Format_Grayscale8)

            pixmap = QPixmap.fromImage(qimage.copy())

            if self.pixmap_item:
                self.scene.removeItem(self.pixmap_item)

            self.pixmap_item = QGraphicsPixmapItem(pixmap)
            # Respect stored raster visibility flag (may be toggled before image loaded)
            try:
                self.pixmap_item.setVisible(self._raster_visible)
            except Exception as e:
                self.logger.debug(f"Failed to set pixmap visibility: {e}")
            self.pixmap_item.setZValue(0)
            self.scene.addItem(self.pixmap_item)
            self.scene.setSceneRect(self.pixmap_item.boundingRect())

            self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self.zoom_factor = 1.0

            # When an image is set, enable left-drag panning and show open-hand cursor
            if not self.measurement_mode:
                try:
                    self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
                    self.setCursor(Qt.CursorShape.OpenHandCursor)
                except Exception as e:
                    # Ignore if platform doesn't support hand cursors
                    self.logger.debug(f"Failed to set drag mode or cursor: {e}")

            # Ensure the view gets keyboard and wheel focus so wheel & drag work immediately
            try:
                self.setFocus()
                self.viewport().setFocus()
            except Exception as e:
                self.logger.debug(f"Failed to set focus: {e}")

            # Also do a short delayed focus in case another widget steals focus right after load
            try:
                QTimer.singleShot(50, self._deferred_focus)
            except Exception as e:
                self.logger.debug(f"Failed to schedule deferred focus: {e}")

        except Exception as e:
            error_msg = f"Error setting image: {e}"
            if hasattr(self, 'logger'):
                self.logger.error(error_msg, exc_info=True)
            import traceback
            traceback.print_exc()

        # After image set, ensure overlay visibility respected
        try:
            if getattr(self, '_overlay_visible', False):
                for it in getattr(self, 'overlay_items', []):
                    try:
                        it.setVisible(True)
                    except Exception as e:
                        self.logger.debug(f"Failed to set overlay item visibility: {e}")
        except Exception as e:
            self.logger.debug(f"Failed to restore overlay visibility: {e}")

    def _all_required_tiles_loaded(self) -> bool:
        """Check if all required tiles for current viewport are loaded in cache."""
        if not self.current_tile_keys:
            return False
        for k in self.current_tile_keys:
            tx, ty = k
            if self.async_tile_loader.get_loaded_pixmap(tx, ty) is None:
                return False
        return True

    def set_smart_rgb_overview(self, pixmap: QPixmap):
        """Update the overview image instantly using a pre-calculated RGB pixmap."""
        if not pixmap or pixmap.isNull():
            return

        self.logger.info(f"[VIEWER] Updating overview with pre-calculated Smart RGB pixmap")
        
        if not self.pixmap_item:
            self.pixmap_item = QGraphicsPixmapItem(pixmap)
            self.pixmap_item.setVisible(self._raster_visible)
            self.pixmap_item.setZValue(2)
            self.scene.addItem(self.pixmap_item)
        else:
            self.pixmap_item.setPixmap(pixmap)
            self.pixmap_item.setVisible(self._raster_visible)

        if self.use_full_resolution and self.tile_manager and self.tile_manager.raster_loader:
            try:
                metadata = self.tile_manager.raster_loader.get_metadata()
                orig_w = metadata.get('width', 0)
                orig_h = metadata.get('height', 0)
                if orig_w > 0 and orig_h > 0 and pixmap.width() > 0 and pixmap.height() > 0:
                    scale_x = orig_w / pixmap.width()
                    scale_y = orig_h / pixmap.height()
                    self.pixmap_item.setPos(0, 0)
                    self.pixmap_item.setTransform(QTransform().scale(scale_x, scale_y))
                    self.pixmap_item.setZValue(2)
                    self.pixmap_item.setOpacity(0.98)
                    self._preview_active = True
                    self.scene.setSceneRect(QRectF(0, 0, orig_w, orig_h))

                    # If all required tiles are already loaded, hide the preview immediately
                    if self._all_required_tiles_loaded():
                        self.pixmap_item.setVisible(False)
                        self._preview_active = False
            except Exception as e:
                self.logger.debug(f"Failed to scale smart RGB overview: {e}")
        else:
            self.pixmap_item.setTransform(QTransform())
            self.scene.setSceneRect(self.pixmap_item.boundingRect())

    def _normalize_band(self, band, band_idx=0):
        """Convert band to uint8 for display with robust Float32 and NoData handling."""
        band = band.astype(np.float32)

        # 1. Abaikan nilai NoData, 0, atau NaN untuk mencari persentil yang akurat
        valid_pixels = band[(band != 0) & (~np.isnan(band))]
        if len(valid_pixels) == 0:
            valid_pixels = band # Fallback jika seluruh piksel bernilai 0

        if not self.use_color_normalization:
            # Penanganan rentang data mentah (True Color / Raw)
            max_val = np.max(valid_pixels) if len(valid_pixels) > 0 else 1.0
            if max_val <= 1.0:
                band = band * 255
            elif max_val <= 255:
                band = np.clip(band, 0, 255)
            elif max_val <= 65535:
                band = (band / 65535.0) * 255
            else:
                band = (band / max_val) * 255 if max_val > 0 else band
            return np.clip(band, 0, 255).astype(np.uint8)

        # 2. Ambil statistik global atau hitung dari valid_pixels
        if self.global_statistics and band_idx < len(self.global_statistics):
            p2 = self.global_statistics[band_idx]['p2']
            p98 = self.global_statistics[band_idx]['p98']
        else:
            p2, p98 = np.percentile(valid_pixels, (2, 98))

        # 3. Mencegah error pembagian dengan nol atau rentang terlalu sempit.
        # PENTING: jangan hanya menambah epsilon kecil ke p98 di sini — kalau rentang
        # [p2, p98] dipaksa jadi nyaris nol lebar, piksel data asli yang jauh di atas p2
        # akan langsung ke-clip ke ujung atas (255) sementara piksel dekat p2 tetap 0,
        # sehingga hasilnya jadi biner hitam-putih, bukan gradasi grayscale. Kalau
        # rentangnya memang degenerate (p98 <= p2), lebih aman tampilkan abu-abu rata.
        if p98 <= p2:
            return np.full_like(band, 128, dtype=np.uint8)

        # 4. Lakukan kliping dan peregangan kontras (contrast stretching) ke 0-255
        band = np.clip(band, p2, p98)
        band = (band - p2) / (p98 - p2) * 255

        return np.clip(band, 0, 255).astype(np.uint8)

    def event(self, event):
        """Fallback event handler to catch Wheel events early.

        Some graphics item configurations or platform widget behaviors can
        prevent wheel events from reaching wheelEvent()/eventFilter. Catching
        Wheel here ensures zoom remains responsive even when tiles are being
        added to the scene.
        """
        try:
            if event.type() == QEvent.Type.Wheel:
                # Call wheelEvent to handle zoom uniformly
                try:
                    self.wheelEvent(event)
                except Exception as e:
                    self.logger.debug(f"Failed to handle wheel event: {e}")
                try:
                    event.accept()
                except Exception as e:
                    self.logger.debug(f"Failed to accept wheel event: {e}")
                return True
        except Exception as e:
            self.logger.debug(f"Error in event handler: {e}")

        return super().event(event)

    def eventFilter(self, source, event):
        """Event filter to intercept wheel events (QGIS style)

        This ensures wheel events are captured even if other widgets try to steal them.
        """
        if event.type() == QEvent.Type.Wheel and source is self.viewport():
            # Log both angleDelta and pixelDelta for diagnostics
            try:
                a = event.angleDelta().y()
                p = event.pixelDelta().y()
            except Exception as e:
                self.logger.debug(f"Failed to get wheel delta: {e}")
                a = 0
                p = 0
            self.logger.debug(f"[EVENT FILTER] Wheel event intercepted: angleDelta={a}, pixelDelta={p}, source={source}")
            # Process the wheel event directly
            self._handle_wheel_zoom(event)
            return True  # Event handled, don't propagate

        return super().eventFilter(source, event)

    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zooming - with debug logging"""
        try:
            a = event.angleDelta().y()
            p = event.pixelDelta().y()
        except Exception as e:
            self.logger.debug(f"Failed to get wheel angle/pixel delta: {e}")
            a = 0
            p = 0
        # PERF: dulu di-log INFO tiap wheel tick (komentar lama bilang "temporary,
        # revert to debug later" tapi kelupaan). Sekarang debug aja + 1 baris
        # (duplikatnya dihapus) -- tidak ada logic yang berubah.
        self.logger.debug(f"[WHEEL EVENT] angleDelta={a}, pixelDelta={p}, current_zoom={self.zoom_factor:.2f}")
        # Mark user activity so we can throttle tile loads while user is
        # continuously interacting (e.g., spinning the touchpad/wheel).
        try:
            self._set_user_active()
        except Exception as e:
            self.logger.debug(f"Failed to set user active state: {e}")

        self._handle_wheel_zoom(event)

    def _handle_wheel_zoom(self, event: QWheelEvent):
        """Core zoom logic - used by both wheelEvent and eventFilter"""
        # Zoom in or out based on wheel direction
        try:
            delta_y = event.angleDelta().y()
        except Exception as e:
            self.logger.debug(f"Failed to get angle delta: {e}")
            delta_y = 0

        # Some devices (touchpads) may report pixelDelta instead of angleDelta
        if delta_y == 0:
            try:
                delta_y = event.pixelDelta().y()
            except Exception as e:
                self.logger.debug(f"Failed to get pixel delta fallback: {e}")
                delta_y = 0

        # Normalize delta to sign only (support small deltas)
        if delta_y > 0:
            zoom_factor = 1.15  # Zoom in
            direction = "IN"
        elif delta_y < 0:
            zoom_factor = 0.85  # Zoom out
            direction = "OUT"
        else:
            # No movement -> ignore
            self.logger.debug("Wheel event with zero delta ignored")
            return

        new_zoom = self.zoom_factor * zoom_factor

        if self.min_zoom <= new_zoom <= self.max_zoom:
            # Apply zoom transformation
            self.scale(zoom_factor, zoom_factor)
            self.zoom_factor = new_zoom
            self._update_render_hints()
            self.viewport_changed.emit()

            # PERF: sama seperti [WHEEL EVENT] di atas, ini jalan tiap wheel tick
            # yang valid -- diturunkan ke debug biar gak spam I/O saat scroll aktif.
            self.logger.debug(f"Zoom {direction}: {self.zoom_factor:.2f}x (factor: {zoom_factor:.2f})")

            if self.use_full_resolution:
                self.is_zooming = True
                # Restart timer - tiles only load after zoom stops
                self.zoom_timer.start(self.zoom_debounce)

                # While user is actively zooming we avoid immediate tile loading.
                # Mark activity (already done in wheelEvent) and ensure a delayed
                # load will run once activity stops.
                try:
                    self.user_activity_timer.start(self.user_activity_debounce)
                except Exception as e:
                    self.logger.debug(f"Failed to start user activity timer: {e}")
        else:
            self.logger.debug(f"Zoom {direction} blocked: new_zoom={new_zoom:.2f} outside limits [{self.min_zoom}, {self.max_zoom}]")

        # Ensure the view has focus (some systems change focus on wheel)
        try:
            self.setFocus()
        except Exception as e:
            self.logger.debug(f"Failed to set focus after zoom: {e}")

        # CRITICAL: Accept event to prevent it from propagating
        try:
            event.accept()
        except Exception as e:
            self.logger.debug(f"Failed to accept event: {e}")

    def _update_render_hints(self):
        if self.zoom_factor > self.pixel_zoom_threshold:
            self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        else:
            self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

    def enterEvent(self, event):
        """Auto-focus viewer when mouse enters (CRITICAL for wheel zoom to work!)"""
        # Automatically give focus to viewer when mouse enters
        # This ensures wheel zoom always works, even after clicking sidebar
        self.setFocus()
        self.logger.debug("[ENTER EVENT] Viewer auto-focused for wheel zoom")
        super().enterEvent(event)

    def mousePressEvent(self, event):
        # Allow graphics items like PolygonVertexItem to handle clicks first if they are movable
        item = self.itemAt(event.pos())
        if hasattr(item, 'flags') and (item.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsMovable):
            super().mousePressEvent(event)
            return

        # Handle polygon drawing mode
        if self.polygon_drawing_mode:
            if event.button() == Qt.MouseButton.LeftButton:
                scene_pos = self.mapToScene(event.pos())
                self._add_polygon_vertex(scene_pos)
                event.accept()
                return
            elif event.button() == Qt.MouseButton.RightButton:
                # Right-click: remove last vertex (undo)
                self._remove_last_polygon_vertex()
                event.accept()
                return

        # Handle measurement mode
        if self.measurement_mode:
            if event.button() == Qt.MouseButton.LeftButton:
                scene_pos = self.mapToScene(event.pos())

                if not self._is_measuring:
                    # First click: start measurement
                    self.measurement_manager.start_measurement(scene_pos)
                    self._is_measuring = True
                    self._measurement_first_point_set = True
                else:
                    # Second click: finish measurement (click-click mode)
                    measurement_info = self.measurement_manager.finish_measurement()
                    self._is_measuring = False
                    self._measurement_first_point_set = False
                    if measurement_info:
                        self.measurement_finished.emit(measurement_info)

                event.accept()
                return

            elif event.button() == Qt.MouseButton.RightButton:
                # Right-click: cancel current measurement or delete last measurement
                if self._is_measuring:
                    # Cancel current measurement
                    self.measurement_manager.cancel_measurement()
                    self._is_measuring = False
                    self._measurement_first_point_set = False
                else:
                    # Delete last measurement from the list
                    self.measurement_manager.delete_last_measurement()

                event.accept()
                return

        # Emit scene_clicked signal for other tools (e.g., centroid add/delete mode)
        if event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            self.scene_clicked.emit(scene_pos.x(), scene_pos.y())

        if event.button() == Qt.MouseButton.MiddleButton:
            self._is_panning = True
            self._pan_start_pos = QPointF(event.pos())
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            # Disable smooth transform during panning for better performance
            self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
            # Set viewport update mode for smoother panning
            self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
            # Start continuous viewport checking
            if self.use_full_resolution:
                self.pan_check_timer.start()
            # mark user activity when starting an explicit pan
            try:
                self._set_user_active()
            except Exception as e:
                self.logger.debug(f"Failed to set user active on pan start: {e}")
        elif event.button() == Qt.MouseButton.LeftButton:
            # Left button with ScrollHandDrag mode
            if self.dragMode() == QGraphicsView.DragMode.ScrollHandDrag and not self.measurement_mode:
                self._is_panning = True
                self._pan_start_pos = QPointF(event.pos())
                # Show closed hand cursor while dragging
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                # Disable smooth transform during panning
                self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
                self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
                if self.use_full_resolution:
                    self.pan_check_timer.start()
                try:
                    self._set_user_active()
                except Exception as e:
                    self.logger.debug(f"Failed to set user active on drag start: {e}")

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        # Get scene position for coordinate tracking
        scene_pos = self.mapToScene(event.pos())
        pixel_x = int(scene_pos.x())
        pixel_y = int(scene_pos.y())

        # Convert pixel to lat/lon coordinates (QGIS style)
        lon, lat = self.measurement_manager.pixel_to_latlon(pixel_x, pixel_y)

        # Emit signal with both pixel and geographic coordinates (lon, lat in degrees)
        self.mouse_moved.emit(pixel_x, pixel_y, lon, lat)

        # Handle measurement mode - update line as cursor moves
        if self.measurement_mode and self._is_measuring:
            self.measurement_manager.update_measurement(scene_pos)
            event.accept()
            return

        if self._is_panning and self._pan_start_pos is not None:
            # Convert to QPointF for consistent type handling
            current_pos = QPointF(event.pos())
            delta = current_pos - self._pan_start_pos
            self._pan_start_pos = current_pos

            self.horizontalScrollBar().setValue(int(self.horizontalScrollBar().value() - delta.x()))
            self.verticalScrollBar().setValue(int(self.verticalScrollBar().value() - delta.y()))

            # Check if viewport has moved significantly
            if self.use_full_resolution:
                self._check_viewport_change()

            # mark user activity while panning
            try:
                self._set_user_active()
            except Exception as e:
                self.logger.debug(f"Failed to set user active while panning: {e}")
        else:
            super().mouseMoveEvent(event)

    def mouseDoubleClickEvent(self, event):
        """Handle double-click to finish polygon drawing"""
        if self.polygon_drawing_mode and event.button() == Qt.MouseButton.LeftButton:
            self._finish_polygon()
            self.polygon_drawing_mode = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return

        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event):
        """Handle keyboard events"""
        # Handle Enter/Return key to finish polygon
        if self.polygon_drawing_mode:
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                if len(self.polygon_vertices) >= 3:
                    self._finish_polygon()
                    self.polygon_drawing_mode = False
                    self.setCursor(Qt.CursorShape.ArrowCursor)
                    event.accept()
                    return
                else:
                    self.logger.warning("Need at least 3 vertices to finish polygon")
                    event.accept()
                    return
            elif event.key() == Qt.Key.Key_Escape:
                # ESC: cancel polygon drawing
                self.clear_polygon()
                self.polygon_drawing_mode = False
                self.setCursor(Qt.CursorShape.ArrowCursor)
                self.logger.info("Polygon drawing cancelled with ESC")
                event.accept()
                return

        super().keyPressEvent(event)

    def mouseReleaseEvent(self, event):
        # Note: Measurement mode now uses click-click (handled in mousePressEvent)
        # No drag-mode handling needed here

        if event.button() == Qt.MouseButton.MiddleButton or event.button() == Qt.MouseButton.LeftButton:
            if self._is_panning:
                self._is_panning = False
                self._pan_start_pos = None
                # Restore open hand cursor if ScrollHandDrag is active, otherwise arrow
                if self.dragMode() == QGraphicsView.DragMode.ScrollHandDrag and not self.measurement_mode:
                    self.setCursor(Qt.CursorShape.OpenHandCursor)
                else:
                    self.setCursor(Qt.CursorShape.ArrowCursor)

                # Stop continuous checking
                self.pan_check_timer.stop()

                # Restore render hints after panning
                self._update_render_hints()
                # Restore viewport update mode
                self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.MinimalViewportUpdate)

                if self.use_full_resolution:
                    # Load tiles immediately after panning finishes
                    self._delayed_tile_load(immediate=True)

        super().mouseReleaseEvent(event)

    def reset_zoom(self):
        if self.pixmap_item:
            self.resetTransform()
            self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self.zoom_factor = 1.0
            self.viewport_changed.emit()

    def zoom_in(self):
        zoom_factor = 1.25
        new_zoom = self.zoom_factor * zoom_factor
        if new_zoom <= self.max_zoom:
            self.scale(zoom_factor, zoom_factor)
            self.zoom_factor = new_zoom
            self._update_render_hints()
            self.viewport_changed.emit()

            if self.use_full_resolution:
                self.is_zooming = True
                self.zoom_timer.start(self.zoom_debounce)

    def zoom_out(self):
        zoom_factor = 0.8
        new_zoom = self.zoom_factor * zoom_factor
        if new_zoom >= self.min_zoom:
            self.scale(zoom_factor, zoom_factor)
            self.zoom_factor = new_zoom
            self._update_render_hints()
            self.viewport_changed.emit()

            if self.use_full_resolution:
                self.is_zooming = True
                self.zoom_timer.start(self.zoom_debounce)

    def set_tile_manager(self, raster_loader):
        if raster_loader and raster_loader.dataset:
            # Increase tile_size to improve per-tile resolution (trades memory for detail).
            # If this causes memory pressure on low-RAM machines, reduce back to 1024.
            self.tile_manager = TileManager(raster_loader, tile_size=512)
            self.async_tile_loader.set_tile_manager(self.tile_manager)
            # Statistics are prepared by QuickRasterPreviewWorker. Never
            # recompute them synchronously while the GUI is being updated.
            self.global_statistics = (
                raster_loader.global_statistics
                if self.use_color_normalization
                else None
            )
            self.tile_manager.global_statistics = self.global_statistics
            # Sync normalization setting to tile loader
            self.async_tile_loader.use_color_normalization = self.use_color_normalization

            # Update measurement metadata for accurate measurements
            metadata = raster_loader.get_metadata()
            self.measurement_manager.set_metadata(metadata)
        else:
            self.tile_manager = None
            self.global_statistics = None
            self.current_render_level = 0
            if hasattr(self, "async_tile_loader"):
                self.async_tile_loader.set_tile_manager(None)
                self.async_tile_loader.clear_cache()

    def enable_full_resolution(self, enabled=True):
        self.use_full_resolution = enabled
        if self.tile_manager:
            self.tile_manager.enable_full_resolution(enabled)

        if enabled:
            metadata = self.tile_manager.raster_loader.get_metadata()
            original_width = metadata['width']
            original_height = metadata['height']

            # Keep a downsampled preview pixmap visible during tile loading so
            # wheel zoom and panning remain interactive. We place it above tiles
            # and hide it once tiles render.
            #
            # IMPORTANT: the preview pixmap was rendered at a reduced resolution
            # (e.g. max 2048px), so its native boundingRect is much smaller than
            # the full-resolution scene we're about to switch to. Without
            # rescaling it, it stays pinned at its small native size in the
            # top-left corner (origin 0,0) instead of covering the whole raster
            # extent -> this is what shows up as a tiny "duplicate" thumbnail
            # stacked on top of the full-size tiled image, and makes the tiled
            # image look shifted off-center. Scale it up to match the full
            # raster extent so it lines up exactly with the tiles.
            if self.pixmap_item:
                try:
                    self.pixmap_item.setPos(0, 0)
                    pixmap = self.pixmap_item.pixmap()
                    if pixmap and not pixmap.isNull() and pixmap.width() > 0 and pixmap.height() > 0:
                        scale_x = original_width / pixmap.width()
                        scale_y = original_height / pixmap.height()
                        self.pixmap_item.setTransform(QTransform().scale(scale_x, scale_y))

                    self.pixmap_item.setZValue(2)
                    try:
                        self.pixmap_item.setOpacity(0.98)
                    except Exception as e:
                        self.logger.debug(f"Failed to set pixmap opacity: {e}")
                    self._preview_active = True

                    if self._all_required_tiles_loaded():
                        self.pixmap_item.setVisible(False)
                        self._preview_active = False
                    else:
                        try:
                            self.pixmap_item.setVisible(self._raster_visible)
                        except Exception as e:
                            self.logger.debug(f"Failed to set pixmap visibility based on flag: {e}")
                except Exception as e:
                    self.logger.debug(f"Failed to prepare preview pixmap: {e}")

            from PyQt6.QtCore import QRectF
            self.scene.setSceneRect(QRectF(0, 0, original_width, original_height))

            # Fit the view to show the entire scene
            self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

            # Update zoom factor after fitInView
            transform = self.transform()
            self.zoom_factor = transform.m11()
            # If fitInView produced a zoom smaller than the configured min_zoom
            # (possible when the window is narrow or sidebars are open), lower
            # the minimum so the user can zoom in from the initial view. This
            # avoids a situation where wheel events are ignored because
            # zoom_factor < min_zoom (reported in user logs when sidebars open).
            try:
                if self.zoom_factor < self.min_zoom:
                    # Reduce min_zoom to a fraction of current zoom so zooming in works
                    self.min_zoom = max(0.0001, self.zoom_factor * 0.5)
                    self.logger.debug(f"Adjusted min_zoom to {self.min_zoom:.6f} after fitInView (current zoom {self.zoom_factor:.6f})")
            except Exception as e:
                self.logger.debug(f"Failed to adjust min_zoom: {e}")
            self.viewport_changed.emit()

            # Ensure drag/pan and focus are enabled in full-resolution mode
            if not self.measurement_mode:
                try:
                    self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
                    self.setCursor(Qt.CursorShape.OpenHandCursor)
                except Exception as e:
                    self.logger.debug(f"Failed to set drag mode/cursor: {e}")

            # Ensure viewport and scrollbars have appropriate focus policies so wheel events
            # aren't stolen by scrollbars or other widgets. Re-install event filter
            # defensively (some operations may remove it) and enforce focus with a delayed
            # fallback to handle race conditions.
            try:
                self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
                self.viewport().setFocusPolicy(Qt.FocusPolicy.StrongFocus)
                self.horizontalScrollBar().setFocusPolicy(Qt.FocusPolicy.NoFocus)
                self.verticalScrollBar().setFocusPolicy(Qt.FocusPolicy.NoFocus)
            except Exception as e:
                self.logger.debug(f"Failed to set focus policies: {e}")

            try:
                # Reinstall event filter defensively
                self.viewport().installEventFilter(self)
            except Exception as e:
                self.logger.debug(f"Failed to reinstall event filter: {e}")

            try:
                self.setFocus()
                self.viewport().setFocus()
                QTimer.singleShot(50, self._deferred_focus)
            except Exception as e:
                self.logger.debug(f"Failed to set focus for full resolution mode: {e}")

            # Overlay subsystem removed: no display dimension update

            self._clear_tile_items()

            # Load tiles immediately without debounce delay
            self._delayed_tile_load(immediate=True)
        else:
            self._clear_tile_items()

            if self.pixmap_item:
                # Reset any scale transform applied while full-resolution mode
                # was active, so the preview goes back to its native size
                # instead of staying stretched to the full raster extent.
                try:
                    self.pixmap_item.setTransform(QTransform())
                except Exception as e:
                    self.logger.debug(f"Failed to reset pixmap transform: {e}")
                # Respect stored raster visibility when leaving full-resolution
                try:
                    self.pixmap_item.setVisible(self._raster_visible)
                except Exception as e:
                    self.logger.debug(f"Failed to set raster visibility, using default: {e}")
                    self.pixmap_item.setVisible(True)
                self.scene.setSceneRect(self.pixmap_item.boundingRect())

            if self.current_display_data is not None:
                if len(self.current_display_data.shape) == 3:
                    display_h, display_w = self.current_display_data.shape[1], self.current_display_data.shape[2]
                else:
                    display_h, display_w = self.current_display_data.shape

                # Overlay subsystem removed: no display dimension update

    def _load_tiles_for_viewport(self, immediate=False):
        if immediate:
            self._delayed_tile_load(immediate=True)
        else:
            self.viewport_update_timer.start(self.debounce_delay)

    def _delayed_tile_load(self, immediate=False):
        if not self.tile_manager:
            return

        if not self.use_full_resolution:
            return

        # If the user is actively interacting (rapid wheel/pan), avoid
        # performing heavy immediate tile loads. Instead, defer until
        # the user has paused (see _on_user_activity_stopped).
        if self._user_active and immediate:
            try:
                # schedule a delayed (user activity debounce) attempt
                self.viewport_update_timer.start(self.user_activity_debounce)
            except Exception as e:
                self.logger.debug(f"Failed to schedule delayed tile load: {e}")
            return
        # If raster layer is hidden, do not request or schedule any tile loads.
        # We avoid clearing cached pixmaps here so re-show is fast; pausing
        # the loader and hiding existing items is done in show_raster().
        if not getattr(self, '_raster_visible', True):
            try:
                self.logger.debug("Raster layer hidden - skipping tile requests (loader paused)")
            except Exception as e:
                self.logger.error(f"Failed to log raster visibility status: {e}")
            return
        try:
            viewport_rect = self.mapToScene(self.viewport().rect()).boundingRect()
            metadata = self.tile_manager.raster_loader.get_metadata()
            render_level = self.tile_manager.raster_loader.select_overview_level(self.zoom_factor)

            if render_level != self.current_render_level:
                # Keep the previous level visible while the new level loads.
                # Individual tile items are replaced as soon as their matching
                # resolution is ready, avoiding a blank canvas during zoom.
                self.current_render_level = render_level

            vp_x = max(0, viewport_rect.x())
            vp_y = max(0, viewport_rect.y())
            vp_w = viewport_rect.width()
            vp_h = viewport_rect.height()

            vp_x = min(vp_x, metadata['width'])
            vp_y = min(vp_y, metadata['height'])

            tile_size = self.tile_manager.tile_size
            # Keep one-tile margin around the viewport so short pans do not
            # immediately require disk reads.
            start_tile_x = max(0, int(vp_x // tile_size) - 1)
            start_tile_y = max(0, int(vp_y // tile_size) - 1)
            end_tile_x = min(
                int(np.ceil((vp_x + vp_w) / tile_size)) + 1,
                int(np.ceil(metadata['width'] / tile_size))
            )
            end_tile_y = min(
                int(np.ceil((vp_y + vp_h) / tile_size)) + 1,
                int(np.ceil(metadata['height'] / tile_size))
            )

            # Visible tiles (priority)
            required_tiles = []
            for ty in range(start_tile_y, end_tile_y):
                for tx in range(start_tile_x, end_tile_x):
                    required_tiles.append((tx, ty))

            # Disable preloading - focus on visible tiles only for better performance
            # Preloading causes lag during zoom out when many tiles are needed

            if not required_tiles:
                return

            new_tile_keys = set(required_tiles)

            tiles_to_remove = self.current_tile_keys - new_tile_keys
            for tx, ty in tiles_to_remove:
                key = (tx, ty)
                item = self.tile_pixmap_items.pop(key, None)
                if item is not None:
                    try:
                        if item.scene() is self.scene:
                            self.scene.removeItem(item)
                    except (RuntimeError, AttributeError):
                        pass

            self.current_tile_keys = new_tile_keys

            # Calculate viewport center in tile coordinates for progressive loading
            viewport_center_x = vp_x + vp_w / 2
            viewport_center_y = vp_y + vp_h / 2
            center_tile_x = int(viewport_center_x // self.tile_manager.tile_size)
            center_tile_y = int(viewport_center_y // self.tile_manager.tile_size)

            self.async_tile_loader.request_tiles(
                self.tile_manager,
                required_tiles,
                self.zoom_factor,
                immediate=immediate,
                viewport_center=(center_tile_x, center_tile_y)
            )

            # Update last viewport position
            self.last_viewport_rect = viewport_rect

        except Exception as e:
            self.logger.error(f"Error loading tiles: {e}", exc_info=True)

    def _on_zoom_finished(self):
        """Called when zoom operation finishes - ONLY load tiles after zoom stops"""
        self.is_zooming = False
        if self.use_full_resolution:
            # Load tiles immediately when zoom finishes for quick response
            self._delayed_tile_load(immediate=True)

    def _set_user_active(self):
        """Mark that the user is actively interacting and restart the
        activity debounce timer."""
        self._user_active = True
        try:
            self.user_activity_timer.start(self.user_activity_debounce)
        except Exception as e:
            self.logger.debug(f"Failed to start user activity timer: {e}")

    def _on_user_activity_stopped(self):
        """Called after a short pause in user activity. Trigger an
        immediate tile load (if in full-resolution mode) so the viewer
        refreshes to high-res tiles once the user stops interacting."""
        self._user_active = False
        if self.use_full_resolution:
            # Load visible tiles now that user stopped interacting
            self._delayed_tile_load(immediate=True)

    def _on_pan_finished(self):
        """Called when pan operation finishes"""
        if self.use_full_resolution:
            self._delayed_tile_load(immediate=False)

    def _on_pan_check_timeout(self):
        """Called periodically during panning to check viewport and load tiles"""
        if self._is_panning and self.use_full_resolution:
            current_rect = self.mapToScene(self.viewport().rect()).boundingRect()

            # If this is first check, just save the position
            if self.last_viewport_rect is None:
                self.last_viewport_rect = current_rect
                return

            # Calculate movement distance
            dx = abs(current_rect.x() - self.last_viewport_rect.x())
            dy = abs(current_rect.y() - self.last_viewport_rect.y())

            # If moved significantly, load tiles immediately
            if dx > self.viewport_move_threshold or dy > self.viewport_move_threshold:
                self.last_viewport_rect = current_rect
                # Load visible tiles only (no preload during active panning)
                self._delayed_tile_load(immediate=True)

    def _check_viewport_change(self):
        """Check if viewport has moved significantly and trigger tile loading"""
        if not self.tile_manager or not self.use_full_resolution:
            return

        current_rect = self.mapToScene(self.viewport().rect()).boundingRect()

        # If this is first check, just save the position
        if self.last_viewport_rect is None:
            self.last_viewport_rect = current_rect
            return

        # Calculate movement distance
        dx = abs(current_rect.x() - self.last_viewport_rect.x())
        dy = abs(current_rect.y() - self.last_viewport_rect.y())

        # If moved more than threshold, trigger tile loading
        if dx > self.viewport_move_threshold or dy > self.viewport_move_threshold:
            # Update last viewport position immediately to avoid multiple triggers
            self.last_viewport_rect = current_rect

            # For fast panning, load immediately without debounce
            # For slower panning, use debounce
            if dx > self.viewport_move_threshold * 2 or dy > self.viewport_move_threshold * 2:
                # Fast panning - load immediately for visible tiles
                self._delayed_tile_load(immediate=True)
            else:
                # Slow panning - use debounce
                self.pan_timer.start(self.pan_debounce)

    def _deferred_focus(self):
        """Delayed focus helper to ensure the viewport receives focus after UI changes."""
        try:
            self.setFocus()
            self.viewport().setFocus()
            # Ensure drag mode and cursor are correct when focus is regained
            if not self.measurement_mode:
                try:
                    self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
                    self.setCursor(Qt.CursorShape.OpenHandCursor)
                except Exception as e:
                    self.logger.debug(f"Failed to set drag mode/cursor on focus: {e}")
            self.logger.debug("[DEFERRED FOCUS] Focus and drag mode enforced")
        except Exception as e:
            self.logger.debug(f"Failed to set deferred focus: {e}")

    def focusInEvent(self, event):
        self.logger.debug("[FOCUS IN] Viewer received focus")
        super().focusInEvent(event)

    def focusOutEvent(self, event):
        self.logger.debug("[FOCUS OUT] Viewer lost focus")
        super().focusOutEvent(event)

    def _on_tiles_ready(self):
        if not self.tile_manager:
            return

        try:
            # If raster layer is hidden, skip rendering any ready tiles.
            # We intentionally do NOT clear cached pixmaps here so that
            # re-show is fast; hiding already sets existing items invisible.
            if not getattr(self, '_raster_visible', True):
                try:
                    self.logger.debug("Raster layer hidden - skipping _on_tiles_ready (loader paused)")
                except Exception as e:
                    self.logger.error(f"Failed to log tile ready status: {e}")
                return

            tile_size = self.tile_manager.tile_size
            overview_decimation = self.tile_manager.raster_loader.get_overview_decimation(
                self.current_render_level
            )
            tiles_added = 0
            max_tiles_per_frame = 32  # Sweet spot: smooth progressive rendering without UI freeze
            has_more_tiles = False

            for tx, ty in self.current_tile_keys:
                key = (tx, ty)

                existing_item = self.tile_pixmap_items.get(key)
                if existing_item is not None:
                    existing_level = existing_item.data(0)
                    if existing_level == self.current_render_level:
                        continue

                pixmap = self.async_tile_loader.get_loaded_pixmap(
                    tx, ty, self.current_render_level
                )

                if pixmap is not None:
                    if existing_item is not None:
                        self.scene.removeItem(existing_item)
                    tile_item = QGraphicsPixmapItem(pixmap)
                    tile_item.setPos(tx * tile_size, ty * tile_size)
                    # Overview pixels are read at a reduced resolution but
                    # retain the full tile footprint in raster coordinates.
                    tile_item.setScale(overview_decimation)
                    tile_item.setData(0, self.current_render_level)
                    # Ensure tile items are non-interactive so they don't steal
                    # focus / mouse / wheel events from the view.
                    try:
                        tile_item.setAcceptHoverEvents(False)
                    except Exception as e:
                        self.logger.debug(f"Failed to set tile hover events: {e}")
                    try:
                        tile_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                    except Exception as e:
                        self.logger.debug(f"Failed to set tile mouse buttons: {e}")
                    try:
                        tile_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsFocusable, False)
                    except Exception as e:
                        self.logger.debug(f"Failed to set tile focusable flag: {e}")
                    # Respect raster visibility when adding new tile items so
                    # hiding the raster prevents newly-loaded tiles from flashing
                    try:
                        tile_item.setVisible(self._raster_visible)
                    except Exception as e:
                        self.logger.debug(f"Failed to set tile visibility: {e}")
                    tile_item.setZValue(-1)
                    self.scene.addItem(tile_item)
                    self.tile_pixmap_items[key] = tile_item

                    tiles_added += 1
                    # Sweet spot - smooth without freezing
                    if tiles_added >= max_tiles_per_frame:
                        has_more_tiles = True
                        break

            # If we hit the limit, there might be more tiles to render
            # Schedule another render pass
            if has_more_tiles:
                from PyQt6.QtCore import QTimer
                QTimer.singleShot(10, self._on_tiles_ready)

            # If we have rendered at least one tile and a preview is active,
            # Hide the downsample preview only when the currently required
            # viewport tiles are all loaded (full coverage). This prevents
            # flicker or showing partially-rendered high-res tiles.
            try:
                if self.use_full_resolution and self._all_required_tiles_loaded():
                    if self.pixmap_item:
                        self.pixmap_item.setVisible(False)
                    self._preview_active = False
            except Exception as e:
                # Best-effort: if anything fails, keep preview as-is
                self.logger.debug(f"Failed to hide preview pixmap: {e}")

        except Exception as e:
            self.logger.error(f"Error rendering tiles: {e}", exc_info=True)

    def _clear_tile_items(self):
        # scene.clear() may already have destroyed the C++ QGraphicsItems.
        # Treat cleanup as idempotent so switching layers cannot abort when a
        # stale Python wrapper points at an already-deleted Qt object.
        for item in list(self.tile_pixmap_items.values()):
            try:
                if item is not None and item.scene() is self.scene:
                    self.scene.removeItem(item)
            except (RuntimeError, AttributeError):
                pass
        self.tile_pixmap_items.clear()
        self.current_tile_keys.clear()
        if hasattr(self, "async_tile_loader"):
            self.async_tile_loader.clear_cache()

    # ===== Overlay methods (tile preview / detection rectangles) =====
    def set_overlay_tiles(self, tile_rects, outline_color=QColor(147, 51, 234), fill_color=QColor(147, 51, 234, 80), outline_width=2):
        """Display semi-transparent rectangles for the provided tile rectangles.

        Args:
            tile_rects: iterable of dicts or tuples with (x,y,w,h) or {'x','y','w','h'}
        """
        # Convert to list if needed (important: do this first to avoid exhausting iterators)
        rect_list = list(tile_rects) if not isinstance(tile_rects, list) else tile_rects

        # Smart clearing: determine overlay type from first rectangle
        # Clear only overlays of the same type we're about to add
        overlay_type_to_clear = 'detection'  # Default to 'detection'
        try:
            if rect_list and isinstance(rect_list[0], dict):
                overlay_type_to_clear = rect_list[0].get('type', 'detection')
                self.logger.info(f"Detected overlay type: '{overlay_type_to_clear}' from first rectangle")
        except Exception as e:
            self.logger.debug(f"Could not detect overlay type, using default 'detection': {e}")

        # Clear only overlays of the same type
        try:
            self.clear_overlay(overlay_type=overlay_type_to_clear)
            self.logger.info(f"Cleared overlays of type '{overlay_type_to_clear}' before adding new ones")
        except Exception as e:
            self.logger.warning(f"Failed to clear overlays: {e}")

        try:
            pen = QPen(outline_color)
            pen.setWidth(outline_width)
        except Exception as e:
            self.logger.debug(f"Failed to create pen with specified color, using fallback: {e}")
            pen = QPen(QColor(147, 51, 234))  # Vivid Purple fallback

        brush = QBrush(fill_color)

        for tr in rect_list:
            try:
                # Support dict entries with optional metadata (class, id)
                meta_class = None
                meta_id = None
                if isinstance(tr, dict):
                    x = tr.get('x', 0)
                    y = tr.get('y', 0)
                    w = tr.get('w', 0)
                    h = tr.get('h', 0)
                    meta_class = tr.get('class', None)
                    meta_id = tr.get('id', None)
                else:
                    # assume tuple/list: (x,y,w,h)
                    x, y, w, h = tr

                # If colors provided per-rect, use them; else use default pen/brush
                outline = tr.get('outline_color', None) if isinstance(tr, dict) else None
                fill = tr.get('fill_color', None) if isinstance(tr, dict) else None
                score = tr.get('score', None) if isinstance(tr, dict) else None

                # Use a small subclass to support hover tooltips and metadata
                class DetectionRect(QGraphicsRectItem):
                    def __init__(self, x, y, w, h):
                        super().__init__(x, y, w, h)
                        try:
                            self.setAcceptHoverEvents(True)
                        except Exception as e:
                            # Some platforms may not support hover events
                            pass

                    def hoverEnterEvent(self, event):
                        try:
                            txt = getattr(self, '_det_tooltip', None)
                            if txt:
                                # Show tooltip at cursor global position
                                try:
                                    QToolTip.showText(event.screenPos().toPoint(), txt)
                                except Exception as e:
                                    QToolTip.showText(QCursor.pos(), txt)
                        except Exception as e:
                            # Tooltip display is best-effort
                            pass
                        super().hoverEnterEvent(event)

                    def hoverLeaveEvent(self, event):
                        try:
                            QToolTip.hideText()
                        except Exception as e:
                            # Tooltip hide is best-effort
                            pass
                        super().hoverLeaveEvent(event)

                rect_item = DetectionRect(x, y, w, h)
                # Diagnostic logging for overlay creation
                try:
                    otype = tr.get('type', 'detection') if isinstance(tr, dict) else 'detection'
                    self.logger.info(f"Adding overlay item: type={otype} class={meta_class} id={meta_id} x={x:.1f} y={y:.1f} w={w:.1f} h={h:.1f} score={score}")
                except Exception as e:
                    self.logger.debug(f"Failed to log overlay creation: {e}")
                # apply colors if present
                try:
                    if outline is not None:
                        pen = QPen(QColor(outline)) if not isinstance(outline, QColor) else QPen(outline)
                    rect_item.setPen(pen)
                except Exception as e:
                    self.logger.debug(f"Failed to set pen, using default: {e}")
                    rect_item.setPen(pen)
                try:
                    if fill is not None:
                        brush = QBrush(QColor(fill)) if not isinstance(fill, QColor) else QBrush(fill)
                    rect_item.setBrush(brush)
                except Exception as e:
                    self.logger.debug(f"Failed to set brush, using default: {e}")
                    rect_item.setBrush(brush)
                rect_item.setZValue(1000)
                rect_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
                rect_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsFocusable, False)
                # Attach detection metadata and overlay type if provided so callers can filter
                try:
                    rect_item._det_class = meta_class
                    rect_item._det_id = meta_id
                    rect_item._det_score = score
                    rect_item._overlay_type = tr.get('type', 'detection') if isinstance(tr, dict) else 'detection'
                    if isinstance(meta_class, str) and score is not None:
                        rect_item._det_tooltip = f"{meta_class} ({score:.2f})"
                    elif isinstance(meta_class, str):
                        rect_item._det_tooltip = meta_class
                    else:
                        # For tile preview, show a simple tooltip
                        if rect_item._overlay_type == 'tile':
                            rect_item._det_tooltip = 'Tile preview'
                        else:
                            rect_item._det_tooltip = None
                except Exception as e:
                    self.logger.debug(f"Failed to set detection metadata: {e}")
                self.scene.addItem(rect_item)
                self.overlay_items.append(rect_item)

                # Add text label for detection boxes (ID number)
                try:
                    if rect_item._overlay_type == 'detection' and meta_id is not None:
                        # Create background rectangle for text
                        text_bg = QGraphicsRectItem()
                        text_bg.setBrush(QBrush(QColor(0, 0, 0, 180)))  # Semi-transparent black
                        text_bg.setPen(QPen(Qt.PenStyle.NoPen))  # No outline
                        text_bg.setZValue(1001)
                        text_bg._overlay_type = 'detection'

                        # Create text label with ID and confidence score using HTML for different sizes
                        text_item = QGraphicsTextItem()

                        if score is not None:
                            # Use HTML to have different font sizes and colors: ID white/bold, confidence yellow/small
                            html_text = f'''
                            <div style="font-family: Arial;">
                                <span style="font-size: 14pt; font-weight: bold; color: white;">{meta_id}</span>
                                <span style="font-size: 7pt; font-weight: normal; color: #FFD700;"> ({score:.2f})</span>
                            </div>
                            '''
                            text_item.setHtml(html_text)
                        else:
                            # Just ID without confidence
                            font = QFont()
                            font.setPointSize(14)
                            font.setBold(True)
                            text_item.setFont(font)
                            text_item.setPlainText(str(meta_id))
                            text_item.setDefaultTextColor(QColor(255, 255, 255))

                        # Calculate text dimensions
                        text_rect = text_item.boundingRect()
                        padding = 4

                        # Position text at top-left corner of bounding box
                        text_x = x + padding
                        text_y = y + padding
                        text_item.setPos(text_x, text_y)

                        # Set background rectangle dimensions and position
                        bg_margin = 2
                        text_bg.setRect(
                            text_x - bg_margin,
                            text_y - bg_margin,
                            text_rect.width() + 2 * bg_margin,
                            text_rect.height() + 2 * bg_margin
                        )

                        # Set z-value: background < text
                        text_item.setZValue(1002)

                        # Store reference to rect_item for coordinated visibility toggling
                        text_bg._parent_rect = rect_item
                        text_item._parent_rect = rect_item

                        # Tag as label for independent toggling
                        text_bg._is_label = True
                        text_item._is_label = True

                        # Add to scene and track in overlay_items
                        self.scene.addItem(text_bg)
                        self.scene.addItem(text_item)
                        self.overlay_items.append(text_bg)
                        self.overlay_items.append(text_item)

                except Exception as e:
                    self.logger.debug(f"Failed to add text label for detection {meta_id}: {e}")
            except Exception as e:
                self.logger.debug(f"Failed to add overlay item: {e}")
                continue

        self._overlay_visible = True

    def clear_overlay(self, overlay_type=None):
        """Remove overlay items from the scene.

        Args:
            overlay_type: If specified, only clear overlays of this type (e.g., 'tile', 'detection').
                         If None, clear all overlays.
        """
        try:
            if overlay_type is None:
                # Clear all overlays
                for it in list(self.overlay_items):
                    try:
                        self.scene.removeItem(it)
                    except Exception as e:
                        self.logger.debug(f"Failed to remove overlay item: {e}")
                self.overlay_items.clear()
                self._overlay_visible = False
            else:
                # Clear only overlays of specified type
                remaining_items = []
                for it in list(self.overlay_items):
                    try:
                        otype = getattr(it, '_overlay_type', None)
                        if otype == overlay_type:
                            self.scene.removeItem(it)
                        else:
                            remaining_items.append(it)
                    except Exception as e:
                        self.logger.debug(f"Failed to check/remove overlay type: {e}")
                        remaining_items.append(it)
                self.overlay_items = remaining_items
                self.logger.info(f"Cleared overlays of type '{overlay_type}', {len(remaining_items)} overlays remaining")
        except Exception as e:
            self.logger.error(f"Error clearing overlay: {e}")

    def show_overlay(self, visible: bool = True):
        """Toggle visibility of existing overlay items."""
        self._overlay_visible = bool(visible)
        try:
            for it in self.overlay_items:
                try:
                    it.setVisible(self._overlay_visible)
                except Exception as e:
                    self.logger.debug(f"Failed to set overlay item visibility: {e}")
        except Exception as e:
            self.logger.error(f"Failed to toggle overlay visibility: {e}")

    # Detection display subsystem removed: overlay methods deleted.

    def set_color_normalization(self, enabled):
        """Enable/disable color normalization (contrast stretching)

        Args:
            enabled: If False, shows raw pixel values (true colors).
                    If True, applies percentile-based contrast stretching.
        """
        self.use_color_normalization = enabled
        self.async_tile_loader.use_color_normalization = enabled

        # Update global statistics based on normalization setting
        if enabled and self.tile_manager:
            self.global_statistics = self.tile_manager.raster_loader.global_statistics
            self.tile_manager.global_statistics = self.global_statistics
        else:
            self.global_statistics = None
            if self.tile_manager:
                self.tile_manager.global_statistics = None

    def show_raster(self, visible: bool = True):
        """Show or hide the main raster image (pixmap_item).

        This is used by the Display Options layer manager checkbox.
        If the pixmap isn't yet created, the visibility preference is stored
        and applied once an image is set.
        """
        self._raster_visible = bool(visible)
        # Toggle preview pixmap visibility (only show preview if preview is active and tiles aren't fully loaded)
        if self.pixmap_item is not None:
            try:
                if not self._raster_visible:
                    self.pixmap_item.setVisible(False)
                else:
                    if getattr(self, '_preview_active', False) and not self._all_required_tiles_loaded():
                        self.pixmap_item.setVisible(True)
                    else:
                        self.pixmap_item.setVisible(False)
            except Exception as e:
                self.logger.debug(f"Failed to toggle pixmap visibility: {e}")

        # Also hide/show any rendered tile pixmap items so full-resolution
        # tiles are not visible when raster layer is hidden.
        try:
            for item in list(self.tile_pixmap_items.values()):
                try:
                    item.setVisible(self._raster_visible)
                except Exception as e:
                    self.logger.debug(f"Failed to toggle tile visibility: {e}")
        except Exception as e:
            self.logger.error(f"Failed to toggle tile visibility for all items: {e}")

        # Additionally, add/remove a blocker rect that covers the whole scene to
        # ensure nothing underneath is visible (this covers edge cases where
        # other items or backgrounds might still show through).
        try:
            if not self._raster_visible:
                if self._raster_blocker is None:
                    rect = self.scene.sceneRect()
                    blocker = QGraphicsRectItem(rect)
                    blocker.setBrush(QBrush(QColor(0, 0, 0)))
                    blocker.setZValue(50)
                    blocker.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
                    blocker.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsFocusable, False)
                    self.scene.addItem(blocker)
                    self._raster_blocker = blocker
            else:
                if self._raster_blocker is not None:
                    try:
                        self.scene.removeItem(self._raster_blocker)
                    except Exception as e:
                        self.logger.debug(f"Failed to remove raster blocker: {e}")
                    self._raster_blocker = None
        except Exception as e:
            self.logger.error(f"Failed to manage raster blocker: {e}")

        # If raster was just hidden: pause loader and keep cached pixmaps/tiles
        # in memory (but invisible) so re-show is fast. This trades memory
        # for responsiveness when toggling visibility frequently.
        if not self._raster_visible:
            try:
                setattr(self.async_tile_loader, 'paused', True)
            except Exception as e:
                self.logger.debug(f"Failed to pause tile loader: {e}")
        else:
            # Unpause loader and, if we already have tile items cached, make
            # them visible immediately. Otherwise trigger an immediate tile
            # load if in full-resolution mode.
            try:
                try:
                    setattr(self.async_tile_loader, 'paused', False)
                except Exception as e:
                    self.logger.debug(f"Failed to unpause tile loader: {e}")

                if len(self.tile_pixmap_items) > 0:
                    try:
                        for item in list(self.tile_pixmap_items.values()):
                            item.setVisible(True)
                    except Exception as e:
                        self.logger.debug(f"Failed to show cached tiles: {e}")
                else:
                    if self.use_full_resolution:
                        self._delayed_tile_load(immediate=True)
            except Exception as e:
                self.logger.error(f"Failed to restore raster visibility: {e}")

    # ===== Measurement Tool Methods =====

    def enable_measurement_mode(self, enabled=True):
        """Enable or disable measurement mode"""
        self.measurement_mode = enabled

        if enabled:
            # Disable pan mode when measuring
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            # Set crosshair cursor
            self.setCursor(Qt.CursorShape.CrossCursor)
            self.logger.info("Measurement mode ENABLED")
        else:
            # Cancel any active measurement
            if self._is_measuring:
                self.measurement_manager.cancel_measurement()
                self._is_measuring = False
                self._measurement_first_point_set = False

            # ALWAYS restore cursor and drag mode when disabling measurement
            # (not just when self._is_measuring is True)
            # Re-enable left-drag panning
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

            # Restore normal cursor (OpenHandCursor for panning)
            try:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
                self.logger.info("Measurement mode DISABLED - cursor restored to OpenHandCursor")
            except Exception as e:
                self.setCursor(Qt.CursorShape.ArrowCursor)
                self.logger.info(f"Measurement mode DISABLED - cursor restored to ArrowCursor (fallback): {e}")

    def clear_measurements(self):
        """Clear all measurements from the scene"""
        self.measurement_manager.clear_all_measurements()

    def remove_measurement(self, measurement_id):
        """Remove one measurement by ID and refresh the view."""
        removed = self.measurement_manager.remove_measurement(measurement_id)
        if removed:
            self.scene.update()
            self.viewport().update()
        return removed

    def removeMeasurement(self, measurement_id):
        """Compatibility alias for ID-based measurement deletion."""
        return self.remove_measurement(measurement_id)

    def get_all_measurements(self):
        """Get info about all measurements"""
        return self.measurement_manager.get_all_measurements_info()

    def update_measurement_metadata(self, metadata):
        """Update raster metadata for accurate measurements"""
        self.measurement_manager.set_metadata(metadata)

    # ========== POLYGON DRAWING METHODS ==========

    def set_polygon_drawing_mode(self, enabled):
        """Enable or disable polygon drawing mode"""
        self.polygon_drawing_mode = enabled
        if enabled:
            self.setCursor(Qt.CursorShape.CrossCursor)
            self.logger.info("Polygon drawing mode ENABLED")
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            # Clear temporary drawing if cancelled
            if len(self.polygon_vertices) > 0 and self.polygon_filled_item is None:
                self._clear_polygon_temp_items()
            self.logger.info("Polygon drawing mode DISABLED")

    def _clear_polygon_temp_items(self):
        """Clear temporary polygon drawing items (vertices and lines during drawing)"""
        for item in self.polygon_vertex_items:
            self.scene.removeItem(item)
        for item in self.polygon_line_items:
            self.scene.removeItem(item)
        if self.polygon_closing_line:
            self.scene.removeItem(self.polygon_closing_line)
            self.polygon_closing_line = None

        self.polygon_vertex_items.clear()
        self.polygon_line_items.clear()
        self.polygon_vertices.clear()

    def _add_polygon_vertex(self, scene_pos):
        """Add a vertex to the polygon being drawn.

        Guards against Qt's double-click event sequence: a double-click always
        fires TWO mousePressEvent calls (at nearly the same pixel) before the
        mouseDoubleClickEvent that finishes the polygon. Without this guard,
        that second press appends a near-duplicate "spike" vertex right before
        closing, which turns the final polygon into a self-intersecting
        (topologically invalid) geometry. QGIS/GEOS silently refuses to render
        invalid geometries - the layer's extent/bbox is still correct (so
        "Zoom to Layer" looks right), but no shape/fill is drawn.
        """
        # Minimum pixel distance between consecutive vertices. Clicks closer
        # than this are treated as an accidental double-click artifact, not
        # an intentional new vertex, and are ignored.
        MIN_VERTEX_DISTANCE_PX = 4.0

        if self.polygon_vertices:
            last_pos = self.polygon_vertices[-1]
            dx = scene_pos.x() - last_pos.x()
            dy = scene_pos.y() - last_pos.y()
            if (dx * dx + dy * dy) ** 0.5 < MIN_VERTEX_DISTANCE_PX:
                self.logger.debug(
                    f"Ignored polygon vertex at ({scene_pos.x():.1f}, {scene_pos.y():.1f}) "
                    f"- too close to previous vertex (likely double-click artifact)"
                )
                return

        self.polygon_vertices.append(scene_pos)

        # Draw vertex marker
        vertex_item = PolygonVertexItem(
            scene_pos,
            self.polygon_vertex_size,
            QPen(self.polygon_vertex_outline_color, 2),
            QBrush(self.polygon_vertex_color),
            self
        )
        self.scene.addItem(vertex_item)
        self.polygon_vertex_items.append(vertex_item)

        # Draw line from previous vertex
        if len(self.polygon_vertices) > 1:
            prev_pos = self.polygon_vertices[-2]
            line_item = self.scene.addLine(
                prev_pos.x(), prev_pos.y(),
                scene_pos.x(), scene_pos.y(),
                QPen(self.polygon_line_color, self.polygon_line_width)
            )
            line_item.setZValue(100)
            self.polygon_line_items.append(line_item)

        # Update closing line (dashed line from last to first)
        if len(self.polygon_vertices) >= 3:
            if self.polygon_closing_line:
                self.scene.removeItem(self.polygon_closing_line)

            first_pos = self.polygon_vertices[0]
            pen = QPen(self.polygon_line_color, self.polygon_line_width, Qt.PenStyle.DashLine)
            self.polygon_closing_line = self.scene.addLine(
                scene_pos.x(), scene_pos.y(),
                first_pos.x(), first_pos.y(),
                pen
            )

        self.logger.debug(f"Added polygon vertex at ({scene_pos.x():.1f}, {scene_pos.y():.1f}) - Total: {len(self.polygon_vertices)}")

    def _remove_last_polygon_vertex(self):
        """Remove the last vertex from polygon (undo)"""
        if not self.polygon_vertices:
            self.logger.debug("No vertices to remove")
            return

        # Remove last vertex from list
        self.polygon_vertices.pop()

        # Remove last vertex marker
        if self.polygon_vertex_items:
            vertex_item = self.polygon_vertex_items.pop()
            self.scene.removeItem(vertex_item)

        # Remove last line
        if self.polygon_line_items:
            line_item = self.polygon_line_items.pop()
            self.scene.removeItem(line_item)

        # Update closing line
        if self.polygon_closing_line:
            self.scene.removeItem(self.polygon_closing_line)
            self.polygon_closing_line = None

        # Redraw closing line if still have >= 3 vertices
        if len(self.polygon_vertices) >= 3:
            first_pos = self.polygon_vertices[0]
            last_pos = self.polygon_vertices[-1]
            pen = QPen(self.polygon_line_color, self.polygon_line_width, Qt.PenStyle.DashLine)
            self.polygon_closing_line = self.scene.addLine(
                last_pos.x(), last_pos.y(),
                first_pos.x(), first_pos.y(),
                pen
            )

        self.logger.debug(f"Removed last polygon vertex - Remaining: {len(self.polygon_vertices)}")

    def _finish_polygon(self):
        """Finish polygon drawing and create filled polygon"""
        if len(self.polygon_vertices) < 3:
            self.logger.warning("Need at least 3 vertices to finish polygon")
            return

        # Remove closing line
        if self.polygon_closing_line:
            self.scene.removeItem(self.polygon_closing_line)
            self.polygon_closing_line = None

        # Add final line from last to first vertex
        first_pos = self.polygon_vertices[0]
        last_pos = self.polygon_vertices[-1]
        line_item = self.scene.addLine(
            last_pos.x(), last_pos.y(),
            first_pos.x(), first_pos.y(),
            QPen(self.polygon_line_color, self.polygon_line_width)
        )
        line_item.setZValue(100)
        self.polygon_line_items.append(line_item)

        # Create filled polygon
        from PyQt6.QtGui import QPolygonF
        polygon_shape = QPolygonF(self.polygon_vertices)
        self.polygon_filled_item = self.scene.addPolygon(
            polygon_shape,
            QPen(self.polygon_line_color, self.polygon_line_width),
            QBrush(QColor(255, 0, 0, 50))  # Semi-transparent red fill
        )
        self.polygon_filled_item.setZValue(99)

        self.logger.info(f"Polygon finished with {len(self.polygon_vertices)} vertices")

        # Emit signal to main window
        self.polygon_finish_requested.emit()

    def get_drawn_polygon_data(self):
        """Get polygon data with pixel and geo coordinates"""
        if len(self.polygon_vertices) < 3:
            self.logger.warning("get_drawn_polygon_data: Less than 3 vertices")
            return None
        pixel_coords = []
        for vertex_item in self.polygon_vertex_items:
            if isinstance(vertex_item, PolygonVertexItem):
                pixel_coords.append((vertex_item.pos().x(), vertex_item.pos().y()))
            else:
                # Fallback for standard items
                rect = vertex_item.rect()
                pixel_coords.append((vertex_item.x() + rect.x() + rect.width()/2, 
                                     vertex_item.y() + rect.y() + rect.height()/2))

        self.logger.debug(f"get_drawn_polygon_data: Converted {len(pixel_coords)} vertices to pixel coords")

        # Convert pixel to geo coordinates using the same affine transform
        # used by the measurement tool.  Keep these coordinates for export,
        # but never use geographic degrees directly for area calculation.
        geo_coords = []
        if self.measurement_manager.is_georeferenced:
            geo_coords = [
                self.measurement_manager.pixel_to_world(px, py)
                for px, py in pixel_coords
            ]
        else:
            metadata = self.measurement_manager.metadata
            transform = metadata.get('transform') if metadata else None
            if transform is not None:
                try:
                    geo_coords = [
                        (
                            transform.c + px * transform.a + py * transform.b,
                            transform.f + px * transform.d + py * transform.e,
                        )
                        for px, py in pixel_coords
                    ]
                except AttributeError:
                    values = list(transform)
                    geo_coords = [
                        (
                            values[2] + px * values[0] + py * values[1],
                            values[5] + px * values[3] + py * values[4],
                        )
                        for px, py in pixel_coords
                    ]

        area_m2 = self.calculate_polygon_area(pixel_coords, geo_coords)
        self.logger.debug(f"get_drawn_polygon_data: Calculated area = {area_m2:.2f} m²")

        return {
            'pixel_coords': pixel_coords,
            'geo_coords': geo_coords,
            'area_m2': area_m2
        }

    @staticmethod
    def _shoelace_area(coords):
        """Return absolute planar area for a closed or open coordinate ring."""
        if len(coords) < 3:
            return 0.0
        return abs(sum(
            coords[i][0] * coords[(i + 1) % len(coords)][1]
            - coords[(i + 1) % len(coords)][0] * coords[i][1]
            for i in range(len(coords))
        )) / 2.0

    def calculate_polygon_area(self, pixel_coords, geo_coords=None):
        """Calculate area in square metres without using lon/lat as metres.

        Projected CRS geometries are measured in their native units and
        converted to metres using the CRS axis unit factors.  Geographic CRS
        geometries are projected to a local UTM CRS before applying the
        planar area formula.  Unreferenced rasters use the calibrated pixel
        size when available.
        """
        if len(pixel_coords) < 3:
            return 0.0

        crs = self.measurement_manager.crs
        if geo_coords and crs:
            try:
                if crs.is_geographic:
                    from pyproj import Transformer
                    lon = sum(point[0] for point in geo_coords) / len(geo_coords)
                    lat = sum(point[1] for point in geo_coords) / len(geo_coords)
                    zone = max(1, min(60, int((lon + 180) // 6) + 1))
                    utm_epsg = (32600 if lat >= 0 else 32700) + zone
                    transformer = Transformer.from_crs(
                        crs, f"EPSG:{utm_epsg}", always_xy=True
                    )
                    projected = [transformer.transform(x, y) for x, y in geo_coords]
                    return self._shoelace_area(projected)

                unit_factors = [
                    axis.unit_conversion_factor
                    for axis in crs.axis_info[:2]
                    if axis.unit_conversion_factor
                ]
                factor_x = unit_factors[0] if unit_factors else 1.0
                factor_y = unit_factors[1] if len(unit_factors) > 1 else factor_x
                return self._shoelace_area(geo_coords) * factor_x * factor_y
            except Exception as exc:
                self.logger.warning("Projected polygon area failed: %s", exc)

        pixel_area = self._shoelace_area(pixel_coords)
        if self.measurement_manager.pixel_size_x and self.measurement_manager.pixel_size_y:
            return pixel_area * abs(
                self.measurement_manager.pixel_size_x
                * self.measurement_manager.pixel_size_y
            )
        return pixel_area

    def polygon_area_text(self, area_m2):
        """Format the area label in the UAT-required m²/ha form."""
        if area_m2 > 10000:
            return f"Area\n{area_m2 / 10000:.2f} ha ({area_m2:,.0f} m²)"
        return f"Area\n{area_m2:,.2f} m²"

    def create_polygon_area_label(self, area_m2, pixel_coords, color=None):
        """Create a scene label for one polygon and position it at its centre."""
        label = QGraphicsTextItem()
        label.setPlainText(self.polygon_area_text(area_m2))
        label.setDefaultTextColor(color or self.polygon_line_color)
        label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        label.setZValue(103)
        label.setFlag(QGraphicsTextItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.scene.addItem(label)
        self.update_polygon_area_label(label, area_m2, pixel_coords)
        return label

    def refreshPolygonGeometry(self, polygon):
        """Update polygon geometry when a vertex is moved"""
        items = polygon.get('items', {})
        vertex_items = items.get('vertex_items', [])
        line_items = items.get('line_items', [])
        filled_item = items.get('filled_item')
        
        if not vertex_items:
            return
            
        # Get new pixel coords
        pixel_coords = []
        for v in vertex_items:
            if isinstance(v, PolygonVertexItem):
                pixel_coords.append((v.pos().x(), v.pos().y()))
            else:
                pixel_coords.append((v.x() + v.rect().x() + v.rect().width()/2, 
                                     v.y() + v.rect().y() + v.rect().height()/2))
            
        polygon['pixel_coords'] = pixel_coords
        
        # Update lines
        from PyQt6.QtCore import QLineF
        for i, line_item in enumerate(line_items):
            if line_item:
                p1 = vertex_items[i].pos() if isinstance(vertex_items[i], PolygonVertexItem) else vertex_items[i].sceneBoundingRect().center()
                p2 = vertex_items[(i + 1) % len(vertex_items)].pos() if isinstance(vertex_items[(i + 1) % len(vertex_items)], PolygonVertexItem) else vertex_items[(i + 1) % len(vertex_items)].sceneBoundingRect().center()
                line_item.setLine(QLineF(p1, p2))
                
        # Update filled polygon
        if filled_item:
            from PyQt6.QtGui import QPolygonF
            from PyQt6.QtCore import QPointF
            qpoly = QPolygonF()
            for p in pixel_coords:
                qpoly.append(QPointF(p[0], p[1]))
            filled_item.setPolygon(qpoly)
            
        # Update Area and Geo Coords
        geo_coords = []
        if hasattr(self, 'measurement_manager') and self.measurement_manager:
            if self.measurement_manager.is_georeferenced:
                geo_coords = [
                    self.measurement_manager.pixel_to_world(px, py)
                    for px, py in pixel_coords
                ]
            else:
                metadata = self.measurement_manager.metadata
                transform = metadata.get('transform') if metadata else None
                if transform is not None:
                    try:
                        geo_coords = [
                            (
                                transform.c + px * transform.a + py * transform.b,
                                transform.f + px * transform.d + py * transform.e,
                            )
                            for px, py in pixel_coords
                        ]
                    except AttributeError:
                        values = list(transform)
                        geo_coords = [
                            (
                                values[2] + px * values[0] + py * values[1],
                                values[5] + px * values[3] + py * values[4],
                            )
                            for px, py in pixel_coords
                        ]
        
        polygon['geo_coords'] = geo_coords
        
        # Re-calculate area
        area_m2 = self.calculatePolygonArea(pixel_coords, geo_coords)
        polygon['area_m2'] = area_m2
        
        # Update label
        self.updateAreaLabel(polygon)

    def calculatePolygonArea(self, pixel_coords, geo_coords=None):
        return self.calculate_polygon_area(pixel_coords, geo_coords)

    def updateAreaLabel(self, polygon):
        area_m2 = polygon.get('area_m2', 0)
        pixel_coords = polygon.get('pixel_coords', [])
        label = polygon.get('items', {}).get('area_label')
        if label:
            self.update_polygon_area_label(label, area_m2, pixel_coords)

    def deletePolygon(self, polygon, main_window=None):
        """Delete a polygon and its items"""
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
            
        if main_window and hasattr(main_window, 'drawn_polygons') and polygon in main_window.drawn_polygons:
            main_window.drawn_polygons.remove(polygon)
            if hasattr(main_window, '_refresh_polygon_list_ui'):
                main_window._refresh_polygon_list_ui()

    def update_polygon_area_label(self, label, area_m2, pixel_coords):
        """Update one polygon label and move it to the polygon centroid."""
        if label is None:
            return
        label.setPlainText(self.polygon_area_text(area_m2))
        if pixel_coords:
            center_x = sum(point[0] for point in pixel_coords) / len(pixel_coords)
            center_y = sum(point[1] for point in pixel_coords) / len(pixel_coords)
            label.setPos(center_x, center_y)

    def clear_polygon(self):
        """Clear all polygon items from scene"""
        # Clear vertex markers
        for item in self.polygon_vertex_items:
            self.scene.removeItem(item)
        self.polygon_vertex_items.clear()

        # Clear lines
        for item in self.polygon_line_items:
            self.scene.removeItem(item)
        self.polygon_line_items.clear()

        # Clear closing line
        if self.polygon_closing_line:
            self.scene.removeItem(self.polygon_closing_line)
            self.polygon_closing_line = None

        # Clear filled polygon
        if self.polygon_filled_item:
            self.scene.removeItem(self.polygon_filled_item)
            self.polygon_filled_item = None

        # Clear data
        self.polygon_vertices.clear()

        self.logger.info("Polygon cleared from scene")

    def set_polygon_vertex_color(self, color):
        """Update polygon vertex fill color"""
        self.polygon_vertex_color = color
        # Update existing vertices
        for item in self.polygon_vertex_items:
            item.setBrush(QBrush(color))

    def set_polygon_vertex_outline_color(self, color):
        """Update polygon vertex outline color"""
        self.polygon_vertex_outline_color = color
        # Update existing vertices
        for item in self.polygon_vertex_items:
            pen = item.pen()
            pen.setColor(color)
            item.setPen(pen)

    def set_polygon_line_color(self, color):
        """Update polygon line color"""
        self.polygon_line_color = color
        # Update existing lines
        for item in self.polygon_line_items:
            pen = item.pen()
            pen.setColor(color)
            item.setPen(pen)
        # Update closing line
        if self.polygon_closing_line:
            pen = self.polygon_closing_line.pen()
            pen.setColor(color)
            self.polygon_closing_line.setPen(pen)
        # Update filled polygon
        if self.polygon_filled_item:
            pen = self.polygon_filled_item.pen()
            pen.setColor(color)
            self.polygon_filled_item.setPen(pen)

    def set_polygon_vertex_size(self, size):
        """Update polygon vertex size"""
        self.polygon_vertex_size = size
        # Update existing vertices
        for i, item in enumerate(self.polygon_vertex_items):
            scene_pos = self.polygon_vertices[i]
            half_size = size / 2
            item.setRect(
                scene_pos.x() - half_size,
                scene_pos.y() - half_size,
                size,
                size
            )

    def set_polygon_line_width(self, width):
        """Update polygon line width"""
        self.polygon_line_width = width
        # Update existing lines
        for item in self.polygon_line_items:
            pen = item.pen()
            pen.setWidth(width)
            item.setPen(pen)
        # Update closing line
        if self.polygon_closing_line:
            pen = self.polygon_closing_line.pen()
            pen.setWidth(width)
            self.polygon_closing_line.setPen(pen)
        # Update filled polygon
        if self.polygon_filled_item:
            pen = self.polygon_filled_item.pen()
            pen.setWidth(width)
            self.polygon_filled_item.setPen(pen)

    def set_polygon_visibility(self, visible):
        """Toggle visibility of polygon items"""
        items_toggled = 0

        # Toggle vertex markers
        for item in self.polygon_vertex_items:
            item.setVisible(visible)
            items_toggled += 1

        # Toggle lines
        for item in self.polygon_line_items:
            item.setVisible(visible)
            items_toggled += 1

        # Toggle closing line
        if self.polygon_closing_line:
            self.polygon_closing_line.setVisible(visible)
            items_toggled += 1

        # Toggle filled polygon
        if self.polygon_filled_item:
            self.polygon_filled_item.setVisible(visible)
            items_toggled += 1

        self.logger.info(f"Polygon visibility set to {visible} - {items_toggled} items toggled "
                        f"(vertices: {len(self.polygon_vertex_items)}, "
                        f"lines: {len(self.polygon_line_items)}, "
                        f"closing_line: {self.polygon_closing_line is not None}, "
                        f"filled: {self.polygon_filled_item is not None})")

    # Alias methods for compatibility with main_window.py
    def update_polygon_vertex_color(self, color):
        """Alias for set_polygon_vertex_color"""
        self.set_polygon_vertex_color(color)

    def update_polygon_vertex_outline_color(self, color):
        """Alias for set_polygon_vertex_outline_color"""
        self.set_polygon_vertex_outline_color(color)

    def update_polygon_line_color(self, color):
        """Alias for set_polygon_line_color"""
        self.set_polygon_line_color(color)

    def update_polygon_vertex_size(self, size):
        """Alias for set_polygon_vertex_size"""
        self.set_polygon_vertex_size(size)

    def update_polygon_line_width(self, width):
        """Alias for set_polygon_line_width"""
        self.set_polygon_line_width(width)

    # ========== VECTOR LAYER RENDERING METHODS ==========

    def render_vector_layer(self, features, transform, raster_crs, vector_crs, style=None):
        """Render vector features (from shapefile/GeoJSON) on the scene

        Args:
            features: List of shapely geometries or feature dicts
            transform: Raster affine transform
            raster_crs: Raster CRS
            vector_crs: Vector CRS
            style: Dict with styling options (stroke_color, stroke_width, fill_color)

        Returns:
            List of QGraphicsItems created
        """
        from PyQt6.QtWidgets import QGraphicsPathItem, QGraphicsEllipseItem, QGraphicsLineItem
        from PyQt6.QtGui import QColor, QPen, QBrush, QPainterPath
        from PyQt6.QtCore import QPointF

        if style is None:
            style = {
                'stroke_color': QColor(255, 0, 0),  # Red
                'stroke_width': 2,
                'fill_color': QColor(255, 0, 0, 50),  # Semi-transparent red
            }

        items = []

        try:
            # Setup coordinate transformation if needed
            transformer = None
            if vector_crs and raster_crs and vector_crs != raster_crs:
                try:
                    from pyproj import CRS, Transformer
                    from_crs = CRS(vector_crs) if not isinstance(vector_crs, str) else CRS.from_user_input(vector_crs)
                    to_crs = CRS(raster_crs) if not isinstance(raster_crs, str) else CRS.from_user_input(raster_crs)
                    transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
                    self.logger.info(f"[VECTOR RENDER] Created transformer: {vector_crs} → {raster_crs}")
                except Exception as e:
                    self.logger.warning(f"[VECTOR RENDER] Failed to create transformer: {e}")

            for feature in features:
                try:
                    # Extract geometry
                    if isinstance(feature, dict):
                        geom = feature.get('geometry')
                    else:
                        geom = feature

                    if geom is None:
                        continue

                    geom_type = geom.geom_type

                    # Render based on geometry type
                    if geom_type == 'Point':
                        item = self._render_point(geom, transform, transformer, style)
                        if item:
                            items.append(item)

                    elif geom_type == 'LineString':
                        item = self._render_linestring(geom, transform, transformer, style)
                        if item:
                            items.append(item)

                    elif geom_type == 'Polygon':
                        item = self._render_polygon(geom, transform, transformer, style)
                        if item:
                            items.append(item)

                    elif geom_type == 'MultiPoint':
                        for pt in geom.geoms:
                            item = self._render_point(pt, transform, transformer, style)
                            if item:
                                items.append(item)

                    elif geom_type == 'MultiLineString':
                        for line in geom.geoms:
                            item = self._render_linestring(line, transform, transformer, style)
                            if item:
                                items.append(item)

                    elif geom_type == 'MultiPolygon':
                        for poly in geom.geoms:
                            item = self._render_polygon(poly, transform, transformer, style)
                            if item:
                                items.append(item)

                    else:
                        self.logger.warning(f"[VECTOR RENDER] Unsupported geometry type: {geom_type}")

                except Exception as e:
                    self.logger.error(f"[VECTOR RENDER] Error rendering feature: {e}", exc_info=True)
                    continue

            self.logger.info(f"[VECTOR RENDER] Rendered {len(items)} vector items")
            return items

        except Exception as e:
            self.logger.error(f"[VECTOR RENDER] Error in render_vector_layer: {e}", exc_info=True)
            return []

    def _geo_to_pixel(self, x, y, transform, transformer):
        """Convert geographic coordinates to pixel coordinates"""
        try:
            # Transform CRS if needed
            if transformer:
                x, y = transformer.transform(x, y)

            # Apply inverse affine transform to get pixel coordinates
            # transform is: (c + a*col + b*row, f + d*col + e*row)
            # We need to solve: x = c + a*col + b*row, y = f + d*col + e*row
            # Using rasterio convention: col=(x-c)/a (assuming b=0), row=(y-f)/e (assuming d=0)

            # For proper affine inversion:
            a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f

            # Inverse transform
            det = a * e - b * d
            if abs(det) < 1e-10:
                return None, None

            col = (e * (x - c) - b * (y - f)) / det
            row = (a * (y - f) - d * (x - c)) / det

            return col, row

        except Exception as e:
            self.logger.debug(f"[VECTOR RENDER] Geo to pixel conversion failed: {e}")
            return None, None

    def _render_point(self, point, transform, transformer, style):
        """Render a point geometry as a visible marker (like palm detection points)"""
        from PyQt6.QtWidgets import QGraphicsEllipseItem
        from PyQt6.QtGui import QPen, QBrush, QColor

        try:
            px, py = self._geo_to_pixel(point.x, point.y, transform, transformer)
            if px is None or py is None:
                return None

            # Get point size from style (default: larger for visibility)
            point_size = style.get('point_size', 10)  # Increased from 5 to 10
            radius = point_size / 2.0

            # Create point marker circle
            item = QGraphicsEllipseItem(px - radius, py - radius, point_size, point_size)

            # Stroke (outline) - make it prominent
            stroke_color = style.get('stroke_color', QColor(255, 0, 0))  # Red
            stroke_width = style.get('stroke_width', 2)
            pen = QPen(stroke_color)
            pen.setWidth(stroke_width)
            item.setPen(pen)

            # Fill - semi-transparent for better visibility
            fill_color = style.get('fill_color', QColor(255, 255, 0, 180))  # Yellow with transparency
            brush = QBrush(fill_color)
            item.setBrush(brush)

            # Set high Z-value to ensure points are above everything
            item.setZValue(100)  # Above raster

            # Make points scalable with zoom
            # item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, False)

            self.scene.addItem(item)

            return item

        except Exception as e:
            self.logger.debug(f"[VECTOR RENDER] Error rendering point: {e}")
            return None

    def _render_linestring(self, linestring, transform, transformer, style):
        """Render a linestring geometry"""
        from PyQt6.QtWidgets import QGraphicsPathItem
        from PyQt6.QtGui import QPen, QPainterPath
        from PyQt6.QtCore import QPointF

        try:
            path = QPainterPath()
            coords = list(linestring.coords)

            if len(coords) < 2:
                return None

            # Convert first point
            px, py = self._geo_to_pixel(coords[0][0], coords[0][1], transform, transformer)
            if px is None or py is None:
                return None

            path.moveTo(QPointF(px, py))

            # Add remaining points
            for x, y in coords[1:]:
                px, py = self._geo_to_pixel(x, y, transform, transformer)
                if px is not None and py is not None:
                    path.lineTo(QPointF(px, py))

            item = QGraphicsPathItem(path)

            pen = QPen(style.get('stroke_color', QColor(255, 0, 0)))
            pen.setWidth(style.get('stroke_width', 2))
            item.setPen(pen)

            item.setZValue(100)
            self.scene.addItem(item)

            return item

        except Exception as e:
            self.logger.debug(f"[VECTOR RENDER] Error rendering linestring: {e}")
            return None

    def _render_polygon(self, polygon, transform, transformer, style):
        """Render a polygon geometry"""
        from PyQt6.QtWidgets import QGraphicsPathItem
        from PyQt6.QtGui import QPen, QBrush, QPainterPath
        from PyQt6.QtCore import QPointF

        try:
            path = QPainterPath()

            # Exterior ring
            exterior_coords = list(polygon.exterior.coords)
            if len(exterior_coords) < 3:
                return None

            # Convert first point
            px, py = self._geo_to_pixel(exterior_coords[0][0], exterior_coords[0][1], transform, transformer)
            if px is None or py is None:
                return None

            path.moveTo(QPointF(px, py))

            # Add remaining points
            for x, y in exterior_coords[1:]:
                px, py = self._geo_to_pixel(x, y, transform, transformer)
                if px is not None and py is not None:
                    path.lineTo(QPointF(px, py))

            path.closeSubpath()

            # Add holes (interior rings)
            for interior in polygon.interiors:
                interior_coords = list(interior.coords)
                if len(interior_coords) < 3:
                    continue

                px, py = self._geo_to_pixel(interior_coords[0][0], interior_coords[0][1], transform, transformer)
                if px is None or py is None:
                    continue

                path.moveTo(QPointF(px, py))

                for x, y in interior_coords[1:]:
                    px, py = self._geo_to_pixel(x, y, transform, transformer)
                    if px is not None and py is not None:
                        path.lineTo(QPointF(px, py))

                path.closeSubpath()

            item = QGraphicsPathItem(path)

            pen = QPen(style.get('stroke_color', QColor(255, 0, 0)))
            pen.setWidth(style.get('stroke_width', 2))
            item.setPen(pen)

            brush = QBrush(style.get('fill_color', QColor(255, 0, 0, 50)))
            item.setBrush(brush)

            item.setZValue(100)
            self.scene.addItem(item)

            return item

        except Exception as e:
            self.logger.debug(f"[VECTOR RENDER] Error rendering polygon: {e}")
            return None

    def clear_vector_items(self, items):
        """Remove vector items from scene"""
        try:
            for item in items:
                try:
                    self.scene.removeItem(item)
                except Exception as e:
                    self.logger.debug(f"Failed to remove vector item: {e}")
            self.logger.debug(f"[VECTOR RENDER] Cleared {len(items)} vector items")
        except Exception as e:
            self.logger.error(f"[VECTOR RENDER] Error clearing vector items: {e}")

    def set_vector_visibility(self, items, visible):
        """Set visibility of vector items"""
        try:
            for item in items:
                try:
                    item.setVisible(visible)
                except Exception as e:
                    self.logger.debug(f"Failed to set vector item visibility: {e}")
            self.logger.debug(f"[VECTOR RENDER] Set {len(items)} vector items visibility: {visible}")
        except Exception as e:
            self.logger.error(f"[VECTOR RENDER] Error setting vector visibility: {e}")

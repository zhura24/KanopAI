"""Tile loader dengan optimasi threading dan caching."""

from typing import Optional, Dict, List, Tuple, Any
from PyQt6.QtCore import QRunnable, QObject, pyqtSignal, QThreadPool, QTimer
from PyQt6.QtGui import QImage, QPixmap
import numpy as np
from numpy.typing import NDArray
from utils.logger_config import get_logger
from collections import OrderedDict
from utils.constants import (
    PERCENTILE_LOW,
    PERCENTILE_HIGH,
    UINT16_MAX_VALUE,
    UINT8_MAX_VALUE,
    MID_GRAY_VALUE,
    REFLECTANCE_MAX_THRESHOLD,
    REFLECTANCE_MIN_THRESHOLD,
    DEFAULT_THREAD_COUNT,
    MAX_TILE_CACHE_SIZE,
    TILE_LOAD_DEBOUNCE_MS,
    TILE_BATCH_SIZE,
    TILE_BATCH_INTERVAL_MS,
    TILE_CACHE_CLEANUP_PERCENT
)


class TileLoadSignals(QObject):
    tile_loaded = pyqtSignal(int, int, int, int, object)
    load_complete = pyqtSignal()
    error_occurred = pyqtSignal(str)


class TileLoadWorker(QRunnable):
    def __init__(self, tile_manager: Any, tile_x: int, tile_y: int, zoom_level: int,
                 generation: int = 0,
                 global_stats: Optional[List[Dict[str, float]]] = None, 
                 use_normalization: bool = False) -> None:
        super().__init__()
        self.tile_manager = tile_manager
        self.tile_x = tile_x
        self.tile_y = tile_y
        self.zoom_level = zoom_level
        self.generation = generation
        self.global_stats = global_stats
        self.use_normalization = use_normalization
        self.signals = TileLoadSignals()
        self.setAutoDelete(True)

    def run(self) -> None:
        try:
            tile_data = self.tile_manager.get_tile(self.tile_x, self.tile_y, self.zoom_level)

            if tile_data is not None:
                # QImage is thread-safe; QPixmap conversion happens on main thread
                qimage = self._convert_to_pixmap(tile_data)
                level = self.tile_manager.raster_loader.select_overview_level(self.zoom_level)
                self.signals.tile_loaded.emit(self.tile_x, self.tile_y, level, self.generation, qimage)
            else:
                level = self.tile_manager.raster_loader.select_overview_level(self.zoom_level)
                self.signals.tile_loaded.emit(self.tile_x, self.tile_y, level, self.generation, None)

        except Exception as e:
            level = self.tile_manager.raster_loader.select_overview_level(self.zoom_level)
            self.signals.tile_loaded.emit(self.tile_x, self.tile_y, level, self.generation, None)
            self.signals.error_occurred.emit(f"Error loading tile ({self.tile_x}, {self.tile_y}): {str(e)}")

    def _convert_to_pixmap(self, tile_data: Optional[NDArray[np.float32]]) -> Optional[QImage]:
        if tile_data is None:
            return None

        try:
            num_bands = tile_data.shape[0]
            height, width = tile_data.shape[1], tile_data.shape[2]

            if hasattr(self, 'tile_manager') and hasattr(self.tile_manager, 'logger'):
                logger = self.tile_manager.logger
                min_val = float(tile_data[0].min())
                max_val = float(tile_data[0].max())
                logger.debug(f"[TILE] Converting tile ({self.tile_x},{self.tile_y}) | Bands: {num_bands} | Size: {width}x{height} | Band 1 range: [{min_val:.4f}, {max_val:.4f}]")

            if tile_data.shape[0] >= 3:
                r = self._normalize_band(tile_data[0], band_idx=0)
                g = self._normalize_band(tile_data[1], band_idx=1)
                b = self._normalize_band(tile_data[2], band_idx=2)

                height, width = r.shape
                rgb_array = np.zeros((height, width, 3), dtype=np.uint8)
                rgb_array[:, :, 0] = r
                rgb_array[:, :, 1] = g
                rgb_array[:, :, 2] = b

                qimage = QImage(rgb_array.data, width, height, width * 3, QImage.Format.Format_RGB888)
            else:
                if num_bands == 2:
                    b1 = tile_data[0].astype(np.float32)
                    b2 = tile_data[1].astype(np.float32)
                    band = self._normalize_band((b1 + b2) / 2.0, band_idx=0)
                else:
                    band = self._normalize_band(tile_data[0], band_idx=0)
                height, width = band.shape
                qimage = QImage(band.data, width, height, width, QImage.Format.Format_Grayscale8)

            return qimage.copy()

        except Exception as e:
            if hasattr(self, 'tile_manager') and hasattr(self.tile_manager, 'logger'):
                self.tile_manager.logger.error(f"[TILE] Error converting tile to pixmap: {e}", exc_info=True)
            return None

    def _normalize_band(self, band: NDArray[np.float32], band_idx: int = 0) -> NDArray[np.uint8]:
        """Convert band to uint8 for display

        If use_normalization is False, uses raw pixel values (true colors).
        If True, applies percentile-based contrast stretching.
        For reflectance data (0-1 range), always applies contrast stretching.
        """
        band = band.astype(np.float32)

        is_reflectance = (band.max() <= REFLECTANCE_MAX_THRESHOLD and band.min() >= REFLECTANCE_MIN_THRESHOLD)

        if is_reflectance:
            if hasattr(self, 'tile_manager') and hasattr(self.tile_manager, 'logger'):
                self.tile_manager.logger.debug(
                    f"[TILE] Reflectance data detected for band {band_idx} | "
                    f"Range: [{band.min():.4f}, {band.max():.4f}] | "
                    f"Applying percentile stretching"
                )

            if self.global_stats and band_idx < len(self.global_stats):
                p2 = self.global_stats[band_idx]['p2']
                p98 = self.global_stats[band_idx]['p98']
            else:
                p2, p98 = np.percentile(band, (PERCENTILE_LOW, PERCENTILE_HIGH))

            band = np.clip(band, p2, p98)

            if p98 > p2:
                band = (band - p2) / (p98 - p2) * UINT8_MAX_VALUE
            else:
                band = np.full_like(band, MID_GRAY_VALUE)

            return band.astype(np.uint8)

        if not self.use_normalization:
            if band.max() <= UINT8_MAX_VALUE:
                band = np.clip(band, 0, UINT8_MAX_VALUE)
            elif band.max() <= UINT16_MAX_VALUE:
                band = (band / UINT16_MAX_VALUE) * UINT8_MAX_VALUE
            else:
                band = (band / band.max()) * UINT8_MAX_VALUE

            return np.clip(band, 0, UINT8_MAX_VALUE).astype(np.uint8)

        if self.global_stats and band_idx < len(self.global_stats):
            p2 = self.global_stats[band_idx]['p2']
            p98 = self.global_stats[band_idx]['p98']
        else:
            p2, p98 = np.percentile(band, (PERCENTILE_LOW, PERCENTILE_HIGH))

        band = np.clip(band, p2, p98)

        if p98 > p2:
            band = (band - p2) / (p98 - p2) * UINT8_MAX_VALUE
        else:
            band = np.zeros_like(band)

        return band.astype(np.uint8)


class AsyncTileLoader(QObject):
    tiles_ready = pyqtSignal()

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.logger = get_logger(__name__)
        self.thread_pool = QThreadPool.globalInstance()
        try:
            self.thread_pool.setMaxThreadCount(DEFAULT_THREAD_COUNT)
        except (AttributeError, RuntimeError) as e:
            self.logger.warning(f"Could not set max thread count: {e}")

        self.pending_tiles = {}
        self.loading_tiles = set()
        self.loaded_pixmaps = OrderedDict()
        self.max_pixmap_cache_bytes = 512 * 1024 * 1024
        self.pixmap_cache_bytes = 0

        self.load_timer = QTimer()
        self.load_timer.setSingleShot(True)
        self.load_timer.timeout.connect(self._process_tile_queue)
        self.debounce_delay = TILE_LOAD_DEBOUNCE_MS
        self.batch_size = TILE_BATCH_SIZE
        self.tile_manager = None
        self.viewport_center = None
        self.use_color_normalization = False
        self.current_zoom_level = 1.0
        self.current_overview_level = 0
        self.generation = 0
        self.paused = False

        self.logger.info(
            f"AsyncTileLoader initialized (High Performance) | "
            f"Max threads: {self.thread_pool.maxThreadCount()} | "
            f"Batch size: {self.batch_size} | "
            f"Debounce: {self.debounce_delay}ms | "
            f"Max pixmap cache: {self.max_pixmap_cache_bytes / (1024 * 1024):.0f} MB"
        )

    def request_tiles(self, tile_manager: Any, required_tiles: List[Tuple[int, int]], 
                     zoom_level: float, immediate: bool = False, 
                     viewport_center: Optional[Tuple[float, float]] = None) -> None:
        """Request loading of tiles for the viewport"""
        if getattr(self, 'paused', False):
            try:
                self.logger.debug("AsyncTileLoader is paused - ignoring tile request")
            except AttributeError:
                pass
            self.tiles_ready.emit()
            return

        self.pending_tiles.clear()
        self.tile_manager = tile_manager
        self.viewport_center = viewport_center
        self.current_zoom_level = zoom_level
        overview_level = tile_manager.raster_loader.select_overview_level(zoom_level)

        self.current_overview_level = overview_level

        new_tiles_needed = 0
        for tx, ty in required_tiles:
            key = (overview_level, tx, ty)
            if key not in self.loaded_pixmaps and key not in self.loading_tiles:
                self.pending_tiles[key] = (overview_level, tx, ty)
                new_tiles_needed += 1

        self.logger.info(
            f"Tile request | "
            f"Required: {len(required_tiles)} | "
            f"Cached: {len(required_tiles) - new_tiles_needed} | "
            f"To load: {new_tiles_needed} | "
            f"Mode: {'IMMEDIATE' if immediate else 'DEBOUNCED'}"
        )

        if immediate:
            self._process_tile_queue()
        else:
            self.load_timer.start(self.debounce_delay)

        self.tiles_ready.emit()

    def _process_tile_queue(self) -> None:
        """Process the tile loading queue"""
        if getattr(self, 'paused', False):
            self.logger.debug("AsyncTileLoader paused - skipping _process_tile_queue")
            return

        if not self.pending_tiles:
            self.logger.debug("Tile queue empty, nothing to process")
            return

        tiles_list = list(self.pending_tiles.items())

        if self.viewport_center is not None:
            try:
                cx, cy = self.viewport_center
                def _dist(item):
                    key, (_, tx, ty) = item
                    return abs(tx - cx) + abs(ty - cy)

                tiles_list.sort(key=_dist)
            except (ValueError, TypeError) as e:
                self.logger.debug(f"Could not sort tiles by distance: {e}")

        tiles_to_load = tiles_list[: self.batch_size]

        self.logger.info(
            f"Processing tile batch | Batch size: {len(tiles_to_load)} | Pending: {len(self.pending_tiles)} | Active threads: {self.thread_pool.activeThreadCount()}"
        )

        # Get global statistics from tile manager
        global_stats = None
        if self.tile_manager and hasattr(self.tile_manager, 'global_statistics'):
            global_stats = self.tile_manager.global_statistics

        for key, (overview_level, tx, ty) in tiles_to_load:
            self.loading_tiles.add(key)
            try:
                del self.pending_tiles[key]
            except KeyError:
                pass

            worker = TileLoadWorker(
                self.tile_manager,
                tx, ty,
                self.current_zoom_level,
                generation=self.generation,
                global_stats=global_stats,
                use_normalization=self.use_color_normalization
            )
            worker.signals.tile_loaded.connect(self._on_tile_loaded)
            self.thread_pool.start(worker)

        if self.pending_tiles:
            QTimer.singleShot(TILE_BATCH_INTERVAL_MS, self._process_tile_queue)

    def _on_tile_loaded(
        self,
        tile_x: int,
        tile_y: int,
        overview_level: int,
        generation: int,
        pixmap: Optional[QImage],
    ) -> None:
        """Handle tile loaded callback"""
        if generation != self.generation:
            self.logger.debug(
                "Ignoring stale tile callback from generation %s (current %s)",
                generation,
                self.generation,
            )
            return
        if pixmap is not None:
            try:
                if isinstance(pixmap, QImage):
                    pix = QPixmap.fromImage(pixmap)
                else:
                    pix = pixmap
            except (RuntimeError, TypeError) as e:
                self.logger.warning(f"Could not convert QImage to QPixmap: {e}")
                pix = None

            key = (overview_level, tile_x, tile_y)
            if pix is not None:
                pixmap_bytes = self._pixmap_size_bytes(pix)
                old_pixmap = self.loaded_pixmaps.pop(key, None)
                if old_pixmap is not None:
                    self.pixmap_cache_bytes -= self._pixmap_size_bytes(old_pixmap)
                self.loaded_pixmaps[key] = pix
                self.pixmap_cache_bytes += pixmap_bytes
            try:
                self.loaded_pixmaps.move_to_end(key)
            except Exception as e:
                self.logger.debug(f"Failed to move tile to end of cache: {e}")

            while self.pixmap_cache_bytes > self.max_pixmap_cache_bytes and self.loaded_pixmaps:
                _, old_pixmap = self.loaded_pixmaps.popitem(last=False)
                self.pixmap_cache_bytes -= self._pixmap_size_bytes(old_pixmap)

            self.logger.debug(f"Tile ready | ({tile_x}, {tile_y}) | Cache size: {len(self.loaded_pixmaps)}")
        else:
            self.logger.warning(f"Tile failed to load | ({tile_x}, {tile_y})")

        key_to_remove = (overview_level, tile_x, tile_y)
        self.loading_tiles.discard(key_to_remove)

        self.tiles_ready.emit()

        if self.pending_tiles:
            remaining = len(self.pending_tiles)
            active_threads = self.thread_pool.activeThreadCount()
            if active_threads < self.thread_pool.maxThreadCount():
                self.logger.debug(f"Continuing tile loading | Remaining: {remaining} | Active threads: {active_threads}")
                self._process_tile_queue()

    def get_loaded_pixmap(
        self,
        tile_x: int,
        tile_y: int,
        overview_level: Optional[int] = None,
    ) -> Optional[QPixmap]:
        if overview_level is None:
            overview_level = self.current_overview_level
        key = (overview_level, tile_x, tile_y)
        return self.loaded_pixmaps.get(key)

    @staticmethod
    def _pixmap_size_bytes(pixmap: QPixmap) -> int:
        """Estimate QPixmap memory without relying on a Qt-version-specific API."""
        if pixmap is None or pixmap.isNull():
            return 0
        return max(1, pixmap.width() * pixmap.height() * 4)

    def cancel_all_pending(self) -> None:
        """Cancel all pending tile load requests (but keep cache)"""
        pending_count = len(self.pending_tiles)
        loading_count = len(self.loading_tiles)

        if pending_count > 0 or loading_count > 0:
            self.logger.info(f"Cancelling tile loads | Pending: {pending_count} | Loading: {loading_count}")

        self.pending_tiles.clear()

        if self.load_timer.isActive():
            self.load_timer.stop()

    def clear_cache(self) -> None:
        """Clear all cached tiles and pending requests"""
        self.generation += 1
        cache_size = len(self.loaded_pixmaps)
        pending_count = len(self.pending_tiles)
        self.logger.info(f"Clearing tile loader cache | Cached: {cache_size} | Pending: {pending_count}")

        self.loaded_pixmaps.clear()
        self.pixmap_cache_bytes = 0
        self.pending_tiles.clear()
        self.loading_tiles.clear()
        self.thread_pool.clear()

    def set_tile_manager(self, tile_manager):
        self.generation += 1
        self.tile_manager = tile_manager
        self.current_overview_level = 0
        self.pending_tiles.clear()
        self.loading_tiles.clear()
        self.loaded_pixmaps.clear()
        self.pixmap_cache_bytes = 0
        if self.load_timer.isActive():
            self.load_timer.stop()

"""Manager untuk tile caching dan loading."""

import numpy as np
from collections import OrderedDict
from PyQt6.QtCore import QObject, pyqtSignal
import threading
from utils.logger_config import get_logger


class TileCache:
    """Thread-safe LRU cache bounded by memory instead of tile count."""
    def __init__(self, max_bytes=512 * 1024 * 1024):
        self.cache = OrderedDict()
        self.max_bytes = max_bytes
        self.current_bytes = 0
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
        return None

    def put(self, key, value):
        if value is None:
            return

        value_bytes = int(getattr(value, "nbytes", 0))
        if value_bytes <= 0 or value_bytes > self.max_bytes:
            return

        with self.lock:
            if key in self.cache:
                old_value = self.cache.pop(key)
                self.current_bytes -= int(getattr(old_value, "nbytes", 0))

            self.cache[key] = value
            self.current_bytes += value_bytes

            while self.current_bytes > self.max_bytes and self.cache:
                _, old_value = self.cache.popitem(last=False)
                self.current_bytes -= int(getattr(old_value, "nbytes", 0))

    def clear(self):
        with self.lock:
            self.cache.clear()
            self.current_bytes = 0

    def get_memory_usage(self):
        with self.lock:
            return self.current_bytes / (1024 * 1024)


class TileManager(QObject):
    tile_loaded = pyqtSignal(int, int, object)

    def __init__(self, raster_loader, tile_size=512):
        super().__init__()
        self.logger = get_logger(__name__)
        self.raster_loader = raster_loader
        self.tile_size = tile_size
        self.tile_cache = TileCache(max_bytes=512 * 1024 * 1024)
        self.use_full_resolution = False
        self.global_statistics = None  # Global statistics for consistent color normalization

        # Which 1-based band numbers to actually read for display, in R,G,B
        # order. None means "use the default": literal band 1,2,3 (or fewer
        # if the raster has less than 3 bands) -- matches the mandatory
        # "no manual band selector, always band 1/2/3" preview spec. This is
        # only ever overridden by the optional Auto RGB Mode toggle.
        self.display_band_indexes = None

        metadata = raster_loader.get_metadata()
        self.logger.info(
            f"TileManager initialized | "
            f"Tile size: {tile_size}x{tile_size} | "
            f"Image: {metadata['width']}x{metadata['height']} | "
            f"Max cache: 512 MB | "
            f"Current cache: {self.tile_cache.get_memory_usage():.1f} MB"
        )

    def enable_full_resolution(self, enabled=True):
        """Enable or disable full resolution tiling"""
        self.use_full_resolution = enabled
        self.logger.info(f"Full resolution mode: {'ENABLED' if enabled else 'DISABLED'}")

        if enabled:
            # Compute global statistics when enabling full resolution
            self.logger.debug("Computing global statistics for full resolution")
            self.global_statistics = self.raster_loader.get_global_statistics()
        if not enabled:
            self.logger.debug("Clearing tile cache")
            self.tile_cache.clear()

    def set_display_band_indices(self, indexes):
        """Set which 1-based band numbers to read for display (R,G,B order).

        Pass None to reset to the default (literal band 1,2,3, or fewer if
        the raster has less than 3 bands). Changing this invalidates the
        tile cache since previously-cached tiles were composited from a
        different set of bands.
        """
        if indexes == self.display_band_indexes:
            return
        self.display_band_indexes = list(indexes) if indexes else None
        self.logger.info(f"Display band indices changed -> {self.display_band_indexes or 'default (literal 1,2,3)'} | Clearing tile cache")
        self.tile_cache.clear()

    def _get_display_band_indexes(self, total_bands):
        """Resolve which 1-based bands to actually read for a tile.

        Always reads ONLY the bands needed for display instead of every
        band in the file -- for a 7-band multispectral raster this reads
        3 bands instead of 7 (or 1 instead of 7 for a single-band preview),
        cutting I/O/RAM roughly proportionally.
        """
        if self.display_band_indexes:
            return [b for b in self.display_band_indexes if 1 <= b <= total_bands] or [1]

        if total_bands >= 3:
            return [1, 2, 3]
        elif total_bands == 2:
            return [1, 2]
        else:
            return [1]

    def get_tile(self, tile_x, tile_y, zoom_level=1.0):
        """Get a tile from cache or load it"""
        band_key = tuple(self.display_band_indexes) if self.display_band_indexes else None
        overview_level = self.raster_loader.select_overview_level(zoom_level)
        cache_key = (tile_x, tile_y, overview_level, self.use_full_resolution, band_key)

        cached = self.tile_cache.get(cache_key)
        if cached is not None:
            self.logger.debug(f"Cache HIT | Tile ({tile_x}, {tile_y})")
            return cached

        self.logger.debug(f"Cache MISS | Tile ({tile_x}, {tile_y}) - loading from disk")
        tile_data = self._load_tile(tile_x, tile_y, zoom_level, overview_level)

        if tile_data is not None:
            self.tile_cache.put(cache_key, tile_data)

        return tile_data

    def _load_tile(self, tile_x, tile_y, zoom_level, overview_level=0):
        """Load a tile from the raster dataset"""
        if not self.raster_loader.dataset:
            self.logger.warning("_load_tile called but no dataset available")
            return None

        try:
            metadata = self.raster_loader.get_metadata()

            x_offset = tile_x * self.tile_size
            y_offset = tile_y * self.tile_size

            width = min(self.tile_size, metadata['width'] - x_offset)
            height = min(self.tile_size, metadata['height'] - y_offset)

            if width <= 0 or height <= 0:
                self.logger.warning(f"Invalid tile dimensions | Tile ({tile_x}, {tile_y}) | Size: {width}x{height}")
                return None

            if self.use_full_resolution:
                scale = 1.0
            else:
                scale = min(1.0, zoom_level)

            band_indexes = self._get_display_band_indexes(metadata['bands'])

            tile_data = self.raster_loader.read_window(
                x_offset, y_offset,
                width, height,
                scale=scale,
                band_indexes=band_indexes,
                overview_level=overview_level,
            )

            if tile_data is not None:
                size_kb = tile_data.nbytes / 1024
                self.logger.debug(f"Tile loaded | ({tile_x}, {tile_y}) | Size: {size_kb:.1f}KB | Scale: {scale:.2f} | Bands: {band_indexes}")

            return tile_data

        except Exception as e:
            self.logger.error(f"Error loading tile ({tile_x}, {tile_y}): {str(e)}", exc_info=True)
            return None

    def get_required_tiles(self, viewport_rect, image_rect, zoom_level):
        tiles = []

        if not self.raster_loader.dataset:
            return tiles

        metadata = self.raster_loader.get_metadata()

        scale_x = image_rect.width() / metadata['width']
        scale_y = image_rect.height() / metadata['height']

        vp_img_x = viewport_rect.x() / scale_x
        vp_img_y = viewport_rect.y() / scale_y
        vp_img_w = viewport_rect.width() / scale_x
        vp_img_h = viewport_rect.height() / scale_y

        start_tile_x = max(0, int(vp_img_x // self.tile_size))
        start_tile_y = max(0, int(vp_img_y // self.tile_size))
        end_tile_x = min(
            int(np.ceil((vp_img_x + vp_img_w) / self.tile_size)),
            int(np.ceil(metadata['width'] / self.tile_size))
        )
        end_tile_y = min(
            int(np.ceil((vp_img_y + vp_img_h) / self.tile_size)),
            int(np.ceil(metadata['height'] / self.tile_size))
        )

        for ty in range(start_tile_y, end_tile_y):
            for tx in range(start_tile_x, end_tile_x):
                tiles.append((tx, ty))

        return tiles

    def clear_cache(self):
        """Clear tile cache"""
        cache_stats = self.get_cache_stats()
        self.logger.info(f"Clearing tile cache | Cached tiles: {cache_stats['cached_tiles']} | Memory: {cache_stats['memory_mb']:.2f}MB")
        self.tile_cache.clear()

    def get_cache_stats(self):
        return {
            'cached_tiles': len(self.tile_cache.cache),
            'memory_mb': self.tile_cache.get_memory_usage(),
            'full_resolution': self.use_full_resolution,
            'max_cache_mb': self.tile_cache.max_bytes / (1024 * 1024),
        }

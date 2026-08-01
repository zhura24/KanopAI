from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from core.raster_loader import RasterLoader
from core.tile_manager import TileCache, TileManager


def create_tiled_raster(path: Path) -> None:
    profile = {
        "driver": "GTiff",
        "height": 2048,
        "width": 2048,
        "count": 3,
        "dtype": "uint8",
        "crs": "EPSG:4326",
        "transform": from_origin(0, 2048, 1, 1),
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.zeros((3, 2048, 2048), dtype=np.uint8))
        dst.build_overviews([2, 4, 8], rasterio.enums.Resampling.average)


def test_tile_cache_evicts_by_bytes():
    cache = TileCache(max_bytes=100)
    cache.put("old", np.zeros(60, dtype=np.uint8))
    cache.put("new", np.zeros(60, dtype=np.uint8))

    assert list(cache.cache) == ["new"]
    assert cache.current_bytes == 60


def test_overview_selection_and_windowed_read(tmp_path):
    raster_path = tmp_path / "large.tif"
    create_tiled_raster(raster_path)

    loader = RasterLoader()
    assert loader.load_file(str(raster_path))

    assert loader.get_overview_decimations()[:3] == [2, 4, 8]
    assert loader.select_overview_level(1.0) == 0
    assert loader.select_overview_level(0.2) == 2

    data = loader.read_window(
        0,
        0,
        512,
        512,
        scale=1.0,
        band_indexes=[1, 2, 3],
        overview_level=2,
    )

    assert data.shape == (3, 128, 128)
    loader.dataset.close()


def test_tile_manager_reports_byte_bounded_cache(tmp_path):
    raster_path = tmp_path / "large.tif"
    create_tiled_raster(raster_path)

    loader = RasterLoader()
    assert loader.load_file(str(raster_path))
    manager = TileManager(loader, tile_size=512)
    manager.get_tile(0, 0, zoom_level=0.2)

    stats = manager.get_cache_stats()
    assert stats["cached_tiles"] == 1
    assert stats["max_cache_mb"] == 512.0
    assert stats["memory_mb"] < 1.0
    loader.dataset.close()

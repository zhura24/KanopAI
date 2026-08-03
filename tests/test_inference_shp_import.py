from pathlib import Path

import numpy as np
import rasterio
import shapefile

from core.inference_engine import load_inference_result_from_shapefile, compute_visible_crop_region


def test_compute_visible_crop_region_clamps_to_raster_bounds():
    bounds = compute_visible_crop_region(2000, 1500, (-50, -20, 250, 1200))
    assert bounds == (0, 0, 250, 1200)


def test_load_inference_result_from_shapefile(tmp_path):
    shp_path = tmp_path / "inference_result.shp"

    with shapefile.Writer(str(shp_path), shapeType=shapefile.POLYGON) as writer:
        writer.field("id", "N", size=10)
        writer.field("kelas", "C", size=20)
        writer.field("confidence", "N", size=10, decimal=4)
        writer.field("model", "C", size=30)
        writer.field("x1_px", "N", size=10, decimal=1)
        writer.field("y1_px", "N", size=10, decimal=1)
        writer.field("x2_px", "N", size=10, decimal=1)
        writer.field("y2_px", "N", size=10, decimal=1)
        writer.field("status", "C", size=20)

        polygon = [[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 50.0], [0.0, 0.0]]
        writer.poly([polygon])
        writer.record(1, "palm", 0.92, "demo", 0.0, 0.0, 100.0, 50.0, "not_corrected")

    result = load_inference_result_from_shapefile(shp_path)

    assert result is not None
    assert result.boxes.shape == (1, 4)
    assert result.scores.shape == (1,)
    assert result.classes.shape == (1,)
    assert result.class_names == ["palm"]


def test_load_inference_result_from_georeferenced_shape_uses_geometry_when_fields_are_stale(tmp_path):
    shp_path = tmp_path / "inference_result.shp"
    raster_path = tmp_path / "demo.tif"

    transform = rasterio.transform.from_origin(100.0, 200.0, 1.0, 1.0)
    height = 400
    width = 500
    data = np.zeros((height, width), dtype=np.uint8)
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data, 1)

    with shapefile.Writer(str(shp_path), shapeType=shapefile.POLYGON) as writer:
        writer.field("id", "N", size=10)
        writer.field("kelas", "C", size=20)
        writer.field("confidence", "N", size=10, decimal=4)
        writer.field("model", "C", size=30)
        writer.field("x1_px", "N", size=10, decimal=1)
        writer.field("y1_px", "N", size=10, decimal=1)
        writer.field("x2_px", "N", size=10, decimal=1)
        writer.field("y2_px", "N", size=10, decimal=1)
        writer.field("status", "C", size=20)

        polygon = [
            [100.0, 198.0],
            [102.0, 198.0],
            [102.0, 195.0],
            [100.0, 195.0],
            [100.0, 198.0],
        ]
        writer.poly([polygon])
        writer.record(1, "palm", 0.77, "demo", 9999.0, 9999.0, 10000.0, 10000.0, "not_corrected")

    result = load_inference_result_from_shapefile(shp_path, raster_path=raster_path)

    assert result.boxes.shape == (1, 4)
    assert result.boxes[0, 0] == 0.0
    assert result.boxes[0, 1] == 2.0
    assert result.boxes[0, 2] == 2.0
    assert result.boxes[0, 3] == 5.0
    assert result.class_names == ["palm"]

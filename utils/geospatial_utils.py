"""Utilitas geospasial untuk transformasi koordinat dan kalkulasi metrik."""

from typing import Optional, Tuple, List, Any
import math
import logging
import numpy as np
from numpy.typing import NDArray
from pyproj import Transformer, CRS, Geod
from shapely.geometry import Polygon
from utils.constants import (
    EARTH_RADIUS_METERS,
    METERS_PER_DEGREE,
    MIN_DETERMINANT_THRESHOLD,
    EPSG_WGS84
)


def normalize_polygon_coordinates(coords: Any) -> List[List[float]]:
    """Normalize polygon coordinates for GeoJSON and shapefile export.

    The export path needs a simple list of [x, y] pairs with a closed ring so
    that GIS viewers can consume the geometry without introducing invalid or
    self-intersecting rings.
    """
    if coords is None:
        return []

    try:
        cleaned: List[List[float]] = []
        for item in coords:
            if item is None:
                continue
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                try:
                    x = float(item[0])
                    y = float(item[1])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(x) and math.isfinite(y):
                    cleaned.append([x, y])

        if not cleaned:
            return []

        if cleaned[0] != cleaned[-1]:
            cleaned.append(cleaned[0][:])

        return cleaned
    except Exception:
        return []


class GeospatialMetrics:
    """Handler untuk transformasi koordinat dan kalkulasi geospasial."""

    def __init__(self, transform: Optional[Any], crs: Optional[Any]) -> None:
        self.transform = transform
        self.crs = crs
        self.transformer: Optional[Transformer] = None
        self.logger = logging.getLogger(__name__)

        if crs is not None:
            try:
                self.transformer = Transformer.from_crs(
                    crs,
                    CRS.from_epsg(EPSG_WGS84),
                    always_xy=True
                )
                self.logger.debug(f"GeospatialMetrics: Created transformer from {crs} to WGS84")
            except Exception as e:
                self.logger.warning(f"GeospatialMetrics: Failed to create transformer: {e}")
                self.transformer = None

    def pixel_to_geo(self, x: float, y: float) -> Tuple[float, float]:
        """Konversi koordinat pixel ke koordinat georeferensi."""
        if self.transform is None:
            return x, y

        geo_x = self.transform[2] + x * self.transform[0] + y * self.transform[1]
        geo_y = self.transform[5] + x * self.transform[3] + y * self.transform[4]
        return geo_x, geo_y

    def geo_to_pixel(self, geo_x: float, geo_y: float) -> Tuple[float, float]:
        """Konversi koordinat georeferensi ke koordinat pixel."""
        if self.transform is None:
            return geo_x, geo_y

        a, b, c = self.transform[0], self.transform[1], self.transform[2]
        d, e, f = self.transform[3], self.transform[4], self.transform[5]

        det = a * e - b * d
        if abs(det) < MIN_DETERMINANT_THRESHOLD:
            self.logger.warning("Transform matrix is singular, returning (0, 0)")
            return 0.0, 0.0

        inv_det = 1.0 / det
        x = inv_det * (e * (geo_x - c) - b * (geo_y - f))
        y = inv_det * (-d * (geo_x - c) + a * (geo_y - f))
        return x, y

    def pixel_to_latlon(self, x: float, y: float) -> Tuple[float, float]:
        """Konversi koordinat pixel ke WGS84 lat/lon."""
        if self.transform is None or self.transformer is None:
            return 0.0, 0.0

        geo_x, geo_y = self.pixel_to_geo(x, y)
        try:
            lon, lat = self.transformer.transform(geo_x, geo_y)
            return lon, lat
        except Exception as e:
            self.logger.warning(f"Transformation failed: {e}")
            return 0.0, 0.0


    def polygon_to_geo(self, pixel_coords: List[Tuple[float, float]]) -> NDArray:
        """Konversi koordinat pixel poligon ke koordinat georeferensi."""
        geo_coords = []
        for x, y in pixel_coords:
            geo_x, geo_y = self.pixel_to_geo(x, y)
            geo_coords.append([geo_x, geo_y])
        return np.array(geo_coords)

    def polygon_to_latlon(self, pixel_coords: List[Tuple[float, float]]) -> NDArray:
        """Konversi koordinat pixel poligon ke WGS84 lat/lon."""
        latlon_coords = []
        for x, y in pixel_coords:
            lon, lat = self.pixel_to_latlon(x, y)
            latlon_coords.append([lon, lat])
        return np.array(latlon_coords)

    def polygon_geo_to_pixel(self, geo_coords: List[Tuple[float, float]]) -> Optional[List[Tuple[float, float]]]:
        """Konversi koordinat georeferensi ke koordinat pixel."""
        px = []
        for gx, gy in geo_coords:
            p = self.geo_to_pixel(gx, gy)
            if p is None:
                return None
            px.append([p[0], p[1]])
        return np.array(px)

    def polygon_latlon_to_pixel(self, latlon_coords):
        """Konversi koordinat lat/lon ke koordinat pixel."""
        if self.crs is None:
            self.logger.warning("No CRS available, assuming coords in dataset CRS")
            return self.polygon_geo_to_pixel(latlon_coords)

        if not latlon_coords:
            self.logger.error("Empty coordinates")
            return None

        try:
            is_wgs84 = False
            try:
                dataset_epsg = self.crs.to_epsg()
                is_wgs84 = (dataset_epsg == EPSG_WGS84)
            except (AttributeError, TypeError):
                is_wgs84 = 'EPSG:4326' in str(self.crs) or '4326' in str(self.crs)

            if is_wgs84:
                return self.polygon_geo_to_pixel(latlon_coords)

            transformer_reverse = Transformer.from_crs(
                CRS.from_epsg(EPSG_WGS84),
                self.crs,
                always_xy=True
            )

            geo_coords = []
            for lon, lat in latlon_coords:
                try:
                    geo_x, geo_y = transformer_reverse.transform(lon, lat)
                    geo_coords.append([geo_x, geo_y])
                except Exception as e:
                    self.logger.error(f"Failed to transform (lon={lon}, lat={lat}): {e}")
                    return None

            return self.polygon_geo_to_pixel(geo_coords)

        except Exception as e:
            self.logger.error(f"polygon_latlon_to_pixel failed: {e}")
            return None

    def compute_area_m2(self, pixel_coords):
        """Hitung luas poligon dalam meter persegi."""
        if not pixel_coords:
            return None

        try:
            geo_coords = self.polygon_to_geo(pixel_coords)
            if self.crs is None:
                self.logger.warning("No CRS, area calculation may be inaccurate")
                return None

            crs_obj = CRS.from_wkt(self.crs.to_wkt()) if hasattr(self.crs, 'to_wkt') else self.crs

            if crs_obj.is_geographic:
                poly_latlon = [(lon, lat) for lon, lat in self.polygon_to_latlon(pixel_coords)]
                poly = Polygon(poly_latlon)
                geod = Geod(ellps='WGS84')
                area, _ = geod.geometry_area_perimeter(poly)
                return abs(area)
            else:
                poly = Polygon(geo_coords)
                area = poly.area
                return abs(area)

        except Exception as e:
            self.logger.error(f"Area computation failed: {e}")
            return None

    def compute_diameter(self, pixel_coords):
        """Hitung diameter poligon (jarak maksimum antar vertex)."""
        if not pixel_coords or len(pixel_coords) < 2:
            return None

        try:
            geo_coords = self.polygon_to_geo(pixel_coords)
            if self.crs is None:
                return None

            crs_obj = CRS.from_wkt(self.crs.to_wkt()) if hasattr(self.crs, 'to_wkt') else self.crs

            if crs_obj.is_geographic:
                latlon_coords = self.polygon_to_latlon(pixel_coords)
                max_dist = 0.0
                for i, (lon1, lat1) in enumerate(latlon_coords):
                    for lon2, lat2 in latlon_coords[i+1:]:
                        dist = self._haversine_distance(lat1, lon1, lat2, lon2)
                        max_dist = max(max_dist, dist)
                return max_dist
            else:
                max_dist = 0.0
                for i, (x1, y1) in enumerate(geo_coords):
                    for x2, y2 in geo_coords[i+1:]:
                        dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        max_dist = max(max_dist, dist)
                return max_dist

        except Exception as e:
            self.logger.error(f"Diameter computation failed: {e}")
            return None

    @staticmethod
    def _haversine_distance(lat1, lon1, lat2, lon2):
        """Hitung jarak great-circle antara dua titik di Bumi."""
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = (math.sin(delta_lat / 2)**2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return EARTH_RADIUS_METERS * c

    def compute_centroid_latlon(self, pixel_coords):
        geo_coords = self.polygon_to_geo(pixel_coords)

        try:
            poly = Polygon(geo_coords)
            centroid = poly.centroid

            if self.transformer is None:
                return centroid.y, centroid.x

            lon, lat = self.transformer.transform(centroid.x, centroid.y)
            return lat, lon
        except (ValueError, TypeError, AttributeError) as e:
            self.logger.warning(f"Centroid computation failed: {e}, using mean coordinates")
            center_x = np.mean([c[0] for c in geo_coords])
            center_y = np.mean([c[1] for c in geo_coords])
            return center_y, center_x

    def get_pixel_size_meters(self):
        if self.transform is None:
            return 1.0, 1.0

        pixel_width = abs(self.transform[0])
        pixel_height = abs(self.transform[4])

        if self.crs is not None and self.crs.is_projected:
            return pixel_width, pixel_height
        else:
            center_y = self.transform[5]
            pixel_width_m = pixel_width * METERS_PER_DEGREE * math.cos(math.radians(center_y))
            pixel_height_m = pixel_height * METERS_PER_DEGREE
            return pixel_width_m, pixel_height_m

    def compute_all_metrics(self, pixel_coords):
        area_m2 = self.compute_area_m2(pixel_coords)
        diameter_m = self.compute_diameter(pixel_coords)
        centroid_lat, centroid_lon = self.compute_centroid_latlon(pixel_coords)

        return {
            'area_m2': area_m2,
            'diameter_m': diameter_m,
            'centroid_lat': centroid_lat,
            'centroid_lon': centroid_lon,
            'perimeter_m': self._compute_perimeter(pixel_coords)
        }

    def _compute_perimeter(self, pixel_coords):
        geo_coords = self.polygon_to_geo(pixel_coords)

        perimeter = 0.0
        for i in range(len(geo_coords)):
            j = (i + 1) % len(geo_coords)
            dx = geo_coords[j][0] - geo_coords[i][0]
            dy = geo_coords[j][1] - geo_coords[i][1]
            perimeter += math.sqrt(dx * dx + dy * dy)

        return perimeter

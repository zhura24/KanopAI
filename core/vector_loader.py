"""Loader untuk file vector (Shapefile, GeoJSON)."""

import numpy as np
from pathlib import Path
from PyQt6.QtCore import QObject, pyqtSignal
from utils.logger_config import get_logger


class VectorLoader(QObject):
    """Loader untuk membaca dan mengelola file vector."""
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)
        self.file_path = None
        self.metadata = {}
        self.features = []
        self.crs = None
        self.bounds = None
        self.geometry_type = None
        self.logger.debug("VectorLoader initialized")

    def load_file(self, file_path):
        """Load file vector (shapefile atau GeoJSON) dan ekstrak metadata."""
        self.logger.info(f"Loading vector file: {file_path}")

        try:
            # Try geopandas first (most comprehensive)
            try:
                import geopandas as gpd
                gdf = gpd.read_file(file_path)

                self.file_path = file_path
                self.features = []
                self.crs = gdf.crs

                # Get bounds
                bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
                self.bounds = {
                    'minx': float(bounds[0]),
                    'miny': float(bounds[1]),
                    'maxx': float(bounds[2]),
                    'maxy': float(bounds[3])
                }

                # Get geometry type
                if len(gdf) > 0:
                    self.geometry_type = gdf.geometry.iloc[0].geom_type
                else:
                    self.geometry_type = "Unknown"

                # Convert to feature list
                for idx, row in gdf.iterrows():
                    feature = {
                        'geometry': row.geometry,
                        'properties': row.drop('geometry').to_dict(),
                        'id': idx
                    }
                    self.features.append(feature)

                self.metadata = {
                    'feature_count': len(self.features),
                    'geometry_type': self.geometry_type,
                    'crs': str(self.crs) if self.crs else None,
                    'bounds': self.bounds,
                    'fields': list(gdf.columns.drop('geometry'))
                }

                self.logger.info(
                    f"Vector file loaded successfully | "
                    f"Features: {len(self.features)} | "
                    f"Type: {self.geometry_type} | "
                    f"CRS: {self.crs}"
                )

                return True

            except ImportError:
                # Fallback to fiona
                self.logger.info("GeoPandas not available, using fiona")
                import fiona
                from shapely.geometry import shape

                with fiona.open(file_path, 'r') as src:
                    self.file_path = file_path
                    self.crs = src.crs
                    self.features = []

                    # Get bounds
                    bounds = src.bounds  # (minx, miny, maxx, maxy)
                    self.bounds = {
                        'minx': float(bounds[0]),
                        'miny': float(bounds[1]),
                        'maxx': float(bounds[2]),
                        'maxy': float(bounds[3])
                    }

                    # Read features
                    for feat in src:
                        geom = shape(feat['geometry'])
                        feature = {
                            'geometry': geom,
                            'properties': feat.get('properties', {}),
                            'id': feat.get('id', len(self.features))
                        }
                        self.features.append(feature)

                        if self.geometry_type is None:
                            self.geometry_type = geom.geom_type

                    self.metadata = {
                        'feature_count': len(self.features),
                        'geometry_type': self.geometry_type or "Unknown",
                        'crs': str(self.crs) if self.crs else None,
                        'bounds': self.bounds,
                        'fields': list(src.schema.get('properties', {}).keys())
                    }

                    self.logger.info(
                        f"Vector file loaded successfully (fiona) | "
                        f"Features: {len(self.features)} | "
                        f"Type: {self.geometry_type}"
                    )

                    return True

        except Exception as e:
            error_msg = f"Error loading vector file: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.error_occurred.emit(error_msg)
            return False

    def get_metadata(self):
        """Get vector metadata"""
        return self.metadata

    def get_features(self):
        """Get all features"""
        return self.features

    def get_bounds(self):
        """Get spatial bounds"""
        return self.bounds

    def get_crs(self):
        """Get coordinate reference system"""
        return self.crs

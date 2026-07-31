"""Handler for centroid detection and canopy analysis operations.

Handles business logic for:
- Converting detections to centroids with GPS coordinates
- Generating canopy circle measurements from bounding boxes
- Exporting centroids and canopy data to shapefiles
- Centroid point management (add/delete)
"""

import logging
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import QMessageBox, QFileDialog
from utils.geospatial_utils import GeospatialMetrics


class CentroidHandler:
    """Handler for centroid and canopy analysis operations."""
    
    def __init__(self, main_window):
        self.main_window = main_window
        self.logger = logging.getLogger(__name__)
    
    def save_centroids_to_shapefile(self):
        """Export centroid points to shapefile with GPS coordinates and canopy measurements.
        
        Exports each centroid as a point feature with attributes:
        - id: Centroid ID
        - lat, lon: GPS coordinates (if CRS available)
        - radius_m: Canopy radius in meters
        - diameter_m: Canopy diameter in meters
        - area_m2: Canopy area in square meters
        """
        if not self.main_window.centroid_points:
            QMessageBox.warning(self.main_window, "No Centroids", "No centroids to save. Please add centroids first.")
            return
        
        # Get save file path
        file_path, _ = QFileDialog.getSaveFileName(
            self.main_window,
            "Save Centroids as Shapefile",
            "",
            "ESRI Shapefile (*.shp)"
        )
        
        if not file_path:
            return
        
        try:
            import fiona
            from fiona.crs import from_epsg
            from shapely.geometry import Point, mapping
            
            # Determine CRS from raster
            crs_code = None
            try:
                if self.main_window.raster_loader and self.main_window.raster_loader.dataset:
                    raster_crs = self.main_window.raster_loader.dataset.crs
                    if raster_crs and raster_crs.to_epsg():
                        crs_code = raster_crs.to_epsg()
                        self.logger.info(f"Using raster CRS: EPSG:{crs_code}")
            except Exception as e:
                self.logger.warning(f"Could not extract CRS from raster: {e}")
            
            # Default to WGS84 if no CRS found
            if not crs_code:
                crs_code = 4326
                self.logger.warning(f"No CRS found in raster, defaulting to EPSG:{crs_code}")
            
            crs = from_epsg(crs_code)
            
            # Define schema
            schema = {
                'geometry': 'Point',
                'properties': {
                    'id': 'int',
                    'lat': 'float',
                    'lon': 'float',
                    'radius_m': 'float',
                    'diameter_m': 'float',
                    'area_m2': 'float'
                }
            }
            
            # Get affine transform for pixel to GPS conversion
            transform = None
            try:
                if self.main_window.raster_loader and self.main_window.raster_loader.dataset:
                    transform = self.main_window.raster_loader.dataset.transform
            except Exception as e:
                self.logger.error(f"Could not get transform: {e}")
                QMessageBox.critical(
                    self.main_window,
                    "Transform Error",
                    "Could not extract geospatial transform from raster. Shapefile export cancelled."
                )
                return
            
            if not transform:
                QMessageBox.critical(
                    self.main_window,
                    "No Transform",
                    "Raster does not have geospatial transform. Cannot export to shapefile."
                )
                return
            
            # Write shapefile
            with fiona.open(file_path, 'w', driver='ESRI Shapefile', crs=crs, schema=schema) as output:
                for idx, pt in enumerate(self.main_window.centroid_points, start=1):
                    pixel_x = pt['x']
                    pixel_y = pt['y']
                    
                    # Convert pixel coordinates to GPS (lon, lat)
                    lon, lat = transform * (pixel_x, pixel_y)
                    
                    # Create point geometry
                    point = Point(lon, lat)
                    
                    # Get canopy measurements (if available)
                    radius_m = pt.get('radius_m', 0.0)
                    diameter_m = radius_m * 2 if radius_m else 0.0
                    area_m2 = np.pi * (radius_m ** 2) if radius_m else 0.0
                    
                    # Create feature
                    feature = {
                        'geometry': mapping(point),
                        'properties': {
                            'id': idx,
                            'lat': lat,
                            'lon': lon,
                            'radius_m': round(radius_m, 2),
                            'diameter_m': round(diameter_m, 2),
                            'area_m2': round(area_m2, 2)
                        }
                    }
                    
                    output.write(feature)
            
            self.logger.info(f"Centroids exported to shapefile: {file_path} ({len(self.main_window.centroid_points)} points)")
            QMessageBox.information(
                self.main_window,
                "Export Complete",
                f"Successfully exported {len(self.main_window.centroid_points)} centroids to:\n{file_path}"
            )
            
        except ImportError as e:
            self.logger.error(f"Shapefile export requires fiona and shapely: {e}")
            QMessageBox.critical(
                self.main_window,
                "Missing Dependencies",
                "Shapefile export requires 'fiona' and 'shapely' libraries.\n"
                "Install with: pip install fiona shapely"
            )
        except Exception as e:
            self.logger.error(f"Failed to export centroids to shapefile: {e}", exc_info=True)
            QMessageBox.critical(
                self.main_window,
                "Export Error",
                f"Failed to export centroids:\n{e}"
            )
    
    def generate_canopy_circles(self):
        """Generate canopy circles from ONNX detection bounding boxes with real-world measurements.
        
        Converts each detection bbox to:
        1. Centroid point (center of bbox)
        2. Canopy circle (inscribed circle in bbox)
        3. GPS coordinates using raster transform
        4. Real-world measurements (radius, diameter, area in meters)
        """
        # Check if detections exist
        if not hasattr(self.main_window, 'onnx_detection_result') or not self.main_window.onnx_detection_result:
            QMessageBox.warning(
                self.main_window,
                "No Detections",
                "No detections found. Please run inference first."
            )
            return
        
        detections = self.main_window.onnx_detection_result.get('detections', [])
        if not detections:
            QMessageBox.warning(
                self.main_window,
                "No Detections",
                "Detection result is empty. Please run inference first."
            )
            return
        
        # Get raster transform for GPS conversion
        transform = None
        try:
            if self.main_window.raster_loader and self.main_window.raster_loader.dataset:
                transform = self.main_window.raster_loader.dataset.transform
        except Exception as e:
            self.logger.debug(f"Could not get raster transform: {e}")
        
        if not transform:
            QMessageBox.warning(
                self.main_window,
                "No Transform",
                "Raster does not have geospatial transform. Cannot calculate real-world measurements."
            )
            return
        
        # Clear existing centroids
        self.main_window.centroid_points = []
        
        # Convert each detection to canopy circle
        for det in detections:
            try:
                # Get bbox coordinates
                if 'box' in det:
                    # Format: [x1, y1, x2, y2]
                    box = det['box']
                    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                elif 'x' in det and 'y' in det and 'w' in det and 'h' in det:
                    # Format: {x, y, w, h} (center + size)
                    cx, cy, w, h = det['x'], det['y'], det['w'], det['h']
                    x1, y1 = cx - w/2, cy - h/2
                    x2, y2 = cx + w/2, cy + h/2
                else:
                    self.logger.warning(f"Unknown detection format: {det.keys()}")
                    continue
                
                # Calculate centroid (center of bbox)
                centroid_x = (x1 + x2) / 2
                centroid_y = (y1 + y2) / 2
                
                # Calculate inscribed circle radius in pixels
                # Use smaller dimension to ensure circle fits inside bbox
                bbox_width = abs(x2 - x1)
                bbox_height = abs(y2 - y1)
                radius_px = min(bbox_width, bbox_height) / 2
                
                # Convert pixel radius to meters using transform
                # Get pixel size in meters (assuming square pixels)
                pixel_width_m = abs(transform.a)  # meters per pixel in X direction
                pixel_height_m = abs(transform.e)  # meters per pixel in Y direction
                pixel_size_m = (pixel_width_m + pixel_height_m) / 2  # average
                
                radius_m = radius_px * pixel_size_m
                
                # Create centroid point with canopy measurements
                centroid_point = {
                    'x': centroid_x,
                    'y': centroid_y,
                    'radius_px': radius_px,
                    'radius_m': radius_m,
                    'diameter_m': radius_m * 2,
                    'area_m2': np.pi * (radius_m ** 2),
                    'source': 'onnx_detection'
                }
                
                self.main_window.centroid_points.append(centroid_point)
                
            except Exception as e:
                self.logger.error(f"Failed to convert detection to canopy: {e}")
                continue
        
        # Update UI
        if hasattr(self.main_window, '_update_centroid_ui'):
            self.main_window._update_centroid_ui()
        if hasattr(self.main_window, '_render_centroids'):
            self.main_window._render_centroids()
        
        # Show summary
        count = len(self.main_window.centroid_points)
        avg_radius = np.mean([pt['radius_m'] for pt in self.main_window.centroid_points]) if count > 0 else 0
        
        self.logger.info(f"Generated {count} canopy circles | Avg radius: {avg_radius:.2f}m")
        QMessageBox.information(
            self.main_window,
            "Canopy Generated",
            f"Generated {count} canopy circles from detections.\n"
            f"Average canopy radius: {avg_radius:.2f} meters"
        )
    
    def save_canopy_to_shapefile(self):
        """Export canopy circles as polygon shapefile with measurements.
        
        Each canopy circle is exported as a polygon feature with attributes:
        - id: Canopy ID
        - center_lat, center_lon: GPS coordinates of center
        - radius_m: Canopy radius in meters
        - diameter_m: Canopy diameter in meters
        - area_m2: Canopy area in square meters
        """
        # Filter centroids that have canopy data
        canopy_centroids = [pt for pt in self.main_window.centroid_points if 'radius_m' in pt and pt['radius_m'] > 0]
        
        if not canopy_centroids:
            QMessageBox.warning(
                self.main_window,
                "No Canopy Data",
                "No canopy circles to save. Please generate canopy circles first."
            )
            return
        
        # Get save file path
        file_path, _ = QFileDialog.getSaveFileName(
            self.main_window,
            "Save Canopy Circles as Shapefile",
            "",
            "ESRI Shapefile (*.shp)"
        )
        
        if not file_path:
            return
        
        try:
            import fiona
            from fiona.crs import from_epsg
            from shapely.geometry import Point, mapping
            from shapely import affinity
            
            # Get CRS from raster
            crs_code = None
            try:
                if self.main_window.raster_loader and self.main_window.raster_loader.dataset:
                    raster_crs = self.main_window.raster_loader.dataset.crs
                    if raster_crs and raster_crs.to_epsg():
                        crs_code = raster_crs.to_epsg()
            except Exception as e:
                self.logger.debug(f"Could not extract CRS from raster: {e}")
            
            if not crs_code:
                crs_code = 4326
                self.logger.warning(f"No CRS found, defaulting to EPSG:{crs_code}")
            
            crs = from_epsg(crs_code)
            
            # Define schema for polygon features
            schema = {
                'geometry': 'Polygon',
                'properties': {
                    'id': 'int',
                    'center_lat': 'float',
                    'center_lon': 'float',
                    'radius_m': 'float',
                    'diameter_m': 'float',
                    'area_m2': 'float'
                }
            }
            
            # Get transform
            transform = None
            try:
                if self.main_window.raster_loader and self.main_window.raster_loader.dataset:
                    transform = self.main_window.raster_loader.dataset.transform
            except Exception as e:
                self.logger.debug(f"Could not get raster transform for canopy export: {e}")
            
            if not transform:
                QMessageBox.critical(
                    self.main_window,
                    "No Transform",
                    "Cannot export without geospatial transform."
                )
                return
            
            # Write shapefile
            with fiona.open(file_path, 'w', driver='ESRI Shapefile', crs=crs, schema=schema) as output:
                for idx, pt in enumerate(canopy_centroids, start=1):
                    # Get centroid GPS coordinates
                    pixel_x = pt['x']
                    pixel_y = pt['y']
                    lon, lat = transform * (pixel_x, pixel_y)
                    
                    # Create circle polygon in GPS coordinates
                    # Convert pixel radius to degree radius (approximate)
                    radius_m = pt['radius_m']
                    
                    # Create point and buffer in meters (requires projected CRS)
                    center_point = Point(lon, lat)
                    
                    # Buffer by radius in meters
                    # Note: This works best with projected CRS (meters), not geographic (degrees)
                    circle_polygon = center_point.buffer(radius_m)
                    
                    diameter_m = pt.get('diameter_m', radius_m * 2)
                    area_m2 = pt.get('area_m2', np.pi * (radius_m ** 2))
                    
                    # Create feature
                    feature = {
                        'geometry': mapping(circle_polygon),
                        'properties': {
                            'id': idx,
                            'center_lat': lat,
                            'center_lon': lon,
                            'radius_m': round(radius_m, 2),
                            'diameter_m': round(diameter_m, 2),
                            'area_m2': round(area_m2, 2)
                        }
                    }
                    
                    output.write(feature)
            
            self.logger.info(f"Canopy exported to shapefile: {file_path} ({len(canopy_centroids)} circles)")
            QMessageBox.information(
                self.main_window,
                "Export Complete",
                f"Successfully exported {len(canopy_centroids)} canopy circles to:\n{file_path}"
            )
            
        except ImportError:
            QMessageBox.critical(
                self.main_window,
                "Missing Dependencies",
                "Shapefile export requires 'fiona' and 'shapely' libraries."
            )
        except Exception as e:
            self.logger.error(f"Failed to export canopy to shapefile: {e}", exc_info=True)
            QMessageBox.critical(
                self.main_window,
                "Export Error",
                f"Failed to export canopy:\n{e}"
            )
    
    def convert_to_centroids(self):
        """Convert ONNX detection bounding boxes to centroid points with GPS coordinates and canopy measurements.
        
        For each detection:
        1. Calculate centroid (center of bbox)
        2. Convert pixel coordinates to GPS using raster transform
        3. Calculate canopy radius from bbox size
        4. Calculate real-world measurements (radius, diameter, area in meters)
        5. Store as centroid point for rendering and export
        """
        # Check if detections exist
        if not hasattr(self.main_window, 'onnx_detection_result') or not self.main_window.onnx_detection_result:
            QMessageBox.warning(
                self.main_window,
                "No Detections",
                "No detections found. Please run ONNX inference first to generate detections."
            )
            return
        
        detections = self.main_window.onnx_detection_result.get('detections', [])
        if not detections:
            QMessageBox.warning(
                self.main_window,
                "Empty Detections",
                "Detection result exists but contains no detections."
            )
            return
        
        # Get raster transform for GPS conversion
        transform = None
        try:
            if self.main_window.raster_loader and self.main_window.raster_loader.dataset:
                transform = self.main_window.raster_loader.dataset.transform
                self.logger.info(f"Using raster transform: {transform}")
        except Exception as e:
            self.logger.error(f"Could not get raster transform: {e}")
        
        if not transform:
            QMessageBox.warning(
                self.main_window,
                "No Geospatial Transform",
                "Raster file does not have geospatial transform information.\n"
                "Centroid conversion will use pixel coordinates only (no GPS coordinates)."
            )
        
        # Clear existing centroids safely
        self.main_window.centroid_points = []
        
        # Convert each detection to centroid
        for det in detections:
            try:
                # Parse detection bbox
                if 'box' in det:
                    # Format: [x1, y1, x2, y2]
                    box = det['box']
                    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                elif 'x' in det and 'y' in det and 'w' in det and 'h' in det:
                    # Format: {x, y, w, h}
                    cx, cy, w, h = det['x'], det['y'], det['w'], det['h']
                    x1, y1 = cx - w/2, cy - h/2
                    x2, y2 = cx + w/2, cy + h/2
                else:
                    self.logger.warning(f"Unknown detection bbox format: {det.keys()}")
                    continue
                
                # Calculate centroid (center of bbox)
                centroid_x = (x1 + x2) / 2
                centroid_y = (y1 + y2) / 2
                
                # Calculate canopy radius from bbox
                bbox_width = abs(x2 - x1)
                bbox_height = abs(y2 - y1)
                radius_px = min(bbox_width, bbox_height) / 2
                
                # Convert to GPS and meters if transform available
                gps_coords = None
                radius_m = None
                
                if transform:
                    try:
                        # Convert pixel coordinates to GPS
                        lon, lat = transform * (centroid_x, centroid_y)
                        gps_coords = {'lon': lon, 'lat': lat}
                        
                        # Convert pixel radius to meters
                        pixel_width_m = abs(transform.a)
                        pixel_height_m = abs(transform.e)
                        pixel_size_m = (pixel_width_m + pixel_height_m) / 2
                        radius_m = radius_px * pixel_size_m
                        
                    except Exception as e:
                        self.logger.error(f"GPS conversion failed: {e}")
                
                # Create centroid point
                centroid_point = {
                    'x': centroid_x,
                    'y': centroid_y,
                    'radius_px': radius_px,
                    'source': 'onnx_detection'
                }
                
                # Add GPS coordinates if available
                if gps_coords:
                    centroid_point['lon'] = gps_coords['lon']
                    centroid_point['lat'] = gps_coords['lat']
                
                # Add real-world measurements if available
                if radius_m is not None:
                    centroid_point['radius_m'] = radius_m
                    centroid_point['diameter_m'] = radius_m * 2
                    centroid_point['area_m2'] = np.pi * (radius_m ** 2)
                
                self.main_window.centroid_points.append(centroid_point)
                
            except Exception as e:
                self.logger.error(f"Failed to convert detection to centroid: {e}")
                continue
        
        # PERSISTENCE: Save to active layer
        try:
            # Modern Refactored Architecture (LayerHandler/Mixin)
            active_layer = None
            if hasattr(self.main_window, 'active_layer_id') and hasattr(self.main_window, 'raster_layers'):
                layer_id = self.main_window.active_layer_id
                if layer_id is not None:
                     for layer in self.main_window.raster_layers:
                        if layer['id'] == layer_id:
                            active_layer = layer
                            break
            
            # Legacy Fallback
            if not active_layer and hasattr(self.main_window, 'layers') and hasattr(self.main_window, '_active_layer_index'):
                if 0 <= self.main_window._active_layer_index < len(self.main_window.layers):
                    active_layer = self.main_window.layers[self.main_window._active_layer_index]
            
            if active_layer:
                active_layer['centroids'] = self.main_window.centroid_points
                self.logger.info(f"Centroids saved to active layer: {active_layer['name']}")
        except Exception as e:
             self.logger.error(f"Failed to save centroids to active layer: {e}")

        # Update UI
        if hasattr(self.main_window, '_update_centroid_ui'):
            self.main_window._update_centroid_ui()
        if hasattr(self.main_window, '_render_centroids'):
            self.main_window._render_centroids()
        
        # Show summary
        count = len(self.main_window.centroid_points)
        has_gps = any('lat' in pt for pt in self.main_window.centroid_points)
        has_measurements = any('radius_m' in pt for pt in self.main_window.centroid_points)
        
        summary = f"Converted {count} detections to centroids."
        if has_gps:
            summary += "\n[+] GPS coordinates calculated"
        if has_measurements:
            avg_radius = np.mean([pt['radius_m'] for pt in self.main_window.centroid_points if 'radius_m' in pt])
            summary += f"\n[+] Canopy measurements: Avg radius {avg_radius:.2f}m"
        
        self.logger.info(f"Converted {count} detections to centroids | GPS: {has_gps} | Measurements: {has_measurements}")
        QMessageBox.information(self.main_window, "Conversion Complete", summary)
    
    def add_centroid_at_click(self, pixel_x, pixel_y):
        """Add a new centroid point at the clicked pixel coordinates.
        
        Args:
            pixel_x: X coordinate in pixel space
            pixel_y: Y coordinate in pixel space
        """
        # Create basic centroid point
        centroid_point = {
            'x': pixel_x,
            'y': pixel_y,
            'source': 'manual'
        }
        
        # Try to add GPS coordinates if transform available
        try:
            if self.main_window.raster_loader and self.main_window.raster_loader.dataset:
                transform = self.main_window.raster_loader.dataset.transform
                if transform:
                    lon, lat = transform * (pixel_x, pixel_y)
                    centroid_point['lon'] = lon
                    centroid_point['lat'] = lat
        except Exception as e:
            self.logger.debug(f"Could not add GPS coords to manual centroid: {e}")
        
        # Add to list
        self.main_window.centroid_points.append(centroid_point)
        
        # Update UI
        if hasattr(self.main_window, '_update_centroid_ui'):
            self.main_window._update_centroid_ui()
        if hasattr(self.main_window, '_render_centroids'):
            self.main_window._render_centroids()
        
        self.logger.info(f"Added manual centroid at ({pixel_x:.1f}, {pixel_y:.1f})")
    
    def delete_centroid_at_click(self, pixel_x, pixel_y, tolerance=10):
        """Delete centroid point near the clicked coordinates.
        
        Args:
            pixel_x: X coordinate in pixel space
            pixel_y: Y coordinate in pixel space
            tolerance: Search radius in pixels (default 10)
        
        Returns:
            bool: True if centroid was deleted, False otherwise
        """
        # Find closest centroid within tolerance
        closest_idx = None
        min_distance = float('inf')
        
        for idx, pt in enumerate(self.main_window.centroid_points):
            dx = pt['x'] - pixel_x
            dy = pt['y'] - pixel_y
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance < min_distance and distance <= tolerance:
                min_distance = distance
                closest_idx = idx
        
        # Delete if found
        if closest_idx is not None:
            deleted_pt = self.main_window.centroid_points.pop(closest_idx)
            self.logger.info(f"Deleted centroid at ({deleted_pt['x']:.1f}, {deleted_pt['y']:.1f})")
            
            # Update UI
            if hasattr(self.main_window, '_update_centroid_ui'):
                self.main_window._update_centroid_ui()
            if hasattr(self.main_window, '_render_centroids'):
                self.main_window._render_centroids()
            
            return True
        else:
            self.logger.debug(f"No centroid found near ({pixel_x:.1f}, {pixel_y:.1f}) within tolerance {tolerance}px")
            return False

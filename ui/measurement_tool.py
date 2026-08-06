"""Tool pengukuran jarak untuk raster viewer."""

from PyQt6.QtWidgets import QGraphicsLineItem, QGraphicsEllipseItem, QGraphicsTextItem
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QPen, QBrush, QColor, QFont
import math
import logging
from pyproj import CRS, Transformer, Geod


class MeasurementLine:
    """Representasi satu garis pengukuran dengan titik awal dan akhir."""

    def __init__(self, start_point, scene, measurement_id):
        self.measurement_id = measurement_id
        self.start_point = start_point
        self.end_point = start_point
        self.scene = scene

        self.line_item = None
        self.start_marker = None
        self.end_marker = None
        self.label_item = None
        self.preview_line = None
        self._finished = False

        self.line_color = QColor(255, 255, 0)
        self.marker_color = QColor(255, 255, 0)
        self.line_width = 2
        self.marker_radius = 5

        self._create_graphics()

    def _create_graphics(self):
        """Buat item grafis untuk garis pengukuran."""
        pen = QPen(self.line_color, self.line_width)
        pen.setCosmetic(True)
        self.line_item = QGraphicsLineItem()
        self.line_item.setPen(pen)
        self.line_item.setZValue(1000)
        self.line_item.setVisible(False)
        self.scene.addItem(self.line_item)

        preview_pen = QPen(self.line_color, self.line_width, Qt.PenStyle.DashLine)
        preview_pen.setCosmetic(True)
        self.preview_line = QGraphicsLineItem()
        self.preview_line.setPen(preview_pen)
        self.preview_line.setZValue(999)
        self.scene.addItem(self.preview_line)

        marker_pen = QPen(self.marker_color, 1)
        marker_pen.setCosmetic(True)
        marker_brush = QBrush(self.marker_color)

        self.start_marker = QGraphicsEllipseItem(
            -self.marker_radius, -self.marker_radius,
            self.marker_radius * 2, self.marker_radius * 2
        )
        self.start_marker.setPen(marker_pen)
        self.start_marker.setBrush(marker_brush)
        self.start_marker.setZValue(1001)
        self.start_marker.setPos(self.start_point)
        # Make marker size constant regardless of zoom
        self.start_marker.setFlag(self.start_marker.GraphicsItemFlag.ItemIgnoresTransformations)
        self.scene.addItem(self.start_marker)

        # End marker
        self.end_marker = QGraphicsEllipseItem(
            -self.marker_radius, -self.marker_radius,
            self.marker_radius * 2, self.marker_radius * 2
        )
        self.end_marker.setPen(marker_pen)
        self.end_marker.setBrush(marker_brush)
        self.end_marker.setZValue(1001)
        self.end_marker.setPos(self.start_point)
        self.end_marker.setFlag(self.end_marker.GraphicsItemFlag.ItemIgnoresTransformations)
        self.scene.addItem(self.end_marker)

        # Label
        self.label_item = QGraphicsTextItem()
        font = QFont("Arial", 12, QFont.Weight.Bold)
        self.label_item.setFont(font)
        self.label_item.setDefaultTextColor(QColor(255, 255, 0))
        self.label_item.setZValue(1002)
        # Make label size constant regardless of zoom
        self.label_item.setFlag(self.label_item.GraphicsItemFlag.ItemIgnoresTransformations)
        self.scene.addItem(self.label_item)

    def update_end_point(self, end_point):
        """Update the end point of the measurement"""
        self.end_point = end_point
        self._update_graphics()

    def _update_graphics(self):
        """Update graphics items to reflect current measurement"""
        line = (
            self.start_point.x(), self.start_point.y(),
            self.end_point.x(), self.end_point.y()
        )
        if self.line_item:
            self.line_item.setLine(*line)
        if self.preview_line:
            self.preview_line.setLine(*line)

        # Update end marker
        self.end_marker.setPos(self.end_point)

        # Update label position (midpoint of line)
        mid_x = (self.start_point.x() + self.end_point.x()) / 2
        mid_y = (self.start_point.y() + self.end_point.y()) / 2
        self.label_item.setPos(mid_x, mid_y - 20)  # Offset above line

    def set_label_text(self, text):
        """Set the label text"""
        self.label_item.setPlainText(text)
        self._update_graphics()

    def finalize(self):
        """Convert the temporary preview into the permanent measurement line."""
        self._finished = True
        if self.preview_line:
            self.scene.removeItem(self.preview_line)
            self.preview_line = None
        if self.line_item:
            self.line_item.setVisible(True)
        self._update_graphics()

    def get_distance_pixels(self):
        """Get distance in pixels"""
        dx = self.end_point.x() - self.start_point.x()
        dy = self.end_point.y() - self.start_point.y()
        return math.sqrt(dx * dx + dy * dy)

    def remove(self):
        """Remove all graphics items from scene"""
        for attr in ('line_item', 'start_marker', 'end_marker', 'label_item', 'preview_line'):
            item = getattr(self, attr, None)
            if item is not None:
                if item.scene() is not None:
                    item.scene().removeItem(item)
                setattr(self, attr, None)
        self.scene.update()
        for view in self.scene.views():
            view.viewport().update()


class MeasurementManager:
    """Manages measurement tools and unit conversions"""

    UNITS = {
        'pixels': {'name': 'Pixels', 'symbol': 'px'},
        'meters': {'name': 'Meters', 'symbol': 'm'},
        'centimeters': {'name': 'Centimeters', 'symbol': 'cm'},
        'millimeters': {'name': 'Millimeters', 'symbol': 'mm'},
        'kilometers': {'name': 'Kilometers', 'symbol': 'km'},
        'feet': {'name': 'Feet', 'symbol': 'ft'},
        'yards': {'name': 'Yards', 'symbol': 'yd'},
        'miles': {'name': 'Miles', 'symbol': 'mi'},
        'inches': {'name': 'Inches', 'symbol': 'in'},
        'degrees': {'name': 'Degrees', 'symbol': '°'}
    }

    MEASUREMENT_MODES = {
        'cartesian': {'name': 'Cartesian', 'description': 'Euclidean distance (straight line)'},
        'ellipsoidal': {'name': 'Ellipsoidal', 'description': 'Geodesic distance (on Earth surface)'}
    }

    def __init__(self, scene, raster_metadata=None):
        self.scene = scene
        self.metadata = raster_metadata
        self.logger = logging.getLogger(__name__)
        self.current_unit = 'meters'
        self.measurement_mode = 'cartesian'  # 'cartesian' or 'ellipsoidal'
        self.measurements = []  # List of MeasurementLine objects
        # Stable ID -> all graphics owned by that measurement.  Keeping the
        # item references together prevents deletion by list index from
        # leaving orphaned scene items behind.
        self.measurement_items = {}
        self._next_measurement_id = 1
        self.current_measurement = None

        # Pixel to meter conversion (from georeferencing or manual calibration)
        self.pixel_size_x = None  # meters per pixel (None = not set)
        self.pixel_size_y = None  # meters per pixel (None = not set)
        self.is_georeferenced = False  # True if file has valid georeferencing
        self.is_calibrated = False  # True if user manually calibrated

        # CRS and transform for georeferencing
        self.crs = None
        self.transform = None  # Original transform from file
        self.transform_original = None  # Keep original for coordinate conversion
        self.geod = None  # For ellipsoidal calculations
        self.transformer_to_wgs84 = None  # Transformer to convert to WGS84 (lat/lon)

        self._update_pixel_size_from_metadata()

    def _update_pixel_size_from_metadata(self):
        """Update pixel size and CRS from raster metadata (georeferencing)"""
        self.is_georeferenced = False
        self.crs = None
        self.transform = None
        self.geod = None

        if self.metadata and 'transform' in self.metadata and self.metadata['transform'] is not None:
            # Convert rasterio Affine object to tuple/list for easier access
            transform_obj = self.metadata['transform']

            # Rasterio Affine can be accessed as tuple: (a, b, c, d, e, f)
            # Or using attributes: a, b, c, d, e, f
            try:
                # Try to convert to tuple
                if hasattr(transform_obj, 'to_gdal'):
                    # Rasterio Affine object - convert to tuple
                    gdal_transform = transform_obj.to_gdal()  # Returns (c, a, b, f, d, e)

                    # Store in standard format [a, b, c, d, e, f]
                    self.transform = [
                        gdal_transform[1],  # a
                        gdal_transform[2],  # b
                        gdal_transform[0],  # c
                        gdal_transform[4],  # d
                        gdal_transform[5],  # e
                        gdal_transform[3]   # f
                    ]
                else:
                    # Already a tuple or list
                    self.transform = list(transform_obj)

                # Keep original transform for coordinate conversion (unchanged)
                self.transform_original = list(self.transform)

                # Extract pixel sizes from transform
                pixel_x = abs(self.transform[0])  # a - pixel width
                pixel_y = abs(self.transform[4])  # e - pixel height

                # Check if CRS uses geographic coordinates (degrees)
                is_geographic = False
                if 'crs' in self.metadata and self.metadata['crs']:
                    try:
                        temp_crs = CRS.from_wkt(self.metadata['crs'].to_wkt())
                        is_geographic = temp_crs.is_geographic
                    except:
                        pass

                # For geographic CRS (lat/lon in degrees), convert to meters
                # 1 degree ≈ 111,320 meters at equator
                if is_geographic and pixel_x < 0.01:  # Likely in degrees
                    pixel_x = pixel_x * 111320.0
                    pixel_y = pixel_y * 111320.0

                # Check if georeferencing values are valid (not default/zero)
                # Valid georeferencing should be > 0.00001 and < 10000 meters (reasonable range)
                if 0.00001 < pixel_x < 10000 and 0.00001 < pixel_y < 10000:
                    self.pixel_size_x = pixel_x
                    self.pixel_size_y = pixel_y

                    # Extract CRS information
                    if 'crs' in self.metadata and self.metadata['crs']:
                        try:
                            self.crs = CRS.from_wkt(self.metadata['crs'].to_wkt())

                            # Setup Geod for ellipsoidal calculations
                            if self.crs.is_geographic:
                                self.geod = Geod(ellps='WGS84')
                            elif self.crs.geodetic_crs:
                                geog_crs = self.crs.geodetic_crs
                                self.geod = Geod(ellps=geog_crs.ellipsoid.name)

                            # Setup transformer to WGS84 (lat/lon) for coordinate display
                            try:
                                self.transformer_to_wgs84 = Transformer.from_crs(
                                    self.crs,
                                    CRS.from_epsg(4326),  # WGS84
                                    always_xy=True
                                )
                            except:
                                self.transformer_to_wgs84 = None
                        except Exception as e:
                            pass  # CRS parsing failed, continue without it

                    self.is_georeferenced = True
                    return True

            except Exception as e:
                pass  # Transform processing failed

        # No valid georeferencing found
        self.pixel_size_x = None
        self.pixel_size_y = None
        return False

    def set_metadata(self, metadata):
        """Update raster metadata and pixel size"""
        self.metadata = metadata
        self._update_pixel_size_from_metadata()

    def pixel_to_latlon(self, pixel_x, pixel_y):
        """Convert pixel coordinates to lat/lon (WGS84) for display

        Args:
            pixel_x: X pixel coordinate
            pixel_y: Y pixel coordinate

        Returns:
            tuple: (lon, lat) in degrees, or (0.0, 0.0) if no georeferencing
        """
        if not self.transform:
            return 0.0, 0.0

        # Convert pixel to projected coordinates
        geo_x = self.transform[2] + pixel_x * self.transform[0] + pixel_y * self.transform[1]
        geo_y = self.transform[5] + pixel_x * self.transform[3] + pixel_y * self.transform[4]

        # If already in geographic coordinates (lat/lon), return directly
        if self.crs and self.crs.is_geographic:
            return geo_x, geo_y

        # Otherwise, transform to WGS84 (lat/lon)
        if self.transformer_to_wgs84:
            try:
                lon, lat = self.transformer_to_wgs84.transform(geo_x, geo_y)
                return lon, lat
            except:
                pass

        # Fallback: return projected coordinates
        return geo_x, geo_y

    def set_manual_pixel_size(self, gsd_cm_per_pixel):
        """Set pixel size manually (calibration)

        Args:
            gsd_cm_per_pixel: Ground Sample Distance in centimeters per pixel
        """
        if gsd_cm_per_pixel > 0:
            # Convert cm to meters
            meters_per_pixel = gsd_cm_per_pixel / 100.0
            self.pixel_size_x = meters_per_pixel
            self.pixel_size_y = meters_per_pixel
            self.is_calibrated = True
            self.is_georeferenced = False  # Manual calibration overrides georeferencing

            # Update all existing measurement labels
            for measurement in self.measurements:
                self._update_measurement_label(measurement)
            return True
        return False

    def has_valid_calibration(self):
        """Check if measurement tool has valid calibration (georeferencing or manual)"""
        return (self.is_georeferenced or self.is_calibrated) and self.pixel_size_x is not None

    def get_calibration_status(self):
        """Get current calibration status"""
        if self.is_georeferenced:
            gsd_cm = self.pixel_size_x * 100 if self.pixel_size_x else 0
            return {
                'status': 'georeferenced',
                'message': f'Georeferenced ({gsd_cm:.2f} cm/px)',
                'gsd_cm': gsd_cm
            }
        elif self.is_calibrated:
            gsd_cm = self.pixel_size_x * 100 if self.pixel_size_x else 0
            return {
                'status': 'calibrated',
                'message': f'Manually calibrated ({gsd_cm:.2f} cm/px)',
                'gsd_cm': gsd_cm
            }
        else:
            return {
                'status': 'not_calibrated',
                'message': 'Not calibrated - physical measurements unavailable',
                'gsd_cm': None
            }

    def set_unit(self, unit):
        """Set the current measurement unit"""
        if unit in self.UNITS:
            self.current_unit = unit
            # Update labels for all existing measurements
            for measurement in self.measurements:
                self._update_measurement_label(measurement)

    def set_measurement_mode(self, mode):
        """Set measurement mode: 'cartesian' or 'ellipsoidal'"""
        if mode in self.MEASUREMENT_MODES:
            self.measurement_mode = mode
            # Update labels for all existing measurements
            for measurement in self.measurements:
                self._update_measurement_label(measurement)

    def pixel_to_world(self, pixel_x, pixel_y):
        """Convert pixel coordinates to world coordinates

        Args:
            pixel_x: Pixel X coordinate
            pixel_y: Pixel Y coordinate

        Returns:
            tuple: (world_x, world_y) in CRS units, or None if not georeferenced
        """
        if not self.is_georeferenced or not self.transform_original:
            return None

        # Use original transform (unchanged) for coordinate conversion
        # Apply affine transform: [a, b, c, d, e, f]
        # world_x = a * pixel_x + b * pixel_y + c
        # world_y = d * pixel_x + e * pixel_y + f
        t = self.transform_original
        world_x = t[0] * pixel_x + t[1] * pixel_y + t[2]
        world_y = t[3] * pixel_x + t[4] * pixel_y + t[5]

        return (world_x, world_y)

    def start_measurement(self, start_point):
        """Start a new measurement"""
        if self.current_measurement:
            self.cancel_measurement()
        measurement_id = self._next_measurement_id
        self._next_measurement_id += 1
        self.current_measurement = MeasurementLine(start_point, self.scene, measurement_id)
        return self.current_measurement

    def update_measurement(self, end_point):
        """Update current measurement end point"""
        if self.current_measurement:
            self.current_measurement.update_end_point(end_point)
            self._update_measurement_label(self.current_measurement)

    def finish_measurement(self):
        """Finish current measurement and add to list"""
        if self.current_measurement:
            measurement = self.current_measurement
            measurement.finalize()
            self.measurements.append(measurement)
            self.measurement_items[measurement.measurement_id] = {
                'measurement': measurement,
                'line': measurement.line_item,
                'start_marker': measurement.start_marker,
                'end_marker': measurement.end_marker,
                'text_label': measurement.label_item,
                'preview_line': measurement.preview_line,
            }
            result = self._get_measurement_info(measurement)
            self.current_measurement = None
            self.scene.update()
            return result
        return None

    def cancel_measurement(self):
        """Cancel current measurement"""
        if self.current_measurement:
            self.current_measurement.remove()
            self.current_measurement = None
            self.scene.update()

    def remove_measurement(self, measurement_id):
        """Remove one completed measurement by its stable ID."""
        try:
            measurement_id = int(measurement_id)
        except (TypeError, ValueError):
            return False

        items = self.measurement_items.pop(measurement_id, None)
        if items is None:
            return False
        measurement = items['measurement']
        if measurement in self.measurements:
            self.measurements.remove(measurement)
        measurement.remove()
        return True

    def removeMeasurement(self, measurement_id):
        """Qt/API-friendly alias for remove_measurement."""
        return self.remove_measurement(measurement_id)

    def clear_all_measurements(self):
        """Remove all measurements"""
        for measurement in list(self.measurements):
            measurement.remove()
        self.measurements.clear()
        self.measurement_items.clear()

        if self.current_measurement:
            self.current_measurement.remove()
            self.current_measurement = None
        self.scene.update()
        for view in self.scene.views():
            view.viewport().update()

    def delete_last_measurement(self):
        """Delete the last completed measurement"""
        if self.measurements:
            return self.remove_measurement(self.measurements[-1].measurement_id)
        return False

    def _update_measurement_label(self, measurement):
        """Update measurement label with current unit"""
        # Calculate distance in meters
        distance_meters = self._calculate_distance_meters(measurement)

        if distance_meters is None:
            label_text = "N/A"
        else:
            # Convert to current unit
            distance_converted = self._convert_distance(distance_meters)
            unit_symbol = self.UNITS[self.current_unit]['symbol']
            label_text = f"{distance_converted:.2f} {unit_symbol}"

        measurement.set_label_text(label_text)

    def _calculate_distance_meters(self, measurement):
        """Calculate distance in meters using current measurement mode

        Args:
            measurement: MeasurementLine object

        Returns:
            float: Distance in meters, or None if not calibrated
        """
        start_px = (measurement.start_point.x(), measurement.start_point.y())
        end_px = (measurement.end_point.x(), measurement.end_point.y())

        if self.is_georeferenced and self.measurement_mode == 'ellipsoidal' and self.geod:
            # Ellipsoidal distance (geodesic on Earth surface)
            start_world = self.pixel_to_world(*start_px)
            end_world = self.pixel_to_world(*end_px)

            if start_world and end_world:
                try:
                    # Calculate geodesic distance
                    # Note: pyproj Geod.inv expects (lon, lat) order
                    _, _, distance_meters = self.geod.inv(
                        start_world[0], start_world[1],
                        end_world[0], end_world[1]
                    )
                    return abs(distance_meters)
                except Exception as e:
                    self.logger.warning(f"Ellipsoidal calculation failed, using Cartesian fallback: {e}")
                    # Fall back to Cartesian
                    pass

        # Cartesian distance (default or fallback)
        if self.has_valid_calibration():
            distance_px = measurement.get_distance_pixels()
            pixel_size_avg = (self.pixel_size_x + self.pixel_size_y) / 2
            return distance_px * pixel_size_avg

        return None

    def _convert_distance(self, distance_meters):
        """Convert distance in meters to current unit

        Args:
            distance_meters: Distance in meters

        Returns:
            Distance in current unit, or None if invalid
        """
        if distance_meters is None:
            return None

        if self.current_unit == 'pixels':
            # Convert meters back to pixels
            if not self.has_valid_calibration():
                return None
            pixel_size_avg = (self.pixel_size_x + self.pixel_size_y) / 2
            return distance_meters / pixel_size_avg

        if self.current_unit == 'degrees':
            # Convert meters to degrees (approximate)
            # 1 degree ≈ 111,320 meters at equator
            return distance_meters / 111320.0

        # Convert meters to target unit
        conversions = {
            'meters': 1.0,
            'centimeters': 100.0,
            'millimeters': 1000.0,
            'kilometers': 0.001,
            'feet': 3.28084,
            'yards': 1.09361,
            'miles': 0.000621371,
            'inches': 39.3701
        }

        multiplier = conversions.get(self.current_unit, 1.0)
        return distance_meters * multiplier

    def _get_measurement_info(self, measurement):
        """Get detailed info about a measurement"""
        # Pixel coordinates
        start_px = (measurement.start_point.x(), measurement.start_point.y())
        end_px = (measurement.end_point.x(), measurement.end_point.y())
        distance_px = measurement.get_distance_pixels()

        # World coordinates (if georeferenced)
        start_world = self.pixel_to_world(*start_px) if self.is_georeferenced else None
        end_world = self.pixel_to_world(*end_px) if self.is_georeferenced else None

        # Basic info
        info = {
            'measurement_id': measurement.measurement_id,
            'start_point_px': start_px,
            'end_point_px': end_px,
            'start_point_world': start_world,
            'end_point_world': end_world,
            'distance_pixels': distance_px,
            'calibrated': self.has_valid_calibration(),
            'georeferenced': self.is_georeferenced,
            'measurement_mode': self.measurement_mode,
            'crs': str(self.crs) if self.crs else None
        }

        # Calculate distance in meters
        distance_meters = self._calculate_distance_meters(measurement)

        if distance_meters is not None:
            # Calculate all unit conversions
            info.update({
                'meters': distance_meters,
                'centimeters': distance_meters * 100,
                'millimeters': distance_meters * 1000,
                'kilometers': distance_meters * 0.001,
                'feet': distance_meters * 3.28084,
                'yards': distance_meters * 1.09361,
                'miles': distance_meters * 0.000621371,
                'inches': distance_meters * 39.3701,
                'degrees': distance_meters / 111320.0  # Approximate conversion
            })
        else:
            # Not calibrated - set physical measurements to None
            info.update({
                'meters': None,
                'centimeters': None,
                'millimeters': None,
                'kilometers': None,
                'feet': None,
                'yards': None,
                'miles': None,
                'inches': None,
                'degrees': None
            })

        return info

    def get_all_measurements_info(self):
        """Get info for all measurements"""
        return [self._get_measurement_info(m) for m in self.measurements]

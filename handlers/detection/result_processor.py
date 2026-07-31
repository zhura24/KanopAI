"""Process and filter inference results."""

import logging
import numpy as np


class ResultProcessor:
    """Process inference results including polygon filtering."""
    
    def __init__(self, main_window):
        self.main_window = main_window
        self.logger = logging.getLogger(__name__)
    
    def process_inference_results(self, result, inference_mode=None, polygons=None):
        """Process inference results with optional polygon filtering.
        
        Args:
            result: Detection results from worker
            inference_mode: 'polygon', 'visible', or 'entire'
            polygons: List of polygon coordinates for filtering
            
        Returns:
            dict: Processed results with filtered detections
        """
        try:
            detections = result.get('detections', []) if isinstance(result, dict) else []
        except Exception as e:
            self.logger.warning(f"Failed to extract detections from result: {e}")
            detections = []
        
        raw_count = len(detections)
        self.logger.info(f"Processing {raw_count} raw detections")
        
        if inference_mode == 'polygon' and polygons:
            detections = self._filter_by_polygons(detections, polygons)
            self.logger.info(f"Polygon filter: {raw_count} -> {len(detections)} detections")
        
        return {
            'detections': detections,
            'raw_count': raw_count,
            'filtered_count': len(detections)
        }
    
    def _filter_by_polygons(self, detections, polygons):
        """Filter detections to only those inside polygons."""
        if not detections or not polygons:
            return detections
        
        original_count = len(detections)
        filtered = []
        
        self._log_polygon_bounds(polygons)
        self._log_detection_samples(detections)
        
        use_shapely, shapely_polys = self._prepare_shapely_polygons(polygons)
        
        for det in detections:
            det_x, det_y = self._get_detection_center(det)
            if det_x is None:
                continue
            
            if self._is_inside_any_polygon(det_x, det_y, polygons, shapely_polys, use_shapely):
                filtered.append(det)
        
        removed = original_count - len(filtered)
        self.logger.info(f"Polygon filter: kept {len(filtered)}, removed {removed} (outside)")
        
        return filtered
    
    def _log_polygon_bounds(self, polygons):
        """Log polygon boundaries for debugging."""
        for i, poly in enumerate(polygons):
            poly_array = np.array(poly) if not isinstance(poly, np.ndarray) else poly
            self.logger.info(f"Polygon {i} bounds: "
                           f"X=[{poly_array[:, 0].min():.1f}, {poly_array[:, 0].max():.1f}], "
                           f"Y=[{poly_array[:, 1].min():.1f}, {poly_array[:, 1].max():.1f}]")
    
    def _log_detection_samples(self, detections):
        """Log sample detection positions."""
        if not detections:
            return
        
        sample = detections[:5]
        for i, det in enumerate(sample):
            x, y = self._get_detection_center(det)
            if x is not None:
                self.logger.info(f"Sample detection {i}: ({x:.1f}, {y:.1f})")
    
    def _prepare_shapely_polygons(self, polygons):
        """Create shapely polygon objects for accurate filtering."""
        try:
            from shapely.geometry import Polygon as ShapelyPolygon
            shapely_polys = []
            for poly in polygons:
                try:
                    shapely_polys.append(ShapelyPolygon(poly))
                except Exception as e:
                    self.logger.warning(f"Failed to create shapely polygon: {e}")
            return True, shapely_polys
        except ImportError:
            self.logger.warning("Shapely not available, using bounding box filtering")
            return False, []
    
    def _get_detection_center(self, det):
        """Extract center coordinates from detection."""
        if 'box' in det:
            box = det['box']
            return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        elif 'x' in det and 'y' in det:
            return det['x'], det['y']
        else:
            self.logger.warning(f"Unknown detection format: {det.keys()}")
            return None, None
    
    def _is_inside_any_polygon(self, x, y, polygons, shapely_polys, use_shapely):
        """Check if point is inside any polygon."""
        if use_shapely and shapely_polys:
            from shapely.geometry import Point
            point = Point(x, y)
            for poly in shapely_polys:
                if poly.contains(point):
                    return True
        else:
            for poly in polygons:
                poly_array = np.array(poly) if not isinstance(poly, np.ndarray) else poly
                minx, miny = poly_array.min(axis=0)
                maxx, maxy = poly_array.max(axis=0)
                if minx <= x <= maxx and miny <= y <= maxy:
                    return True
        return False

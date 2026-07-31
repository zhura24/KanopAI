"""Handler untuk ekspor data training."""

import logging
import traceback
import json
import numpy as np
from PyQt6.QtWidgets import (QFileDialog, QMessageBox, QDialog, QVBoxLayout,
                              QLabel, QRadioButton, QCheckBox, QDialogButtonBox)
from pathlib import Path


class ExportHandler:
    """Handler untuk ekspor deteksi ke format YOLO."""
    
    def __init__(self, main_window):
        self.main_window = main_window
        self.logger = logging.getLogger(__name__)
    
    def browse_export_directory(self):
        self.logger.info("User initiated export directory selection")
        
        directory = QFileDialog.getExistingDirectory(
            self.main_window,
            "Select Export Directory",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        
        if directory:
            self.main_window.export_directory = directory
            dir_name = Path(directory).name
            parent_name = Path(directory).parent.name
            abbreviated = f".../{parent_name}/{dir_name}"
            self.main_window.label_export_dir.setText(abbreviated)
            self.main_window.label_export_dir.setStyleSheet(
                "QLabel { color: green; }"
            )
            self.logger.info(f"Export directory selected: {directory}")
            
            if hasattr(self.main_window, '_update_export_button_state'):
                self.main_window._update_export_button_state()
        else:
            self.logger.info("User cancelled export directory selection")
    
    def export_training_data(self):
        if not self.main_window.export_directory:
            self.main_window.show_error_detailed(
                "Please select an export directory first"
            )
            return
        
        if self.main_window.current_data is None:
            self.main_window.show_error_detailed("No raster data loaded")
            return
        
        self.logger.info("Starting training data export")
        
        self.main_window.btn_export_training_data.setEnabled(False)
        self.main_window.footer_progress_bar.setVisible(True)
        self.main_window.footer_progress_bar.setValue(0)
        
        try:
            import os
            
            export_tiles = self.main_window.check_export_tiles.isChecked()
            export_mask = self.main_window.check_export_mask.isChecked()
            export_grayscale = self.main_window.check_export_grayscale.isChecked()
            tile_size = self.main_window.spin_tile_size.value()
            resolution = self.main_window.spin_resolution.value()
            
            if self.main_window.radio_overlap_percent.isChecked():
                overlap_percent = self.main_window.spin_overlap_percent.value()
                overlap_pixels = int(tile_size * overlap_percent / 100)
            else:
                overlap_pixels = self.main_window.spin_overlap_dx.value()
            
            # current_data is now a RasterLoader instance (BigTIFF-safe).
            # Load pixel data on-demand here, right before we need it, rather
            # than at layer-switch time.  For very large rasters this reads a
            # downsampled version (max 8192px on longest edge); the export tiles
            # will be sliced from that array.  If you need true full-resolution
            # export tiles, switch to the per-tile rasterio windowed-read approach.
            if self.main_window.detection_data is not None:
                input_data = self.main_window.detection_data
            else:
                loader = self.main_window.current_data
                if hasattr(loader, 'read_for_export'):
                    self.logger.info("Loading raster data for export (BigTIFF-safe decimated read)...")
                    input_data = loader.read_for_export(max_dimension=8192)
                    if input_data is None:
                        self.main_window.show_error_detailed("Failed to read raster data for export.")
                        return
                else:
                    # Fallback: old path where current_data was already a numpy array
                    input_data = loader
            
            tiles_dir = Path(self.main_window.export_directory) / "tiles"
            masks_dir = Path(self.main_window.export_directory) / "masks"
            
            if export_tiles:
                tiles_dir.mkdir(exist_ok=True)
            if export_mask:
                masks_dir.mkdir(exist_ok=True)
            
            self.logger.info(
                f"Export parameters | "
                f"Tiles: {export_tiles} | "
                f"Masks: {export_mask} | "
                f"Grayscale: {export_grayscale} | "
                f"Tile size: {tile_size}px | "
                f"Overlap: {overlap_pixels}px | "
                f"Resolution: {resolution} cm/px"
            )
            
            stride = tile_size - overlap_pixels
            
            if len(input_data.shape) == 3:
                channels, height, width = input_data.shape
            else:
                height, width = input_data.shape
                channels = 1
            
            num_tiles_x = max(1, (width - overlap_pixels) // stride)
            num_tiles_y = max(1, (height - overlap_pixels) // stride)
            total_tiles = num_tiles_x * num_tiles_y
            
            self.logger.info(
                f"Exporting {total_tiles} tiles ({num_tiles_x}x{num_tiles_y})"
            )
            
            tile_count = 0
            
            for ty in range(num_tiles_y):
                for tx in range(num_tiles_x):
                    x_start = tx * stride
                    y_start = ty * stride
                    x_end = min(x_start + tile_size, width)
                    y_end = min(y_start + tile_size, height)
                    
                    if len(input_data.shape) == 3:
                        tile_data = input_data[:, y_start:y_end, x_start:x_end]
                    else:
                        tile_data = input_data[y_start:y_end, x_start:x_end]
                    
                    if (export_grayscale and export_tiles and 
                        len(tile_data.shape) == 3 and tile_data.shape[0] >= 3):
                        r = tile_data[0].astype(np.float32)
                        g = tile_data[1].astype(np.float32)
                        b = tile_data[2].astype(np.float32)
                        
                        grayscale = 0.299 * r + 0.587 * g + 0.114 * b
                        
                        if tile_data.dtype == np.uint8:
                            tile_data = grayscale.astype(np.uint8)
                        elif tile_data.dtype == np.uint16:
                            tile_data = grayscale.astype(np.uint16)
                        else:
                            tile_data = grayscale.astype(tile_data.dtype)
                    
                    if export_tiles:
                        tile_filename = f"tile_{ty:04d}_{tx:04d}.tif"
                        tile_path = tiles_dir / tile_filename
                        self._save_tile(tile_data, tile_path)
                    
                    if (export_mask and 
                        self.main_window.detection_result is not None):
                        mask_data = \
                            self.main_window.detection_result['binary_mask']
                        if mask_data is not None:
                            mask_tile = mask_data[y_start:y_end, x_start:x_end]
                            mask_filename = f"mask_{ty:04d}_{tx:04d}.tif"
                            mask_path = masks_dir / mask_filename
                            self._save_tile(mask_tile, mask_path)
                    
                    tile_count += 1
                    progress = int((tile_count / total_tiles) * 100)
                    self.main_window.footer_progress_bar.setValue(progress)
            
            metadata = {
                'tile_size_px': tile_size,
                'overlap_px': overlap_pixels,
                'resolution_cm_px': resolution,
                'stride_px': stride,
                'num_tiles_x': num_tiles_x,
                'num_tiles_y': num_tiles_y,
                'total_tiles': total_tiles,
                'image_width': width,
                'image_height': height,
                'image_channels': channels,
                'exported_tiles': export_tiles,
                'exported_masks': export_mask,
                'exported_as_grayscale': export_grayscale
            }
            
            metadata_path = Path(self.main_window.export_directory)
            metadata_path = metadata_path / "export_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(
                f"Training data export completed | {tile_count} tiles exported"
            )
            
            self.main_window.footer_progress_bar.setVisible(False)
            self.main_window.btn_export_training_data.setEnabled(True)
            
            format_info = "Grayscale" if export_grayscale else "RGB"
            QMessageBox.information(
                self.main_window,
                "Export Complete",
                f"Training data exported successfully!\\n\\n"
                f"Total tiles: {tile_count}\\n"
                f"Tile size: {tile_size}x{tile_size}px\\n"
                f"Overlap: {overlap_pixels}px\\n"
                f"Format: {format_info}\\n"
                f"Location: {self.main_window.export_directory}"
            )
        
        except Exception as e:
            error_msg = f"Export failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.main_window.footer_progress_bar.setVisible(False)
            self.main_window.btn_export_training_data.setEnabled(True)
            self.main_window.show_error_detailed(
                error_msg,
                details=traceback.format_exc()
            )
    
    def _save_tile(self, tile_data, file_path):
        try:
            import rasterio
            from rasterio.transform import from_bounds
            
            if len(tile_data.shape) == 2:
                tile_data = np.expand_dims(tile_data, axis=0)
                count = 1
            else:
                count = tile_data.shape[0]
            
            height, width = tile_data.shape[-2:]
            
            if tile_data.dtype == np.float32 or tile_data.dtype == np.float64:
                if tile_data.max() <= 1.0:
                    tile_data = (tile_data * 255).astype(np.uint8)
                else:
                    tile_data = tile_data.astype(np.uint8)
            elif tile_data.dtype != np.uint8 and tile_data.dtype != np.uint16:
                tile_data = tile_data.astype(np.uint8)
            
            with rasterio.open(
                file_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=count,
                dtype=tile_data.dtype,
                compress='lzw'
            ) as dst:
                if count == 1:
                    dst.write(tile_data[0], 1)
                else:
                    for i in range(count):
                        dst.write(tile_data[i], i + 1)
        
        except Exception as e:
            self.logger.warning(
                f"Rasterio save failed, using numpy fallback: {e}"
            )
            from PIL import Image
            
            if len(tile_data.shape) == 3:
                if tile_data.shape[0] == 1:
                    tile_data = tile_data[0]
                else:
                    tile_data = np.moveaxis(tile_data, 0, -1)
            
            tile_data = np.clip(tile_data, 0, 255).astype(np.uint8)
            img = Image.fromarray(tile_data)
            img.save(file_path)
    
    def export_polygons(self):
        from PyQt6.QtWidgets import QDialog
        from utils.geospatial_utils import GeospatialMetrics
        
        polygons_source = None
        is_onnx = False
        
        if (self.main_window.detection_result is not None and 
            'polygons' in self.main_window.detection_result):
            polygons_source = self.main_window.detection_result
        elif (self.main_window.onnx_detection_result is not None and 
              'polygons' in self.main_window.onnx_detection_result):
            polygons_source = self.main_window.onnx_detection_result
            is_onnx = True
        
        if polygons_source is None:
            self.main_window.show_error_detailed(
                "No detection results to export"
            )
            return
        
        if not is_onnx:
            QMessageBox.information(
                self.main_window,
                "Feature Removed",
                "Classic canopy detection export has been removed. "
                "Use ONNX detections for export."
            )
            return
        
        class ExportOptionsDialog(QDialog):
            def __init__(self, parent, raster_has_crs):
                super().__init__(parent)
                self.setWindowTitle('Export Options')
                self.selected_format = 'geojson'
                self.use_raster_crs = False
                
                v = QVBoxLayout(self)
                
                fmt_label = QLabel('Choose export format:')
                v.addWidget(fmt_label)
                
                self.rb_geojson = QRadioButton('GeoJSON (lat/lon)')
                self.rb_geojson.setChecked(True)
                self.rb_json = QRadioButton('ONNX Raw JSON')
                self.rb_shp = QRadioButton('Shapefile')
                
                v.addWidget(self.rb_geojson)
                v.addWidget(self.rb_json)
                v.addWidget(self.rb_shp)
                
                self.chk_use_raster_crs = QCheckBox(
                    'Use raster CRS for Shapefile (if available)'
                )
                self.chk_use_raster_crs.setEnabled(raster_has_crs)
                v.addWidget(self.chk_use_raster_crs)
                
                buttons = QDialogButtonBox(
                    QDialogButtonBox.StandardButton.Ok |
                    QDialogButtonBox.StandardButton.Cancel
                )
                v.addWidget(buttons)
                
                def on_accept():
                    if self.rb_json.isChecked():
                        self.selected_format = 'json'
                    elif self.rb_shp.isChecked():
                        self.selected_format = 'shp'
                    else:
                        self.selected_format = 'geojson'
                    self.use_raster_crs = bool(
                        self.chk_use_raster_crs.isChecked()
                    )
                    self.accept()
                
                def on_reject():
                    self.reject()
                
                buttons.accepted.connect(on_accept)
                buttons.rejected.connect(on_reject)
        
        md = (self.main_window.raster_loader.get_metadata() 
              if self.main_window.raster_loader else None)
        transform = md.get('transform') if md else None
        crs = md.get('crs') if md else None
        
        dlg = ExportOptionsDialog(self.main_window, raster_has_crs=(crs is not None))
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        
        if dlg.selected_format == 'json':
            filt = 'ONNX Raw JSON (*.json)'
            default_suffix = '.json'
        elif dlg.selected_format == 'shp':
            filt = 'Shapefile (*.shp)'
            default_suffix = '.shp'
        else:
            filt = 'GeoJSON (*.geojson)'
            default_suffix = '.geojson'
        
        file_path, _ = QFileDialog.getSaveFileName(
            self.main_window,
            'Export ONNX Detections',
            '',
            filt
        )
        if not file_path:
            return
        
        pixel_polygons = []
        for i, det in enumerate(polygons_source['detections']):
            box = det.get('box', [])
            if len(box) < 4:
                continue
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            pixel_coords = [
                [x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]
            ]
            pixel_polygons.append({
                'id': i,
                'pixel_coords': pixel_coords,
                'class': det.get('class'),
                'score': det.get('score')
            })
        
        try:
            if dlg.selected_format == 'json' or file_path.lower().endswith('.json'):
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(polygons_source['detections'], f, indent=2)
                QMessageBox.information(
                    self.main_window,
                    'Export Complete',
                    f'ONNX raw JSON exported to {file_path}'
                )
                return
            
            from utils.geospatial_utils import GeospatialMetrics, normalize_polygon_coordinates
            geo = GeospatialMetrics(transform, crs)
            
            features = []
            for item in pixel_polygons:
                pid = item['id']
                pix = item['pixel_coords']
                props = {
                    'id': pid,
                    'class': item['class'],
                    'score': item['score']
                }
                
                if dlg.selected_format == 'geojson':
                    try:
                        latlon = geo.polygon_to_latlon(pix)
                        coords = [list(map(float, p)) for p in latlon]
                    except Exception as e:
                        self.logger.warning(f"Failed to convert polygon to lat/lon, using pixel coords: {e}")
                        coords = pix

                    coords = normalize_polygon_coordinates(coords)

                    feat = {
                        'type': 'Feature',
                        'properties': props,
                        'geometry': {
                            'type': 'Polygon',
                            'coordinates': [coords]
                        }
                    }
                    features.append(feat)
                
                elif dlg.selected_format == 'shp':
                    try:
                        if dlg.use_raster_crs and crs is not None:
                            geo_coords = geo.polygon_to_geo(pix)
                            coords = [
                                [float(c[0]), float(c[1])] for c in geo_coords
                            ]
                        else:
                            latlon = geo.polygon_to_latlon(pix)
                            coords = [list(map(float, p)) for p in latlon]
                    except Exception as e:
                        self.logger.warning(f"Failed to convert polygon coordinates for shapefile: {e}")
                        coords = pix

                    coords = normalize_polygon_coordinates(coords)
                    
                    feat = {
                        'type': 'Feature',
                        'properties': props,
                        'geometry': {
                            'type': 'Polygon',
                            'coordinates': [coords]
                        }
                    }
                    features.append(feat)
            
            if dlg.selected_format == 'shp' or file_path.lower().endswith('.shp'):
                try:
                    try:
                        import fiona
                        from fiona.crs import from_epsg
                        
                        schema = {
                            'geometry': 'Polygon',
                            'properties': {
                                'id': 'int',
                                'class': 'str',
                                'score': 'float'
                            }
                        }
                        
                        out_crs = None
                        if dlg.use_raster_crs and crs is not None:
                            try:
                                eps = None
                                try:
                                    eps = (int(crs.to_epsg()) 
                                           if hasattr(crs, 'to_epsg') else None)
                                except Exception as e:
                                    self.logger.debug(f"Could not extract EPSG code from CRS: {e}")
                                    eps = None
                                if eps:
                                    out_crs = from_epsg(eps)
                                else:
                                    out_crs = from_epsg(4326)
                            except Exception as e:
                                self.logger.warning(f"Failed to create CRS from EPSG, defaulting to WGS84: {e}")
                                out_crs = from_epsg(4326)
                        else:
                            out_crs = from_epsg(4326)
                        
                        with fiona.open(
                            file_path,
                            'w',
                            driver='ESRI Shapefile',
                            crs=out_crs,
                            schema=schema
                        ) as dst:
                            for feat in features:
                                dst.write({
                                    'geometry': feat['geometry'],
                                    'properties': feat['properties']
                                })
                        
                        QMessageBox.information(
                            self.main_window,
                            'Export Complete',
                            f'Shapefile exported to {file_path}'
                        )
                        return
                    except Exception as e:
                        self.logger.warning(f"Fiona export failed, falling back to pyshp: {e}")
                        import shapefile
                        
                        shp_path = Path(file_path)
                        w = shapefile.Writer(str(shp_path.with_suffix('')))
                        w.autoBalance = 1
                        w.field('id', 'N')
                        w.field('class', 'C', size=50)
                        w.field('score', 'F', decimal=6)
                        
                        for feat in features:
                            coords = feat['geometry']['coordinates'][0]
                            pts = [(float(p[0]), float(p[1])) for p in coords]
                            if len(pts) > 0 and pts[0] != pts[-1]:
                                pts.append(pts[0])
                            w.poly([pts])
                            props = feat['properties']
                            w.record(
                                props.get('id', -1),
                                str(props.get('class', '')),
                                float(props.get('score', 0.0))
                            )
                        
                        w.close()
                        
                        try:
                            if dlg.use_raster_crs and crs is not None:
                                try:
                                    prj_text = (crs.to_wkt() 
                                                if hasattr(crs, 'to_wkt') 
                                                else None)
                                except Exception as e:
                                    self.logger.debug(f"Failed to convert CRS to WKT: {e}")
                                    prj_text = None
                            else:
                                prj_text = None
                            
                            if not prj_text:
                                try:
                                    from pyproj import CRS
                                    prj_text = CRS.from_epsg(4326).to_wkt()
                                except Exception as e:
                                    self.logger.debug(f"Could not create WKT from pyproj, using hardcoded WGS84: {e}")
                                    prj_text = (
                                        'GEOGCS["WGS 84",'
                                        'DATUM["WGS_1984",'
                                        'SPHEROID["WGS 84",6378137,298.257223563]],'
                                        'PRIMEM["Greenwich",0],'
                                        'UNIT["degree",0.0174532925199433]]'
                                    )
                            
                            try:
                                prj_file = shp_path.with_suffix('.prj')
                                prj_file.write_text(prj_text, encoding='utf-8')
                            except Exception as e:
                                self.logger.warning(f"Failed to write .prj file: {e}")
                        
                        except Exception as e:
                            self.logger.warning(f"Failed to create projection file: {e}")
                        
                        QMessageBox.information(
                            self.main_window,
                            'Export Complete',
                            f'Shapefile exported to {file_path}'
                        )
                        return
                except Exception as e:
                    self.main_window.show_error_detailed(
                        f'Shapefile export failed: {e}',
                        details=traceback.format_exc()
                    )
                    return
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'type': 'FeatureCollection',
                    'features': features
                }, f, indent=2)
            
            QMessageBox.information(
                self.main_window,
                'Export Complete',
                f'GeoJSON exported to {file_path}'
            )
        
        except Exception as e:
            self.main_window.show_error_detailed(
                f'Export failed: {e}',
                details=traceback.format_exc()
            )

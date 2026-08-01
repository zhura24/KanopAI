"""Raster Operations Mixin

Handles raster file operations, background quick preview loading,
and safe multi-band float32 normalization for RGB display.
"""

from pathlib import Path
import numpy as np
import rasterio
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import QMessageBox
from core.tile_manager import TileManager
import logging


class QuickRasterPreviewWorker(QObject):
    """Worker untuk meload preview awal raster besar di background thread."""
    ready = pyqtSignal(object, object, object)
    failed = pyqtSignal(str)

    def __init__(self, loader):
        super().__init__()
        self.loader = loader

    def run(self):
        try:
            preview_data = self.loader.get_overview(max_dimension=2048)
            if self.loader.global_statistics is None:
                self.loader.global_statistics = self.loader.get_global_statistics()
            metadata = self.loader.get_metadata()
            self.ready.emit(preview_data, metadata, self.loader)
        except Exception as e:
            self.failed.emit(str(e))


class RasterMixin:
    """Mixin for raster file operations and multi-layer management."""
    
    def __init__(self):
        super().__init__()
        self.preview_thread = None
        self.preview_worker = None

    def add_raster_layer(self):
        self.logger.debug("RasterMixin.add_raster_layer() -> delegating to layer_handler")
        if hasattr(self, 'layer_handler'):
            return self.layer_handler.add_raster_layer()
        else:
            self.logger.error("layer_handler not initialized!")
            QMessageBox.critical(self, "Error", "Layer handler not initialized")
    
    def _get_active_layer(self):
        if not self.active_layer_id:
            return None
        for layer in self.raster_layers:
            if layer['id'] == self.active_layer_id:
                return layer
        return None
    
    def _update_active_layer_references(self):
        active = self._get_active_layer()
        if active:
            self.drawn_polygons = active['polygons']
            self.centroid_points = active['centroids']
            self.onnx_detection_result = active['detections']
            self.current_metadata = active['metadata']
            self.raster_loader = active['loader']
        else:
            self.drawn_polygons = []
            self.centroid_points = []
            self.onnx_detection_result = None
            self.current_metadata = {}
    
    def _load_and_display_layer(self, layer):
        try:
            loader = layer['loader']
            metadata = layer['metadata']
            
            if hasattr(self, '_update_footer_metadata'):
                self._update_footer_metadata(metadata)

            if self.preview_thread is not None and self.preview_thread.isRunning():
                try:
                    self.preview_thread.quit()
                    self.preview_thread.wait(1000)
                except RuntimeError:
                    pass

            self.preview_thread = QThread()
            self.preview_worker = QuickRasterPreviewWorker(loader)
            self.preview_worker.moveToThread(self.preview_thread)
            
            self.preview_thread.started.connect(self.preview_worker.run)
            self.preview_worker.ready.connect(lambda data, meta, ldr: self._on_preview_ready(layer, data, meta, ldr))
            self.preview_worker.failed.connect(self._on_preview_failed)
            
            self.preview_worker.ready.connect(self.preview_thread.quit)
            self.preview_worker.failed.connect(self.preview_thread.quit)
            
            self.preview_thread.finished.connect(self.preview_worker.deleteLater)
            self.preview_thread.finished.connect(self.preview_thread.deleteLater)
            
            self.preview_thread.start()
            self.logger.info(f"Background preview loading started for layer: {layer['name']}")
                
        except Exception as e:
            self.logger.error(f"Failed to initiate layer loading: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to load layer: {e}")

    def _on_preview_ready(self, layer, data, metadata, loader):
        try:
            if layer not in getattr(self, 'raster_layers', []):
                self.logger.info("Ignoring preview for a layer that was already removed")
                loader.close()
                return
            if getattr(self, 'active_layer_id', layer.get('id')) != layer.get('id'):
                self.logger.debug("Ignoring preview for an inactive layer: %s", layer.get('name'))
                return

            if data is not None:
                # NOTE: previously this method manually picked bands (2,1,0),
                # normalized them, and packed them into an (H, W, 3) array
                # before calling viewer.set_image(). That caused two problems:
                #   1) It used a reversed band order (band3->R, band2->G,
                #      band1->B) that did NOT match the literal band-1,2,3
                #      order used everywhere else (full-resolution tiles),
                #      so the small overview showed different colors than
                #      the big tiled image once it loaded.
                #   2) set_image() expects (Bands, H, W) — an (H, W, 3)
                #      array made it misinterpret image rows as "bands",
                #      corrupting the preview.
                # Fix: just hand the raw (Bands, H, W) data straight to
                # set_image(), which already does literal band1->R,
                # band2->G, band3->B (see raster_viewer.set_image), exactly
                # matching the tile pipeline and the required spec.
                if data.ndim == 3 and data.shape[0] not in (1, 2, 3, 4, 5, 6, 7, 8):
                    # Defensive: only true (H, W, Bands) rasters hit this;
                    # normal rasterio reads are already (Bands, H, W).
                    data = np.transpose(data, (2, 0, 1))

                self.current_data = data

                self.tile_manager = TileManager(loader, tile_size=512)

                if hasattr(self, 'viewer') and self.viewer:
                    self.viewer.set_tile_manager(loader)
                    self.viewer.set_geospatial_metadata(
                        metadata.get('transform'),
                        metadata.get('crs')
                    )
                    self.viewer.set_color_normalization(True)
                    self.viewer.set_image(data)
                    self.viewer.enable_full_resolution(True)

                self.logger.info(f"Layer successfully displayed: {layer['name']}")

                # Notify InferencePanel to refresh raster info display
                try:
                    if hasattr(self, 'inference_panel') and self.inference_panel:
                        self.inference_panel.refresh_raster_info()
                except Exception as _rr_e:
                    self.logger.debug(f"Failed to refresh inference panel raster info: {_rr_e}")
        except Exception as e:
            self.logger.error(f"Error handling preview ready: {e}", exc_info=True)

    def _on_preview_failed(self, error_message):
        self.logger.error(f"Quick preview failed: {error_message}")
        QMessageBox.warning(self, "Warning", f"Failed to load raster preview: {error_message}")

    def _update_layer_list_ui(self):
        if not hasattr(self, 'layer_list_layout'):
            return
        while self.layer_list_layout.count():
            child = self.layer_list_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        has_layers = len(self.raster_layers) > 0
        if hasattr(self, 'lbl_no_layers'):
            self.lbl_no_layers.setVisible(not has_layers)
        for layer in reversed(self.raster_layers):
            layer_widget = self._create_layer_list_item(layer)
            self.layer_list_layout.addWidget(layer_widget)

    def _create_layer_list_item(self, layer):
        from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox
        from PyQt6.QtCore import Qt
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        top_layout = QHBoxLayout()
        name_label = QLabel(layer['name'])
        is_active = layer['id'] == self.active_layer_id
        if is_active:
            name_label.setStyleSheet("QLabel { color: #4CAF50; font-weight: bold; font-size: 11px; }")
        else:
            name_label.setStyleSheet("QLabel { color: #DDD; font-size: 11px; }")
        name_label.mousePressEvent = lambda e, lid=layer['id']: self._set_active_layer(lid)
        name_label.setCursor(Qt.CursorShape.PointingHandCursor)
        top_layout.addWidget(name_label, 1)
        chk_visible = QCheckBox()
        chk_visible.setChecked(layer['visible'])
        chk_visible.stateChanged.connect(lambda state, lid=layer['id']: self._toggle_layer_visibility(lid, state))
        top_layout.addWidget(chk_visible)
        layout.addLayout(top_layout)
        metadata = layer['metadata']
        info_label = QLabel(f"{metadata['width']}×{metadata['height']} | {metadata['bands']} band(s)")
        info_label.setStyleSheet("QLabel { color: #888; font-size: 9px; }")
        layout.addWidget(info_label)
        widget.setLayout(layout)
        widget.setStyleSheet("QWidget { background-color: #3a4a3a; border-left: 3px solid #4CAF50; border-radius: 4px; }" if is_active else "QWidget { background-color: #3a3a3a; border-radius: 4px; }")
        return widget
    
    def _update_file_label(self, file_path):
        from pathlib import Path
        if hasattr(self, 'file_panel'):
            filename = Path(file_path).name
            self.file_panel.update_file_label(filename)
            self.file_panel.set_new_session_visible(True)
    
    def _update_footer_metadata(self, metadata):
        if hasattr(self, 'label_size'):
            self.label_size.setText(f"Size: {metadata['width']}×{metadata['height']}")
        if hasattr(self, 'label_bands'):
            self.label_bands.setText(f"Bands: {metadata['bands']}")
        if hasattr(self, '_update_crs_label'):
            self._update_crs_label(metadata)
    
    def _enable_tools(self):
        if hasattr(self, 'measurement_panel'):
            self.measurement_panel.set_tools_enabled(True)
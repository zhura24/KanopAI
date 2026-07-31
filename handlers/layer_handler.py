"""Handler untuk pengelolaan layer raster dan vector."""

from typing import Any
import logging


class LayerHandler:
    """Handler untuk menambah, menghapus, dan mengelola layer."""
    
    
    def __init__(self, main_window: Any) -> None:
        self.main_window = main_window
        self.logger = logging.getLogger(__name__)
        self.layers = []  # Own the layer state
        self.layer_counter = 0
    
    def add_raster_layer(self) -> None:
        """Tambah layer baru (raster atau vector)."""
        from PyQt6.QtWidgets import QFileDialog, QMessageBox, QProgressDialog
        from PyQt6.QtCore import Qt, QCoreApplication
        from pathlib import Path
        import os
        from core.raster_loader import RasterLoader
        from core.vector_loader import VectorLoader
        
        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "Add Layer",
            "",
            "All Supported (*.tif *.tiff *.img *.jpg *.png *.shp *.geojson);;Raster Files (*.tif *.tiff *.img *.jpg *.png);;Vector Files (*.shp *.geojson);;All Files (*.*)"
        )

        if not file_path:
            return

        # Detect file type
        file_ext = Path(file_path).suffix.lower()
        is_vector = file_ext in ['.shp', '.geojson', '.json']

        layer_type = "vector" if is_vector else "raster"
        self.logger.info(f"Adding {layer_type} layer: {file_path}")

        # Create progress dialog
        progress = QProgressDialog("Loading layer...", None, 0, 100, self.main_window)
        progress.setWindowTitle("Loading")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        QCoreApplication.processEvents()

        try:
            # Create appropriate loader based on file type
            if is_vector:
                loader = VectorLoader()
            else:
                loader = RasterLoader()

            progress.setValue(20)
            QCoreApplication.processEvents()

            if not loader.load_file(file_path):
                QMessageBox.critical(self.main_window, "Error", f"Failed to load {layer_type} file:\n{file_path}")
                return

            metadata = loader.get_metadata()
            progress.setValue(50)
            QCoreApplication.processEvents()

            # Generate layer ID and name
            self.layer_counter += 1
            layer_id = self.layer_counter
            layer_name = f"Layer {layer_id}: {os.path.basename(file_path)}"

            # Create pixmap item for this layer (will be created during rendering)
            pixmap_item = None  # Will be created during rendering

            # Create layer data structure
            new_layer = {
                'id': layer_id,
                'name': layer_name,
                'file_path': file_path,
                'layer_type': layer_type,  # 'raster' or 'vector'
                'loader': loader,
                'metadata': metadata,
                'visible': True,
                'opacity': 1.0,  # 0.0 to 1.0
                'pixmap_item': pixmap_item,
                'is_active': False,
                # Layer-specific data (each layer has its own)
                'full_data': None,  # Cached full raster data (raster only)
                'polygons': [],  # Polygon drawings for this layer
                'measurements': [],  # Measurements for this layer
                'detections': None,  # ONNX detection results for this layer
                'centroids': [],  # Centroid points for this layer
                # Vector-specific
                'vector_items': [],  # QGraphicsItems for vector features
                'vector_style': self.main_window._get_default_vector_style(layer_type, metadata)
            }

            # Add to layers list
            self.layers.append(new_layer)

            progress.setValue(70)
            QCoreApplication.processEvents()

            # Set as active layer if this is the first layer
            # Access self.layers instead of main_window.raster_layers
            if len(self.layers) == 1:
                self.main_window._set_active_layer(layer_id)
                # Update viewer with first layer (includes enable_full_resolution and show_raster)
                self.main_window._update_viewer_for_active_layer()

                # Enable tools after first layer is loaded
                self.main_window._enable_tools_for_layer()

                # Pre-load tiles for first layer AFTER initial display
                # This ensures subsequent switches to other layers and back are instant
                self.main_window._preload_layer_tiles_background(new_layer)
            else:
                # Pre-load tiles for non-active layers to speed up switching
                self.main_window._preload_layer_tiles_background(new_layer)

            progress.setValue(90)
            QCoreApplication.processEvents()

            # Refresh layer list UI
            self.main_window._refresh_layer_list_ui()

            # Enable/disable buttons
            if hasattr(self.main_window, 'btn_remove_layer'):
                 self.main_window.btn_remove_layer.setEnabled(True)
            if hasattr(self.main_window, 'btn_clear_layers'):
                 self.main_window.btn_clear_layers.setEnabled(True)

            # Show New Session button after first layer is loaded
            # Show New Session button after first layer is loaded
            if hasattr(self.main_window, 'file_panel'):
                self.main_window.file_panel.set_new_session_visible(True)
                self.logger.info("New Session button shown")
            elif hasattr(self.main_window, 'btn_new_session') and not self.main_window.btn_new_session.isVisible():
                self.main_window.btn_new_session.setVisible(True)
                self.logger.info("New Session button shown (legacy)")

            progress.setValue(100)
            progress.close()

            self.logger.info(f"Layer added successfully: {layer_name} (ID: {layer_id})")

            QMessageBox.information(
                self.main_window, "Layer Added",
                f"Layer added successfully:\n\n{layer_name}\n\n"
                f"Total layers: {len(self.layers)}"
            )

        except Exception as e:
            progress.close()
            QMessageBox.critical(self.main_window, "Error", f"Failed to add layer:\n{str(e)}")
            self.logger.error(f"Failed to add layer: {e}", exc_info=True)

    
    
    def remove_raster_layer(self, layer_id):
        self.logger.info(f"LayerHandler.remove_raster_layer({layer_id})")
        
        # Since MainWindow only has remove_active_layer, we must:
        # 1. Activate the target layer (using internal method)
        # 2. Call remove_active_layer
        
        if hasattr(self.main_window, '_set_active_layer'):
            self.main_window._set_active_layer(layer_id)
            
        if hasattr(self.main_window, 'remove_active_layer'):
            return self.main_window.remove_active_layer()
        else:
            self.logger.warning("main_window does not have remove_active_layer method")
    
    def toggle_layer_visibility(self, layer_id, visible):
        self.logger.info(f"LayerHandler.toggle_layer_visibility({layer_id}, {visible})")
        if hasattr(self.main_window, 'toggle_layer_visibility'):
            return self.main_window.toggle_layer_visibility(layer_id, visible)
    
    def set_layer_opacity(self, layer_id, opacity):
        self.logger.info(f"LayerHandler.set_layer_opacity({layer_id}, {opacity})")
        if hasattr(self.main_window, 'set_layer_opacity'):
            return self.main_window.set_layer_opacity(layer_id, opacity)
    
    def set_active_layer(self, layer_id):
        self.logger.info(f"LayerHandler.set_active_layer({layer_id})")
        # MainWindow uses _set_active_layer (internal method)
        if hasattr(self.main_window, '_set_active_layer'):
            return self.main_window._set_active_layer(layer_id)
        elif hasattr(self.main_window, 'set_active_layer'):
            return self.main_window.set_active_layer(layer_id)

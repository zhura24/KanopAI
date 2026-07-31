"""Detection Operations Mixin

Handles ONNX model loading and inference operations.
"""

from typing import List, Dict, Any, Optional
from PyQt6.QtWidgets import QMessageBox, QFileDialog
from PyQt6.QtCore import QCoreApplication
import logging
import traceback
from pathlib import Path


class DetectionMixin:
    """Mixin for ONNX detection operations.
    
    Provides UI components and delegates detection logic to DetectionHandler.
    """
    
    def _display_detections_as_overlay(self, detections: List[Dict[str, Any]]) -> None:
        """Display detection results as bounding box overlays.
        
        Args:
            detections: List of detection dicts with 'box', 'score', 'class', 'id'
        """
        if not detections:
            self.logger.warning("No detections to display")
            return
        
        # Get raster metadata for coordinate validation
        try:
            metadata = self.raster_loader.get_metadata()
            img_w = metadata.get('width', 0)
            img_h = metadata.get('height', 0)
        except Exception as e:
            self.logger.error(f"Failed to get raster metadata for validation: {e}")
            img_w = 10000
            img_h = 10000
        
        # Define contrasting colors for detection visualization
        from PyQt6.QtGui import QColor
        PALM_OUTLINE_COLOR = QColor(147, 51, 234)  # Vivid Purple
        PALM_FILL_COLOR = QColor(147, 51, 234, 80)  # Semi-transparent
        OUTLINE_WIDTH = 3
        
        overlay_rects = []
        valid_count = 0
        invalid_count = 0
        
        for det in detections:
            try:
                box = det.get('box', [])
                score = det.get('score', 0.0)
                cls = det.get('class', 'palm')
                det_id = det.get('id', None)
                
                if not box or len(box) < 4:
                    invalid_count += 1
                    continue
                
                try:
                    x1, y1, x2, y2 = [float(b) for b in box[:4]]
                except (ValueError, TypeError):
                    invalid_count += 1
                    continue
                
                import numpy as np
                if not all(np.isfinite([x1, y1, x2, y2])):
                    invalid_count += 1
                    continue
                
                w = x2 - x1
                h = y2 - y1
                
                if w <= 0 or h <= 0:
                    invalid_count += 1
                    continue
                
                # Clamp to image bounds
                if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
                    x1 = max(0.0, min(float(img_w), x1))
                    y1 = max(0.0, min(float(img_h), y1))
                    x2 = max(0.0, min(float(img_w), x2))
                    y2 = max(0.0, min(float(img_h), y2))
                    w = x2 - x1
                    h = y2 - y1
                    
                    if w <= 0 or h <= 0:
                        invalid_count += 1
                        continue
                
                overlay_rects.append({
                    'x': x1,
                    'y': y1,
                    'w': w,
                    'h': h,
                    'outline_color': PALM_OUTLINE_COLOR,
                    'fill_color': PALM_FILL_COLOR,
                    'class': cls,
                    'score': score,
                    'id': det_id,
                    'type': 'detection'
                })
                valid_count += 1
                
            except Exception as e:
                self.logger.error(f"Error processing detection for overlay: {e}", exc_info=True)
                invalid_count += 1
                continue
        
        self.logger.info(
            f"Detection overlay validation: {valid_count} valid, {invalid_count} invalid/skipped "
            f"(total {len(detections)} detections)"
        )
        
        if not overlay_rects:
            QMessageBox.warning(
                self, "Display Warning",
                "No valid detection boxes could be displayed.\n"
                "Check logs for coordinate validation details."
            )
            return
        
        # Display overlay rectangles in viewer
        try:
            self.viewer.set_overlay_tiles(
                overlay_rects,
                outline_color=PALM_OUTLINE_COLOR,
                fill_color=PALM_FILL_COLOR,
                outline_width=OUTLINE_WIDTH
            )
            
            # Auto-enable detection layer checkbox
            if hasattr(self, 'chk_detector_overlay'):
                self.chk_detector_overlay.setChecked(True)
                self.chk_detector_overlay.setEnabled(True)
                self.chk_detector_overlay.setText(f"Detections ({len(overlay_rects)})")
            
            self.logger.info(f"Successfully displayed {len(overlay_rects)} detection overlays")
            
        except Exception as e:
            self.logger.error(f"Failed to display overlay rectangles: {e}", exc_info=True)
            QMessageBox.critical(
                self, "Display Error",
                f"Failed to display detection overlays: {e}\n\nSee logs for details."
            )
    
    def detection_error(self, error_msg):
        """Handle detection error from worker (classic canopy detection removed)."""
        # Classic canopy detection removed; show basic error and return
        self.logger.error(f"detection_error: {error_msg}")
        try:
            self.footer_progress_bar.setVisible(False)
        except Exception as e:
            self.logger.debug(f"Failed to hide progress bar: {e}")
        try:
            # Show a detailed error dialog (includes traceback)
            self.show_error_detailed(error_msg, details=traceback.format_exc())
        except Exception as e:
            self.logger.debug(f"Failed to show detailed error dialog: {e}")
            # Fallback to simple critical dialog if something goes wrong
            try:
                QMessageBox.critical(self, "Error", error_msg)
            except Exception as e2:
                self.logger.error(f"Failed to show error message box: {e2}")

    def browse_onnx_model(self):
        """Browse and load ONNX model - delegated to handler."""
        return self.detection_handler.browse_onnx_model()
    
    def reload_onnx_model(self):
        """Reload ONNX model - delegated to handler."""
        return self.detection_handler.reload_onnx_model()
    
    def load_default_params(self):
        """Load default params - delegated to handler."""
        return self.detection_handler.load_default_params()

    def inference_finished(self, result):
        """Handle detections returned by DetectionWorker - delegated to handler."""
        # Process results through handler (filtering, storage)
        processed = self.detection_handler.handle_inference_finished(result)
        
        # Restore UI state
        try:
            self.footer_progress_bar.setRange(0, 100)
            self.footer_progress_bar.setFormat("Detection: %p%")
            self.footer_progress_bar.setVisible(False)
            self.label_detection.setText("Detection: -")
            self.btn_run_inference.setEnabled(False)
            self.btn_cancel_inference.setEnabled(False)
            self.btn_cancel_inference.setVisible(False)
            self._inference_running = False
        except Exception as e:
            self.logger.warning(f"Failed to restore UI state after inference: {e}")
        
        # Handle errors
        if not processed.get('success'):
            error = processed.get('error', 'Unknown error')
            QMessageBox.critical(self, "Result Error", f"Failed to store detection results: {error}")
            return
        
        detections = processed.get('detections', [])
        
        # No detections found
        if not detections:
            QMessageBox.information(self, "No detections", "No detections were produced by the model.")
            return
        
        # Enable action buttons
        try:
            self.btn_save_detections.setEnabled(True)
            if hasattr(self, 'detector_model_session') and self.detector_model_session is not None:
                self.btn_run_inference.setEnabled(True)
            if hasattr(self, 'btn_convert_to_centroids'):
                self.btn_convert_to_centroids.setEnabled(True)
            if hasattr(self, 'btn_generate_canopy'):
                self.btn_generate_canopy.setEnabled(True)
            if hasattr(self, 'label_last_detections'):
                self.label_last_detections.setText(f"Last detections: {len(detections)} (not saved)")
        except Exception as e:
            self.logger.debug(f"Failed to enable buttons: {e}")
        
        # Auto-visualize detections
        try:
            self._display_detections_as_overlay(detections)
            self.logger.info(f"Displayed {len(detections)} detections as overlay")
            
            # Update checkboxes
            if hasattr(self, 'chk_detector_overlay'):
                self.chk_detector_overlay.blockSignals(True)
                self.chk_detector_overlay.setChecked(True)
                self.chk_detector_overlay.blockSignals(False)
            
            if hasattr(self, 'chk_detection_labels'):
                self.chk_detection_labels.blockSignals(True)
                self.chk_detection_labels.setChecked(True)
                self.chk_detection_labels.blockSignals(False)
            
            # Auto-hide tile preview
            if hasattr(self, 'chk_tile_preview'):
                self.chk_tile_preview.blockSignals(True)
                self.chk_tile_preview.setChecked(False)
                self.chk_tile_preview.blockSignals(False)
                self._on_tile_preview_toggled(False)
            
            # Auto-expand detection result subsection
            if hasattr(self, 'detection_result_subsection'):
                if not self.detection_result_subsection.toggle_button.isChecked():
                    self.detection_result_subsection.toggle()
            
        except Exception as e:
            self.logger.error(f"Failed to display detections: {e}", exc_info=True)
        
        # Notify user
        try:
            QMessageBox.information(
                self, "Inference Complete",
                f"Inference finished — {len(detections)} objects detected."
            )
        except Exception as e:
            self.logger.debug(f"Failed to show inference complete message: {e}")


    def inference_error(self, error_msg: str):
        """Handle inference error - delegated to handler."""
        return self.detection_handler.handle_inference_error(error_msg)

    def cancel_inference(self):
        """Cancel running inference - delegated to handler."""
        return self.detection_handler.cancel_inference()

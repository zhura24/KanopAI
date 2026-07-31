"""Handler untuk menjalankan deteksi memakai model YOLO multispektral (.pt).

Pengganti handlers/detection_handler.py (ONNX). Method surface SENGAJA dibuat
identik dengan DetectionHandler lama (run_detection, browse_onnx_model,
reload_onnx_model, load_default_params, cancel_inference,
handle_inference_finished, handle_inference_error, update_progress,
detection_finished) supaya nanti tinggal 1 baris di main_window_impl.py:

    self.detection_handler = MultispectralDetectionHandler(self)

tanpa perlu ubah main_window_impl.py / detection_mixin.py sama sekali.

Panel UI (detection_panel.py) BELUM diubah di tahap ini -- tombol "Browse"
yang sudah ada dipakai untuk memilih file .pt DAN band_stats.json secara
berurutan (dua file dialog), supaya bisa langsung dites end-to-end dari UI
yang sekarang sebelum panel baru dibuat.
"""

from typing import Optional, Any
import logging
from pathlib import Path

from PyQt6.QtWidgets import QMessageBox, QFileDialog, QApplication

from ui.dialogs.band_mismatch_dialog import resolve_band_matching
from core.multispectral_worker import MultispectralInferenceWorker
from core.inference_engine import InferenceResult


class MultispectralDetectionHandler:
    """Handler untuk menjalankan dan mengelola deteksi YOLO multispektral."""

    def __init__(self, main_window: Any) -> None:
        self.main_window = main_window
        self.logger = logging.getLogger(__name__)

        # State model -- pengganti main_window.detector_model_path (ONNX)
        # Dipakai 2 path terpisah: model .pt dan band_stats.json
        if not hasattr(main_window, "detector_model_path"):
            main_window.detector_model_path = None
        if not hasattr(main_window, "band_stats_path"):
            main_window.band_stats_path = None

        self._worker: Optional[MultispectralInferenceWorker] = None
        self._last_result: Optional[InferenceResult] = None

    # ==================================================================
    # PARAMETER (sementara: nilai default, belum ada field UI khusus)
    # ==================================================================
    def _get_run_params(self) -> dict:
        """Ambil parameter run() dari UI kalau field-nya sudah ada (dipakai
        ulang dari panel ONNX lama: spin_tile_size, spin_confidence,
        spin_iou, spin_batch_size), fallback ke default engine kalau
        field belum ada / belum sesuai konteks multispektral."""
        mw = self.main_window
        get = lambda attr, default: (
            mw.__dict__.get(attr).value() if attr in mw.__dict__ and hasattr(mw.__dict__.get(attr), "value") else default
        )
        return {
            "conf": float(getattr(mw, "spin_confidence", None).value()) if hasattr(mw, "spin_confidence") else 0.25,
            "tile_size": int(getattr(mw, "spin_tile_size", None).value()) if hasattr(mw, "spin_tile_size") else 640,
            "overlap": 64,
            "iou_threshold": float(getattr(mw, "spin_iou", None).value()) if hasattr(mw, "spin_iou") else 0.5,
            "batch_size": int(getattr(mw, "spin_batch_size", None).value()) if hasattr(mw, "spin_batch_size") else 4,
        }

    # ==================================================================
    # MAIN ENTRY POINT
    # ==================================================================
    def run_detection(self) -> None:
        """Jalankan deteksi YOLO multispektral - method orchestrator."""
        success, error = self._validate_detection_prerequisites()
        if not success:
            return

        self._clear_inference_session()

        raster_path = self._get_active_raster_path()
        if not raster_path:
            QMessageBox.critical(self.main_window, "No raster", "No raster loaded to run detection on.")
            return

        params = self._get_run_params()

        manual_mapping, adaptive_fallback, proceed = resolve_band_matching(
            self.main_window,
            raster_path,
            self.main_window.band_stats_path,
        )
        if not proceed:
            return

        try:
            self._worker = MultispectralInferenceWorker(
                model_path=self.main_window.detector_model_path,
                band_stats_path=self.main_window.band_stats_path,
                raster_path=raster_path,
                conf=params["conf"],
                tile_size=params["tile_size"],
                overlap=params["overlap"],
                iou_threshold=params["iou_threshold"],
                batch_size=params["batch_size"],
                output_dir=str(Path(raster_path).parent / "output"),
                manual_band_mapping=manual_mapping,
                enable_adaptive_fallback=adaptive_fallback,
            )

            self._worker.log.connect(self._on_worker_log)
            self._worker.progress.connect(self.update_progress)
            self._worker.finished.connect(self.detection_finished)
            self._worker.error.connect(self.detection_error)
            self._worker.cancelled.connect(self._on_worker_cancelled)

            self.main_window.footer_progress_bar.setVisible(True)
            self.main_window.footer_progress_bar.setValue(0)
            self.main_window._current_detection_worker = self._worker

            if hasattr(self.main_window, "btn_run_inference"):
                self.main_window.btn_run_inference.setEnabled(False)
            if hasattr(self.main_window, "btn_cancel_inference"):
                self.main_window.btn_cancel_inference.setVisible(True)
                self.main_window.btn_cancel_inference.setEnabled(True)
            if hasattr(self.main_window, "label_detection"):
                self.main_window.label_detection.setText("Detection: Running...")

            self._worker.start()
            self.logger.info("MultispectralInferenceWorker started")

        except Exception as e:
            self.logger.error(f"Failed to start worker: {e}", exc_info=True)
            QMessageBox.critical(self.main_window, "Error", f"Failed to start detection: {e}")

    # ==================================================================
    # HELPERS
    # ==================================================================
    def _validate_detection_prerequisites(self):
        if not getattr(self.main_window, "detector_model_path", None):
            error_msg = "Please load a model (.pt) first."
            QMessageBox.warning(self.main_window, "No model", error_msg)
            return False, error_msg
        if not getattr(self.main_window, "band_stats_path", None):
            error_msg = "Please load band_stats.json first."
            QMessageBox.warning(self.main_window, "No band stats", error_msg)
            return False, error_msg
        return True, None

    def _clear_inference_session(self):
        try:
            self.logger.info("=== NEW INFERENCE SESSION: Clearing previous overlays ===")
            if hasattr(self.main_window, "viewer") and self.main_window.viewer:
                self.main_window.viewer.clear_overlay()
            if hasattr(self.main_window, "chk_detector_overlay"):
                self.main_window.chk_detector_overlay.setChecked(True)
            self.main_window.onnx_detection_result = None
            if hasattr(self.main_window, "centroid_points"):
                self.main_window.centroid_points.clear()
                if hasattr(self.main_window, "_clear_centroid_rendering"):
                    self.main_window._clear_centroid_rendering()
        except Exception as e:
            self.logger.warning(f"Failed to clear previous session overlays: {e}")

    def _get_active_raster_path(self) -> Optional[str]:
        """Ambil path file raster yang sedang aktif (dibutuhkan engine baru
        karena dia baca langsung dari file lewat rasterio, bukan dari array
        yang sudah di-load viewer)."""
        mw = self.main_window
        try:
            if hasattr(mw, "raster_layers") and hasattr(mw, "active_layer_id"):
                for layer in mw.raster_layers:
                    if layer.get("id") == mw.active_layer_id:
                        loader = layer.get("loader")
                        if loader and getattr(loader, "file_path", None):
                            return loader.file_path
            if hasattr(mw, "raster_loader") and getattr(mw.raster_loader, "file_path", None):
                return mw.raster_loader.file_path
        except Exception as e:
            self.logger.error(f"Failed to resolve active raster path: {e}")
        return None

    # ==================================================================
    # RESULT CONVERSION -- InferenceResult -> format 'detections' lama
    # ==================================================================
    def _convert_result_to_detections(self, result: InferenceResult) -> list:
        """Konversi InferenceResult (boxes/scores/classes, numpy) ke list of
        dict {'box','score','class'} yang sudah dipakai kode overlay/filter
        polygon/export yang ADA (result_processor.py dst), supaya kode itu
        tidak perlu diubah sama sekali."""
        detections = []
        if result is None or result.boxes is None or len(result.boxes) == 0:
            return detections

        from core.inference_engine import resolve_class_name

        for box, score, cls in zip(result.boxes, result.scores, result.classes):
            x1, y1, x2, y2 = [float(v) for v in box]
            detections.append({
                "box": [x1, y1, x2, y2],
                "score": float(score),
                "class": resolve_class_name(cls, result.class_names),
            })
        return detections

    # ==================================================================
    # SIGNAL HANDLERS (dari worker)
    # ==================================================================
    def _on_worker_log(self, message: str) -> None:
        self.logger.info(f"[inference] {message}")

    def update_progress(self, current: int, total: int) -> None:
        pct = int((current / total) * 100) if total else 0
        self.main_window.footer_progress_bar.setValue(pct)
        if hasattr(self.main_window, "label_detection"):
            self.main_window.label_detection.setText(f"Detection: {current}/{total} tile")

    def detection_finished(self, result: InferenceResult) -> None:
        self._last_result = result
        detections = self._convert_result_to_detections(result)
        wrapped = {"detections": detections}

        if hasattr(self.main_window, "inference_finished"):
            self.main_window.inference_finished(wrapped)
        else:
            self.handle_inference_finished(wrapped)

        self._restore_ui_idle(status=f"Detection: {len(detections)} objek ditemukan")

    def detection_error(self, error_msg: str) -> None:
        self.logger.error(f"Detection error: {error_msg}", exc_info=True)
        if hasattr(self.main_window, "inference_error"):
            self.main_window.inference_error(error_msg)
        else:
            self.handle_inference_error(error_msg)

    def _on_worker_cancelled(self) -> None:
        self._restore_ui_idle(status="Detection: Cancelled")

    def _restore_ui_idle(self, status: str = "Detection: -") -> None:
        try:
            self.main_window.footer_progress_bar.setVisible(False)
            if hasattr(self.main_window, "btn_run_inference"):
                self.main_window.btn_run_inference.setEnabled(True)
            if hasattr(self.main_window, "btn_cancel_inference"):
                self.main_window.btn_cancel_inference.setEnabled(False)
                self.main_window.btn_cancel_inference.setVisible(False)
            if hasattr(self.main_window, "label_detection"):
                self.main_window.label_detection.setText(status)
        except Exception as e:
            self.logger.debug(f"Failed to restore UI state: {e}")
        self.main_window._current_detection_worker = None

    # ==================================================================
    # CANCEL
    # ==================================================================
    def cancel_detection(self):
        self.logger.info("Cancelling detection...")
        if getattr(self.main_window, "_current_detection_worker", None):
            self.main_window._current_detection_worker.stop()
        self._restore_ui_idle(status="Detection: Cancelling...")

    def cancel_inference(self):
        if not self._worker:
            return
        reply = QMessageBox.question(
            self.main_window, "Confirm Cancellation",
            "Are you sure you want to cancel the running inference task?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self.logger.info("User requested inference cancellation")
        self._worker.stop()
        if not self._worker.wait(5000):
            self.logger.warning("Worker thread did not finish within timeout, forcing terminate")
            self._worker.terminate()
            self._worker.wait(2000)
        self._restore_ui_idle(status="Detection: Cancelled")

    # ==================================================================
    # MODEL MANAGEMENT (dipakai lewat tombol "Browse" yang sudah ada)
    # ==================================================================
    def browse_onnx_model(self):
        """Reuse tombol Browse yang ada: minta file .pt lalu band_stats.json.

        NOTE: nama method sengaja dipertahankan (bukan 'onnx' beneran)
        supaya wiring lama (detection_panel.py, detection_mixin.py) tidak
        perlu diubah di tahap ini. Ganti nama + tambah 2 field terpisah
        kalau panel baru sudah dibuat.
        """
        pt_path, _ = QFileDialog.getOpenFileName(
            self.main_window, "Select Model (.pt)", "", "YOLO Model (*.pt);;All Files (*.*)"
        )
        if not pt_path:
            return

        stats_path, _ = QFileDialog.getOpenFileName(
            self.main_window, "Select band_stats.json", "", "JSON (*.json);;All Files (*.*)"
        )
        if not stats_path:
            return

        self.main_window.detector_model_path = pt_path
        self.main_window.band_stats_path = stats_path

        if hasattr(self.main_window, "label_model_path"):
            self.main_window.label_model_path.setText(
                f"{Path(pt_path).name}  |  {Path(stats_path).name}"
            )
        if hasattr(self.main_window, "label_model_info"):
            self.main_window.label_model_info.setText("Model & band stats siap. Klik Run Inference.")
        if hasattr(self.main_window, "btn_run_inference"):
            self.main_window.btn_run_inference.setEnabled(True)

        self.logger.info(f"Model dimuat: {pt_path} | band_stats: {stats_path}")
        QApplication.processEvents()

    def reload_onnx_model(self):
        if not getattr(self.main_window, "detector_model_path", None):
            return
        try:
            self.browse_onnx_model()
        except Exception as e:
            self.logger.error(f"Failed to reload model: {e}")
            QMessageBox.information(self.main_window, "Reload Failed", f"Failed to reload model: {e}")

    def load_default_params(self):
        QMessageBox.information(
            self.main_window, "Not Implemented",
            "Loading default model parameters is not implemented in this build.",
        )

    # ==================================================================
    # HASIL / ERROR -- format sama seperti DetectionHandler lama
    # ==================================================================
    def handle_inference_finished(self, result) -> dict:
        from handlers.detection.result_processor import ResultProcessor

        try:
            detections = result.get("detections", []) if isinstance(result, dict) else []
        except Exception:
            detections = []

        self.logger.info(f"INFERENCE_RAW_DETS count={len(detections)} sample={detections[:10]}")

        processor = ResultProcessor(self.main_window)
        processed = processor.process_inference_results(result, None, None)
        detections = processed["detections"]

        self.logger.info(f"Inference finished | Detections: {len(detections)}")

        if not detections:
            return {"success": True, "detections": [], "count": 0, "error": None}

        try:
            self.main_window.onnx_detection_result = {"polygons": True, "detections": detections}
            active_layer = self._get_active_layer()
            if active_layer:
                active_layer["detections"] = self.main_window.onnx_detection_result
                self.logger.info(f"Detection results saved to layer: {active_layer['name']}")
            if hasattr(self.main_window, "_update_layer_info_panel"):
                self.main_window._update_layer_info_panel()
            return {"success": True, "detections": detections, "count": len(detections), "error": None}
        except Exception as e:
            self.logger.error(f"Failed to store detection results: {e}", exc_info=True)
            return {"success": False, "detections": [], "count": 0, "error": str(e)}

    def handle_inference_error(self, error_msg):
        self.logger.error(f"Inference error: {error_msg}")
        self._restore_ui_idle(status="Detection: Error")
        try:
            QMessageBox.critical(self.main_window, "Inference Error", f"Detection failed:\n{error_msg}")
        except Exception as e:
            self.logger.error(f"Failed to show error dialog: {e}")

    def _get_active_layer(self):
        try:
            if hasattr(self.main_window, "active_layer_id") and hasattr(self.main_window, "raster_layers"):
                layer_id = self.main_window.active_layer_id
                if layer_id is not None:
                    for layer in self.main_window.raster_layers:
                        if layer["id"] == layer_id:
                            return layer
            if hasattr(self.main_window, "layers") and hasattr(self.main_window, "_active_layer_index"):
                if 0 <= self.main_window._active_layer_index < len(self.main_window.layers):
                    return self.main_window.layers[self.main_window._active_layer_index]
        except Exception as e:
            self.logger.debug(f"Failed to get active layer: {e}")
        return None

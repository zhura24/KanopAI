"""Worker thread untuk inference multispektral (Ultralytics YOLO .pt).

Menggantikan core/detection_worker.py (ONNX). Membungkus InferenceEngine
dari core/inference_engine.py supaya proses inference (yang bisa makan
waktu lama untuk raster besar) tidak membekukan UI PyQt6, sama seperti
pola DetectionWorker lama, tapi API-nya mengikuti InferenceEngine.run().
"""

from typing import Optional, Dict, Any
import logging

from PyQt6.QtCore import QThread, pyqtSignal

from core.inference_engine import InferenceEngine, InferenceResult, CancelledError

logger = logging.getLogger(__name__)


class MultispectralInferenceWorker(QThread):
    """Worker thread untuk menjalankan InferenceEngine.run() di background.

    Emits:
      - log(str): baris log dari engine (log_fn)
      - progress(int, int): (current, total) tile yang sudah diproses
      - finished(object): InferenceResult kalau sukses
      - error(str): pesan error kalau gagal
      - cancelled(): kalau proses dibatalkan pengguna
    """

    log = pyqtSignal(str)
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(
        self,
        model_path: str,
        band_stats_path: str,
        raster_path: str,
        conf: float = 0.25,
        tile_size: int = 640,
        overlap: int = 64,
        iou_threshold: float = 0.5,
        output_dir: Optional[str] = None,
        batch_size: int = 4,
        out_name: Optional[str] = None,
        aoi_shp_path: Optional[str] = None,
        exclude_shp_path: Optional[str] = None,
        aoi_polygons_px: Optional[list] = None,
        exclude_polygons_px: Optional[list] = None,
        db_path: Optional[str] = None,
        manual_band_mapping: Optional[Dict[int, int]] = None,
        enable_adaptive_fallback: bool = False,
    ) -> None:
        super().__init__()
        self.model_path = model_path
        self.band_stats_path = band_stats_path
        self.raster_path = raster_path

        # Parameter run() -- lihat InferenceEngine.run() untuk detail lengkap
        self.conf = conf
        self.tile_size = tile_size
        self.overlap = overlap
        self.iou_threshold = iou_threshold
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.out_name = out_name
        self.aoi_shp_path = aoi_shp_path
        self.exclude_shp_path = exclude_shp_path
        self.aoi_polygons_px = aoi_polygons_px
        self.exclude_polygons_px = exclude_polygons_px
        self.db_path = db_path
        self.manual_band_mapping = manual_band_mapping
        self.enable_adaptive_fallback = enable_adaptive_fallback

        self._should_cancel = False
        self.engine: Optional[InferenceEngine] = None

    # ------------------------------------------------------------------
    # Kontrol dari luar (dipanggil dari thread GUI)
    # ------------------------------------------------------------------
    def stop(self) -> None:
        """Minta engine berhenti di titik cek berikutnya (antar-tile)."""
        self._should_cancel = True

    def _should_cancel_fn(self) -> bool:
        return self._should_cancel

    def _log_fn(self, message: str) -> None:
        # log_fn dipanggil dari thread worker ini sendiri (bukan main
        # thread), makanya WAJIB pakai sinyal Qt (bukan langsung update
        # widget) supaya thread-safe.
        self.log.emit(str(message))

    def _progress_fn(self, current: int, total: int) -> None:
        self.progress.emit(int(current), int(total))

    # ------------------------------------------------------------------
    # Entry point QThread
    # ------------------------------------------------------------------
    def run(self) -> None:
        try:
            self.engine = InferenceEngine(
                model_path=self.model_path,
                band_stats_path=self.band_stats_path,
                log_fn=self._log_fn,
                progress_fn=self._progress_fn,
                should_cancel=self._should_cancel_fn,
            )
            self.engine.load()

            result: InferenceResult = self.engine.run(
                raster_path=self.raster_path,
                conf=self.conf,
                tile_size=self.tile_size,
                overlap=self.overlap,
                iou_threshold=self.iou_threshold,
                output_dir=self.output_dir,
                batch_size=self.batch_size,
                out_name=self.out_name,
                aoi_shp_path=self.aoi_shp_path,
                exclude_shp_path=self.exclude_shp_path,
                aoi_polygons_px=self.aoi_polygons_px,
                exclude_polygons_px=self.exclude_polygons_px,
                db_path=self.db_path,
                manual_band_mapping=self.manual_band_mapping,
                enable_adaptive_fallback=self.enable_adaptive_fallback,
            )
            self.finished.emit(result)

        except CancelledError:
            logger.info("Inference dibatalkan oleh pengguna.")
            self.cancelled.emit()

        except Exception as e:
            logger.error(f"Inference gagal: {e}", exc_info=True)
            self.error.emit(str(e))

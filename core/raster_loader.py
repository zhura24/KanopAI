"""Loader untuk file raster dengan thread-safe operations."""

from typing import Optional, Dict, Tuple, Any, List
import rasterio
from rasterio.windows import Window
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from PyQt6.QtCore import QObject, pyqtSignal, QMutex, QMutexLocker
from utils.logger_config import get_logger, PerformanceLogger


class RasterLoader(QObject):
    """Loader untuk membaca dan mengelola file raster."""
    error_occurred = pyqtSignal(str)

    # Overview levels yang dibangun — sama seperti QGIS default (2, 4, 8, 16, 32, 64, 128)
    OVR_LEVELS = [2, 4, 8, 16, 32, 64, 128]

    def __init__(self) -> None:
        super().__init__()
        self.logger = get_logger(__name__)
        self.dataset: Optional[Any] = None
        self.file_path: Optional[str] = None
        self.metadata: Dict[str, Any] = {}
        self.global_statistics: Optional[List[Dict[str, float]]] = None  # Cache min/max per band
        self._ovr_ready: bool = False          # True setelah .ovr terkonfirmasi ada/sudah dibangun
        self._read_mutex = QMutex()            # Thread-safe lock for rasterio reads
        self.logger.debug("RasterLoader initialized")

    def load_file(self, file_path: str) -> bool:
        """Load file raster dan ekstrak metadata."""
        self.logger.info(f"Loading raster file: {file_path}")

        try:
            with PerformanceLogger(self.logger, f"Load file: {Path(file_path).name}"):
                if self.dataset:
                    self.logger.debug("Closing previous dataset")
                    self.close()

                self.dataset = rasterio.open(file_path)
                self.file_path = file_path
                self._ovr_ready = False

                self.metadata = {
                    'width': self.dataset.width,
                    'height': self.dataset.height,
                    'bands': self.dataset.count,
                    'dtype': self.dataset.dtypes[0],
                    'crs': self.dataset.crs,
                    'transform': self.dataset.transform,
                    'bounds': self.dataset.bounds
                }

                self.logger.info(
                    f"File loaded successfully | "
                    f"Size: {self.metadata['width']}x{self.metadata['height']} | "
                    f"Bands: {self.metadata['bands']} | "
                    f"Type: {self.metadata['dtype']}"
                )

                # Overview construction is intentionally deferred to the
                # background preview worker so opening a large raster never
                # blocks the GUI thread.

                return True

        except rasterio.errors.RasterioIOError as e:
            error_msg = "File tidak dapat dibuka. Format tidak didukung."
            self.logger.error(f"{error_msg} | {str(e)}")
            self.error_occurred.emit(error_msg)
            return False
        except Exception as e:
            error_msg = f"Error loading file: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.error_occurred.emit(error_msg)
            return False

    def read_window(self, x_offset: int, y_offset: int, width: int, height: int,
                   scale: float = 1.0, band_indexes: Optional[List[int]] = None,
                   overview_level: int = 0) -> Optional[NDArray]:
        """Read a window from the raster dataset (thread-safe).

        Args:
            band_indexes: Optional list of 1-based band numbers to read, in the
                order they should be returned (e.g. [1, 2, 3] for R,G,B).
                When None, ALL bands are read (legacy behaviour). Passing only
                the bands actually needed for display avoids reading unused
                bands (e.g. bands 4-7 of a 7-band multispectral tile that will
                never be drawn), cutting I/O and RAM roughly in proportion to
                bands_read / bands_total.
        """
        if not self.dataset:
            self.logger.warning("read_window called but no dataset loaded")
            return None

        # Use mutex to ensure thread-safe access to rasterio dataset
        # This prevents "TIFFReadEncodedStrip: Seek error" from concurrent reads
        with QMutexLocker(self._read_mutex):
            try:
                self.logger.debug(
                    f"Reading window | "
                    f"Offset: ({x_offset}, {y_offset}) | "
                    f"Size: {width}x{height} | "
                    f"Scale: {scale:.2f} | "
                    f"Bands: {band_indexes if band_indexes else 'ALL'}"
                )

                window = Window(x_offset, y_offset, width, height)

                if overview_level > 0:
                    decimation = self.get_overview_decimations()[overview_level - 1]
                    # The tile still occupies its native raster footprint in
                    # the scene, so the overview read must be expanded back
                    # to that footprint by the viewer.
                    scale = 1.0 / decimation

                out_shape = (
                    max(1, int(height * scale)),
                    max(1, int(width * scale))
                )

                band_count = len(band_indexes) if band_indexes else self.dataset.count

                data = self.dataset.read(
                    indexes=band_indexes,
                    window=window,
                    out_shape=(band_count, *out_shape),
                    resampling=rasterio.enums.Resampling.bilinear
                )

                return data

            except Exception as e:
                error_msg = f"Error reading window: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                self.error_occurred.emit(error_msg)
                return None

    def get_overview_decimations(self) -> List[int]:
        """Return available overview factors, including the base level separately."""
        if not self.dataset:
            return []
        try:
            return list(self.dataset.overviews(1))
        except Exception as e:
            self.logger.debug(f"Unable to read overview metadata: {e}")
            return []

    def select_overview_level(self, zoom_factor: float) -> int:
        """Select the closest pyramid level for pixels-per-screen-pixel."""
        overviews = self.get_overview_decimations()
        if not overviews or zoom_factor >= 1.0:
            return 0

        requested_decimation = 1.0 / max(zoom_factor, 1e-6)
        candidates = [(0, 1)] + list(enumerate(overviews, start=1))
        return min(candidates, key=lambda item: abs(item[1] - requested_decimation))[0]

    def get_overview_decimation(self, overview_level: int) -> int:
        """Return the source-to-overview scale factor for a selected level."""
        if overview_level <= 0:
            return 1
        overviews = self.get_overview_decimations()
        if overview_level > len(overviews):
            return 1
        return int(overviews[overview_level - 1])

    def read_full_downsampled(self, max_dimension: int = 2048) -> Optional[NDArray]:
        if not self.dataset:
            return None

        with QMutexLocker(self._read_mutex):
            try:
                scale = min(1.0, max_dimension / max(self.metadata['width'], self.metadata['height']))

                out_shape = (
                    int(self.metadata['height'] * scale),
                    int(self.metadata['width'] * scale)
                )

                data = self.dataset.read(
                    out_shape=(self.dataset.count, *out_shape),
                    resampling=rasterio.enums.Resampling.bilinear
                )

                return data

            except Exception as e:
                self.error_occurred.emit(f"Error membaca data: {str(e)}")
                return None

    # ------------------------------------------------------------------
    # Overview / pyramid helpers  (pendekatan QGIS: .ovr eksternal)
    # ------------------------------------------------------------------

    def _ovr_path(self) -> Optional[Path]:
        """Kembalikan path file .ovr di sebelah raster asli."""
        if not self.file_path:
            return None
        return Path(self.file_path).with_suffix(Path(self.file_path).suffix + '.ovr')

    def _ensure_overviews(self) -> bool:
        """Build/reopen overviews while excluding concurrent raster reads."""
        with QMutexLocker(self._read_mutex):
            return self._ensure_overviews_locked()

    def _ensure_overviews_locked(self) -> bool:
        """Pastikan file .ovr ada di disk.  Kalau belum ada, bangun sekarang.

        Persis seperti yang QGIS lakukan saat kamu klik
        "Layer → Pyramids → Build Pyramids" — hasilnya disimpan sebagai
        file .ovr di sebelah file raster asli, sehingga kali berikutnya
        file dibuka GDAL langsung memakai pyramid tanpa rebuild.

        Returns:
            True jika overview sudah siap (ada di disk atau baru dibangun).
        """
        if self._ovr_ready:
            return True

        if not self.dataset or not self.file_path:
            return False

        ovr = self._ovr_path()

        # Sudah ada dari sesi sebelumnya → langsung pakai
        if ovr and ovr.exists():
            self.logger.info(f"Overview file sudah ada: {ovr.name} ({ovr.stat().st_size / (1024*1024):.1f} MB)")
            # Reopen dataset supaya GDAL "melihat" .ovr yang ada
            try:
                self.dataset.close()
                self.dataset = rasterio.open(self.file_path, overview_level=None)
            except Exception:
                self.dataset = rasterio.open(self.file_path)
            self._ovr_ready = True
            return True

        # Belum ada → bangun sekarang
        self.logger.info(
            f"Membangun overview pyramid (.ovr) untuk: {Path(self.file_path).name} | "
            f"Levels: {self.OVR_LEVELS} | "
            f"Ini hanya dilakukan sekali, hasilnya disimpan di disk."
        )

        try:
            with PerformanceLogger(self.logger, "Build .ovr pyramid"):
                # rasterio.open dengan mode 'r+' untuk nulis overview ke file asli
                # (untuk BigTIFF read-only, kita pakai external .ovr via GDAL_TIFF_OVR_BLOCKSIZE)
                with rasterio.open(self.file_path, 'r+') as dst:
                    dst.build_overviews(self.OVR_LEVELS, rasterio.enums.Resampling.average)
                    dst.update_tags(ns='rio_overview', resampling='average')

            # Setelah build, reopen supaya GDAL load overview index
            self.dataset.close()
            self.dataset = rasterio.open(self.file_path)
            self._ovr_ready = True

            if ovr and ovr.exists():
                self.logger.info(f"Overview .ovr berhasil dibangun: {ovr.name} ({ovr.stat().st_size / (1024*1024):.1f} MB)")
            else:
                # rasterio mungkin menyimpan overview internal (bukan .ovr eksternal)
                self.logger.info("Overview berhasil dibangun (internal/embedded dalam file raster).")

            return True

        except Exception as e:
            # File read-only (misal di DVD/network share) → fallback ke decimated read
            self.logger.warning(
                f"Tidak bisa menulis overview ke disk (file mungkin read-only): {e} | "
                f"Fallback ke decimated read — masih aman untuk BigTIFF tapi lebih lambat dari .ovr."
            )
            self._ovr_ready = False  # Tandai supaya get_overview() tetap jalan via fallback
            return False

    def get_overview(self, max_dimension: int = 2048) -> Optional[NDArray]:
        """Baca overview raster sesuai max_dimension.

        Kalau .ovr sudah ada di disk, GDAL secara otomatis memilih level pyramid
        yang paling cocok dan baca hanya piksel yang diperlukan — sangat cepat
        bahkan untuk BigTIFF multi-GB.  Kalau .ovr belum ada (file read-only
        atau build gagal), fallback ke decimated read biasa.

        Args:
            max_dimension: Panjang sisi terpanjang array yang dikembalikan (piksel).
        Returns:
            NDArray shape (bands, h, w) atau None jika error.
        """
        if not self.dataset:
            return None

        with QMutexLocker(self._read_mutex):
            try:
                scale = min(1.0, max_dimension / max(self.metadata['width'], self.metadata['height']))
                out_h = max(1, int(self.metadata['height'] * scale))
                out_w = max(1, int(self.metadata['width'] * scale))

                self.logger.info(
                    f"Reading overview | "
                    f"Source: {self.metadata['width']}x{self.metadata['height']} | "
                    f"Output: {out_w}x{out_h} | "
                    f"OVR on disk: {self._ovr_ready}"
                )

                # Dengan .ovr di disk, GDAL memilih pyramid level yang tepat secara
                # otomatis — ini yang membuat read cepat persis seperti di QGIS.
                data = self.dataset.read(
                    out_shape=(self.dataset.count, out_h, out_w),
                    resampling=rasterio.enums.Resampling.average
                )

                size_mb = data.nbytes / (1024 * 1024)
                self.logger.info(f"Overview read selesai | Size: {size_mb:.1f} MB")
                return data

            except Exception as e:
                error_msg = f"Error reading overview: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                self.error_occurred.emit(error_msg)
                return None

    def read_for_export(self, max_dimension: int = 8192) -> Optional[NDArray]:
        """Read raster data for export features (training data export etc.).
        Uses decimated read — safe for BigTIFF, never loads more than
        max_dimension^2 pixels per band into RAM.

        This is the *only* place that should be called by export_handler;
        it replaces the old read_window(0, 0, full_w, full_h, scale=1.0) pattern.
        """
        return self.get_overview(max_dimension)

    def read_adaptive(self, max_dimension_display: int = 2048, 
                     max_dimension_detection: int = 4096) -> Tuple[Optional[NDArray], Optional[NDArray]]:
        """Read adaptive resolution data for display and detection"""
        if not self.dataset:
            self.logger.warning("read_adaptive called but no dataset loaded")
            return None, None

        with QMutexLocker(self._read_mutex):
            try:
                with PerformanceLogger(self.logger, "Read adaptive resolution data"):
                    display_scale = min(1.0, max_dimension_display / max(self.metadata['width'], self.metadata['height']))
                    detection_scale = min(1.0, max_dimension_detection / max(self.metadata['width'], self.metadata['height']))

                    display_shape = (
                        int(self.metadata['height'] * display_scale),
                        int(self.metadata['width'] * display_scale)
                    )

                    detection_shape = (
                        int(self.metadata['height'] * detection_scale),
                        int(self.metadata['width'] * detection_scale)
                    )

                    self.logger.info(
                        f"Reading adaptive data | "
                        f"Display: {display_shape[1]}x{display_shape[0]} (scale: {display_scale:.3f}) | "
                        f"Detection: {detection_shape[1]}x{detection_shape[0]} (scale: {detection_scale:.3f})"
                    )

                    display_data = self.dataset.read(
                        out_shape=(self.dataset.count, *display_shape),
                        resampling=rasterio.enums.Resampling.bilinear
                    )

                    if detection_scale > display_scale:
                        self.logger.debug("Reading separate detection data")
                        detection_data = self.dataset.read(
                            out_shape=(self.dataset.count, *detection_shape),
                            resampling=rasterio.enums.Resampling.bilinear
                        )
                    else:
                        self.logger.debug("Using display data for detection")
                        detection_data = display_data

                    return display_data, detection_data

            except Exception as e:
                error_msg = f"Error reading adaptive data: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                self.error_occurred.emit(error_msg)
                return None, None

    def get_optimal_dimension(self, target_memory_mb: int = 200) -> int:
        if not self.dataset:
            return 2048

        width = self.metadata['width']
        height = self.metadata['height']
        bands = self.metadata['bands']

        bytes_per_pixel = 1
        if 'uint16' in str(self.metadata['dtype']):
            bytes_per_pixel = 2
        elif 'float' in str(self.metadata['dtype']):
            bytes_per_pixel = 4

        total_pixels = width * height
        current_size_mb = (total_pixels * bands * bytes_per_pixel) / (1024 * 1024)

        if current_size_mb <= target_memory_mb:
            return max(width, height)

        scale = (target_memory_mb / current_size_mb) ** 0.5
        return int(max(width, height) * scale)

    def read_band(self, band_index: int, max_dimension: int = 4096) -> Optional[NDArray]:
        if not self.dataset or band_index < 1 or band_index > self.dataset.count:
            return None

        try:
            scale = min(1.0, max_dimension / max(self.metadata['width'], self.metadata['height']))

            out_shape = (
                int(self.metadata['height'] * scale),
                int(self.metadata['width'] * scale)
            )

            data = self.dataset.read(
                band_index,
                out_shape=out_shape,
                resampling=rasterio.enums.Resampling.bilinear
            )

            return data

        except Exception as e:
            self.error_occurred.emit(f"Error membaca band: {str(e)}")
            return None

    def get_metadata(self) -> Dict[str, Any]:
        return self.metadata

    def compute_global_statistics(self, sample_dimension: int = 2048) -> Optional[List[Dict[str, float]]]:
        """Compute global statistics for consistent color normalization across tiles"""
        if not self.dataset:
            self.logger.warning("compute_global_statistics called but no dataset loaded")
            return None

        if self.global_statistics is not None:
            self.logger.debug("Returning cached global statistics")
            return self.global_statistics

        with QMutexLocker(self._read_mutex):
            try:
                with PerformanceLogger(self.logger, "Compute global statistics"):
                    # Read a downsampled version to compute statistics efficiently
                    scale = min(1.0, sample_dimension / max(self.metadata['width'], self.metadata['height']))

                    out_shape = (
                        int(self.metadata['height'] * scale),
                        int(self.metadata['width'] * scale)
                    )

                    self.logger.info(f"Computing global statistics from {out_shape[1]}x{out_shape[0]} sample")

                    sample_data = self.dataset.read(
                        out_shape=(self.dataset.count, *out_shape),
                        resampling=rasterio.enums.Resampling.bilinear
                    )

                    # Compute percentiles for each band
                    # PENTING: NoData/piksel kosong (biasanya 0, atau nilai nodata dataset)
                    # harus dibuang dulu sebelum hitung percentile. Kalau tidak, area hitam
                    # di pinggir raster besar (orthomosaic/BigTIFF) bisa mendominasi sample
                    # sehingga p98 ikut ~0 (sama dengan p2). Efeknya rentang stretch jadi
                    # nyaris nol lebar, dan hasil akhirnya jadi biner hitam-putih (bukan
                    # gradasi grayscale) karena hampir semua piksel data asli langsung
                    # ke-clip ke ujung atas (255) sementara NoData tetap di 0.
                    nodata_val = self.dataset.nodata
                    stats = []
                    for band_idx in range(sample_data.shape[0]):
                        band = sample_data[band_idx].astype(np.float32)

                        valid_mask = ~np.isnan(band)
                        if nodata_val is not None:
                            valid_mask &= (band != nodata_val)
                        else:
                            # Tidak ada nodata eksplisit di metadata -> asumsikan 0 adalah
                            # nilai kosong/border, konsisten dengan fallback per-tile.
                            valid_mask &= (band != 0)

                        valid_pixels = band[valid_mask]
                        if valid_pixels.size == 0:
                            # Seluruh sample kosong/nodata -> fallback ke semua piksel
                            # supaya tidak error, walau hasilnya memang tidak informatif.
                            valid_pixels = band

                        p2, p98 = np.percentile(valid_pixels, (2, 98))
                        stats.append({
                            'p2': float(p2),
                            'p98': float(p98),
                            'min': float(np.min(valid_pixels)),
                            'max': float(np.max(valid_pixels))
                        })

                        self.logger.debug(
                            f"Band {band_idx + 1} statistics | "
                            f"p2={p2:.2f}, p98={p98:.2f}, "
                            f"min={np.min(valid_pixels):.2f}, max={np.max(valid_pixels):.2f} | "
                            f"valid pixels: {valid_pixels.size}/{band.size}"
                        )

                    self.global_statistics = stats
                    self.logger.info("Global statistics computed and cached")
                    return stats

            except Exception as e:
                error_msg = f"Error computing statistics: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                self.error_occurred.emit(error_msg)
                return None

    def get_global_statistics(self) -> Optional[List[Dict[str, float]]]:
        """Get cached global statistics, compute if not available"""
        if self.global_statistics is None:
            return self.compute_global_statistics()
        return self.global_statistics

    def close(self) -> None:
        """Close the GDAL dataset safely after active reads finish."""
        with QMutexLocker(self._read_mutex):
            if self.dataset is not None:
                self.logger.info(
                    f"Closing dataset: {Path(self.file_path).name if self.file_path else 'unknown'}"
                )
                self.dataset.close()
                self.dataset = None
            self.file_path = None
            self.metadata = {}
            self.global_statistics = None
            self._ovr_ready = False
            self.logger.debug("Dataset closed and cache cleared")

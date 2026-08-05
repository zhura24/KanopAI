"""Handler untuk fitur eHara: ekstraksi nilai piksel raster (mean per band)
di sekitar titik pusat bounding box / point hasil deteksi.

Alur:
1. Cek apakah ada raster aktif (untuk diekstrak nilainya).
2. Cek apakah ada bounding box / hasil inferensi yang sedang ditampilkan
   (inference_overlay_handler.box_items). Jika kosong -> tampilkan pesan
   agar user import bounding box atau menjalankan inferensi terlebih dahulu.
3. Untuk tiap box: hitung titik pusat (centroid) dalam koordinat piksel,
   konversi ke koordinat geografis menggunakan transform raster, lalu buat
   poligon persegi (buffer) di sekitar titik tersebut sesuai radius yang
   ditentukan user.
4. Untuk tiap band raster, hitung nilai rata-rata piksel di dalam tiap
   poligon persegi (nodata diabaikan).
5. Simpan hasil ke file Excel (.xlsx) yang dipilih user; jika file sudah
   ada, akan langsung di-override (dialog simpan bawaan Qt sudah meminta
   konfirmasi overwrite).
"""

import datetime
import logging

import numpy as np


class EHaraHandler:
    """Handler untuk ekstraksi nilai piksel raster ala eHara."""

    def __init__(self, main_window):
        self.main_window = main_window
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_active_boxes(self):
        """Ambil daftar bounding box aktif (tidak eliminated, visible) dari
        inference_overlay_handler yang sedang menampilkan hasil deteksi."""
        handler = getattr(self.main_window, "inference_overlay_handler", None)
        if not handler or not getattr(handler, "box_items", None):
            return []

        return [
            item for item in handler.box_items
            if item.status != "eliminated" and item.isVisible()
        ]

    def _get_active_dataset(self):
        """Ambil rasterio dataset dari raster aktif, jika ada."""
        loader = getattr(self.main_window, "raster_loader", None)
        if loader is None:
            return None
        return getattr(loader, "dataset", None)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run_extraction(self):
        """Jalankan ekstraksi nilai piksel eHara untuk bounding box yang aktif."""
        from PyQt6.QtWidgets import QMessageBox, QFileDialog, QProgressDialog
        from PyQt6.QtCore import Qt, QCoreApplication
        from shapely.geometry import Polygon
        import rasterio
        from rasterio.mask import mask
        import pandas as pd

        # 1. Cek raster aktif
        dataset = self._get_active_dataset()
        if dataset is None:
            QMessageBox.warning(
                self.main_window,
                "Tidak Ada Raster Aktif",
                "Tidak ada raster aktif. Silakan buka/pilih layer raster terlebih dahulu."
            )
            return

        # 2. Cek bounding box / hasil inferensi yang sedang ditampilkan
        active_boxes = self._get_active_boxes()
        if not active_boxes:
            QMessageBox.warning(
                self.main_window,
                "Tidak Ada Bounding Box",
                "Tidak ada bounding box/hasil inference. Tolong import bounding box "
                "atau jalankan inference."
            )
            return

        # 3. Ambil radius buffer dari panel (default 2.0 m)
        radius = 2.0
        panel = getattr(self.main_window, "ehara_panel", None)
        if panel is not None and hasattr(panel, "spin_ehara_radius"):
            radius = panel.spin_ehara_radius.value()

        # Peringatan jika CRS raster geografis (derajat), radius dalam meter tidak akan valid
        try:
            crs = dataset.crs
            if crs is not None and crs.is_geographic:
                reply = QMessageBox.question(
                    self.main_window,
                    "CRS Geografis Terdeteksi",
                    "Raster ini menggunakan CRS geografis (satuan derajat), bukan meter.\n"
                    f"Radius buffer {radius} akan diterapkan sebagai derajat, bukan meter, "
                    "sehingga hasilnya kemungkinan tidak sesuai.\n\n"
                    "Lanjutkan proses ekstraksi?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.No:
                    return
        except Exception as e:
            self.logger.debug(f"Gagal cek tipe CRS: {e}")

        transform = dataset.transform
        band_count = dataset.count
        nodata_value = dataset.nodata if dataset.nodata is not None else -9999

        # 4. Bangun poligon persegi untuk tiap box (dari titik pusat bbox)
        ids = []
        eastings = []
        northings = []
        polygons = []

        for item in active_boxes:
            x1, y1, x2, y2 = item.get_box_coords()
            px_center = (x1 + x2) / 2.0
            py_center = (y1 + y2) / 2.0

            # Konversi pixel -> koordinat geografis (mengikuti konvensi yang
            # sudah dipakai di centroid_handler.convert_to_centroids)
            gx, gy = transform * (px_center, py_center)

            y_plus = gy + radius
            y_minus = gy - radius
            x_plus = gx + radius
            x_minus = gx - radius

            polygon = Polygon([
                (x_minus, y_plus),
                (x_plus, y_plus),
                (x_plus, y_minus),
                (x_minus, y_minus),
                (x_minus, y_plus),
            ])

            ids.append(item.box_id)
            eastings.append(gx)
            northings.append(gy)
            polygons.append(polygon)

        # 5. Progress dialog karena proses per band bisa agak lama
        progress = QProgressDialog(
            "Mengekstrak nilai piksel...", "Batal", 0, band_count, self.main_window
        )
        progress.setWindowTitle("eHara - Ekstraksi Piksel")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        QCoreApplication.processEvents()

        start_time = datetime.datetime.now()

        result_data = {
            "ID": ids,
            "Easting": eastings,
            "Northing": northings,
        }

        try:
            for band_idx in range(1, band_count + 1):
                if progress.wasCanceled():
                    self.logger.info("Ekstraksi eHara dibatalkan oleh user")
                    return

                mean_values = []
                for geom in polygons:
                    try:
                        out_image, _ = mask(
                            dataset, [geom], crop=True, indexes=band_idx, all_touched=True
                        )
                        data = out_image[0]
                        data = np.where(data == nodata_value, np.nan, data)
                        mean_values.append(float(np.nanmean(data)))
                    except Exception as e:
                        self.logger.warning(f"Gagal memproses poligon band {band_idx}: {e}")
                        mean_values.append(np.nan)

                result_data[f"Band_{band_idx}_mean"] = mean_values

                progress.setValue(band_idx)
                QCoreApplication.processEvents()

        except Exception as e:
            progress.close()
            self.logger.error(f"Ekstraksi eHara gagal: {e}", exc_info=True)
            QMessageBox.critical(
                self.main_window, "Error", f"Ekstraksi piksel gagal:\n{e}"
            )
            return

        progress.close()

        df_result = pd.DataFrame(result_data)

        # 6. Pilih lokasi simpan (override jika file sudah ada)
        output_path, _ = QFileDialog.getSaveFileName(
            self.main_window,
            "Simpan Hasil Ekstraksi eHara",
            "Pixel_Extraction.xlsx",
            "Excel Files (*.xlsx)"
        )
        if not output_path:
            return
        if not output_path.lower().endswith(".xlsx"):
            output_path += ".xlsx"

        try:
            df_result.to_excel(output_path, index=False)
        except Exception as e:
            self.logger.error(f"Gagal menyimpan hasil eHara: {e}", exc_info=True)
            QMessageBox.critical(
                self.main_window, "Error", f"Gagal menyimpan file Excel:\n{e}"
            )
            return

        elapsed = datetime.datetime.now() - start_time
        self.logger.info(
            f"Ekstraksi eHara selesai | {len(polygons)} titik | {band_count} band | "
            f"Disimpan di {output_path} | Waktu: {elapsed}"
        )

        QMessageBox.information(
            self.main_window,
            "Ekstraksi Selesai",
            f"Ekstraksi piksel eHara selesai.\n\n"
            f"Titik diproses: {len(polygons)}\n"
            f"Band: {band_count}\n"
            f"Disimpan di:\n{output_path}\n\n"
            f"Waktu proses: {elapsed}"
        )

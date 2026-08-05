"""Handler for the eHara feature: raster pixel value extraction (mean per
band) around the center point of the bounding box / detection point.

Flow:
1. Check whether there is an active raster (to extract values from).
2. Check whether there are bounding boxes / inference results currently
   shown (inference_overlay_handler.box_items). If empty -> show a message
   telling the user to import bounding boxes or run inference first.
3. For each box: compute the center point (centroid) in pixel coordinates,
   convert it to geographic coordinates using the raster transform, then
   build a square polygon (buffer) around that point based on the radius
   the user specified.
4. For each raster band, compute the mean pixel value inside each square
   polygon (nodata is ignored).
5. Save the result to the Excel (.xlsx) file chosen by the user; if the
   file already exists it will be overwritten directly (Qt's built-in save
   dialog already asks for overwrite confirmation).
"""

import datetime
import logging

import numpy as np


class EHaraHandler:
    """Handler for eHara-style raster pixel value extraction."""

    def __init__(self, main_window):
        self.main_window = main_window
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_active_boxes(self):
        """Get the list of active bounding boxes (not eliminated, visible)
        from the inference_overlay_handler currently showing detection
        results."""
        handler = getattr(self.main_window, "inference_overlay_handler", None)
        if not handler or not getattr(handler, "box_items", None):
            return []

        return [
            item for item in handler.box_items
            if item.status != "eliminated" and item.isVisible()
        ]

    def _get_active_dataset(self):
        """Get the rasterio dataset from the active raster, if any."""
        loader = getattr(self.main_window, "raster_loader", None)
        if loader is None:
            return None
        return getattr(loader, "dataset", None)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run_extraction(self):
        """Run eHara pixel value extraction for the active bounding boxes."""
        from PyQt6.QtWidgets import QMessageBox, QFileDialog, QProgressDialog
        from PyQt6.QtCore import Qt, QCoreApplication
        from shapely.geometry import Polygon
        import rasterio
        from rasterio.mask import mask
        import pandas as pd

        # 1. Check for an active raster
        dataset = self._get_active_dataset()
        if dataset is None:
            QMessageBox.warning(
                self.main_window,
                "No Active Raster",
                "There is no active raster. Please open/select a raster layer first."
            )
            return

        # 2. Check for bounding boxes / inference results currently shown
        active_boxes = self._get_active_boxes()
        if not active_boxes:
            QMessageBox.warning(
                self.main_window,
                "No Bounding Boxes",
                "There are no bounding boxes/inference results. Please import "
                "bounding boxes or run inference."
            )
            return

        # 3. Get the buffer radius from the panel (default 2.0 m)
        radius = 2.0
        panel = getattr(self.main_window, "ehara_panel", None)
        if panel is not None and hasattr(panel, "spin_ehara_radius"):
            radius = panel.spin_ehara_radius.value()

        # Warn if the raster CRS is geographic (degrees), the radius in meters won't be valid
        try:
            crs = dataset.crs
            if crs is not None and crs.is_geographic:
                reply = QMessageBox.question(
                    self.main_window,
                    "Geographic CRS Detected",
                    "This raster uses a geographic CRS (degree units), not meters.\n"
                    f"The buffer radius {radius} will be applied as degrees, not meters, "
                    "so the result may be inaccurate.\n\n"
                    "Continue with extraction anyway?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.No:
                    return
        except Exception as e:
            self.logger.debug(f"Failed to check CRS type: {e}")

        transform = dataset.transform
        band_count = dataset.count
        nodata_value = dataset.nodata if dataset.nodata is not None else -9999

        # 4. Build a square polygon for each box (from the bbox center point)
        ids = []
        eastings = []
        northings = []
        polygons = []

        for item in active_boxes:
            x1, y1, x2, y2 = item.get_box_coords()
            px_center = (x1 + x2) / 2.0
            py_center = (y1 + y2) / 2.0

            # Convert pixel -> geographic coordinates (following the same
            # convention already used in centroid_handler.convert_to_centroids)
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

        # 5. Progress dialog since the per-band process can take a while
        progress = QProgressDialog(
            "Extracting pixel values...", "Cancel", 0, band_count, self.main_window
        )
        progress.setWindowTitle("eHara - Pixel Extraction")
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
                    self.logger.info("eHara extraction canceled by user")
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
                        self.logger.warning(f"Failed to process polygon for band {band_idx}: {e}")
                        mean_values.append(np.nan)

                result_data[f"Band_{band_idx}_mean"] = mean_values

                progress.setValue(band_idx)
                QCoreApplication.processEvents()

        except Exception as e:
            progress.close()
            self.logger.error(f"eHara extraction failed: {e}", exc_info=True)
            QMessageBox.critical(
                self.main_window, "Error", f"Pixel extraction failed:\n{e}"
            )
            return

        progress.close()

        df_result = pd.DataFrame(result_data)

        # 6. Choose save location (overwrite if the file already exists)
        output_path, _ = QFileDialog.getSaveFileName(
            self.main_window,
            "Save eHara Extraction Result",
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
            self.logger.error(f"Failed to save eHara result: {e}", exc_info=True)
            QMessageBox.critical(
                self.main_window, "Error", f"Failed to save Excel file:\n{e}"
            )
            return

        elapsed = datetime.datetime.now() - start_time
        self.logger.info(
            f"eHara extraction complete | {len(polygons)} points | {band_count} bands | "
            f"Saved to {output_path} | Time: {elapsed}"
        )

        QMessageBox.information(
            self.main_window,
            "Extraction Complete",
            f"eHara pixel extraction complete.\n\n"
            f"Points processed: {len(polygons)}\n"
            f"Bands: {band_count}\n"
            f"Saved to:\n{output_path}\n\n"
            f"Processing time: {elapsed}"
        )

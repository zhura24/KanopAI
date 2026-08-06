"""Handler for the eHara feature: raster pixel value extraction (mean per
band) around the center point of the bounding box / detection point, plus
NDVI/GNDVI/SR calculation and optional N/P/K/Mg leaf nutrient prediction.

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
5. Using the three band indices chosen by the user (Band 1/2/3, mapped to
   NDVI/GNDVI/SR formulas — see core.hara_regression), compute NDVI, GNDVI
   and SR for each point.
6. If the user has loaded a training Excel dataset (historical ground
   truth: ID, X, Y, N, P, K, Mg, band1, band2, band3, NDVI, GNDVI, SR), fit
   a PCA + Linear Regression calibration on the fly and predict N/P/K/Mg
   leaf nutrient content for each extracted point.
7. Save the result to the Excel (.xlsx) file chosen by the user; if the
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

        from core.hara_regression import (
            calculate_ndvi, calculate_gndvi, calculate_sr,
            load_training_data, HaraRegressionModel, HaraRegressionError,
        )

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

        # 3. Get the buffer radius + band indices + training data from the panel
        radius = 2.0
        band1_idx, band2_idx, band3_idx = 1, 2, 3
        training_data_path = None
        ndvi_enabled = True    # default: compute spectral indices
        training_enabled = True  # default: run nutrient prediction
        panel = getattr(self.main_window, "ehara_panel", None)
        if panel is not None:
            if hasattr(panel, "spin_ehara_radius"):
                radius = panel.spin_ehara_radius.value()
            # Only read band indices when the NDVI section is enabled
            ndvi_enabled = getattr(panel, "_ndvi_enabled", True)
            if ndvi_enabled:
                if hasattr(panel, "spin_ehara_band1"):
                    band1_idx = panel.spin_ehara_band1.value()
                if hasattr(panel, "spin_ehara_band2"):
                    band2_idx = panel.spin_ehara_band2.value()
                if hasattr(panel, "spin_ehara_band3"):
                    band3_idx = panel.spin_ehara_band3.value()
            # Only use training data when the training section is enabled
            training_enabled = getattr(panel, "_training_enabled", True)
            if training_enabled:
                training_data_path = getattr(panel, "training_data_path", None)

        band_count = dataset.count

        # Validate the chosen band indices against the actual raster — only
        # when the NDVI section is active (indices are needed only then).
        if ndvi_enabled:
            for label, idx in (("Band 1", band1_idx), ("Band 2", band2_idx), ("Band 3", band3_idx)):
                if idx < 1 or idx > band_count:
                    QMessageBox.warning(
                        self.main_window,
                        "Invalid Band Index",
                        f"{label} index ({idx}) is out of range for this raster, "
                        f"which has {band_count} band(s). Please adjust the band "
                        "index in the eHara panel."
                    )
                    return

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

        # 4. If training data was provided (and the training section is enabled),
        # load + validate it up-front so we fail fast before doing the
        # (potentially slow) per-band extraction below.
        hara_model = None
        if training_enabled and training_data_path:
            try:
                training_df = load_training_data(training_data_path)
                dropped = training_df.attrs.get("rows_dropped", 0)
                if dropped:
                    self.logger.warning(
                        f"eHara training data: dropped {dropped} row(s) with "
                        "missing/non-numeric values."
                    )
                hara_model = HaraRegressionModel(training_df)
            except HaraRegressionError as e:
                QMessageBox.critical(
                    self.main_window, "Training Data Error",
                    f"Could not use the loaded training data:\n{e}\n\n"
                    "Extraction will continue without N/P/K/Mg prediction."
                )
                hara_model = None
            except Exception as e:
                self.logger.error(f"Failed to fit eHara regression model: {e}", exc_info=True)
                QMessageBox.critical(
                    self.main_window, "Training Data Error",
                    f"Unexpected error while fitting the nutrient regression model:\n{e}\n\n"
                    "Extraction will continue without N/P/K/Mg prediction."
                )
                hara_model = None

        transform = dataset.transform
        nodata_value = dataset.nodata if dataset.nodata is not None else -9999

        # 5. Build a square polygon for each box (from the bbox center point)
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

        # 6. Progress dialog since the per-band process can take a while
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

        # 7. Compute NDVI/GNDVI/SR — only when the spectral index section is
        # enabled.  When disabled the Excel output only contains the band mean
        # columns (ID, Easting, Northing, Band_1_mean, Band_2_mean, …).
        if ndvi_enabled:
            df_result["band1"] = df_result[f"Band_{band1_idx}_mean"]
            df_result["band2"] = df_result[f"Band_{band2_idx}_mean"]
            df_result["band3"] = df_result[f"Band_{band3_idx}_mean"]
            df_result["NDVI"] = calculate_ndvi(df_result["band1"], df_result["band3"])
            df_result["GNDVI"] = calculate_gndvi(df_result["band2"], df_result["band3"])
            df_result["SR"] = calculate_sr(df_result["band1"], df_result["band3"])

        # 8. Optional N/P/K/Mg nutrient prediction — only when the training
        # section is enabled AND a model was successfully fitted.
        if training_enabled and hara_model is not None:
            try:
                nutrient_df = hara_model.predict(df_result)
                df_result = pd.concat([df_result, nutrient_df], axis=1)
            except HaraRegressionError as e:
                QMessageBox.warning(
                    self.main_window, "Nutrient Prediction Skipped",
                    f"Could not predict N/P/K/Mg for the extracted points:\n{e}\n\n"
                    "The Excel file will still be saved without those columns."
                )
            except Exception as e:
                self.logger.error(f"eHara nutrient prediction failed: {e}", exc_info=True)
                QMessageBox.warning(
                    self.main_window, "Nutrient Prediction Skipped",
                    f"Unexpected error while predicting N/P/K/Mg:\n{e}\n\n"
                    "The Excel file will still be saved without those columns."
                )

        # 9. Choose save location (overwrite if the file already exists)
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
        nutrient_note = " with N/P/K/Mg prediction" if hara_model is not None and \
            any(c.endswith("Leaf (%)") for c in df_result.columns) else ""
        self.logger.info(
            f"eHara extraction complete{nutrient_note} | {len(polygons)} points | "
            f"{band_count} bands | Saved to {output_path} | Time: {elapsed}"
        )

        QMessageBox.information(
            self.main_window,
            "Extraction Complete",
            f"eHara pixel extraction complete{nutrient_note}.\n\n"
            f"Points processed: {len(polygons)}\n"
            f"Bands: {band_count}\n"
            f"Saved to:\n{output_path}\n\n"
            f"Processing time: {elapsed}"
        )

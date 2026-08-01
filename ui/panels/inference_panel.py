"""YOLO Multispectral Inference Panel.

Collapsible panel to configure multispectral inference parameters, select a model,
load band_stats metadata, run MultispectralInferenceWorker in a background thread,
show real-time feedback (progress, ETA, console log), and provide bounding box
correction plus shapefile export for raw and corrected detections.
"""

from typing import Optional, Dict, Any, List
import logging
import shutil
import sys
from pathlib import Path
import time

from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QWidget,
    QFormLayout, QSpinBox, QDoubleSpinBox, QCheckBox, QLineEdit,
    QProgressBar, QPlainTextEdit, QFileDialog, QMessageBox, QGroupBox,
    QScrollArea, QComboBox, QInputDialog, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QColor, QFont, QCursor

from ui.widgets.collapsible_box import CollapsibleBox
from core.multispectral_worker import MultispectralInferenceWorker
from core.inference_engine import InferenceResult, register_model_in_sqlite, load_model_from_sqlite, list_models_in_sqlite
from ui.dialogs.band_mismatch_dialog import resolve_band_matching
from ui.dialogs.correction_metadata_dialog import request_correction_metadata


class InferencePanel(CollapsibleBox):
    """Collapsible panel for the Multispectral Detector module."""

    def __init__(self, main_window: Any) -> None:
        super().__init__("Multispectral Detector")
        self.main_window = main_window
        self.logger = logging.getLogger(__name__)

        self._worker: Optional[MultispectralInferenceWorker] = None
        self._last_result: Optional[InferenceResult] = None

        # State paths
        self.model_path: Optional[str] = None
        self.band_stats_path: Optional[str] = None
        self.model_description: str = ""
        self._manual_band_mapping: Optional[Dict[int, int]] = None
        self._enable_adaptive_fallback: bool = False
        self._model_loaded_from_db: bool = False
        self.correction_reviewer_name: str = ""
        self.correction_date: str = ""

        # AOI/Exclude polygon selection (from drawn polygons)
        self._aoi_polygon_ids: List[int] = []
        self._exclude_polygon_ids: List[int] = []

        # SQLite database path
        self.db_path = str(self._get_database_path())

        # Timer untuk kalkulasi ETA
        self._start_time: float = 0.0
        self._timer_eta = QTimer(self)
        self._timer_eta.timeout.connect(self._update_eta_label)

        self.init_ui()

    def _get_database_path(self) -> Path:
        """Use a database beside the app, with a writable-user fallback."""
        if getattr(sys, "frozen", False):
            app_dir = Path(sys.executable).resolve().parent
        else:
            app_dir = Path(__file__).resolve().parents[2]

        app_db_dir = app_dir / "data"
        app_db_path = app_db_dir / "kanopai_models.db"
        legacy_db_path = Path.home() / ".kanopai" / "kanopai_models.db"
        bundled_db_path = (
            Path(getattr(sys, "_MEIPASS", app_dir))
            / "data"
            / "kanopai_models.db"
        )

        try:
            app_db_dir.mkdir(parents=True, exist_ok=True)
            if not app_db_path.exists():
                if bundled_db_path.exists():
                    shutil.copy2(bundled_db_path, app_db_path)
                elif legacy_db_path.exists():
                    shutil.copy2(legacy_db_path, app_db_path)
            return app_db_path
        except OSError as error:
            self.logger.warning(
                "App folder is not writable; using user database directory: %s", error
            )
            legacy_db_path.parent.mkdir(parents=True, exist_ok=True)
            return legacy_db_path

    def init_ui(self) -> None:
        """Initialize UI components for the multispectral detector panel."""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(10)

        # Style helper
        section_style = "QLabel { color: #4CAF50; font-weight: bold; font-size: 11px; margin-top: 4px; }"
        info_sub_style = "QLabel { color: #aaa; font-size: 10px; font-family: monospace; }"

        # =========================================================================
        # 1. MODEL SELECTION (.pt & band_stats.json)
        # =========================================================================
        model_group = QGroupBox("Multispectral Model (.pt)")
        model_group.setStyleSheet("QGroupBox { color: #ddd; font-weight: bold; font-size: 11px; border: 1px solid #444; border-radius: 4px; margin-top: 4px; padding-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 8px; }")
        model_layout = QVBoxLayout(model_group)
        model_layout.setSpacing(6)

        btn_model_layout = QHBoxLayout()
        btn_model_layout.setSpacing(4)

        # SQLite model management
        db_layout = QHBoxLayout()
        db_layout.setSpacing(4)

        self.combo_db_models = QComboBox()
        self.combo_db_models.setPlaceholderText("Load model from SQLite database...")
        self.combo_db_models.setStyleSheet("QComboBox { font-size: 10px; background-color: #1e1e1e; color: #ccc; }")
        db_layout.addWidget(self.combo_db_models, 3)

        self.btn_load_db_model = QPushButton("Load from DB")
        self.btn_load_db_model.setMinimumHeight(26)
        self.btn_load_db_model.setToolTip("Load the selected model directly from the SQLite database for inference.")
        self.btn_load_db_model.clicked.connect(self._load_model_from_db)
        db_layout.addWidget(self.btn_load_db_model)

        self.btn_import_db_model = QPushButton("Import to DB")
        self.btn_import_db_model.setMinimumHeight(26)
        self.btn_import_db_model.clicked.connect(self._import_model_to_db)
        db_layout.addWidget(self.btn_import_db_model)

        model_layout.addLayout(db_layout)

        self.lbl_model_file = QLabel("Model: None selected")
        self.lbl_model_file.setWordWrap(True)
        self.lbl_model_file.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.lbl_model_file.setStyleSheet(info_sub_style)
        model_layout.addWidget(self.lbl_model_file)

        self.lbl_stats_file = QLabel("Band Stats: None selected")
        self.lbl_stats_file.setWordWrap(True)
        self.lbl_stats_file.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.lbl_stats_file.setStyleSheet(info_sub_style)
        model_layout.addWidget(self.lbl_stats_file)

        self.lbl_model_meta = QLabel("Info: Load .pt and stats to view metadata")
        self.lbl_model_meta.setWordWrap(True)
        self.lbl_model_meta.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)
        self.lbl_model_meta.setStyleSheet("QLabel { color: #38bdf8; font-size: 11px; padding: 8px; background-color: #141d2e; border-radius: 4px; }")
        self.lbl_model_meta.setMinimumHeight(80)
        model_layout.addWidget(self.lbl_model_meta)

        main_layout.addWidget(model_group)

        # =========================================================================
        # 3. PARAMETER CONTROLS
        # =========================================================================
        param_label = QLabel("Inference Parameters:")
        param_label.setStyleSheet(section_style)
        main_layout.addWidget(param_label)

        param_form = QFormLayout()
        param_form.setContentsMargins(2, 2, 2, 2)
        param_form.setSpacing(6)

        self.spin_confidence = QDoubleSpinBox()
        self.spin_confidence.setRange(0.01, 1.00)
        self.spin_confidence.setSingleStep(0.05)
        self.spin_confidence.setValue(0.25)
        self.spin_confidence.setToolTip("Confidence threshold for object detection (0.01 - 1.00)")
        param_form.addRow("Confidence:", self.spin_confidence)

        self.spin_iou = QDoubleSpinBox()
        self.spin_iou.setRange(0.01, 1.00)
        self.spin_iou.setSingleStep(0.05)
        self.spin_iou.setValue(0.50)
        self.spin_iou.setToolTip("Intersection over Union (IoU) threshold for NMS deduplication")
        param_form.addRow("IoU Threshold:", self.spin_iou)

        self.spin_tile_size = QSpinBox()
        self.spin_tile_size.setRange(64, 2048)
        self.spin_tile_size.setSingleStep(64)
        self.spin_tile_size.setValue(640)
        self.spin_tile_size.setToolTip("Sub-image tile size in pixels (e.g., 640)")
        param_form.addRow("Tile Size (px):", self.spin_tile_size)

        self.spin_overlap = QSpinBox()
        self.spin_overlap.setRange(0, 512)
        self.spin_overlap.setSingleStep(16)
        self.spin_overlap.setValue(64)
        self.spin_overlap.setToolTip("Tile boundary overlap width in pixels")
        param_form.addRow("Tile Overlap (px):", self.spin_overlap)

        self.spin_batch_size = QSpinBox()
        self.spin_batch_size.setRange(1, 64)
        self.spin_batch_size.setValue(4)
        self.spin_batch_size.setToolTip("Batch size for GPU/CPU parallel inference")
        param_form.addRow("Batch Size:", self.spin_batch_size)

        main_layout.addLayout(param_form)

        # =========================================================================
        # 4. ADVANCED BACKEND CONTROLS (AOI, Exclude, Fallback, Output)
        # =========================================================================
        self.adv_box = CollapsibleBox("Advanced Backend Options")
        adv_layout = QVBoxLayout()
        adv_layout.setContentsMargins(4, 4, 4, 4)
        adv_layout.setSpacing(6)

        # AOI Shapefile / Polygon
        aoi_header = QLabel("AOI (Area of Interest):")
        aoi_header.setStyleSheet("QLabel { color: #94a3b8; font-size: 10px; }")
        adv_layout.addWidget(aoi_header)

        aoi_layout = QHBoxLayout()
        self.txt_aoi_path = QLineEdit()
        self.txt_aoi_path.setPlaceholderText("AOI Shapefile (*.shp) - Optional")
        self.txt_aoi_path.setStyleSheet("QLineEdit { font-size: 10px; background-color: #1e1e1e; border: 1px solid #444; color: #ccc; }")
        btn_aoi = QPushButton("...")
        btn_aoi.setFixedWidth(24)
        btn_aoi.clicked.connect(self._browse_aoi_shp)
        btn_aoi_polygon = QPushButton("From Polygon")
        btn_aoi_polygon.setFixedHeight(24)
        btn_aoi_polygon.setToolTip("Use a polygon already drawn in the left-side polygon tool")
        btn_aoi_polygon.clicked.connect(lambda: self._select_polygon_for("aoi"))
        aoi_layout.addWidget(self.txt_aoi_path)
        aoi_layout.addWidget(btn_aoi)
        aoi_layout.addWidget(btn_aoi_polygon)
        adv_layout.addLayout(aoi_layout)

        self.lbl_aoi_polygon = QLabel("Polygon AOI: None")
        self.lbl_aoi_polygon.setStyleSheet("QLabel { color: #64748b; font-size: 9px; }")
        adv_layout.addWidget(self.lbl_aoi_polygon)

        # Exclude Shapefile / Polygon
        exclude_header = QLabel("Exclude Zone:")
        exclude_header.setStyleSheet("QLabel { color: #94a3b8; font-size: 10px; }")
        adv_layout.addWidget(exclude_header)

        exclude_layout = QHBoxLayout()
        self.txt_exclude_path = QLineEdit()
        self.txt_exclude_path.setPlaceholderText("Exclude Zone Shapefile (*.shp) - Optional")
        self.txt_exclude_path.setStyleSheet("QLineEdit { font-size: 10px; background-color: #1e1e1e; border: 1px solid #444; color: #ccc; }")
        btn_exclude = QPushButton("...")
        btn_exclude.setFixedWidth(24)
        btn_exclude.clicked.connect(self._browse_exclude_shp)
        btn_exclude_polygon = QPushButton("From Polygon")
        btn_exclude_polygon.setFixedHeight(24)
        btn_exclude_polygon.setToolTip("Use a polygon already drawn in the left-side polygon tool")
        btn_exclude_polygon.clicked.connect(lambda: self._select_polygon_for("exclude"))
        exclude_layout.addWidget(self.txt_exclude_path)
        exclude_layout.addWidget(btn_exclude)
        exclude_layout.addWidget(btn_exclude_polygon)
        adv_layout.addLayout(exclude_layout)

        self.lbl_exclude_polygon = QLabel("Polygon Exclude: None")
        self.lbl_exclude_polygon.setStyleSheet("QLabel { color: #64748b; font-size: 9px; }")
        adv_layout.addWidget(self.lbl_exclude_polygon)

        # Adaptive Fallback
        self.chk_adaptive_fallback = QCheckBox("Enable Adaptive Band Fallback")
        self.chk_adaptive_fallback.setToolTip("Auto-fallback for non-standard sensor band order")
        self.chk_adaptive_fallback.setStyleSheet("QCheckBox { color: #ccc; font-size: 10px; }")
        adv_layout.addWidget(self.chk_adaptive_fallback)

        # Output Dir Override
        out_layout = QHBoxLayout()
        self.txt_output_dir = QLineEdit()
        self.txt_output_dir.setPlaceholderText("Output Dir (Default: auto output subfolder)")
        self.txt_output_dir.setStyleSheet("QLineEdit { font-size: 10px; background-color: #1e1e1e; border: 1px solid #444; color: #ccc; }")
        btn_out_dir = QPushButton("...")
        btn_out_dir.setFixedWidth(24)
        btn_out_dir.clicked.connect(self._browse_output_dir)
        out_layout.addWidget(self.txt_output_dir)
        out_layout.addWidget(btn_out_dir)
        adv_layout.addLayout(out_layout)

        # Custom Output Name
        self.txt_output_name = QLineEdit()
        self.txt_output_name.setPlaceholderText("Custom Output Name Stem (Optional)")
        self.txt_output_name.setStyleSheet("QLineEdit { font-size: 10px; background-color: #1e1e1e; border: 1px solid #444; color: #ccc; }")
        adv_layout.addWidget(self.txt_output_name)

        self.adv_box.setContentLayout(adv_layout)
        main_layout.addWidget(self.adv_box)

        # =========================================================================
        # 5. EXECUTION CONTROLS & REAL-TIME FEEDBACK
        # =========================================================================
        exec_label = QLabel("Execution & Progress:")
        exec_label.setStyleSheet(section_style)
        main_layout.addWidget(exec_label)

        btn_exec_layout = QHBoxLayout()
        btn_exec_layout.setSpacing(6)

        self.btn_run = QPushButton("Run Inference")
        self.btn_run.setEnabled(False)
        self.btn_run.setMinimumHeight(34)
        self.btn_run.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_run.setStyleSheet("""
            QPushButton {
                background-color: #16a34a;
                color: white;
                font-weight: bold;
                font-size: 12px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #15803d; }
            QPushButton:disabled { background-color: #334155; color: #64748b; }
        """)
        self.btn_run.clicked.connect(self.run_inference)
        btn_exec_layout.addWidget(self.btn_run, 3)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setVisible(False)
        self.btn_cancel.setMinimumHeight(34)
        self.btn_cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_cancel.setStyleSheet("""
            QPushButton {
                background-color: #dc2626;
                color: white;
                font-weight: bold;
                font-size: 11px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #b91c1c; }
            QPushButton:disabled { background-color: #450a0a; color: #7f1d1d; }
        """)
        self.btn_cancel.clicked.connect(self.cancel_inference)
        btn_exec_layout.addWidget(self.btn_cancel, 1)

        main_layout.addLayout(btn_exec_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(18)
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #444;
                border-radius: 3px;
                background-color: #1e1e1e;
                text-align: center;
                color: #fff;
                font-size: 10px;
                font-weight: bold;
            }
            QProgressBar::chunk { background-color: #2563eb; border-radius: 2px; }
        """)
        main_layout.addWidget(self.progress_bar)

        # Progress info stats
        self.lbl_progress_stats = QLabel("Tile: 0 / 0 | Detections: 0")
        self.lbl_progress_stats.setStyleSheet("QLabel { color: #a3e635; font-size: 10px; font-weight: bold; }")
        main_layout.addWidget(self.lbl_progress_stats)

        self.lbl_eta = QLabel("Elapsed: 00:00 | ETA: --:--")
        self.lbl_eta.setStyleSheet("QLabel { color: #94a3b8; font-size: 10px; }")
        main_layout.addWidget(self.lbl_eta)

        # Real-time log terminal
        log_header = QLabel("Inference Console Log:")
        log_header.setStyleSheet("QLabel { color: #888; font-size: 10px; margin-top: 4px; }")
        main_layout.addWidget(log_header)

        self.log_terminal = QPlainTextEdit()
        self.log_terminal.setReadOnly(True)
        self.log_terminal.setMaximumHeight(110)
        self.log_terminal.setStyleSheet("""
            QPlainTextEdit {
                background-color: #0f172a;
                color: #4ade80;
                font-family: Consolas, Monospace, Courier;
                font-size: 9px;
                border: 1px solid #1e293b;
                border-radius: 3px;
            }
        """)
        main_layout.addWidget(self.log_terminal)

        # =========================================================================
        # 6. VECTOR CORRECTION TOOLS & EXPORT SYSTEM
        # =========================================================================
        corr_label = QLabel("Manual Box Correction & Export:")
        corr_label.setStyleSheet(section_style)
        main_layout.addWidget(corr_label)

        self.lbl_summary = QLabel("Results: No active detection session")
        self.lbl_summary.setWordWrap(True)
        self.lbl_summary.setStyleSheet("QLabel { color: #f59e0b; font-size: 10px; font-weight: bold; }")
        main_layout.addWidget(self.lbl_summary)

        # Vector Tool Buttons
        tools_layout = QHBoxLayout()
        tools_layout.setSpacing(4)

        self.btn_add_box = QPushButton("Add Box")
        self.btn_add_box.setCheckable(True)
        self.btn_add_box.setEnabled(False)
        self.btn_add_box.setMinimumHeight(28)
        self.btn_add_box.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_add_box.setStyleSheet("QPushButton { background-color: #334155; color: #e2e8f0; font-size: 10px; border-radius: 3px; } QPushButton:checked { background-color: #2563eb; font-weight: bold; color: white; }")
        self.btn_add_box.clicked.connect(self._toggle_add_box_mode)
        tools_layout.addWidget(self.btn_add_box)

        self.btn_edit_box = QPushButton("Select/Edit")
        self.btn_edit_box.setCheckable(True)
        self.btn_edit_box.setEnabled(False)
        self.btn_edit_box.setMinimumHeight(28)
        self.btn_edit_box.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_edit_box.setStyleSheet("QPushButton { background-color: #334155; color: #e2e8f0; font-size: 10px; border-radius: 3px; } QPushButton:checked { background-color: #d97706; font-weight: bold; color: white; }")
        self.btn_edit_box.clicked.connect(self._toggle_edit_box_mode)
        tools_layout.addWidget(self.btn_edit_box)

        self.btn_delete_box = QPushButton("Delete Box")
        self.btn_delete_box.setEnabled(False)
        self.btn_delete_box.setMinimumHeight(28)
        self.btn_delete_box.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_delete_box.setStyleSheet("QPushButton { background-color: #991b1b; color: white; font-size: 10px; border-radius: 3px; } QPushButton:hover { background-color: #b91c1c; } QPushButton:disabled { background-color: #450a0a; color: #7f1d1d; }")
        self.btn_delete_box.clicked.connect(self._delete_selected_box)
        tools_layout.addWidget(self.btn_delete_box)

        main_layout.addLayout(tools_layout)

        self.lbl_box_info = QLabel("Box Info: Select 'Edit' then click a box")
        self.lbl_box_info.setWordWrap(True)
        self.lbl_box_info.setMinimumHeight(60)
        self.lbl_box_info.setStyleSheet("QLabel { color: #38bdf8; font-size: 11px; font-family: monospace; padding: 8px; background-color: #0f172a; border-radius: 3px; }")
        main_layout.addWidget(self.lbl_box_info)

        # Undo / Redo Row
        undo_layout = QHBoxLayout()
        undo_layout.setSpacing(4)

        self.btn_undo = QPushButton("↺ Undo")
        self.btn_undo.setEnabled(False)
        self.btn_undo.setMinimumHeight(24)
        self.btn_undo.clicked.connect(self._undo_box_action)
        self.btn_undo.setStyleSheet("QPushButton { background-color: #1e293b; color: #94a3b8; font-size: 10px; border: 1px solid #334155; border-radius: 3px; } QPushButton:enabled { color: #f1f5f9; }")
        undo_layout.addWidget(self.btn_undo)

        self.btn_redo = QPushButton("↻ Redo")
        self.btn_redo.setEnabled(False)
        self.btn_redo.setMinimumHeight(24)
        self.btn_redo.clicked.connect(self._redo_box_action)
        self.btn_redo.setStyleSheet("QPushButton { background-color: #1e293b; color: #94a3b8; font-size: 10px; border: 1px solid #334155; border-radius: 3px; } QPushButton:enabled { color: #f1f5f9; }")
        undo_layout.addWidget(self.btn_redo)

        main_layout.addLayout(undo_layout)

        # Export Buttons System (2 shapefiles: raw & corrected)
        self.btn_export_shapefiles = QPushButton("Export Shapefiles (Raw, Corrected & Metrics)")
        self.btn_export_shapefiles.setEnabled(False)
        self.btn_export_shapefiles.setMinimumHeight(32)
        self.btn_export_shapefiles.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_export_shapefiles.setStyleSheet("""
            QPushButton {
                background-color: #7c3aed;
                color: white;
                font-weight: bold;
                font-size: 11px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #6d28d9; }
            QPushButton:disabled { background-color: #3b0764; color: #7e22ce; }
        """)
        self.btn_export_shapefiles.clicked.connect(self.export_shapefiles)
        main_layout.addWidget(self.btn_export_shapefiles)

        # Convert to Centroids button
        self.btn_convert_to_centroids = QPushButton("Convert to Centroids")
        self.btn_convert_to_centroids.setEnabled(False)
        self.btn_convert_to_centroids.setMinimumHeight(32)
        self.btn_convert_to_centroids.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_convert_to_centroids.setToolTip(
            "Convert bounding boxes from Multispectral Detector to centroid points (Centroid Detector)"
        )
        self.btn_convert_to_centroids.setStyleSheet("""
            QPushButton {
                background-color: #0f766e;
                color: white;
                font-weight: bold;
                font-size: 11px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #0d9488; }
            QPushButton:disabled { background-color: #134e4a; color: #2d6a65; }
        """)
        self.btn_convert_to_centroids.clicked.connect(self._convert_to_centroids)
        main_layout.addWidget(self.btn_convert_to_centroids)

        self.setContentLayout(main_layout)
        self._refresh_db_model_list()

    # =========================================================================
    # SQLITE MODEL MANAGEMENT
    # =========================================================================
    def _refresh_db_model_list(self) -> None:
        self.combo_db_models.clear()
        self.combo_db_models.addItem("-- Select model from database --", "")
        try:
            for m in list_models_in_sqlite(Path(self.db_path)):
                label = m["model_name"]
                if m.get("description"):
                    label += f" — {m['description'][:40]}"
                self.combo_db_models.addItem(label, m["model_name"])
        except Exception as e:
            self.logger.debug(f"Failed to list DB models: {e}")

    def _import_model_to_db(self) -> None:
        if not self.model_path or not self.band_stats_path:
            self._browse_model()

        if not self.model_path or not self.band_stats_path:
            QMessageBox.warning(
                self.main_window, "No Model",
                "Please select a .pt model and matching band_stats.json before importing to the database."
            )
            return

        default_name = Path(self.model_path).stem
        model_name, ok = QInputDialog.getText(
            self.main_window, "Import Model to Database",
            "Model name:", text=default_name
        )
        if not ok or not model_name.strip():
            return

        description, ok2 = QInputDialog.getMultiLineText(
            self.main_window, "Model Purpose / Description",
            "Model purpose / description (required):",
            text=self.model_description or ""
        )
        if not ok2:
            return

        success = register_model_in_sqlite(
            Path(self.db_path),
            model_name.strip(),
            Path(self.model_path),
            Path(self.band_stats_path),
            description.strip(),
        )
        if success:
            self.model_description = description.strip()
            self._model_loaded_from_db = False
            self.lbl_model_meta.setText(
                f"Imported to DB: {model_name} | {description.strip()[:60]}"
            )
            self._refresh_db_model_list()
            QMessageBox.information(self.main_window, "Import Success", f"Model '{model_name}' has been saved to SQLite.")
        else:
            QMessageBox.critical(self.main_window, "Import Failed", "Failed to save the model to the database.")

    def _load_model_from_db(self) -> None:
        model_name = self.combo_db_models.currentData()
        if not model_name:
            QMessageBox.warning(self.main_window, "No Selection", "Please select a model from the database list first.")
            return
        try:
            data = load_model_from_sqlite(Path(self.db_path), model_name)
            self.model_path = str(data["pt_path"])
            self.main_window.detector_model_path = self.model_path

            import json, tempfile, os
            temp_stats = os.path.join(tempfile.gettempdir(), f"kanopai_stats_{model_name}.json")
            with open(temp_stats, "w", encoding="utf-8") as f:
                json.dump(data["band_stats"], f)
            self._set_band_stats_path(temp_stats)

            self.model_description = data.get("description", "")
            self._model_loaded_from_db = True
            self.lbl_model_file.setText(f"Model: {model_name} (from DB)")
            self.lbl_model_meta.setText(
                f"Loaded from DB: {self.model_description or '(no description)'} | "
                f"Channels: {len(data['band_stats'])} bands"
            )
            self._check_ready_state()
            self.logger.info(f"Loaded model '{model_name}' from SQLite database.")
        except Exception as e:
            QMessageBox.critical(self.main_window, "Load Failed", f"Failed to load the model from the database:\n{e}")

    def _select_polygon_for(self, target: str) -> None:
        """Select a drawn polygon for AOI or Exclude."""
        polygons = getattr(self.main_window, "drawn_polygons", [])
        if not polygons:
            QMessageBox.warning(
                self.main_window, "No Polygons",
                "There are no drawn polygons yet. Please draw a polygon first using the left-side polygon tool."
            )
            return

        items = [f"{p['name']} ({len(p['pixel_coords'])} vertices, {p.get('area_m2', 0):.0f} m²)" for p in polygons]
        choice, ok = QInputDialog.getItem(
            self.main_window,
            f"Select Polygon for {'AOI' if target == 'aoi' else 'Exclude'}",
            "Select polygon:",
            items,
            0,
            False,
        )
        if not ok:
            return

        idx = items.index(choice)
        poly_id = polygons[idx]["id"]

        if target == "aoi":
            self._aoi_polygon_ids = [poly_id]
            self.lbl_aoi_polygon.setText(f"Polygon AOI: {polygons[idx]['name']} (crop → pad → scan)")
        else:
            self._exclude_polygon_ids = [poly_id]
            self.lbl_exclude_polygon.setText(f"Polygon Exclude: {polygons[idx]['name']}")

    def _get_polygon_coords_by_ids(self, ids: List[int]) -> List[List[tuple]]:
        polygons = getattr(self.main_window, "drawn_polygons", [])
        result = []
        for p in polygons:
            if p["id"] in ids:
                result.append(p["pixel_coords"])
        return result

    # =========================================================================
    # RASTER & MODEL BROWSER / INFO MANAGEMENT
    # =========================================================================
    def refresh_raster_info(self) -> None:
        """Called when a new raster is loaded — update readiness state."""
        self._check_ready_state()

    def _get_active_raster_path(self) -> Optional[str]:
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
            self.logger.debug(f"Failed to resolve active raster path: {e}")
        return None


    def _browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self.main_window, "Select YOLO Multispectral Model (.pt)", "", "YOLO Model (*.pt);;All Files (*.*)"
        )
        if not path:
            return

        self.model_path = path
        self.main_window.detector_model_path = path
        self.lbl_model_file.setText(f"Model: {Path(path).name} (import candidate)")
        self._model_loaded_from_db = False

        # Otomatis cari band_stats.json di folder yang sama
        p_stats = Path(path).parent / "band_stats.json"
        p_stats_stem = Path(path).parent / f"{Path(path).stem}_band_stats.json"

        if p_stats.exists():
            self._set_band_stats_path(str(p_stats))
        elif p_stats_stem.exists():
            self._set_band_stats_path(str(p_stats_stem))
        else:
            self.lbl_stats_file.setText("Band Stats: Please select band_stats.json")
            QMessageBox.information(
                self.main_window, "Band Stats Required",
                "Model (.pt) loaded. Now please select the matching band_stats.json file."
            )
            self._browse_band_stats()

        self._update_model_metadata()
        self._check_ready_state()

    def _browse_band_stats(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self.main_window, "Select band_stats.json", "", "JSON (*.json);;All Files (*.*)"
        )
        if path:
            self._set_band_stats_path(path)
            self._model_loaded_from_db = False
            self._update_model_metadata()
            self._check_ready_state()
            self._model_loaded_from_db = False
            self._update_model_metadata()
            self._check_ready_state()

    def _set_band_stats_path(self, path: str) -> None:
        self.band_stats_path = path
        self.main_window.band_stats_path = path
        self.lbl_stats_file.setText(f"Band Stats: {Path(path).name}")

    def _update_model_metadata(self) -> None:
        if not self.model_path or not self.band_stats_path:
            self.lbl_model_meta.setText("Info: Incomplete model configuration")
            return

        try:
            import json
            with open(self.band_stats_path, "r", encoding="utf-8") as f:
                stats = json.load(f)

            n_channels = len(stats) if isinstance(stats, dict) else 0
            if self._model_loaded_from_db:
                self.lbl_model_meta.setText(f"Channels: {n_channels} bands registered | Ready for inference")
            else:
                self.lbl_model_meta.setText(
                    f"Imported candidate ({n_channels} bands). Load from DB before running inference."
                )
        except Exception as e:
            self.lbl_model_meta.setText(f"Metadata read warning: {e}")

    def _browse_aoi_shp(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self.main_window, "Select AOI Shapefile", "", "Shapefile (*.shp)")
        if path:
            self.txt_aoi_path.setText(path)

    def _browse_exclude_shp(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self.main_window, "Select Exclude Shapefile", "", "Shapefile (*.shp)")
        if path:
            self.txt_exclude_path.setText(path)

    def _browse_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self.main_window, "Select Output Directory")
        if path:
            self.txt_output_dir.setText(path)

    def _check_ready_state(self) -> None:
        raster_path = self._get_active_raster_path()
        has_raster = raster_path is not None and Path(raster_path).exists()
        has_model = self.model_path is not None and Path(self.model_path).exists()
        has_stats = self.band_stats_path is not None and Path(self.band_stats_path).exists()

        ready = has_raster and has_model and has_stats and self._model_loaded_from_db
        self.btn_run.setEnabled(ready)

    # =========================================================================
    # INFERENCE EXECUTION WORKFLOW
    # =========================================================================
    def run_inference(self) -> None:
        """Run the inference worker thread."""
        raster_path = self._get_active_raster_path()
        if not raster_path:
            QMessageBox.critical(self.main_window, "No Raster", "No valid raster file loaded.")
            return

        if not self.model_path or not self.band_stats_path:
            QMessageBox.critical(self.main_window, "No Model", "Please select a model candidate (.pt) and band_stats.json first, then load a model from the SQLite database.")
            return

        if not self._model_loaded_from_db:
            QMessageBox.critical(
                self.main_window,
                "Model Not Loaded from DB",
                "Please load a model from the SQLite database before running inference."
            )
            return

        # Each inference result gets its own correction audit metadata.
        self.correction_reviewer_name = ""
        self.correction_date = ""

        # Band mismatch dialog — forced / manual matching
        manual_mapping, adaptive_fallback, proceed = resolve_band_matching(
            self.main_window, raster_path, self.band_stats_path
        )
        if not proceed:
            return
        self._manual_band_mapping = manual_mapping
        self._enable_adaptive_fallback = adaptive_fallback or self.chk_adaptive_fallback.isChecked()

        # Prepare parameters
        conf = self.spin_confidence.value()
        iou = self.spin_iou.value()
        tile_size = self.spin_tile_size.value()
        overlap = self.spin_overlap.value()
        batch_size = self.spin_batch_size.value()

        aoi_shp = self.txt_aoi_path.text().strip() or None
        exclude_shp = self.txt_exclude_path.text().strip() or None
        aoi_polygons_px = self._get_polygon_coords_by_ids(self._aoi_polygon_ids) or None
        exclude_polygons_px = self._get_polygon_coords_by_ids(self._exclude_polygon_ids) or None
        enable_adaptive = self._enable_adaptive_fallback

        out_dir = self.txt_output_dir.text().strip() or str(Path(raster_path).parent / "output") if raster_path else "output"
        out_name = self.txt_output_name.text().strip() or None

        self.log_terminal.clear()
        self.append_log(f"--- STARTING MULTISPECTRAL INFERENCE ---")
        self.append_log(f"Raster: {Path(raster_path).name}")
        self.append_log(f"Model: {Path(self.model_path).name}")
        if self.model_description:
            self.append_log(f"Model purpose: {self.model_description}")
        if manual_mapping:
            self.append_log(f"Band mapping: MANUAL {manual_mapping}")
        elif enable_adaptive:
            self.append_log("Band mapping: FORCED (adaptive fallback)")
        if aoi_polygons_px:
            self.append_log(f"AOI from polygon: {len(aoi_polygons_px)} polygon(s) — crop → pad → scan")
        if exclude_polygons_px:
            self.append_log(f"Exclude from polygon: {len(exclude_polygons_px)} polygon(s)")
        self.append_log(f"Conf: {conf:.2f} | IoU: {iou:.2f} | Tile: {tile_size}px | Overlap: {overlap}px | Batch: {batch_size}")

        try:
            self._worker = MultispectralInferenceWorker(
                model_path=self.model_path,
                band_stats_path=self.band_stats_path,
                raster_path=raster_path,
                conf=conf,
                tile_size=tile_size,
                overlap=overlap,
                iou_threshold=iou,
                output_dir=out_dir,
                batch_size=batch_size,
                out_name=out_name,
                aoi_shp_path=aoi_shp,
                exclude_shp_path=exclude_shp,
                aoi_polygons_px=aoi_polygons_px,
                exclude_polygons_px=exclude_polygons_px,
                db_path=self.db_path,
                manual_band_mapping=manual_mapping,
                enable_adaptive_fallback=enable_adaptive,
            )

            self._worker.log.connect(self.append_log)
            self._worker.progress.connect(self.on_progress)
            self._worker.finished.connect(self.on_finished)
            self._worker.error.connect(self.on_error)
            self._worker.cancelled.connect(self.on_cancelled)

            # Update UI state for running
            self._set_ui_running_state(True)

            self._start_time = time.time()
            self._timer_eta.start(1000)

            self._worker.start()
            self.logger.info("MultispectralInferenceWorker thread started successfully.")

        except Exception as e:
            self.logger.error(f"Failed to start inference worker: {e}", exc_info=True)
            QMessageBox.critical(self.main_window, "Execution Error", f"Failed to start worker: {e}")
            self._set_ui_running_state(False)

    def cancel_inference(self) -> None:
        if not self._worker:
            return

        reply = QMessageBox.question(
            self.main_window, "Confirm Cancellation",
            "Are you sure you want to cancel the running inference process?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.append_log(">>> USER CANCELLATION REQUESTED <<<")
        self.btn_cancel.setEnabled(False)
        self._worker.stop()

    def append_log(self, text: str) -> None:
        """Append a log message to the inference console window."""
        self.log_terminal.appendPlainText(text)
        sb = self.log_terminal.verticalScrollBar()
        sb.setValue(sb.maximum())

    def on_progress(self, current: int, total: int) -> None:
        """Callback progress dari worker."""
        if total > 0:
            pct = int((current / total) * 100)
            self.progress_bar.setValue(pct)
            self.lbl_progress_stats.setText(f"Tile: {current} / {total} ({pct}%)")

            if hasattr(self.main_window, "footer_progress_bar"):
                self.main_window.footer_progress_bar.setVisible(True)
                self.main_window.footer_progress_bar.setValue(pct)
            if hasattr(self.main_window, "label_detection"):
                self.main_window.label_detection.setText(f"Detection: {current}/{total} tiles")

    def _update_eta_label(self) -> None:
        if self._start_time <= 0:
            return

        elapsed = time.time() - self._start_time
        el_m, el_s = divmod(int(elapsed), 60)
        str_elapsed = f"{el_m:02d}:{el_s:02d}"

        val = self.progress_bar.value()
        if val > 0:
            total_est = (elapsed / val) * 100.0
            eta_rem = total_est - elapsed
            eta_m, eta_s = divmod(max(0, int(eta_rem)), 60)
            str_eta = f"{eta_m:02d}:{eta_s:02d}"
        else:
            str_eta = "--:--"

        self.lbl_eta.setText(f"Elapsed: {str_elapsed} | ETA: {str_eta}")

    def on_finished(self, result: InferenceResult) -> None:
        """Callback saat inference selesai sukses."""
        self._last_result = result
        self._timer_eta.stop()
        self._set_ui_running_state(False)

        n_boxes = len(result.boxes) if result.boxes is not None else 0
        self.append_log(f"--- INFERENCE COMPLETE ---")
        self.append_log(f"Objects detected: {n_boxes} | Time: {result.elapsed_seconds:.2f}s")
        self.append_log("Results are ready. Press Export Shapefiles to save them.")

        self.lbl_summary.setText(f"Results: {n_boxes} detections found")

        # Enable correction tools
        self.btn_add_box.setEnabled(True)
        self.btn_edit_box.setEnabled(True)
        self.btn_delete_box.setEnabled(True)
        self.btn_export_shapefiles.setEnabled(True)
        self.btn_convert_to_centroids.setEnabled(n_boxes > 0)

        # Delegate overlay rendering & session management
        if hasattr(self.main_window, "handle_inference_finished"):
            self.main_window.handle_inference_finished(result)
        elif hasattr(self.main_window, "detection_handler"):
            # Format detections wrapped
            detections = []
            if result.boxes is not None:
                from core.inference_engine import resolve_class_name
                for box, score, cls in zip(result.boxes, result.scores, result.classes):
                    x1, y1, x2, y2 = [float(v) for v in box]
                    detections.append({
                        "box": [x1, y1, x2, y2],
                        "score": float(score),
                        "class": resolve_class_name(cls, result.class_names),
                    })
            self.main_window.detection_handler.handle_inference_finished({"detections": detections})

        msg = (
            f"Inference complete! Found {n_boxes} objects.\n\n"
            "Results are currently displayed only.\n"
            "Press Export Shapefiles to save the raw and corrected results."
        )
        QMessageBox.information(self.main_window, "Inference Finished", msg)

    def on_error(self, error_msg: str) -> None:
        self._timer_eta.stop()
        self._set_ui_running_state(False)
        self.append_log(f"[ERROR] Inference failed: {error_msg}")
        QMessageBox.critical(self.main_window, "Inference Error", f"Inference process failed:\n{error_msg}")

    def on_cancelled(self) -> None:
        self._timer_eta.stop()
        self._set_ui_running_state(False)
        self.append_log("--- INFERENCE CANCELLED BY USER ---")
        self.lbl_progress_stats.setText("Tile: Cancelled")

    def _set_ui_running_state(self, running: bool) -> None:
        self.btn_run.setEnabled(not running)
        self.btn_cancel.setVisible(running)
        self.btn_cancel.setEnabled(running)

        # Disable DB model controls while inference is running
        self.btn_load_db_model.setEnabled(not running)
        self.btn_import_db_model.setEnabled(not running)
        self.spin_confidence.setEnabled(not running)
        self.spin_iou.setEnabled(not running)
        self.spin_tile_size.setEnabled(not running)
        self.spin_overlap.setEnabled(not running)
        self.spin_batch_size.setEnabled(not running)

        self.progress_bar.setVisible(running)

        if hasattr(self.main_window, "footer_progress_bar"):
            self.main_window.footer_progress_bar.setVisible(running)
        if hasattr(self.main_window, "btn_cancel_inference"):
            self.main_window.btn_cancel_inference.setVisible(running)
            self.main_window.btn_cancel_inference.setEnabled(running)

    # =========================================================================
    # VECTOR BOX CORRECTION TOOLS & EXPORT (RAW + CORRECTED)
    # =========================================================================
    def _toggle_add_box_mode(self, checked: bool) -> None:
        if checked:
            self.btn_edit_box.setChecked(False)
            if not self._ensure_correction_metadata():
                self.btn_add_box.setChecked(False)
                return
        if hasattr(self.main_window, "set_inference_box_mode"):
            self.main_window.set_inference_box_mode("add" if checked else "none")

    def _toggle_edit_box_mode(self, checked: bool) -> None:
        if checked:
            self.btn_add_box.setChecked(False)
            if not self._ensure_correction_metadata():
                self.btn_edit_box.setChecked(False)
                return
        if hasattr(self.main_window, "set_inference_box_mode"):
            self.main_window.set_inference_box_mode("edit" if checked else "none")
        if not checked:
            self.lbl_box_info.setText("Box Info: Select 'Edit' then click a box")

    def _delete_selected_box(self) -> None:
        if hasattr(self.main_window, "delete_selected_inference_box"):
            self.main_window.delete_selected_inference_box()

    def _undo_box_action(self) -> None:
        if hasattr(self.main_window, "undo_inference_box_action"):
            self.main_window.undo_inference_box_action()

    def _redo_box_action(self) -> None:
        if hasattr(self.main_window, "redo_inference_box_action"):
            self.main_window.redo_inference_box_action()

    def _ensure_correction_metadata(self) -> bool:
        """Ask once before editing and reuse metadata for the corrected export."""
        if self.correction_reviewer_name and self.correction_date:
            return True
        result = request_correction_metadata(
            self.main_window,
            self.correction_reviewer_name,
        )
        if result is None:
            return False
        self.correction_reviewer_name, self.correction_date = result
        return True

    def update_undo_redo_states(self, can_undo: bool, can_redo: bool) -> None:
        self.btn_undo.setEnabled(can_undo)
        self.btn_redo.setEnabled(can_redo)

    def export_shapefiles(self) -> None:
        """Export two shapefile datasets: raw_detection.shp and corrected_detection.shp."""
        if not self._ensure_correction_metadata():
            return
        if hasattr(self.main_window, "export_inference_shapefiles"):
            self.main_window.export_inference_shapefiles(
                self.correction_reviewer_name,
                self.correction_date,
            )
        else:
            QMessageBox.information(self.main_window, "Export", "Exporting shapefiles...")

    def _convert_to_centroids(self) -> None:
        """Convert bounding boxes from Multispectral Detector to centroid points."""
        handler = getattr(self.main_window, "inference_overlay_handler", None)
        if not handler or not handler.box_items:
            QMessageBox.warning(
                self.main_window, "No Detections",
                "No detection bounding boxes are available. Please run inference first."
            )
            return

        # Use active boxes only; eliminated boxes remain in the audit export.
        active_boxes = [
            item.get_box_coords()
            for item in handler.box_items
            if item.status != "eliminated" and item.isVisible()
        ]

        if not active_boxes:
            QMessageBox.warning(
                self.main_window, "No Active Boxes",
                "All detection bounding boxes were eliminated. Nothing available to convert."
            )
            return

        # Format sebagai detection dicts agar kompatibel dengan centroid_handler
        detections = [
            {"box": box}
            for box in active_boxes
        ]

        # Simpan ke onnx_detection_result agar centroid_handler.convert_to_centroids() bisa membacanya
        self.main_window.onnx_detection_result = {"detections": detections}

        # Panggil convert_to_centroids pada centroid_handler
        if hasattr(self.main_window, "centroid_handler"):
            self.main_window.centroid_handler.convert_to_centroids()
        elif hasattr(self.main_window, "convert_to_centroids"):
            self.main_window.convert_to_centroids()
        else:
            QMessageBox.warning(
                self.main_window, "Not Available",
                "Centroid handler is not available."
            )
            return

        # Aktifkan tombol centroid panel
        if hasattr(self.main_window, "btn_convert_to_centroids"):
            self.main_window.btn_convert_to_centroids.setEnabled(True)
        if hasattr(self.main_window, "btn_add_centroid"):
            self.main_window.btn_add_centroid.setEnabled(True)
        if hasattr(self.main_window, "btn_delete_centroid"):
            self.main_window.btn_delete_centroid.setEnabled(True)
        if hasattr(self.main_window, "btn_save_centroids"):
            self.main_window.btn_save_centroids.setEnabled(True)

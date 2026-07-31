"""Panel deteksi menggunakan model ONNX."""

import logging
from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QCheckBox, QComboBox, QWidget, QFormLayout, QSizePolicy,
    QScrollArea, QFrame, QRadioButton, QSpinBox, QDoubleSpinBox
)
from PyQt6.QtCore import Qt
from ui.widgets.collapsible_box import CollapsibleBox


class DetectionPanel(CollapsibleBox):
    """Panel untuk konfigurasi dan eksekusi deteksi ONNX."""
    def __init__(self, main_window):
        super().__init__("Detector")
        self.main_window = main_window
        self.logger = logging.getLogger(__name__)
        self.init_ui()

    def init_ui(self):
        """Inisialisasi UI panel deteksi."""

        det_layout = QVBoxLayout()
        det_layout.setContentsMargins(6, 6, 6, 6)
        det_layout.setSpacing(8)

        # 1. Input Data sub-section
        self.input_sub = CollapsibleBox("Input Data", nested=True)
        input_layout = QFormLayout()
        input_layout.setContentsMargins(6, 6, 6, 6)
        input_layout.setSpacing(6)

        # Input layer (combo_input_layer is shared)
        input_layout.addRow("Input layer:", self.main_window.combo_input_layer)

        # Processed area mask dropdown
        self.combo_processed_area = QComboBox()
        self.combo_processed_area.addItems(["Visible part", "Entire layer", "From polygons (shapefile)"])
        input_layout.addRow("Processed area:", self.combo_processed_area)

        self.input_sub.setContentLayout(input_layout)
        self.input_sub.toggle_button.setChecked(False)
        self.input_sub.toggle()
        det_layout.addWidget(self.input_sub)

        # 2. ONNX Model sub-section
        self.model_sub = CollapsibleBox("ONNX Model", nested=True)
        model_layout = QVBoxLayout()
        model_layout.setContentsMargins(6, 6, 6, 6)
        model_layout.setSpacing(6)

        model_row = QWidget()
        mr_layout = QHBoxLayout()
        mr_layout.setContentsMargins(0, 0, 0, 0)
        self.label_model_path = QLabel("No model loaded")
        self.label_model_path.setWordWrap(True)
        mr_layout.addWidget(self.label_model_path, 1)
        self.btn_browse_model = QPushButton("Browse")
        self.btn_browse_model.setMaximumWidth(100)
        self.btn_browse_model.clicked.connect(self.main_window.browse_onnx_model)
        mr_layout.addWidget(self.btn_browse_model)
        model_row.setLayout(mr_layout)

        self.label_model_info = QLabel("Model info: -")
        self.label_model_info.setWordWrap(True)
        self.label_model_info.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.label_model_info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        # Wrap model info in a scroll area
        model_info_scroll = QScrollArea()
        model_info_scroll.setWidget(self.label_model_info)
        model_info_scroll.setWidgetResizable(True)
        model_info_scroll.setMaximumHeight(120)
        model_info_scroll.setMinimumHeight(60)
        model_info_scroll.setFrameShape(QFrame.Shape.NoFrame)
        model_info_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        model_info_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        model_layout.addWidget(model_row)
        model_layout.addWidget(model_info_scroll)
        self.model_sub.setContentLayout(model_layout)
        self.model_sub.toggle_button.setChecked(False)
        self.model_sub.toggle()
        det_layout.addWidget(self.model_sub)

        # 3. Input Channels Mapping sub-section
        self.mapping_sub = CollapsibleBox("Input Channels Mapping", nested=True)
        mapping_layout = QVBoxLayout()
        mapping_layout.setContentsMargins(6, 6, 6, 6)
        mapping_layout.setSpacing(6)

        self.label_image_bands_info = QLabel("Image Input Bands: -")
        self.label_image_bands_info.setStyleSheet("QLabel { color: #aaa; font-size: 10px; margin-bottom: 4px; }")
        mapping_layout.addWidget(self.label_image_bands_info)

        mode_layout = QHBoxLayout()
        mode_label = QLabel("Mapping Mode:")
        self.radio_map_default = QRadioButton("Default (Sequential)")
        self.radio_map_advanced = QRadioButton("Advanced (Custom)")
        self.radio_map_default.setChecked(True)
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.radio_map_default)
        mode_layout.addWidget(self.radio_map_advanced)
        mode_layout.addStretch()
        mapping_layout.addLayout(mode_layout)

        self.radio_map_default.toggled.connect(lambda checked: self.main_window._on_mapping_mode_changed("Default" if checked else "Advanced"))

        self.channel_mapping_widget = QWidget()
        self.channel_mapping_layout = QVBoxLayout()
        self.channel_mapping_layout.setContentsMargins(0, 0, 0, 0)
        self.channel_mapping_layout.setSpacing(4)
        self.channel_mapping_widget.setLayout(self.channel_mapping_layout)
        self.channel_mapping_widget.setVisible(False)
        mapping_layout.addWidget(self.channel_mapping_widget)

        self.label_mapping_preview = QLabel("Current Mapping: Default (Band 1→R, Band 2→G, Band 3→B)")
        self.label_mapping_preview.setWordWrap(True)
        self.label_mapping_preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.label_mapping_preview.setMinimumHeight(40)
        self.label_mapping_preview.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.label_mapping_preview.setStyleSheet("QLabel { background-color: #2a2a2a; color: #aaa; padding: 8px; border-radius: 4px; font-size: 10px; line-height: 1.6; }")
        mapping_layout.addWidget(self.label_mapping_preview)

        self.mapping_sub.setContentLayout(mapping_layout)
        self.mapping_sub.toggle_button.setChecked(False)
        self.mapping_sub.toggle()
        det_layout.addWidget(self.mapping_sub)

        # 4. Processing Parameters sub-section
        self.proc_sub = CollapsibleBox("Processing Parameters", nested=True)
        proc_layout = QFormLayout()
        proc_layout.setContentsMargins(6, 6, 6, 6)
        proc_layout.setSpacing(6)

        proc_layout.addRow("Resolution (cm/px):", self.main_window.spin_resolution)
        proc_layout.addRow("Tile size (px):", self.main_window.spin_tile_size)

        self.spin_batch_size = QSpinBox()
        self.spin_batch_size.setRange(1, 1024)
        self.spin_batch_size.setValue(8)
        proc_layout.addRow("Batch size:", self.spin_batch_size)

        self.chk_qgis_preproc = QCheckBox("Letterbox")
        self.chk_qgis_preproc.setChecked(True)
        proc_layout.addRow("Preprocessing:", self.chk_qgis_preproc)

        overlap_widget = QWidget()
        overlap_h = QHBoxLayout()
        overlap_h.setContentsMargins(0, 0, 0, 0)
        overlap_h.addWidget(self.main_window.radio_overlap_percent)
        overlap_h.addWidget(self.main_window.spin_overlap_percent)
        overlap_h.addWidget(self.main_window.radio_overlap_dx)
        overlap_h.addWidget(self.main_window.spin_overlap_dx)
        overlap_widget.setLayout(overlap_h)
        proc_layout.addRow("Tiles overlap:", overlap_widget)

        self.proc_sub.setContentLayout(proc_layout)
        self.proc_sub.toggle_button.setChecked(False)
        self.proc_sub.toggle()
        det_layout.addWidget(self.proc_sub)

        # 5. Detection Parameters sub-section
        self.detect_sub = CollapsibleBox("Detection Parameters", nested=True)
        detect_layout = QFormLayout()
        detect_layout.setContentsMargins(6, 6, 6, 6)
        detect_layout.setSpacing(6)

        self.combo_model_type = QComboBox()
        self.combo_model_type.addItems(["YOLO_Ultralytics"])
        self.combo_model_type.setToolTip("Optimized for YOLOv11/v8 models (xyxy format)")
        detect_layout.addRow("Model Type:", self.combo_model_type)

        self.spin_confidence = QDoubleSpinBox()
        self.spin_confidence.setRange(0.0, 1.0)
        self.spin_confidence.setSingleStep(0.01)
        self.spin_confidence.setValue(0.01)
        detect_layout.addRow("Confidence:", self.spin_confidence)

        self.spin_iou = QDoubleSpinBox()
        self.spin_iou.setRange(0.0, 1.0)
        self.spin_iou.setSingleStep(0.01)
        self.spin_iou.setValue(0.45)
        detect_layout.addRow("IoU Threshold:", self.spin_iou)

        self.detect_sub.setContentLayout(detect_layout)
        self.detect_sub.toggle_button.setChecked(False)
        self.detect_sub.toggle()
        det_layout.addWidget(self.detect_sub)

        # Placeholder for ExportPanel (injected by MainWindow)
        self.export_container = QWidget()
        self.export_container.setLayout(QVBoxLayout())
        self.export_container.layout().setContentsMargins(0, 0, 0, 0)
        det_layout.addWidget(self.export_container)

        # 6. Buttons
        self.btn_run_inference = QPushButton("Run Inference")
        self.btn_run_inference.setEnabled(False)
        self.btn_run_inference.clicked.connect(self.main_window.run_onnx_inference)
        det_layout.addWidget(self.btn_run_inference)

        self.btn_save_detections = QPushButton("Save Last Detections")
        self.btn_save_detections.setEnabled(False)
        self.btn_save_detections.clicked.connect(self.main_window.save_last_detections)
        det_layout.addWidget(self.btn_save_detections)

        self.label_last_detections = QLabel("Last detections: None (not saved)")
        self.label_last_detections.setStyleSheet("QLabel { color: #ccc; font-size: 11px; padding: 6px; }")
        det_layout.addWidget(self.label_last_detections)

        self.setContentLayout(det_layout)

    def set_export_panel(self, panel):
        self.export_container.layout().addWidget(panel)

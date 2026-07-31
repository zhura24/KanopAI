"""Panel opsi tampilan untuk UI."""

import logging
from pathlib import Path
from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QScrollArea, QWidget, QRadioButton, QCheckBox,
    QSlider, QRadioButton
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QColor, QBrush, QPen
from ui.widgets.collapsible_box import CollapsibleBox
from ui.widgets.flow_layout import FlowLayout


class DisplayPanel(CollapsibleBox):
    """Panel untuk mengatur opsi tampilan raster dan layer."""

    def __init__(self, main_window):
        super().__init__("Display Options")
        self.main_window = main_window
        self.logger = logging.getLogger(__name__)
        self.init_ui()

    def init_ui(self):
        """Inisialisasi UI panel display."""
        self.toggle_button.setChecked(False)
        self.toggle()

        display_layout = QVBoxLayout()
        display_layout.setContentsMargins(6, 6, 6, 6)
        display_layout.setSpacing(8)

        # ===== Sub-section: Raster Layers (Multi-Layer Support) =====
        self.layer_sub = CollapsibleBox("Raster Layers", nested=True)
        layer_sub_layout = QVBoxLayout()
        layer_sub_layout.setContentsMargins(6, 6, 6, 6)
        layer_sub_layout.setSpacing(6)

        # Layer list container (scrollable)
        self.layer_list_widget = QWidget()
        self.layer_list_layout = QVBoxLayout()
        self.layer_list_layout.setContentsMargins(0, 0, 0, 0)
        self.layer_list_layout.setSpacing(8)
        self.layer_list_widget.setLayout(self.layer_list_layout)

        # Scroll area for layer list
        self.layer_scroll = QScrollArea()
        self.layer_scroll.setWidgetResizable(True)
        self.layer_scroll.setMinimumHeight(100)
        self.layer_scroll.setMaximumHeight(350)
        self.layer_scroll.setWidget(self.layer_list_widget)
        self.layer_scroll.setStyleSheet("QScrollArea { border: 1px solid #555; background-color: #2b2b2b; }")
        self.layer_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.layer_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        layer_sub_layout.addWidget(self.layer_scroll)

        # "No layers" placeholder
        self.lbl_no_layers = QLabel("No raster layers loaded")
        self.lbl_no_layers.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_no_layers.setStyleSheet("QLabel { color: #888; font-style: italic; padding: 15px; }")
        self.layer_list_layout.addWidget(self.lbl_no_layers)

        # Layer info panel
        self.lbl_layer_info = QLabel("Active Layer: None")
        self.lbl_layer_info.setWordWrap(True)
        self.lbl_layer_info.setStyleSheet("""
            QLabel {
                color: #AAA;
                font-size: 9pt;
                padding: 8px;
                background-color: #2b2b2b;
                border: 1px solid #555;
                border-radius: 3px;
            }
        """)
        self.lbl_layer_info.setToolTip("Shows data (polygons, detections, centroids) for the active layer")
        layer_sub_layout.addWidget(self.lbl_layer_info)

        # Layer management buttons
        layer_btn_row = QHBoxLayout()
        self.btn_add_layer = QPushButton("Add Layer")
        self.btn_add_layer.clicked.connect(self.main_window.add_raster_layer)
        layer_btn_row.addWidget(self.btn_add_layer)

        self.btn_remove_layer = QPushButton("Remove")
        self.btn_remove_layer.setEnabled(False)
        self.btn_remove_layer.clicked.connect(self.main_window.remove_active_layer)
        layer_btn_row.addWidget(self.btn_remove_layer)

        self.btn_clear_layers = QPushButton("Clear All")
        self.btn_clear_layers.setEnabled(False)
        self.btn_clear_layers.clicked.connect(self.main_window.clear_all_layers)
        layer_btn_row.addWidget(self.btn_clear_layers)
        layer_sub_layout.addLayout(layer_btn_row)

        self.layer_sub.setContentLayout(layer_sub_layout)
        self.layer_sub.toggle_button.setChecked(True)
        display_layout.addWidget(self.layer_sub)

        # ===== Sub-section: Polygon Drawing =====
        self.polygon_drawing_sub = CollapsibleBox("Polygon Drawing", nested=True)
        polygon_drawing_layout = QVBoxLayout()
        polygon_drawing_layout.setContentsMargins(6, 6, 6, 6)
        polygon_drawing_layout.setSpacing(6)

        self.chk_polygon_drawing = QCheckBox("Show Polygon")
        self.chk_polygon_drawing.setChecked(True)
        self.chk_polygon_drawing.setEnabled(False)
        self.chk_polygon_drawing.toggled.connect(self.main_window._on_polygon_drawing_toggled)
        polygon_drawing_layout.addWidget(self.chk_polygon_drawing)

        self.polygon_drawing_sub.setContentLayout(polygon_drawing_layout)
        self.polygon_drawing_sub.toggle_button.setChecked(False)
        self.polygon_drawing_sub.toggle()
        display_layout.addWidget(self.polygon_drawing_sub)

        # ===== Sub-section: Detection Result =====
        self.detection_result_sub = CollapsibleBox("Detection Result", nested=True)
        detection_result_layout = QVBoxLayout()
        detection_result_layout.setContentsMargins(6, 6, 6, 6)
        detection_result_layout.setSpacing(6)

        self.chk_tile_preview = QCheckBox("Tile Preview")
        self.chk_tile_preview.setChecked(False)
        self.chk_tile_preview.stateChanged.connect(self.main_window._on_tile_preview_toggled)
        detection_result_layout.addWidget(self.chk_tile_preview)

        self.chk_detection_labels = QCheckBox("Detection Labels")
        self.chk_detection_labels.setChecked(True)
        self.chk_detection_labels.stateChanged.connect(self.main_window._on_detection_labels_toggled)
        detection_result_layout.addWidget(self.chk_detection_labels)

        self.chk_detector_overlay = QCheckBox("Detections")
        self.chk_detector_overlay.setChecked(False)
        self.chk_detector_overlay.stateChanged.connect(self.main_window._on_detector_overlay_toggled)
        detection_result_layout.addWidget(self.chk_detector_overlay)

        self.detection_class_container = QWidget()
        det_class_v = QVBoxLayout()
        det_class_v.setContentsMargins(12, 0, 0, 0)
        det_class_v.setSpacing(4)
        self.detection_class_container.setLayout(det_class_v)
        self.detection_class_container.setVisible(False)
        detection_result_layout.addWidget(self.detection_class_container)

        self.detection_result_sub.setContentLayout(detection_result_layout)
        self.detection_result_sub.toggle_button.setChecked(False)
        self.detection_result_sub.toggle()
        display_layout.addWidget(self.detection_result_sub)

        # ===== Sub-section: Centroid Detection =====
        self.centroid_detection_sub = CollapsibleBox("Centroid Detection", nested=True)
        centroid_detection_layout = QVBoxLayout()
        centroid_detection_layout.setContentsMargins(6, 6, 6, 6)
        centroid_detection_layout.setSpacing(6)

        self.chk_centroid_layer = QCheckBox("Centroids (0)")
        self.chk_centroid_layer.setChecked(True)
        self.chk_centroid_layer.setEnabled(False)
        self.chk_centroid_layer.stateChanged.connect(self.main_window._on_centroid_layer_toggled)
        centroid_detection_layout.addWidget(self.chk_centroid_layer)

        self.chk_show_canopy_circles = QCheckBox("Show Canopy Circles")
        self.chk_show_canopy_circles.setChecked(True)
        self.chk_show_canopy_circles.stateChanged.connect(self.main_window._on_canopy_circle_toggled)
        centroid_detection_layout.addWidget(self.chk_show_canopy_circles)

        self.chk_show_measurement_labels = QCheckBox("Show Measurement Labels")
        self.chk_show_measurement_labels.setChecked(True)
        self.chk_show_measurement_labels.stateChanged.connect(self.main_window._on_measurement_labels_toggled)
        centroid_detection_layout.addWidget(self.chk_show_measurement_labels)

        canopy_display_separator = QLabel("─" * 30)
        canopy_display_separator.setStyleSheet("QLabel { color: #444; margin: 6px 0; }")
        centroid_detection_layout.addWidget(canopy_display_separator)

        canopy_display_label = QLabel("Canopy Display Format:")
        canopy_display_label.setStyleSheet("QLabel { color: #aaa; font-size: 10px; margin-top: 4px; }")
        centroid_detection_layout.addWidget(canopy_display_label)

        self.radio_show_radius = QRadioButton("Radius only")
        self.radio_show_radius.toggled.connect(self.main_window._on_canopy_display_changed)
        centroid_detection_layout.addWidget(self.radio_show_radius)

        self.radio_show_diameter = QRadioButton("Diameter only")
        self.radio_show_diameter.setChecked(True)
        self.radio_show_diameter.toggled.connect(self.main_window._on_canopy_display_changed)
        centroid_detection_layout.addWidget(self.radio_show_diameter)

        self.radio_show_both = QRadioButton("Both (Radius & Diameter)")
        self.radio_show_both.toggled.connect(self.main_window._on_canopy_display_changed)
        centroid_detection_layout.addWidget(self.radio_show_both)

        self.centroid_detection_sub.setContentLayout(centroid_detection_layout)
        self.centroid_detection_sub.toggle_button.setChecked(False)
        self.centroid_detection_sub.toggle()
        display_layout.addWidget(self.centroid_detection_sub)

        # Layers container for future use
        self.layers_container = QWidget()
        self.layers_container.setLayout(QVBoxLayout())
        self.layers_container.layout().setContentsMargins(0, 0, 0, 0)
        self.layers_container.layout().setSpacing(4)
        display_layout.addWidget(self.layers_container)

        display_widget = QWidget()
        display_widget.setLayout(display_layout)
        self.addWidget(display_widget)

    def refresh_layer_list(self, layers):
        """Refresh daftar layer di UI."""
        while self.layer_list_layout.count():
            item = self.layer_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not layers:
            self.lbl_no_layers = QLabel("No raster layers loaded")
            self.lbl_no_layers.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.lbl_no_layers.setStyleSheet("QLabel { color: #888; font-style: italic; padding: 15px; }")
            self.layer_list_layout.addWidget(self.lbl_no_layers)
            self.layer_scroll.setFixedHeight(100)
        else:
            for layer in reversed(layers):
                layer_widget = self._create_layer_list_item(layer)
                self.layer_list_layout.addWidget(layer_widget)

            # Dynamic height adjustment
            item_height = 60
            spacing = 8
            num_layers = len(layers)
            calculated_height = (num_layers * item_height) + ((num_layers - 1) * spacing) + 20
            target_height = max(100, min(calculated_height, 350))
            self.layer_scroll.setFixedHeight(target_height)

    def _create_layer_list_item(self, layer):
        """Buat widget untuk item layer di daftar."""
        from PyQt6.QtWidgets import QRadioButton, QToolButton
        
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        top_row = QHBoxLayout()
        top_row.setSpacing(4)
        radio = QRadioButton()
        radio.setChecked(layer['is_active'])
        radio.toggled.connect(lambda checked, lid=layer['id']: self.main_window._on_layer_selected(lid, checked))
        top_row.addWidget(radio)

        name_display = layer['name']
        if len(name_display) > 18:
            name_display = name_display[:15] + "..."
        lbl_name = QLabel(name_display)
        lbl_name.setStyleSheet("QLabel { color: #DDD; font-weight: bold; }" if layer['is_active'] else "QLabel { color: #AAA; }")
        # Keep the tooltip short (just the filename) so it doesn't render as a
        # wide popup that visually overlaps the visibility/delete buttons next to it.
        lbl_name.setToolTip(Path(layer['file_path']).name)
        lbl_name.setMinimumWidth(0)
        top_row.addWidget(lbl_name, 1)

        btn_visibility = QPushButton("Hide" if layer['visible'] else "Show")
        btn_visibility.setFixedSize(45, 25)
        btn_visibility.setStyleSheet("QPushButton { font-size: 10px; padding: 0px; }")
        btn_visibility.setToolTip("Show/Hide raster")
        btn_visibility.clicked.connect(lambda checked, lid=layer['id']: self.main_window._toggle_layer_visibility(lid))
        top_row.addWidget(btn_visibility, 0)

        btn_delete = QPushButton("✕")
        btn_delete.setFixedSize(25, 25)
        btn_delete.setStyleSheet("QPushButton { font-size: 12px; padding: 0px; color: #E66; font-weight: bold; }")
        btn_delete.setToolTip("Remove layer")
        btn_delete.clicked.connect(lambda checked, lid=layer['id']: self.main_window._delete_layer(lid))
        top_row.addWidget(btn_delete, 0)
        layout.addLayout(top_row)

        layer_type = layer.get('layer_type', 'raster')
        if layer_type == 'vector':
            info_row = QHBoxLayout()
            info_row.setContentsMargins(24, 4, 4, 4)
            info_row.setSpacing(8)
            geom_type = layer['metadata'].get('geometry_type', 'Unknown')
            feature_count = layer['metadata'].get('feature_count', 0)
            vector_icon = QLabel("[VECTOR]")
            vector_icon.setStyleSheet("QLabel { color: #4A9; font-size: 9pt; font-weight: bold; }")
            info_row.addWidget(vector_icon)
            info_row.addWidget(QLabel(f"{geom_type} ({feature_count})"))
            info_row.addStretch()
            layout.addLayout(info_row)
        else:
            # FlowLayout so band swatches (B1, B2, B3, ...) wrap to a new
            # line instead of overflowing/getting clipped by the sidebar
            # width when a raster has many bands.
            band_container = QWidget()
            info_row = FlowLayout(band_container, margin=0, h_spacing=10, v_spacing=4)
            band_container.setLayout(info_row)

            num_bands = layer['metadata'].get('bands', 0)
            band_configs = self._get_band_configs(num_bands)
            for config in band_configs:
                chip = QWidget()
                chip_layout = QHBoxLayout(chip)
                chip_layout.setContentsMargins(0, 0, 0, 0)
                chip_layout.setSpacing(4)

                color_box = QLabel()
                color_box.setFixedSize(12, 12)
                color_box.setStyleSheet(f"background-color: {config['color']}; border: 1px solid #555; border-radius: 2px;")
                chip_layout.addWidget(color_box)

                label = QLabel(config['short_name'])
                chip_layout.addWidget(label)

                info_row.addWidget(chip)

            band_wrapper = QHBoxLayout()
            band_wrapper.setContentsMargins(24, 4, 4, 4)
            band_wrapper.addWidget(band_container)
            layout.addLayout(band_wrapper)

        bg_color = "#4a4a4a" if layer['is_active'] else "#3a3a3a"
        border_color = "#666" if layer['is_active'] else "#555"
        widget.setLayout(layout)
        widget.setStyleSheet(f"QWidget {{ background-color: {bg_color}; border: 1px solid {border_color}; border-radius: 5px; }}")
        return widget

    def _get_band_configs(self, num_bands):
        """Auto-detect konfigurasi band."""
        if num_bands == 1:
            return [{'name': 'Band 1 (Grayscale)', 'short_name': 'Gray', 'color': '#999999'}]
        elif num_bands == 2:
            return [{'name': 'Band 1 (Red)', 'short_name': 'R', 'color': '#c0392b'}, {'name': 'Band 2 (Green)', 'short_name': 'G', 'color': '#27ae60'}]
        elif num_bands == 3:
            return [{'name': 'Band 1 (Red)', 'short_name': 'R', 'color': '#c0392b'}, {'name': 'Band 2 (Green)', 'short_name': 'G', 'color': '#27ae60'}, {'name': 'Band 3 (Blue)', 'short_name': 'B', 'color': '#2980b9'}]
        elif num_bands == 4:
            return [{'name': 'Band 1 (Red)', 'short_name': 'R', 'color': '#c0392b'}, {'name': 'Band 2 (Green)', 'short_name': 'G', 'color': '#27ae60'}, {'name': 'Band 3 (Blue)', 'short_name': 'B', 'color': '#2980b9'}, {'name': 'Band 4 (NIR/Alpha)', 'short_name': 'NIR', 'color': '#8e44ad'}]
        else:
            colors = ['#c0392b', '#27ae60', '#2980b9', '#f39c12', '#8e44ad', '#16a085', '#e74c3c', '#3498db', '#9b59b6', '#1abc9c']
            return [{'name': f'Band {i+1}', 'short_name': f'B{i+1}', 'color': colors[i % len(colors)]} for i in range(num_bands)]

    def update_layer_info(self, active_layer):
        """Update panel info layer dengan data active layer."""
        if not active_layer:
            self.lbl_layer_info.setText("Active Layer: None")
            return

        # Get current counts from active layer
        layer_polygons = active_layer.get('polygons', [])
        layer_detections = active_layer.get('detections', None)
        layer_centroids = active_layer.get('centroids', [])

        # Calculate canopy statistics if available
        canopy_info = ""
        if len(layer_centroids) > 0:
            canopy_avg = self.main_window._get_canopy_avg_text(layer_centroids)
            if canopy_avg:
                canopy_info = f" ({canopy_avg})"

        # Update info text
        info_text = f"<b>{active_layer['name']}</b><br/>"
        info_text += f"Polygons: {len(layer_polygons)} | "
        info_text += f"Detections: {'Yes' if layer_detections else 'No'} | "
        info_text += f"Centroids: {len(layer_centroids)}{canopy_info}"

        self.lbl_layer_info.setText(info_text)

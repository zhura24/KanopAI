"""Panel gambar poligon untuk UI."""

from PyQt6.QtWidgets import (QPushButton, QLabel, QVBoxLayout, QHBoxLayout,
                             QWidget, QScrollArea, QFormLayout, QSpinBox, QMessageBox,
                             QCheckBox)
from PyQt6.QtCore import Qt
from ui.widgets.collapsible_box import CollapsibleBox


class PolygonPanel(CollapsibleBox):
    """Panel untuk menggambar dan mengelola poligon."""
    def __init__(self, main_window):
        super().__init__("Polygon Drawing")
        self.main_window = main_window
        self.init_ui()

    def init_ui(self):
        """Inisialisasi UI panel poligon."""

        polygon_layout = QVBoxLayout()
        polygon_layout.setContentsMargins(6, 6, 6, 6)
        polygon_layout.setSpacing(8)

        # Button row 1: Draw / Cancel / Finish
        btn_row1 = QHBoxLayout()
        self.btn_draw_polygon = QPushButton("Draw Polygon")
        self.btn_draw_polygon.clicked.connect(self.main_window.polygon_handler.toggle_draw_polygon_mode)
        btn_row1.addWidget(self.btn_draw_polygon)

        self.btn_cancel_polygon = QPushButton("Cancel")
        self.btn_cancel_polygon.setEnabled(False)
        # Using lambda to ensuring it calls with brackets if needed, though toggle_draw_polygon_mode doesn't expect args usually
        self.btn_cancel_polygon.clicked.connect(lambda: self.main_window.polygon_handler.toggle_draw_polygon_mode())
        btn_row1.addWidget(self.btn_cancel_polygon)

        self.btn_finish_polygon = QPushButton("Finish")
        self.btn_finish_polygon.setEnabled(False)
        self.btn_finish_polygon.clicked.connect(self.main_window.polygon_handler.finish_polygon_drawing)
        btn_row1.addWidget(self.btn_finish_polygon)
        polygon_layout.addLayout(btn_row1)

        # Button row 2: Clear / Save
        btn_row2 = QHBoxLayout()
        self.btn_clear_polygon = QPushButton("Clear All")
        self.btn_clear_polygon.setEnabled(False)
        self.btn_clear_polygon.clicked.connect(self.main_window.polygon_handler.clear_drawn_polygon)
        self.btn_clear_polygon.setToolTip("Clear all polygons")
        btn_row2.addWidget(self.btn_clear_polygon)

        self.btn_save_polygon = QPushButton("Export All")
        self.btn_save_polygon.setEnabled(False)
        self.btn_save_polygon.clicked.connect(self.main_window.polygon_handler.save_polygon_to_file)
        self.btn_save_polygon.setToolTip("Export all polygons to file")
        btn_row2.addWidget(self.btn_save_polygon)
        polygon_layout.addLayout(btn_row2)

        # Polygon List
        polygon_list_label = QLabel("Drawn Polygons:")
        polygon_list_label.setStyleSheet("QLabel { font-weight: bold; margin-top: 8px; }")
        polygon_layout.addWidget(polygon_list_label)

        # Scrollable list container for polygons
        self.polygon_list_widget = QWidget()
        self.polygon_list_layout = QVBoxLayout()
        self.polygon_list_layout.setContentsMargins(0, 0, 0, 0)
        self.polygon_list_layout.setSpacing(4)
        self.polygon_list_widget.setLayout(self.polygon_list_layout)

        # Scroll area for polygon list
        polygon_scroll = QScrollArea()
        polygon_scroll.setWidgetResizable(True)
        polygon_scroll.setMinimumHeight(80)   # Minimum height for "No polygons" message
        polygon_scroll.setMaximumHeight(200)  # Maximum height
        polygon_scroll.setWidget(self.polygon_list_widget)
        polygon_scroll.setStyleSheet("QScrollArea { border: 1px solid #555; background-color: #2b2b2b; }")
        polygon_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        polygon_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        polygon_layout.addWidget(polygon_scroll)

        # Store reference for dynamic height adjustment
        self.polygon_scroll_area = polygon_scroll

        # "No polygons" placeholder
        self.lbl_no_polygons = QLabel("No polygons drawn yet")
        self.lbl_no_polygons.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_no_polygons.setStyleSheet("QLabel { color: #888; font-style: italic; padding: 20px; }")
        self.polygon_list_layout.addWidget(self.lbl_no_polygons)

        # Buttons for polygon list management
        list_btn_row = QHBoxLayout()
        self.btn_select_all_polygons = QPushButton("Select All")
        self.btn_select_all_polygons.setEnabled(False)
        self.btn_select_all_polygons.clicked.connect(self.main_window.polygon_handler.select_all_polygons)
        list_btn_row.addWidget(self.btn_select_all_polygons)

        self.btn_deselect_all_polygons = QPushButton("Deselect All")
        self.btn_deselect_all_polygons.setEnabled(False)
        self.btn_deselect_all_polygons.clicked.connect(self.main_window.polygon_handler.deselect_all_polygons)
        list_btn_row.addWidget(self.btn_deselect_all_polygons)

        self.btn_delete_selected_polygons = QPushButton("Delete Selected")
        self.btn_delete_selected_polygons.setEnabled(False)
        self.btn_delete_selected_polygons.clicked.connect(self.main_window.polygon_handler.delete_selected_polygons)
        list_btn_row.addWidget(self.btn_delete_selected_polygons)
        polygon_layout.addLayout(list_btn_row)

        # Color and size controls
        style_layout = QFormLayout()

        # Vertex color
        vertex_color_layout = QHBoxLayout()
        self.btn_vertex_color = QPushButton()
        self.btn_vertex_color.setFixedSize(30, 20)
        # Assuming main_window has these attributes initialized
        if hasattr(self.main_window, 'polygon_vertex_color'):
            self.btn_vertex_color.setStyleSheet(f"background-color: {self.main_window.polygon_vertex_color.name()};")
        self.btn_vertex_color.clicked.connect(self.main_window._pick_vertex_color)
        vertex_color_layout.addWidget(self.btn_vertex_color)
        vertex_color_layout.addStretch()
        style_layout.addRow("Vertex Color:", vertex_color_layout)

        # Vertex outline color
        outline_color_layout = QHBoxLayout()
        self.btn_vertex_outline_color = QPushButton()
        self.btn_vertex_outline_color.setFixedSize(30, 20)
        if hasattr(self.main_window, 'polygon_vertex_outline_color'):
            self.btn_vertex_outline_color.setStyleSheet(f"background-color: {self.main_window.polygon_vertex_outline_color.name()};")
        self.btn_vertex_outline_color.clicked.connect(self.main_window._pick_vertex_outline_color)
        outline_color_layout.addWidget(self.btn_vertex_outline_color)
        outline_color_layout.addStretch()
        style_layout.addRow("Vertex Outline:", outline_color_layout)

        # Line color
        line_color_layout = QHBoxLayout()
        self.btn_line_color = QPushButton()
        self.btn_line_color.setFixedSize(30, 20)
        if hasattr(self.main_window, 'polygon_line_color'):
            self.btn_line_color.setStyleSheet(f"background-color: {self.main_window.polygon_line_color.name()};")
        self.btn_line_color.clicked.connect(self.main_window._pick_line_color)
        line_color_layout.addWidget(self.btn_line_color)
        line_color_layout.addStretch()
        style_layout.addRow("Line Color:", line_color_layout)

        # Vertex size
        self.spin_vertex_size = QSpinBox()
        self.spin_vertex_size.setRange(10, 100)
        if hasattr(self.main_window, 'polygon_vertex_size'):
            self.spin_vertex_size.setValue(self.main_window.polygon_vertex_size)
        self.spin_vertex_size.setSuffix(" px")
        self.spin_vertex_size.valueChanged.connect(self.main_window._on_vertex_size_changed)
        style_layout.addRow("Vertex Size:", self.spin_vertex_size)

        # Line width
        self.spin_line_width = QSpinBox()
        self.spin_line_width.setRange(1, 20)
        if hasattr(self.main_window, 'polygon_line_width'):
            self.spin_line_width.setValue(self.main_window.polygon_line_width)
        self.spin_line_width.setSuffix(" px")
        self.spin_vertex_size.valueChanged.connect(self.main_window._on_line_width_changed)
        style_layout.addRow("Line Width:", self.spin_line_width)

        polygon_layout.addLayout(style_layout)

        # Status label and help button row
        status_row = QHBoxLayout()
        self.lbl_polygon_status = QLabel("Status: No polygon")
        self.lbl_polygon_status.setStyleSheet("QLabel { color: #888; font-style: italic; }")
        status_row.addWidget(self.lbl_polygon_status, 1)  # Stretch to fill space

        # Help button with "?" icon
        self.btn_polygon_help = QPushButton("?")
        self.btn_polygon_help.setFixedSize(25, 25)
        self.btn_polygon_help.setStyleSheet("""
            QPushButton {
                background-color: #4a4a4a;
                border: 1px solid #666;
                border-radius: 12px;
                color: #FFF;
                font-weight: bold;
                font-size: 12pt;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
                border: 1px solid #888;
            }
            QPushButton:pressed {
                background-color: #3a3a3a;
            }
        """)
        self.btn_polygon_help.setToolTip("Click for drawing instructions")
        self.btn_polygon_help.clicked.connect(self.main_window._show_polygon_instructions)
        status_row.addWidget(self.btn_polygon_help)

        polygon_layout.addLayout(status_row)

        polygon_widget = QWidget()
        polygon_widget.setLayout(polygon_layout)
        self.addWidget(polygon_widget)

    # UI Update Methods
    def update_status(self, text):
        self.lbl_polygon_status.setText(text)

    def set_drawing_buttons_state(self, is_drawing):
        self.btn_draw_polygon.setEnabled(not is_drawing)
        self.btn_cancel_polygon.setEnabled(is_drawing)
        self.btn_finish_polygon.setEnabled(is_drawing)

    def set_action_buttons_enabled(self, enabled):
        self.btn_clear_polygon.setEnabled(enabled)
        self.btn_save_polygon.setEnabled(enabled)
        self.btn_select_all_polygons.setEnabled(enabled)
        self.btn_deselect_all_polygons.setEnabled(enabled)
        self.btn_delete_selected_polygons.setEnabled(enabled and hasattr(self.main_window, 'selected_polygon_ids') and len(self.main_window.selected_polygon_ids) > 0)

    def refresh_polygon_list(self, polygons, selected_ids):
        """Refresh daftar poligon dengan penyesuaian tinggi dinamis."""
        # Clear existing widgets
        while self.polygon_list_layout.count():
            item = self.polygon_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Check if we have polygons
        if not polygons:
            # Show "no polygons" message
            self.lbl_no_polygons = QLabel("No polygons drawn yet")
            self.lbl_no_polygons.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.lbl_no_polygons.setStyleSheet("QLabel { color: #888; font-style: italic; padding: 20px; }")
            self.polygon_list_layout.addWidget(self.lbl_no_polygons)

            # Disable buttons
            self.set_action_buttons_enabled(False)

            # Set scroll area to minimum height when empty
            if hasattr(self, 'polygon_scroll_area'):
                self.polygon_scroll_area.setFixedHeight(80)
        else:
            # Create polygon list items
            for polygon in polygons:
                polygon_widget = self._create_polygon_list_item(polygon, selected_ids)
                self.polygon_list_layout.addWidget(polygon_widget)

            # Enable buttons
            self.set_action_buttons_enabled(True)
            self.btn_delete_selected_polygons.setEnabled(len(selected_ids) > 0)

            # Dynamic height adjustment
            if hasattr(self, 'polygon_scroll_area'):
                item_height = 40
                num_polygons = len(polygons)
                calculated_height = (num_polygons * item_height) + 10
                min_height = 80
                max_height = 200
                target_height = max(min_height, min(calculated_height, max_height))
                self.polygon_scroll_area.setFixedHeight(target_height)

    def _create_polygon_list_item(self, polygon, selected_ids):
        """Buat widget untuk item poligon di daftar."""
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # Checkbox for selection
        chk = QCheckBox()
        chk.setChecked(polygon['id'] in selected_ids)
        if hasattr(self.main_window, '_on_polygon_selection_changed'):
            chk.stateChanged.connect(lambda state, pid=polygon['id']: self.main_window._on_polygon_selection_changed(pid, state))
        layout.addWidget(chk)

        # Color indicator
        color_label = QLabel()
        color_label.setFixedSize(20, 20)
        color_label.setStyleSheet(f"background-color: {polygon['color'].name()}; border: 1px solid #555;")
        layout.addWidget(color_label)

        # Polygon name and info
        area_m2 = polygon.get('area_m2', 0)
        area_ha = area_m2 / 10000 if area_m2 > 0 else 0
        vertex_count = len(polygon.get('pixel_coords', []))

        info_text = f"{polygon['name']}"
        if area_ha > 0:
            info_text += f" ({area_ha:.2f} ha, {vertex_count} vertices)"
        else:
            info_text += f" ({vertex_count} vertices)"

        info_label = QLabel(info_text)
        info_label.setStyleSheet("QLabel { color: #DDD; }")
        layout.addWidget(info_label, 1)

        # Visibility toggle button
        btn_visibility = QPushButton("👁" if polygon['visible'] else "👁‍🗨")
        btn_visibility.setFixedSize(30, 25)
        btn_visibility.setToolTip("Toggle visibility")
        if hasattr(self.main_window, '_toggle_polygon_visibility'):
            btn_visibility.clicked.connect(lambda checked, pid=polygon['id']: self.main_window._toggle_polygon_visibility(pid))
        layout.addWidget(btn_visibility)

        # Delete button
        btn_delete = QPushButton("🗑")
        btn_delete.setFixedSize(30, 25)
        btn_delete.setToolTip("Delete polygon")
        if hasattr(self.main_window, '_delete_single_polygon'):
            btn_delete.clicked.connect(lambda checked, pid=polygon['id']: self.main_window._delete_single_polygon(pid))
        layout.addWidget(btn_delete)

        widget.setLayout(layout)
        widget.setStyleSheet("QWidget { background-color: #3a3a3a; border-radius: 4px; }")

        return widget

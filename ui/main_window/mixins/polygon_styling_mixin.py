"""
Polygon Styling Mixin for MainWindow

This mixin handles all polygon styling and visual configuration:
- Color pickers (vertex, vertex outline, line colors)
- Size and width adjustments (vertex size, line width)
- Polygon list UI item creation
- Polygon instruction dialog

Extracted to improve MainWindow modularity - focused on polygon visual configuration.
"""

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QPushButton, QCheckBox,
    QColorDialog, QMessageBox
)
from PyQt6.QtCore import Qt


class PolygonStylingMixin:
    """Handles polygon visual styling and configuration UI"""

    def _create_polygon_list_item(self, polygon):
        """Create a widget for a polygon list item"""
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # Checkbox for selection
        chk = QCheckBox()
        chk.setChecked(polygon['id'] in self.selected_polygon_ids)
        chk.stateChanged.connect(lambda state, pid=polygon['id']: self._on_polygon_selection_changed(pid, state))
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
        btn_visibility.clicked.connect(lambda checked, pid=polygon['id']: self._toggle_polygon_visibility(pid))
        layout.addWidget(btn_visibility)

        # Delete button
        btn_delete = QPushButton("🗑")
        btn_delete.setFixedSize(30, 25)
        btn_delete.setToolTip("Delete polygon")
        btn_delete.clicked.connect(lambda checked, pid=polygon['id']: self._delete_single_polygon(pid))
        layout.addWidget(btn_delete)

        widget.setLayout(layout)
        widget.setStyleSheet("QWidget { background-color: #3a3a3a; border-radius: 4px; }")

        return widget

    def _pick_vertex_color(self):
        """Open color picker for vertex color"""
        color = QColorDialog.getColor(self.polygon_vertex_color, self, "Select Vertex Color")
        if color.isValid():
            self.polygon_vertex_color = color
            if hasattr(self, 'polygon_panel'):
                self.polygon_panel.btn_vertex_color.setStyleSheet(f"background-color: {color.name()};")
            elif hasattr(self, 'btn_vertex_color'):
                self.btn_vertex_color.setStyleSheet(f"background-color: {color.name()};")
            if hasattr(self, 'viewer') and self.viewer:
                self.viewer.set_polygon_vertex_color(color)

    def _pick_vertex_outline_color(self):
        """Open color picker for vertex outline color"""
        color = QColorDialog.getColor(self.polygon_vertex_outline_color, self, "Select Vertex Outline Color")
        if color.isValid():
            self.polygon_vertex_outline_color = color
            if hasattr(self, 'polygon_panel'):
                self.polygon_panel.btn_vertex_outline_color.setStyleSheet(f"background-color: {color.name()};")
            elif hasattr(self, 'btn_vertex_outline_color'):
                self.btn_vertex_outline_color.setStyleSheet(f"background-color: {color.name()};")
            if hasattr(self, 'viewer') and self.viewer:
                self.viewer.set_polygon_vertex_outline_color(color)

    def _pick_line_color(self):
        """Open color picker for line color"""
        color = QColorDialog.getColor(self.polygon_line_color, self, "Select Line Color")
        if color.isValid():
            self.polygon_line_color = color
            if hasattr(self, 'polygon_panel'):
                self.polygon_panel.btn_line_color.setStyleSheet(f"background-color: {color.name()};")
            elif hasattr(self, 'btn_line_color'):
                self.btn_line_color.setStyleSheet(f"background-color: {color.name()};")
            if hasattr(self, 'viewer') and self.viewer:
                self.viewer.set_polygon_line_color(color)

    def _on_vertex_size_changed(self, value):
        """Update polygon vertex size"""
        self.polygon_vertex_size = value
        if hasattr(self, 'viewer') and self.viewer:
            self.viewer.set_polygon_vertex_size(value)

    def _on_line_width_changed(self, value):
        """Update polygon line width"""
        self.polygon_line_width = value
        if hasattr(self, 'viewer') and self.viewer:
            self.viewer.set_polygon_line_width(value)

    def _show_polygon_instructions(self):
        """Show polygon drawing instructions in a popup dialog"""
        instructions_text = """
<h3 style='color: #4CAF50;'>How to Draw Polygons</h3>

<p><b>Starting:</b></p>
<ul>
  <li>Click <b>'Draw Polygon'</b> button to start drawing mode</li>
  <li>Your cursor will be ready to place vertices on the raster</li>
</ul>

<p><b>Adding Vertices:</b></p>
<ul>
  <li><b>Left-click</b> on the raster to add vertices</li>
  <li><b>Right-click</b> to undo the last vertex</li>
  <li>Draw at least 3 vertices to create a polygon</li>
</ul>

<p><b>Finishing the Polygon:</b></p>
<ul>
  <li><b>Double-click</b> on the raster to finish</li>
  <li>Or press <b>Enter</b> key to finish</li>
  <li>Click <b>'Finish'</b> button to complete</li>
</ul>

<p><b>Canceling:</b></p>
<ul>
  <li>Press <b>ESC</b> key to cancel drawing</li>
  <li>Click <b>'Cancel'</b> button to abort</li>
</ul>

<p><b>Using Polygons for Detection:</b></p>
<ul>
  <li>Drawn polygons are <b>auto-detected</b></li>
  <li>Select polygons from the list (checkbox)</li>
  <li>Choose <b>'From polygon'</b> mode in Detector</li>
  <li>Click <b>'Run Inference'</b> to detect only within selected polygons</li>
</ul>

<p><b>Managing Multiple Polygons:</b></p>
<ul>
  <li>You can draw <b>multiple polygons</b></li>
  <li>Each polygon gets a different color automatically</li>
  <li>Use checkboxes to select which polygons to use</li>
  <li>Toggle visibility with 👁 button</li>
  <li>Delete individual polygons with 🗑 button</li>
</ul>

<p style='color: #888; font-size: 9pt; margin-top: 15px;'>
<i>Tip: You can manage all your polygons from the list below the drawing controls.</i>
</p>
        """

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Polygon Drawing Instructions")
        msg_box.setTextFormat(Qt.TextFormat.RichText)
        msg_box.setText(instructions_text)
        msg_box.setIcon(QMessageBox.Icon.Information)
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg_box.exec()

        self.logger.info("Polygon drawing instructions shown to user")

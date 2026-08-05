"""Centroid Operations Mixin

Handles centroid detection, canopy circle generation, and measurement operations.
"""

from PyQt6.QtWidgets import QMessageBox, QColorDialog,  QInputDialog
from PyQt6.QtGui import QColor
import logging


class CentroidMixin:
    """Mixin for centroid detection and canopy analysis operations.
    
    Pure UI rendering - keeps UI methods, delegates business logic to handlers.
    """
    
    def _render_centroids(self):
        """Render centroid points, canopy circles, and measurement labels on the viewer."""
        if not hasattr(self, 'viewer') or not self.viewer:
            return
        
        # Clear previous rendering
        self._clear_centroid_rendering()
        
        from PyQt6.QtWidgets import QGraphicsEllipseItem, QGraphicsTextItem, QGraphicsItem
        from PyQt6.QtGui import QPen, QBrush, QColor, QFont
        
        for pt in self.centroid_points:
            x, y = pt['x'], pt['y']
            
            # 1. Render canopy circle (if checkbox enabled and radius exists)
            if hasattr(self, 'chk_show_canopy_circles') and self.chk_show_canopy_circles.isChecked():
                if 'radius_px' in pt and pt['radius_px'] > 0:
                    radius_px = pt['radius_px']
                    
                    # Create canopy circle
                    canopy_circle = QGraphicsEllipseItem(
                        x - radius_px,
                        y - radius_px,
                        radius_px * 2,
                        radius_px * 2
                    )
                    
                    # Style canopy circle (semi-transparent green)
                    canopy_pen = QPen(QColor(0, 255, 0, 150))
                    canopy_pen.setWidth(2)
                    canopy_circle.setPen(canopy_pen)
                    
                    canopy_brush = QBrush(QColor(0, 255, 0, 30))
                    canopy_circle.setBrush(canopy_brush)
                    
                    # Tag as centroid canopy for filtering
                    canopy_circle._is_centroid_canopy = True
                    
                    # Add to scene and tracking list
                    self.viewer.scene.addItem(canopy_circle)
                    self.centroid_items.append(canopy_circle)
            
            # 2. Render centroid point (always shown)
            point_size = self.centroid_size
            
            # Create point ellipse centered at (x, y)
            ellipse = QGraphicsEllipseItem(
                x - point_size/2,
                y - point_size/2,
                point_size,
                point_size
            )
            
            # Set styling
            pen = QPen(self.centroid_color)
            pen.setWidth(2)
            ellipse.setPen(pen)
            
            brush = QBrush(self.centroid_color)
            ellipse.setBrush(brush)
            
            # Tag as centroid for filtering
            ellipse._is_centroid = True
            
            # Make clickable for delete mode
            ellipse.setAcceptHoverEvents(True)
            try:
                ellipse.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
            except Exception as e:
                self.logger.debug(f"Failed to set ellipse selectable flag: {e}")
            
            # Add to scene and tracking list
            self.viewer.scene.addItem(ellipse)
            self.centroid_items.append(ellipse)
            
            # 3. Render measurement label (if checkbox enabled and canopy data exists)
            if hasattr(self, 'chk_show_measurement_labels') and self.chk_show_measurement_labels.isChecked():
                if 'radius_m' in pt:
                    # Get measurement text based on user preference
                    measurement_text = self._get_canopy_display_text(pt['radius_m'])
                    
                    # Create text label
                    text_item = QGraphicsTextItem(measurement_text)
                    
                    # Style the text
                    font = QFont("Arial", 9, QFont.Weight.Bold)
                    text_item.setFont(font)
                    text_item.setDefaultTextColor(QColor(255, 255, 255))
                    
                    # Add black background for better visibility
                    text_item.setHtml(
                        f'<div style="background-color: rgba(0, 0, 0, 180); padding: 2px 4px; border-radius: 3px;">'
                        f'{measurement_text}'
                        f'</div>'
                    )
                    
                    # Position label below the centroid point
                    text_width = text_item.boundingRect().width()
                    text_height = text_item.boundingRect().height()
                    text_item.setPos(x - text_width/2, y + point_size/2 + 5)
                    
                    # Tag as centroid label
                    text_item._is_centroid_label = True
                    
                    # Add to scene and tracking list
                    self.viewer.scene.addItem(text_item)
                    self.centroid_items.append(text_item)
        
        self.logger.info(f"Rendered {len(self.centroid_points)} centroid points with measurements")
    
    def add_centroid_at_click(self, pixel_x, pixel_y):
        """Add centroid at clicked position - delegated to handler."""
        result = self.centroid_handler.add_centroid_at_click(pixel_x, pixel_y)
        if result:
            self._render_centroids()
            self._update_centroid_ui()
        return result
    
    def delete_centroid_at_click(self, pixel_x, pixel_y):
        """Delete centroid near clicked position - delegated to handler."""
        result = self.centroid_handler.delete_centroid_at_click(pixel_x, pixel_y)
        if result:
            self._render_centroids()
            self._update_centroid_ui()
        return result
    
    def pick_centroid_color(self):
        """Open color picker dialog for centroid color."""
        color = QColorDialog.getColor(self.centroid_color, self, "Select Centroid Color")
        
        if color.isValid():
            self.centroid_color = color
            
            # Update color button
            if hasattr(self, 'btn_centroid_color'):
                self.btn_centroid_color.setStyleSheet(
                    f"background-color: {color.name()}; border: 1px solid #666;"
                )
            
            # Re-render centroids with new color
            self._render_centroids()
            
            self.logger.info(f"Centroid color changed to {color.name()}")
    
    def update_centroid_styling(self):
        """Update centroid point styling (size)."""
        if hasattr(self, 'spin_centroid_size'):
            self.centroid_size = self.spin_centroid_size.value()
        
        # Re-render centroids with new size
        self._render_centroids()
        
        self.logger.info(f"Centroid size changed to {self.centroid_size}px")
    
    def save_centroids_to_shapefile(self):
        """Export centroids to shapefile - delegated to handler."""
        return self.centroid_handler.save_centroids_to_shapefile()

    def run_ehara_extraction(self):
        """Run eHara pixel extraction - delegated to handler."""
        return self.ehara_handler.run_extraction()
    
    def generate_canopy_circles(self):
        """Generate canopy circles from detections - delegated to handler."""
        result = self.centroid_handler.generate_canopy_circles()
        if result:
            self._render_canopy_circles()
            self._update_canopy_ui()
        return result
    
    def save_canopy_to_shapefile(self):
        """Export canopy to shapefile - delegated to handler."""
        return self.centroid_handler.save_canopy_to_shapefile()
    
    def pick_canopy_color(self):
        """Open color picker dialog for canopy circle color."""
        color = QColorDialog.getColor(self.canopy_color, self, "Select Canopy Circle Color")

        if color.isValid():
            self.canopy_color = color

            # Update button color
            if hasattr(self, 'btn_canopy_color'):
                self.btn_canopy_color.setStyleSheet(
                    f"background-color: {color.name()}; border: 1px solid #666;"
                )

            # Update label
            if hasattr(self, 'label_canopy_color'):
                self.label_canopy_color.setText(color.name())

            # Re-render canopy circles with new color
            self._render_canopy_circles()

            self.logger.info(f"Canopy color changed to {color.name()}")
    
    def update_canopy_styling(self):
        """Update canopy circle styling (line width)."""
        # Re-render canopy circles with new line width
        self._render_canopy_circles()

        line_width = self.spin_canopy_line_width.value() if hasattr(self, 'spin_canopy_line_width') else 2
        self.logger.info(f"Canopy line width changed to {line_width}px")
    
    def convert_to_centroids(self):
        """Convert detections to centroids - delegated to handler."""
        result = self.centroid_handler.convert_to_centroids()
        if result:
            self._render_centroids()
            self._update_centroid_ui()
        return result
    
    def _render_canopy_circles(self):
        """Render canopy circles on the viewer as QGraphicsEllipseItem."""
        from PyQt6.QtWidgets import QGraphicsEllipseItem, QGraphicsItem
        from PyQt6.QtGui import QPen, QBrush, QColor

        if not hasattr(self, 'viewer') or not self.viewer:
            return

        # Clear existing canopy items
        self._clear_canopy_rendering()

        # Render each canopy circle
        for canopy in self.canopy_circles:
            center_x = canopy['center_x']
            center_y = canopy['center_y']
            radius_px = canopy['radius_px']

            # Create ellipse centered at (center_x, center_y) with radius
            # QGraphicsEllipseItem uses top-left corner, so we offset by radius
            ellipse = QGraphicsEllipseItem(
                center_x - radius_px,
                center_y - radius_px,
                radius_px * 2,
                radius_px * 2
            )

            # Set styling
            pen_color = self.canopy_color if hasattr(self, 'canopy_color') else QColor(0, 255, 0, 150)
            line_width = self.spin_canopy_line_width.value() if hasattr(self, 'spin_canopy_line_width') else 2

            pen = QPen(pen_color)
            pen.setWidth(line_width)
            ellipse.setPen(pen)

            # Set fill with transparency
            brush_color = QColor(pen_color)
            brush_color.setAlpha(50)  # More transparent fill
            brush = QBrush(brush_color)
            ellipse.setBrush(brush)

            # Tag as canopy for filtering
            ellipse._is_canopy = True
            ellipse._canopy_id = canopy['id']

            # Make selectable
            try:
                ellipse.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
            except Exception as e:
                self.logger.debug(f"Failed to set canopy selectable flag: {e}")

            # Add to scene and tracking list
            self.viewer.scene.addItem(ellipse)
            self.canopy_items.append(ellipse)

            # Add measurement label (radius and diameter)
            self._add_canopy_label(canopy, center_x, center_y, radius_px)

        self.logger.info(f"Rendered {len(self.canopy_items)} canopy circles")
        self.logger.info(f"Created {len(self.canopy_label_items)} canopy measurement labels")

    def _add_canopy_label(self, canopy, center_x, center_y, radius_px):
        """Add diameter line with measurement labels inside canopy circle.

        Design:
        - Horizontal diameter line across circle center
        - Diameter label ABOVE the line
        - Radius label BELOW the line
        - Smaller font (7pt) for compact display
        """
        from PyQt6.QtWidgets import QGraphicsTextItem, QGraphicsRectItem, QGraphicsLineItem
        from PyQt6.QtGui import QFont, QColor, QBrush, QPen
        from PyQt6.QtCore import Qt

        try:
            # Get measurements
            radius_m = canopy.get('radius_m', 0.0)
            diameter_m = canopy.get('diameter_m', 0.0)

            self.logger.debug(f"Creating labels: Ø{diameter_m:.1f}m, R{radius_m:.1f}m at ({center_x:.1f}, {center_y:.1f})")

            # === 1. DIAMETER LINE ===
            # Draw horizontal line across circle diameter
            line_start_x = center_x - radius_px
            line_end_x = center_x + radius_px
            line_y = center_y

            diameter_line = QGraphicsLineItem(line_start_x, line_y, line_end_x, line_y)
            line_pen = QPen(QColor(255, 200, 0, 200))  # Orange-yellow, semi-transparent
            line_pen.setWidth(2)
            line_pen.setStyle(Qt.PenStyle.SolidLine)
            diameter_line.setPen(line_pen)
            diameter_line.setZValue(2)  # Above circle, below text

            # === 2. DIAMETER LABEL (ABOVE LINE) ===
            diameter_text = f"Ø{diameter_m:.1f}m"
            diameter_label = QGraphicsTextItem(diameter_text)
            diameter_label.setDefaultTextColor(QColor(255, 255, 255))  # White

            # Smaller font - 7pt for compact display
            font = QFont("Arial", 7, QFont.Weight.Bold)
            diameter_label.setFont(font)

            # Position diameter label centered above line
            diameter_rect = diameter_label.boundingRect()
            diameter_x = center_x - diameter_rect.width() / 2
            diameter_y = center_y - diameter_rect.height() - 2  # 2px above line

            diameter_label.setPos(diameter_x, diameter_y)
            diameter_label.setZValue(3)

            # Background for diameter label
            padding = 1.5
            diameter_bg = QGraphicsRectItem(
                diameter_x - padding,
                diameter_y - padding,
                diameter_rect.width() + 2 * padding,
                diameter_rect.height() + 2 * padding
            )
            diameter_bg.setBrush(QBrush(QColor(0, 0, 0, 140)))
            diameter_bg.setPen(QPen(Qt.PenStyle.NoPen))
            diameter_bg.setZValue(2.5)

            # === 3. RADIUS LABEL (BELOW LINE) ===
            radius_text = f"R{radius_m:.1f}m"
            radius_label = QGraphicsTextItem(radius_text)
            radius_label.setDefaultTextColor(QColor(255, 255, 255))  # White
            radius_label.setFont(font)  # Same 7pt font

            # Position radius label centered below line
            radius_rect = radius_label.boundingRect()
            radius_x = center_x - radius_rect.width() / 2
            radius_y = center_y + 2  # 2px below line

            radius_label.setPos(radius_x, radius_y)
            radius_label.setZValue(3)

            # Background for radius label
            radius_bg = QGraphicsRectItem(
                radius_x - padding,
                radius_y - padding,
                radius_rect.width() + 2 * padding,
                radius_rect.height() + 2 * padding
            )
            radius_bg.setBrush(QBrush(QColor(0, 0, 0, 140)))
            radius_bg.setPen(QPen(Qt.PenStyle.NoPen))
            radius_bg.setZValue(2.5)

            # === 4. ADD ALL TO SCENE ===
            self.viewer.scene.addItem(diameter_line)
            self.viewer.scene.addItem(diameter_bg)
            self.viewer.scene.addItem(diameter_label)
            self.viewer.scene.addItem(radius_bg)
            self.viewer.scene.addItem(radius_label)

            # Store in tracking list: (line, diameter_bg, diameter_label, radius_bg, radius_label)
            self.canopy_label_items.append((diameter_line, diameter_bg, diameter_label, radius_bg, radius_label))

        except Exception as e:
            self.logger.error(f"Failed to add canopy labels: {e}", exc_info=True)

    def _clear_canopy_rendering(self):
        """Clear canopy circle rendering from viewer."""
        if not hasattr(self, 'viewer') or not self.viewer:
            return

        # Clear canopy circle items
        for item in self.canopy_items:
            try:
                self.viewer.scene.removeItem(item)
            except Exception as e:
                self.logger.debug(f"Failed to remove canopy item: {e}")

        self.canopy_items.clear()

        # Clear canopy label items (line, backgrounds, labels)
        for item_tuple in self.canopy_label_items:
            try:
                # Handle multiple formats for backward compatibility
                if len(item_tuple) == 5:
                    # New format: (line, diameter_bg, diameter_label, radius_bg, radius_label)
                    for item in item_tuple:
                        self.viewer.scene.removeItem(item)
                elif len(item_tuple) == 3:
                    # Previous format: (line, bg, text)
                    for item in item_tuple:
                        self.viewer.scene.removeItem(item)
                elif len(item_tuple) == 2:
                    # Legacy format: (bg, text)
                    for item in item_tuple:
                        self.viewer.scene.removeItem(item)
            except Exception as e:
                self.logger.debug(f"Failed to remove canopy label items: {e}")

        self.canopy_label_items.clear()

    def _update_canopy_ui(self):
        """Update canopy-related UI elements."""
        count = len(self.canopy_circles)

        # Update statistics label in Canopy Analysis section
        if hasattr(self, 'label_canopy_stats'):
            if count > 0:
                avg_radius = sum(c['radius_m'] for c in self.canopy_circles) / count
                avg_diameter = avg_radius * 2
                self.label_canopy_stats.setText(
                    f"Canopy Circles: {count} | Avg Ø: {avg_diameter:.2f}m"
                )
            else:
                self.label_canopy_stats.setText("Canopy Circles: 0")

        # Enable/disable buttons
        has_canopy = count > 0

        if hasattr(self, 'btn_save_canopy'):
            self.btn_save_canopy.setEnabled(has_canopy)

        if hasattr(self, 'btn_generate_canopy'):
            # Enable if we have detections
            has_detections = bool(self.onnx_detection_result)
            self.btn_generate_canopy.setEnabled(has_detections)


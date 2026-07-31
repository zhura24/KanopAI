from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QFileDialog, QLabel, QProgressBar,
                             QMessageBox, QComboBox, QSpinBox, QDoubleSpinBox,
                             QFormLayout, QCheckBox, QRadioButton, QButtonGroup,
                             QScrollArea, QToolButton, QFrame, QApplication,
                             QDialog, QListWidget, QDialogButtonBox, QListWidgetItem,
                             QSizePolicy, QGraphicsPixmapItem, QGraphicsView, 
                             QColorDialog, QSlider)
from PyQt6.QtCore import Qt, QTimer, QCoreApplication, pyqtSignal, QPropertyAnimation, QEasingCurve, QEvent, QPointF, QSize, QThread, QObject
from PyQt6.QtGui import QFont, QWheelEvent, QColor, QBrush, QPen
from core.raster_loader import RasterLoader
from core.vector_loader import VectorLoader
from ui.raster_viewer import RasterViewer
import os 
DL_AVAILABLE = False
ONNXInferenceEngine = None
ONNXInferenceWorker = None
class ModelType:
    DETECTOR = 'detector'
    SEGMENTOR = 'segmentor'
    RECOGNITION = 'recognition'
    REGRESSOR = 'regressor'
    SUPERRESOLUTION = 'superresolution'
import numpy as np
import logging
import json
from pathlib import Path
from utils.logger_config import setup_logging, get_logger, PerformanceLogger
import traceback
import datetime
from utils.geospatial_utils import GeospatialMetrics

from handlers import (
    LayerHandler,
    PolygonHandler,
    DetectionHandler,
    CentroidHandler,
    MeasurementHandler,
    ExportHandler,
    ViewHandler
)

from ui.widgets.collapsible_box import CollapsibleBox
from ui.panels.file_panel import FilePanel
from ui.panels.polygon_panel import PolygonPanel
from ui.panels.display_panel import DisplayPanel
from ui.panels.measurement_panel import MeasurementPanel
from ui.panels.export_panel import ExportPanel
from ui.panels.detection_panel import DetectionPanel
from ui.panels.centroid_panel import CentroidPanel
from ui.panels.inference_panel import InferencePanel


from .mixins import (
    RasterMixin,
    PolygonMixin,
    CentroidMixin,
    CentroidUIHandlersMixin,
    DetectionMixin,
    SignalsMixin,
    ViewMixin,
    SidebarMixin,
    ChannelMappingMixin,
    ExportUIMixin,
    TilePreviewMixin,
    DisplayControlsMixin,
    StatusBarMixin,
    LayerGraphicsMixin,
    LayerUIMixin,
    LayerManagementMixin,
    PolygonStylingMixin,
    EventHandlersMixin,
)

class _LayerPreloadWorker(QObject):
    """Background worker: warms the tile cache and pre-computes the RGB
    overview array for a newly-added layer. Runs entirely on a QThread --
    only numpy/rasterio work, no QPixmap/QImage/GUI object creation here
    (those must stay on the GUI thread; see _on_layer_preload_ready).
    """
    finished = pyqtSignal(object)  # emits a result dict

    def __init__(self, loader, metadata):
        super().__init__()
        self.loader = loader
        self.metadata = metadata

    def run(self):
        result = {
            'loaded_tile_count': 0,
            'rgb_array': None,
            'needs_color_normalization': False,
            'error': None,
        }
        try:
            width = self.metadata.get('width', 0)
            height = self.metadata.get('height', 0)
            num_bands = self.metadata.get('bands', 0)

            tile_size = 2048  # Match the tile size used in TileManager
            center_tile_x = (width // 2) // tile_size
            center_tile_y = (height // 2) // tile_size

            tiles_priority = [
                (center_tile_x, center_tile_y),
                (center_tile_x - 1, center_tile_y),
                (center_tile_x + 1, center_tile_y),
                (center_tile_x, center_tile_y - 1),
                (center_tile_x, center_tile_y + 1),
                (center_tile_x - 1, center_tile_y - 1),
                (center_tile_x + 1, center_tile_y - 1),
                (center_tile_x - 1, center_tile_y + 1),
                (center_tile_x + 1, center_tile_y + 1),
            ]

            # Only warm the cache with the bands the preview actually needs
            # (literal band 1,2,3 -> R,G,B, or fewer) instead of every band --
            # a 7-band multispectral tile only needs 3 bands read here.
            band_indexes = [1, 2, 3] if num_bands >= 3 else (list(range(1, num_bands + 1)) or None)

            loaded_count = 0
            for tx, ty in tiles_priority:
                if tx < 0 or ty < 0:
                    continue
                try:
                    x_offset = tx * tile_size
                    y_offset = ty * tile_size
                    tile_width = min(tile_size, width - x_offset)
                    tile_height = min(tile_size, height - y_offset)
                    if tile_width > 0 and tile_height > 0:
                        _ = self.loader.read_window(
                            x_offset, y_offset, tile_width, tile_height,
                            scale=1.0, band_indexes=band_indexes
                        )
                        loaded_count += 1
                except Exception as e:
                    pass  # Skip failed tiles, non-fatal
            result['loaded_tile_count'] = loaded_count

            if num_bands > 3:
                # Pre-calculate a downsampled RGB overview (literal band 1,2,3
                # -> R,G,B -- must match the tile pipeline exactly, see
                # core/tile_loader.py).
                ov_data = self.loader.read_full_downsampled(max_dimension=2048)
                if ov_data is not None and ov_data.shape[0] >= 3:
                    stats = self.loader.get_global_statistics()

                    def normalize(band, idx):
                        b = band.astype(np.float32)
                        valid = b[(b != 0) & (~np.isnan(b))]
                        if len(valid) == 0:
                            valid = b
                        if stats and idx < len(stats):
                            p2, p98 = stats[idx]['p2'], stats[idx]['p98']
                        else:
                            p2, p98 = np.percentile(valid, (2, 98))
                        if p98 <= p2:
                            p2, p98 = np.min(valid), np.max(valid)
                            if p98 <= p2:
                                p98 = p2 + 1e-5
                        b = np.clip(b, p2, p98)
                        return ((b - p2) / (p98 - p2) * 255).astype(np.uint8)

                    r = normalize(ov_data[0], 0)
                    g = normalize(ov_data[1], 1)
                    b = normalize(ov_data[2], 2)

                    h, w = r.shape
                    rgb = np.zeros((h, w, 3), dtype=np.uint8)
                    rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2] = r, g, b
                    result['rgb_array'] = np.ascontiguousarray(rgb)
            else:
                # 1-3 band raster: no RGB overview needed, but float/reflectance
                # data (range 0-1) still needs contrast stretching or it will
                # render as black/white instead of a grayscale gradient.
                dtype_str = str(self.metadata.get('dtype', ''))
                try:
                    sample = self.loader.read_full_downsampled(max_dimension=256)
                    is_float_reflectance = (
                        sample is not None and
                        ('float' in dtype_str or sample.max() <= 1.0)
                    )
                except Exception:
                    is_float_reflectance = 'float' in dtype_str
                result['needs_color_normalization'] = is_float_reflectance

        except Exception as e:
            result['error'] = f"Error in background preload: {e}"

        self.finished.emit(result)


class MainWindow(
    RasterMixin,
    PolygonMixin,
    CentroidMixin,
    CentroidUIHandlersMixin,
    DetectionMixin,
    SignalsMixin,
    ViewMixin,
    SidebarMixin,
    ChannelMappingMixin,
    ExportUIMixin,
    TilePreviewMixin,
    DisplayControlsMixin,
    StatusBarMixin,
    LayerGraphicsMixin,
    LayerUIMixin,
    LayerManagementMixin,
    PolygonStylingMixin,
    EventHandlersMixin,
    QMainWindow
):
    """Main application window with modular mixin architecture.
    
    Mixins provide:
    - RasterMixin: Multi-layer raster management
    - PolygonMixin: Polygon operations and export
    - CentroidMixin: Centroid detection and canopy analysis
    - DetectionMixin: ONNX inference visualization
    - SignalsMixin: Signal connections and helpers
    - ViewMixin: Zoom, pan, coordinate display
    - SidebarMixin: Sidebar toggle animations
    """
    def __init__(self):
        super().__init__()

        setup_logging(log_level=logging.INFO, log_to_file=True)
        self.logger = get_logger(__name__)
        self.logger.info("="*80)
        self.logger.info("KanopAI Starting")
        self.logger.info("="*80)

        self.setWindowTitle("KanopAI")
        self.setGeometry(100, 100, 1200, 800)

        try:
            from PyQt6.QtGui import QPixmap, QIcon, QPainter
            from PyQt6.QtCore import QSize
            import os
            import sys

            possible_paths = [
                os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logo", "logo.png")),
                os.path.abspath(os.path.join(os.path.dirname(sys.executable), "logo", "logo.png")),
                os.path.abspath(os.path.join(os.getcwd(), "logo", "logo.png")),
            ]

            logo_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    logo_path = path
                    break

            if logo_path and os.path.exists(logo_path):
                original = QPixmap(logo_path)
                self.logo_pixmap = original

                self.logger.info(f"Logo loaded: {logo_path} (size: {original.width()}x{original.height()})")

                icon = QIcon()

                if original.width() != original.height():
                    self.logger.warning(f"Logo is not square ({original.width()}x{original.height()}), creating square version")

                    for size in [16, 32, 48, 64, 128, 256]:
                        square = QPixmap(size, size)
                        square.fill(Qt.GlobalColor.transparent)

                        target_size = int(size * 0.95)

                        scaled = original.scaled(target_size, target_size,
                                                Qt.AspectRatioMode.KeepAspectRatio,
                                                Qt.TransformationMode.SmoothTransformation)

                        painter = QPainter(square)
                        x = (size - scaled.width()) // 2
                        y = (size - scaled.height()) // 2
                        painter.drawPixmap(x, y, scaled)
                        painter.end()

                        icon.addPixmap(square)
                else:
                    for size in [16, 32, 48, 64, 128, 256]:
                        scaled = original.scaled(size, size,
                                                Qt.AspectRatioMode.IgnoreAspectRatio,
                                                Qt.TransformationMode.SmoothTransformation)
                        icon.addPixmap(scaled)

                self.setWindowIcon(icon)

                from PyQt6.QtWidgets import QApplication
                app = QApplication.instance()
                if app:
                    app.setWindowIcon(icon)

                self.logger.info(f"Window icon set successfully from: {logo_path}")
            else:
                self.logo_pixmap = None
                self.logger.warning(f"Logo file not found. Tried paths: {', '.join(possible_paths)}")
        except Exception as e:
            self.logger.error(f"Failed to load logo: {e}", exc_info=True)
            self.logo_pixmap = None

        self.raster_loader = RasterLoader()  # Legacy single loader (kept for backward compatibility)

        # Initialize core handlers EARLY to allow property delegation
        self.layer_handler = LayerHandler(self)
        
        # self.raster_layers and self.layer_counter are now properties delegating to layer_handler
        # self.raster_layers = []  <-- Removed, handled by layer_handler
        # self.layer_counter = 0   <-- Removed, handled by layer_handler
        self.active_layer_id = None  # Currently selected/active layer

        self.onnx_engine = None
        self.detection_worker = None
        self.current_data = None
        self.detection_data = None
        self.detection_result = None
        self.onnx_detection_result = None
        self.showing_mask = False
        self.channel_mapping_combos = []  # List of channel mapping comboboxes
        self.export_directory = None  # Training data export directory
        self.last_measurement_info = None  # Store last measurement info for unit updates

        self.centroid_points = []  # List of centroid points: [{x, y, gx, gy}, ...]
        self.centroid_items = []  # List of QGraphicsEllipseItem for rendering
        self.centroid_edit_mode = None  # 'add' or 'delete' or None
        self.centroid_color = QColor(255, 255, 0)  # Default yellow
        self.centroid_size = 6  # Default size

        self.polygon_drawing_mode = False  # Whether polygon drawing is active
        self.polygon_vertices = []  # List of pixel coordinates: [(x, y), ...]
        self.polygon_vertex_items = []  # List of QGraphicsEllipseItem for rendering vertices
        self.polygon_line_items = []  # List of QGraphicsLineItem for rendering edges
        self.polygon_item = None  # QGraphicsPolygonItem for final closed polygon
        self.polygon_filled_item = None  # Reference to filled polygon item
        self.polygon_closing_line = None  # Temporary line connecting last vertex to first

        self.drawn_polygons = []  # List of polygon data dicts: [{id, name, pixel_coords, geo_coords, area_m2, color, visible, items}, ...]
        self.polygon_counter = 0  # Counter for generating unique polygon IDs
        self.selected_polygon_ids = set()  # Set of polygon IDs selected for inference

        self.polygon_colors = [
            QColor(255, 0, 0),    # Red
            QColor(0, 255, 0),    # Green
            QColor(0, 0, 255),    # Blue
            QColor(255, 255, 0),  # Yellow
            QColor(255, 0, 255),  # Magenta
            QColor(0, 255, 255),  # Cyan
            QColor(255, 128, 0),  # Orange
            QColor(128, 0, 255),  # Purple
        ]

        self.polygon_vertex_color = QColor(255, 255, 0)  # Yellow fill for vertices
        self.polygon_vertex_outline_color = QColor(255, 0, 0)  # Red outline for vertices
        self.polygon_line_color = QColor(255, 0, 0)  # Red line color
        self.polygon_vertex_size = 30  # Vertex marker size in pixels
        self.polygon_line_width = 5  # Line width in pixels

        self.canopy_circles = []  # List of canopy circles: [{center_x, center_y, radius_px, radius_m, diameter_m, area_m2, gx, gy}, ...]
        self.canopy_items = []  # List of QGraphicsEllipseItem for rendering canopy circles
        self.canopy_label_items = []  # List of (line, diameter_bg, diameter_label, radius_bg, radius_label) - 5-tuple
        self.canopy_color = QColor(0, 255, 0, 100)  # Default green with transparency

        self.logger.info("MainWindow initialized | Resolution: 1200x800")

        # self.layer_handler = LayerHandler(self) <-- Moved up
        self.polygon_handler = PolygonHandler(self)
        from handlers.multispectral_detection_handler import MultispectralDetectionHandler
        self.detection_handler = MultispectralDetectionHandler(self)
        self.centroid_handler = CentroidHandler(self)
        self.measurement_handler = MeasurementHandler(self)
        self.export_handler = ExportHandler(self)
        self.view_handler = ViewHandler(self)
        self.logger.info("Handlers initialized successfully")

        self._init_ui()
        self._connect_signals()
        self._init_inference_overlay_handler()
        self._detect_device()


    @property
    def raster_layers(self):
        """Delegate raster_layers access to LayerHandler."""
        return self.layer_handler.layers
        
    @raster_layers.setter
    def raster_layers(self, value):
        self.layer_handler.layers = value

    @property
    def layer_counter(self):
        """Delegate layer_counter access to LayerHandler."""
        return self.layer_handler.layer_counter

    @layer_counter.setter
    def layer_counter(self, value):
        self.layer_handler.layer_counter = value

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_vertical_layout = QVBoxLayout(central_widget)
        main_vertical_layout.setContentsMargins(0, 0, 0, 0)
        main_vertical_layout.setSpacing(0)

        content_widget = QWidget()
        main_layout = QHBoxLayout(content_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.viewer = RasterViewer()
        self.viewer.logger = self.logger  # Share logger with viewer for better error tracking

        self.left_sidebar_container = QWidget()
        left_sidebar_layout = QVBoxLayout(self.left_sidebar_container)
        left_sidebar_layout.setContentsMargins(0, 0, 0, 0)
        left_sidebar_layout.setSpacing(0)

        self.btn_toggle_left_sidebar = QToolButton()
        self.btn_toggle_left_sidebar.setAutoRaise(True)
        self.btn_toggle_left_sidebar.setText('❮')
        self.btn_toggle_left_sidebar.setFixedHeight(30)
        self.btn_toggle_left_sidebar.setMinimumWidth(30)
        self.btn_toggle_left_sidebar.setMaximumWidth(30)
        self.btn_toggle_left_sidebar.setToolTip("Toggle left sidebar")
        self.btn_toggle_left_sidebar.setStyleSheet("""
            QToolButton {
                background-color: #333333;
                border: 1px solid #444444;
                color: #DDDDDD;
                border-radius: 4px;
                font-weight: bold;
                font-size: 14px;
            }
            QToolButton:hover {
                background-color: #3e3e3e;
                border: 1px solid #5a5a5a;
            }
            QToolButton:pressed {
                background-color: #2f2f2f;
            }
        """)


        self.btn_toggle_left_sidebar.clicked.connect(self.toggle_left_sidebar)
        toggle_left_layout = QHBoxLayout()
        toggle_left_layout.addStretch()
        toggle_left_layout.addWidget(self.btn_toggle_left_sidebar)
        toggle_left_layout.setContentsMargins(5, 5, 5, 5)
        left_sidebar_layout.addLayout(toggle_left_layout)

        left_panel = self._create_left_panel()
        self.left_panel_scroll = left_panel
        left_sidebar_layout.addWidget(left_panel, 1)

        if hasattr(self, 'logo_pixmap') and self.logo_pixmap is not None:
            logo_label = QLabel()
            logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            logo_label.setContentsMargins(10, 15, 10, 15)
            logo_label.setStyleSheet("QLabel { background-color: #2b2b2b; }")  # Match sidebar color

            scaled_logo = self.logo_pixmap.scaled(
                200, 60,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            logo_label.setPixmap(scaled_logo)
            left_sidebar_layout.addWidget(logo_label)

        self.right_sidebar_container = QWidget()
        right_sidebar_layout = QVBoxLayout(self.right_sidebar_container)
        right_sidebar_layout.setContentsMargins(0, 0, 0, 0)
        right_sidebar_layout.setSpacing(0)

        self.btn_toggle_right_sidebar = QToolButton()
        self.btn_toggle_right_sidebar.setAutoRaise(True)
        self.btn_toggle_right_sidebar.setText('❯')
        self.btn_toggle_right_sidebar.setFixedHeight(30)
        self.btn_toggle_right_sidebar.setMinimumWidth(30)
        self.btn_toggle_right_sidebar.setMaximumWidth(30)
        self.btn_toggle_right_sidebar.setToolTip("Toggle right sidebar")
        self.btn_toggle_right_sidebar.setStyleSheet(self.btn_toggle_left_sidebar.styleSheet())
        self.btn_toggle_right_sidebar.clicked.connect(self.toggle_right_sidebar)
        toggle_right_layout = QHBoxLayout()
        toggle_right_layout.addWidget(self.btn_toggle_right_sidebar)
        toggle_right_layout.addStretch()
        toggle_right_layout.setContentsMargins(5, 5, 5, 5)
        right_sidebar_layout.addLayout(toggle_right_layout)

        right_panel = self._create_right_panel()
        self.right_panel_scroll = right_panel
        right_sidebar_layout.addWidget(right_panel, 1)

        self.left_sidebar_visible = True
        self.right_sidebar_visible = False

        self.left_panel_scroll.setVisible(True)
        self.left_sidebar_container.setMaximumWidth(350)
        self.btn_toggle_left_sidebar.setText('❮')
        try:
            startup_w = 350
            self.btn_toggle_left_sidebar.setMinimumWidth(startup_w)
            self.btn_toggle_left_sidebar.setMaximumWidth(startup_w)
        except Exception as e:
            self.logger.debug(f"Failed to set left sidebar button width: {e}")

        self.right_panel_scroll.setVisible(False)
        self.right_sidebar_container.setMaximumWidth(40)
        self.btn_toggle_right_sidebar.setText('❯')
        try:
            self.btn_toggle_right_sidebar.setMinimumWidth(30)
            self.btn_toggle_right_sidebar.setMaximumWidth(30)
        except Exception as e:
            self.logger.debug(f"Failed to set right sidebar button width: {e}")

        main_layout.addWidget(self.left_sidebar_container, 1)  # Left sidebar
        main_layout.addWidget(self.viewer, 4)  # Viewer in center
        main_layout.addWidget(self.right_sidebar_container, 1)  # Right sidebar

        footer_widget = QWidget()
        footer_widget.setStyleSheet("QWidget { background-color: #2b2b2b; border-top: 1px solid #444; }")
        footer_widget.setFixedHeight(30)
        footer_layout = QHBoxLayout(footer_widget)
        footer_layout.setContentsMargins(10, 5, 10, 5)
        footer_layout.setSpacing(15)

        self.label_size = QLabel("Size: -")
        self.label_size.setStyleSheet("QLabel { color: #888; font-size: 11px; background-color: transparent; }")
        footer_layout.addWidget(self.label_size)

        self.label_bands = QLabel("Bands: -")
        self.label_bands.setStyleSheet("QLabel { color: #888; font-size: 11px; background-color: transparent; }")
        footer_layout.addWidget(self.label_bands)

        self.label_detection = QLabel("Detection: -")
        self.label_detection.setStyleSheet("QLabel { color: #888; font-size: 11px; background-color: transparent; }")
        footer_layout.addWidget(self.label_detection)

        self.label_display = QLabel("Display: -")
        self.label_display.setStyleSheet("QLabel { color: #888; font-size: 11px; background-color: transparent; }")
        footer_layout.addWidget(self.label_display)

        self.footer_progress_bar = QProgressBar()
        self.footer_progress_bar.setFixedWidth(200)
        self.footer_progress_bar.setFixedHeight(18)
        self.footer_progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555;
                border-radius: 3px;
                background-color: #1e1e1e;
                text-align: center;
                color: #aaa;
                font-size: 10px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 2px;
            }
        """)
        self.footer_progress_bar.setFormat("Detection: %p%")
        self.footer_progress_bar.setVisible(False)
        footer_layout.addWidget(self.footer_progress_bar)

        self.btn_cancel_inference = QPushButton("Cancel Inference")
        self.btn_cancel_inference.setEnabled(False)
        self.btn_cancel_inference.setVisible(False)
        try:
            self.btn_cancel_inference.clicked.connect(self.cancel_inference)
        except Exception as e:
            self.logger.debug(f"Failed to connect cancel_inference signal: {e}")
        footer_layout.addWidget(self.btn_cancel_inference)

        footer_layout.addStretch()

        self.label_device = QLabel("Device: -")
        self.label_device.setStyleSheet("QLabel { color: #888; font-size: 11px; background-color: transparent; }")
        footer_layout.addWidget(self.label_device)

        self.label_zoom = QLabel("Zoom: 1.0x")
        self.label_zoom.setStyleSheet("QLabel { color: #888; font-size: 11px; background-color: transparent; }")
        footer_layout.addWidget(self.label_zoom)

        self.label_scale = QLabel("Scale: -")
        self.label_scale.setStyleSheet("QLabel { color: #888; font-size: 11px; background-color: transparent; }")
        footer_layout.addWidget(self.label_scale)

        self.label_coordinates = QLabel("Lon: - | Lat: -")
        self.label_coordinates.setStyleSheet("QLabel { color: #888; font-size: 11px; background-color: transparent; }")
        self.label_coordinates.setMinimumWidth(200)  # Reduced from 250
        footer_layout.addWidget(self.label_coordinates)

        self.label_crs = QLabel("EPSG: -")
        self.label_crs.setStyleSheet("QLabel { color: #888; font-size: 11px; background-color: transparent; }")
        self.label_crs.setMinimumWidth(90)  # Reduced from 120
        footer_layout.addWidget(self.label_crs)

        main_vertical_layout.addWidget(content_widget, 1)
        main_vertical_layout.addWidget(footer_widget)

        self.viewer.setFocus()

    def _create_left_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.file_panel = FilePanel(self)
        layout.addWidget(self.file_panel)

        self.polygon_panel = PolygonPanel(self)
        layout.addWidget(self.polygon_panel)

        self.display_panel = DisplayPanel(self)
        layout.addWidget(self.display_panel)

        self.lbl_no_layers = self.display_panel.lbl_no_layers
        self.lbl_layer_info = self.display_panel.lbl_layer_info
        self.btn_add_layer = self.display_panel.btn_add_layer
        self.btn_remove_layer = self.display_panel.btn_remove_layer
        self.btn_clear_layers = self.display_panel.btn_clear_layers
        self.chk_polygon_drawing = self.display_panel.chk_polygon_drawing
        self.chk_tile_preview = self.display_panel.chk_tile_preview
        self.chk_detection_labels = self.display_panel.chk_detection_labels
        self.chk_detector_overlay = self.display_panel.chk_detector_overlay
        self.detection_class_container = self.display_panel.detection_class_container
        self.chk_centroid_layer = self.display_panel.chk_centroid_layer
        self.chk_show_canopy_circles = self.display_panel.chk_show_canopy_circles
        self.chk_show_measurement_labels = self.display_panel.chk_show_measurement_labels
        self.radio_show_radius = self.display_panel.radio_show_radius
        self.radio_show_diameter = self.display_panel.radio_show_diameter
        self.radio_show_both = self.display_panel.radio_show_both
        
        self.polygon_drawing_subsection = self.display_panel.polygon_drawing_sub
        self.detection_result_subsection = self.display_panel.detection_result_sub
        self.centroid_detection_subsection = self.display_panel.centroid_detection_sub
        
        self.layer_scroll_area = self.display_panel.layer_scroll
        self.layer_list_layout = self.display_panel.layer_list_layout

        self.measurement_panel = MeasurementPanel(self)
        layout.addWidget(self.measurement_panel)
        
        self.check_measurement_mode = self.measurement_panel.check_measurement_mode
        self.combo_unit = self.measurement_panel.combo_unit
        self.label_measurement = self.measurement_panel.label_measurement
        self.btn_clear_measurements = self.measurement_panel.btn_clear_measurements

        panel.setMaximumWidth(350)
        panel.setMinimumWidth(320)
        scroll.setMaximumWidth(350)
        scroll.setMinimumWidth(320)
        scroll.setWidget(panel)
        return scroll

    def _create_right_panel(self):
        """Create right panel with Deep Learning and Training Data Export"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        try:
            scroll.setStyleSheet('''
                QScrollBar:vertical {
                    width: 18px;
                    background: #272727;
                    margin: 0px 0px 0px 0px;
                }
                QScrollBar::handle:vertical {
                    background: #555555;
                    min-height: 24px;
                    border-radius: 6px;
                }
                QScrollBar::add-line, QScrollBar::sub-line { height: 0px; }
                QScrollBar::add-page, QScrollBar::sub-page { background: none; }
            ''')
        except Exception as e:
            self.logger.debug(f"Failed to set scrollbar stylesheet: {e}")

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        from PyQt6.QtWidgets import QRadioButton, QSpinBox, QDoubleSpinBox, QComboBox
        
        self.radio_overlap_percent = QRadioButton("Percentage")
        self.radio_overlap_percent.setChecked(True)
        self.radio_overlap_dx = QRadioButton("Distance (px)")
        self.radio_overlap_percent.toggled.connect(self.on_overlap_mode_changed)

        self.spin_overlap_percent = QSpinBox()
        self.spin_overlap_percent.setRange(0, 100)
        self.spin_overlap_percent.setValue(10)

        self.spin_overlap_dx = QSpinBox()
        self.spin_overlap_dx.setRange(0, 10000)
        self.spin_overlap_dx.setValue(50)

        self.spin_tile_size = QSpinBox()
        self.spin_tile_size.setRange(640, 640)
        self.spin_tile_size.setValue(640)
        self.spin_tile_size.setEnabled(False)

        self.spin_resolution = QDoubleSpinBox()
        self.spin_resolution.setRange(0.1, 10000.0)
        self.spin_resolution.setDecimals(2)
        self.spin_resolution.setValue(10.0)

        self.combo_input_layer = QComboBox()
        self.combo_input_layer.addItem("Raster")
        self.combo_input_layer.setEnabled(False)

        self.spin_overlap_percent.valueChanged.connect(self.update_export_info_labels)
        self.spin_overlap_dx.valueChanged.connect(self.update_export_info_labels)
        self.spin_tile_size.valueChanged.connect(self.update_export_info_labels)
        self.spin_resolution.valueChanged.connect(self.update_export_info_labels)
        self.combo_input_layer.currentTextChanged.connect(self.update_export_info_labels)

        self.detection_panel = DetectionPanel(self)
        layout.addWidget(self.detection_panel)

        self.export_panel = ExportPanel(self)
        self.detection_panel.set_export_panel(self.export_panel)

        # Inference panel (new): positioned between Detector and Centroid
        self.inference_panel = InferencePanel(self)
        layout.addWidget(self.inference_panel)

        self.centroid_panel = CentroidPanel(self)
        layout.addWidget(self.centroid_panel)

        self.input_sub = self.detection_panel.input_sub
        self.combo_processed_area = self.detection_panel.combo_processed_area
        self.model_sub = self.detection_panel.model_sub
        self.label_model_path = self.detection_panel.label_model_path
        self.label_model_info = self.detection_panel.label_model_info
        self.mapping_sub = self.detection_panel.mapping_sub
        self.label_image_bands_info = self.detection_panel.label_image_bands_info
        self.radio_map_default = self.detection_panel.radio_map_default
        self.radio_map_advanced = self.detection_panel.radio_map_advanced
        self.channel_mapping_widget = self.detection_panel.channel_mapping_widget
        self.channel_mapping_layout = self.detection_panel.channel_mapping_layout
        self.label_mapping_preview = self.detection_panel.label_mapping_preview
        self.proc_sub = self.detection_panel.proc_sub
        self.spin_batch_size = self.detection_panel.spin_batch_size
        self.chk_qgis_preproc = self.detection_panel.chk_qgis_preproc
        self.detect_sub = self.detection_panel.detect_sub
        self.combo_model_type = self.detection_panel.combo_model_type
        self.spin_confidence = self.detection_panel.spin_confidence
        self.spin_iou = self.detection_panel.spin_iou
        self.btn_run_inference = self.detection_panel.btn_run_inference
        self.btn_save_detections = self.detection_panel.btn_save_detections
        self.label_last_detections = self.detection_panel.label_last_detections
        
        self.label_export_dir = self.export_panel.label_export_dir
        self.check_export_tiles = self.export_panel.check_export_tiles
        self.check_export_mask = self.export_panel.check_export_mask
        self.check_export_grayscale = self.export_panel.check_export_grayscale
        self.label_export_overlap = self.export_panel.label_export_overlap
        self.label_export_tile_size = self.export_panel.label_export_tile_size
        self.label_export_resolution = self.export_panel.label_export_resolution
        self.btn_export_training_data = self.export_panel.btn_export_training_data
        
        self.btn_convert_to_centroids = self.centroid_panel.btn_convert_to_centroids
        self.label_centroid_count = self.centroid_panel.label_centroid_count
        self.btn_add_centroid = self.centroid_panel.btn_add_centroid
        self.btn_delete_centroid = self.centroid_panel.btn_delete_centroid
        self.btn_save_centroids = self.centroid_panel.btn_save_centroids
        self.spin_point_radius = self.centroid_panel.spin_point_radius
        self.btn_centroid_color = self.centroid_panel.btn_centroid_color
        self.label_stats_total = self.centroid_panel.label_stats_total
        self.label_stats_avg = self.centroid_panel.label_stats_avg
        self.label_stats_density = self.centroid_panel.label_stats_density

        try:
            self.chk_qgis_preproc.toggled.connect(lambda v: self.logger.info(f"QGIS-like preprocessing set to {v}"))
        except Exception as e:
            self.logger.debug(f"Failed to connect QGIS preprocessing checkbox: {e}")

        panel.setMaximumWidth(420)
        panel.setMinimumWidth(360)
        scroll.setMaximumWidth(420)
        scroll.setMinimumWidth(360)
        scroll.setWidget(panel)
        return scroll

    # =========================================================================
    # INFERENCE OVERLAY HANDLER -- PUBLIC INTERFACE
    # (Delegated to by InferencePanel & connected via signal/slot wiring)
    # =========================================================================
    def _init_inference_overlay_handler(self) -> None:
        """Instantiate InferenceOverlayHandler after viewer is ready."""
        try:
            from handlers.inference_overlay_handler import InferenceOverlayHandler
            self.inference_overlay_handler = InferenceOverlayHandler(self)
            self.logger.info("InferenceOverlayHandler initialised.")
        except Exception as e:
            self.logger.error(f"Failed to init InferenceOverlayHandler: {e}", exc_info=True)
            self.inference_overlay_handler = None

    def handle_inference_finished(self, result) -> None:
        """Main entry point called by InferencePanel.on_finished with InferenceResult."""
        handler = getattr(self, 'inference_overlay_handler', None)
        if handler is None:
            self._init_inference_overlay_handler()
            handler = getattr(self, 'inference_overlay_handler', None)
        if handler:
            handler.display_results(result)
        else:
            self.logger.warning("inference_overlay_handler not available -- overlays not rendered.")

    def set_inference_box_mode(self, mode: str) -> None:
        """Set interactive box editing mode ('none', 'add', 'edit')."""
        handler = getattr(self, 'inference_overlay_handler', None)
        if handler is None:
            self._init_inference_overlay_handler()
            handler = getattr(self, 'inference_overlay_handler', None)
        if handler:
            handler.set_mode(mode)
            if mode == "none":
                if hasattr(self, "inference_panel"):
                    self.inference_panel.btn_add_box.setChecked(False)
                    self.inference_panel.btn_edit_box.setChecked(False)

    def delete_selected_inference_box(self) -> None:
        handler = getattr(self, 'inference_overlay_handler', None)
        if handler:
            handler.delete_selected_box()

    def undo_inference_box_action(self) -> None:
        handler = getattr(self, 'inference_overlay_handler', None)
        if handler:
            handler.undo_action()

    def redo_inference_box_action(self) -> None:
        handler = getattr(self, 'inference_overlay_handler', None)
        if handler:
            handler.redo_action()

    def export_inference_shapefiles(self) -> None:
        handler = getattr(self, 'inference_overlay_handler', None)
        if handler:
            handler.export_shapefiles()

    def _reset_detection_state(self):
        self.detection_result = None
        self.detection_data = None
        self.showing_mask = False

    def open_file(self):
        return self.add_raster_layer()

    def update_progress(self, value):
        return self.detection_handler.update_progress(value)

    def detection_finished(self, result):
        return self.detection_handler.detection_finished(result)

    def save_last_detections(self):
        """Save latest detection results via ExportHandler"""
        return self.export_handler.export_polygons()

    def toggle_measurement_mode(self, state):
        return self.measurement_handler.toggle_measurement_mode(state)

    def _update_band_display(self, metadata):
        """Legacy method - Band display is now shown in individual layer boxes.

        This method is kept for backward compatibility but no longer updates the UI.
        Bands are now displayed directly in each layer's box in the Raster Layers list.
        """
        try:
            num_bands = metadata.get('bands', 0)

            if num_bands == 0:
                self.logger.warning("No bands detected in raster metadata")
                return

            self.logger.info(f"[BAND DETECTION] Raster has {num_bands} band(s)")

            if num_bands == 1:
                self.logger.info(f"[BAND TYPE] Single band (Grayscale) image")
            elif num_bands == 2:
                self.logger.info(f"[BAND TYPE] Two-band image")
            elif num_bands == 3:
                self.logger.info(f"[BAND TYPE] RGB image (3 bands)")
            elif num_bands == 4:
                self.logger.info(f"[BAND TYPE] 4-band image (RGB + NIR/Alpha)")
            else:
                self.logger.info(f"[BAND TYPE] Multispectral image ({num_bands} bands)")

        except Exception as e:
            self.logger.error(f"Failed to log band info: {e}", exc_info=True)

    def on_unit_changed(self, index):
        return self.measurement_handler.on_unit_changed(index)

    def clear_measurements(self):
        return self.measurement_handler.clear_measurements()

    def _display_measurement_result(self, measurement_info):
        return self.measurement_handler.display_measurement_result(measurement_info)

    def on_measurement_finished(self, measurement_info):
        return self.measurement_handler.on_measurement_finished(measurement_info)

    def browse_export_directory(self):
        return self.export_handler.browse_export_directory()

    def export_training_data(self):
        return self.export_handler.export_training_data()

    def _save_tile(self, tile_data, file_path):
        return self.export_handler._save_tile(tile_data, file_path)

    def run_onnx_inference(self):
        """Run ONNX detection (Delegated to DetectionHandler)."""
        if hasattr(self, 'detection_handler') and self.detection_handler:
            self.detection_handler.run_detection()
        else:
            QMessageBox.critical(self, "Error", "Detection handler not initialized.")

    def toggle_draw_polygon_mode(self):
        return self.polygon_handler.toggle_draw_polygon_mode()

    def _update_current_data_for_active_layer(self):
        """Update self.current_data to point to the active layer's raster loader.

        IMPORTANT — BigTIFF safe: we no longer load the entire raster into RAM here.
        self.current_data now holds the RasterLoader instance for the active layer
        (not a numpy array). Features that need pixel data must call the appropriate
        loader method themselves:
          - Export training data  → loader.read_for_export(max_dimension=8192)
          - Quick display check   → loader.get_overview(max_dimension=2048)
          - Channel mapping info  → loader.get_metadata()['bands']

        The old pattern  `loader.read_window(0, 0, w, h, scale=1.0)`  that pulled
        the whole raster into RAM has been removed; it was the cause of OOM crashes
        on BigTIFF files.
        """
        active_layer = self._get_active_layer()
        if not active_layer:
            self.current_data = None
            self._sync_layer_specific_data_to_ui()
            return

        try:
            loader = active_layer['loader']
            metadata = active_layer['metadata']

            if not loader or not metadata:
                self.current_data = None
                self._sync_layer_specific_data_to_ui()
                return

            # Expose loader so downstream code can call read_for_export() or
            # get_overview() on demand, without blocking the UI at layer-switch time.
            self.current_data = loader
            self.logger.info(
                f"Active layer data updated (lazy) | "
                f"Layer: {active_layer['name']} | "
                f"Size: {metadata.get('width', '?')}x{metadata.get('height', '?')} | "
                f"Bands: {metadata.get('bands', '?')}"
            )

            self._sync_layer_specific_data_to_ui()

        except Exception as e:
            self.logger.error(f"Error updating active layer data reference: {e}", exc_info=True)
            self.current_data = None
            self._sync_layer_specific_data_to_ui()

    def _sync_layer_specific_data_to_ui(self):
        """Sync layer-specific data (polygons, measurements, detections, centroids) to UI.

        When switching layers, each layer has its own:
        - Polygons
        - Measurements
        - Detection results
        - Centroid points

        This method updates the global UI variables to show the active layer's data.
        """
        active_layer = self._get_active_layer()

        if not active_layer:
            self.drawn_polygons = []
            self.onnx_detection_result = None
            self.centroid_points = []
            self._clear_polygon_graphics()
            self._clear_centroid_graphics()
            self._clear_detection_overlay()
            self.logger.info("[LAYER SYNC] No active layer - cleared all layer data")
            self._update_layer_info_panel()
            self._update_export_button_state()
            return

        layer_polygons = active_layer.get('polygons', [])
        if 'polygons' not in active_layer:
            active_layer['polygons'] = []
        self.drawn_polygons = active_layer['polygons']  # Use direct reference

        layer_detections = active_layer.get('detections', None)
        self.onnx_detection_result = layer_detections

        self.logger.info(f"[LAYER SYNC] Layer has detections: {layer_detections is not None}")
        if layer_detections:
            if isinstance(layer_detections, dict):
                det_count = len(layer_detections.get('detections', []))
                self.logger.info(f"[LAYER SYNC] Detection count in layer: {det_count}")
            else:
                self.logger.info(f"[LAYER SYNC] Detection data type: {type(layer_detections)}")

        layer_centroids = active_layer.get('centroids', [])
        if 'centroids' not in active_layer:
            active_layer['centroids'] = []
        self.centroid_points = active_layer['centroids']  # Use direct reference

        self._redraw_layer_specific_graphics()

        if hasattr(self, 'polygon_panel'):
            if layer_polygons:
                self.polygon_panel.update_status(f"Status: {len(layer_polygons)} polygon(s) drawn")
            else:
                self.polygon_panel.update_status("Status: Ready to draw")
        elif hasattr(self, 'lbl_polygon_status'):
            if layer_polygons:
                self.lbl_polygon_status.setText(f"Status: {len(layer_polygons)} polygon(s) drawn")
            else:
                self.lbl_polygon_status.setText("Status: Ready to draw")

        self._update_layer_info_panel()

        self._update_export_button_state()

        self.logger.info(
            f"[LAYER SYNC] Active layer '{active_layer['name']}' | "
            f"Polygons: {len(layer_polygons)} | "
            f"Detections: {'Yes' if layer_detections else 'No'} | "
            f"Centroids: {len(layer_centroids)}"
        )

    def _refresh_layer_list_ui(self):
        """Refresh the layer list UI via DisplayPanel"""
        if hasattr(self, 'display_panel'):
            self.display_panel.refresh_layer_list(self.raster_layers)
        else:
            self.logger.warning("display_panel not found, skipping _refresh_layer_list_ui")

    def _preload_layer_tiles_background(self, layer):
        """Pre-load tiles for instant layer switching -- runs on a background
        QThread so it never blocks the GUI while opening large rasters.

        NOTE (audit fix): this used to run synchronously on the GUI thread
        (see git history) -- reading up to 9 full tiles plus a 2048px
        downsampled overview plus global statistics, all before control
        returned to Qt's event loop. For a multi-GB BigTIFF that froze the
        UI for seconds right after "Add Layer", which is the opposite of
        the lazy/progressive-loading goal. All raster I/O and numpy work
        below now happens inside _LayerPreloadWorker.run() on a QThread;
        only the final QPixmap/viewer calls (which must happen on the GUI
        thread) run in _on_layer_preload_ready.
        """
        loader = layer.get('loader')
        metadata = layer.get('metadata')
        if not loader or not metadata or not metadata.get('width') or not metadata.get('height'):
            return

        if not hasattr(self, '_preload_threads'):
            self._preload_threads = []

        thread = QThread()
        worker = _LayerPreloadWorker(loader, metadata)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(lambda result: self._on_layer_preload_ready(layer, result))
        worker.finished.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self._preload_threads.remove((thread, worker))
                                 if (thread, worker) in self._preload_threads else None)

        self._preload_threads.append((thread, worker))
        thread.start()
        self.logger.info(f"[PRE-LOAD] Background preload started for: {layer['name']}")

    def _on_layer_preload_ready(self, layer, result):
        """Runs on the GUI thread once _LayerPreloadWorker finishes. Only does
        cheap Qt object creation (QImage/QPixmap) and viewer calls -- no raster I/O.
        """
        try:
            layer['_preloaded'] = True
            self.logger.info(
                f"[PRE-LOAD] Loaded {result['loaded_tile_count']}/9 tiles for: {layer['name']}"
            )

            if result.get('error'):
                self.logger.warning(f"[PRE-LOAD] {result['error']}")

            rgb_array = result.get('rgb_array')
            if rgb_array is not None:
                from PyQt6.QtGui import QImage, QPixmap
                h, w = rgb_array.shape[:2]
                qimg = QImage(rgb_array.data, w, h, w * 3, QImage.Format.Format_RGB888)
                layer['rgb_overview_pixmap'] = QPixmap.fromImage(qimg.copy())
                self.logger.info(f"[PRE-LOAD] RGB overview cached for {layer['name']}")

                if hasattr(self, 'viewer') and self.viewer:
                    self.viewer.set_color_normalization(True)
                    self.viewer.set_smart_rgb_overview(layer['rgb_overview_pixmap'])
                    self.viewer.async_tile_loader.clear_cache()
                    self.viewer._delayed_tile_load(immediate=True)
                    self.logger.info(f"[PRE-LOAD] RGB overview displayed immediately for {layer['name']}")

            elif result.get('needs_color_normalization') and hasattr(self, 'viewer') and self.viewer:
                self.viewer.set_color_normalization(True)
                self.viewer.async_tile_loader.clear_cache()
                self.viewer._delayed_tile_load(immediate=True)
                self.logger.info(
                    f"[PRE-LOAD] Color normalization auto-enabled for float/reflectance raster: {layer['name']}"
                )

        except Exception as e:
            self.logger.error(f"Error handling preload result: {e}", exc_info=True)

    def _get_default_vector_style(self, layer_type, metadata):
        """Get default styling for vector layer based on geometry type

        Args:
            layer_type: 'vector' or 'raster'
            metadata: Layer metadata dict

        Returns:
            Dict with style parameters
        """
        if layer_type != 'vector':
            return {
                'stroke_color': QColor(255, 0, 0),
                'stroke_width': 2,
                'fill_color': QColor(255, 0, 0, 50),
            }

        geom_type = metadata.get('geometry_type', 'Unknown')

        if 'Point' in geom_type:
            return {
                'stroke_color': QColor(255, 0, 0),      # Red outline
                'stroke_width': 2,                       # 2px stroke
                'fill_color': QColor(255, 255, 0, 180), # Yellow fill with transparency
                'point_size': 10,                        # 10px diameter (larger for visibility)
            }

        elif 'Line' in geom_type:
            return {
                'stroke_color': QColor(0, 100, 255),    # Blue
                'stroke_width': 3,
                'fill_color': QColor(0, 100, 255, 0),   # No fill for lines
            }

        elif 'Polygon' in geom_type:
            return {
                'stroke_color': QColor(255, 0, 0),      # Red outline
                'stroke_width': 2,
                'fill_color': QColor(255, 0, 0, 50),   # Semi-transparent red fill
            }

        else:
            return {
                'stroke_color': QColor(255, 0, 0),
                'stroke_width': 2,
                'fill_color': QColor(255, 0, 0, 80),
            }

        info_row.addStretch()
        layout.addLayout(info_row)

        bg_color = "#4a4a4a" if layer['is_active'] else "#3a3a3a"
        border_color = "#666" if layer['is_active'] else "#555"
        widget.setLayout(layout)
        widget.setStyleSheet(
            f"QWidget {{ "
            f"background-color: {bg_color}; "
            f"border: 1px solid {border_color}; "
            f"border-radius: 5px; "
            f"}}"
        )

        return widget

    def _on_layer_selected(self, layer_id, checked):
        """Handle layer radio button selection"""
        if checked:
            self._set_active_layer(layer_id)
            self._update_viewer_for_active_layer()
            self._refresh_layer_list_ui()

    def _on_layer_opacity_changed(self, layer_id, value):
        """Handle layer opacity slider change"""
        layer = self._get_layer_by_id(layer_id)
        if not layer:
            return

        opacity = value / 100.0
        layer['opacity'] = opacity

        if layer['pixmap_item']:
            layer['pixmap_item'].setOpacity(opacity)

        for i in range(self.layer_list_layout.count()):
            widget = self.layer_list_layout.itemAt(i).widget()
            if widget:
                for child in widget.findChildren(QSlider):
                    if hasattr(child, 'opacity_label'):
                        child.opacity_label.setText(f"{value}%")

        self.logger.debug(f"Layer {layer_id} opacity: {opacity}")

    def _update_viewer_for_active_layer(self):
        """Update viewer to display the active layer"""
        active_layer = self._get_active_layer()
        if not active_layer:
            return

        layer_type = active_layer.get('layer_type', 'raster')
        metadata = active_layer['metadata']

        if hasattr(self, 'viewer') and self.viewer:
            self.logger.info(f"[LAYER SWITCH] Switching to {layer_type} layer: {active_layer['name']}")

            try:
                if hasattr(self.viewer, 'async_tile_loader'):
                    self.viewer.async_tile_loader.cancel_all_pending()

                for layer in self.raster_layers:
                    vector_items = layer.get('vector_items', [])
                    if vector_items:
                        self.viewer.clear_vector_items(vector_items)
                        layer['vector_items'] = []

                if hasattr(self.viewer, '_clear_tile_items'):
                    self.viewer._clear_tile_items()
                else:
                    if hasattr(self.viewer, 'tile_pixmap_items'):
                        self.viewer.tile_pixmap_items.clear()
                    if hasattr(self.viewer, 'current_tile_keys'):
                        self.viewer.current_tile_keys.clear()

                if getattr(self.viewer, 'pixmap_item', None) is not None:
                    try:
                        self.viewer.scene.removeItem(self.viewer.pixmap_item)
                    except Exception:
                        pass

                self.viewer.pixmap_item = None
                self.viewer._raster_blocker = None
                self.viewer._preview_active = False

                layer_detections = active_layer.get('inference_result', None) or active_layer.get('detections', None)
                if layer_detections:
                    if hasattr(self, 'inference_overlay_handler') and self.inference_overlay_handler and hasattr(layer_detections, 'boxes'):
                        self.inference_overlay_handler.display_results(layer_detections, replace_existing=False)
                    else:
                        self._redraw_detection_overlay(layer_detections)
                else:
                    if hasattr(self, 'inference_overlay_handler') and self.inference_overlay_handler:
                        self.inference_overlay_handler.clear_overlay()
                    self._clear_detection_overlay()

                if layer_type == 'vector':
                    self.logger.info(f"[VECTOR] Rendering vector layer: {active_layer['name']}")

                    reference_transform = None
                    reference_crs = None

                    for layer in self.raster_layers:
                        if layer.get('layer_type') == 'raster':
                            ref_metadata = layer.get('metadata', {})
                            reference_transform = ref_metadata.get('transform')
                            reference_crs = ref_metadata.get('crs')
                            if reference_transform:
                                self.logger.info(f"[VECTOR] Using raster layer '{layer['name']}' for coordinate transform")
                                break

                    if reference_transform is None:
                        bounds = metadata.get('bounds', {})
                        if bounds:
                            from rasterio.transform import from_bounds
                            width = 1000  # arbitrary pixel width
                            minx = bounds.get('minx', 0)
                            maxx = bounds.get('maxx', 1)
                            miny = bounds.get('miny', 0)
                            maxy = bounds.get('maxy', 1)
                            height = int(width * (maxy - miny) / (maxx - minx))
                            reference_transform = from_bounds(minx, miny, maxx, maxy, width, height)
                            self.logger.info(f"[VECTOR] Created synthetic transform for vector-only display")

                    loader = active_layer['loader']
                    features = loader.get_features()
                    vector_crs = loader.get_crs()
                    style = active_layer.get('vector_style', {})

                    if reference_transform:
                        vector_items = self.viewer.render_vector_layer(
                            features,
                            reference_transform,
                            reference_crs,
                            vector_crs,
                            style
                        )

                        active_layer['vector_items'] = vector_items

                        self.logger.info(f"[VECTOR] Rendered {len(vector_items)} vector features")

                        if vector_items:
                            self._auto_zoom_to_vector_extent(active_layer, reference_transform, vector_crs, reference_crs)
                    else:
                        self.logger.warning("[VECTOR] Cannot render vector layer: no coordinate transform available")

                else:
                    self.viewer.set_tile_manager(active_layer['loader'])

                    self.viewer.enable_full_resolution(True)

                    is_visible = active_layer.get('visible', True)
                    self.viewer.show_raster(is_visible)

                    if hasattr(self.viewer, '_load_tiles_for_viewport'):
                        self.viewer._load_tiles_for_viewport(immediate=True)

            except Exception as e:
                self.logger.error(f"Error switching layer: {e}", exc_info=True)

        if hasattr(self, 'file_panel'):
            layer_type_display = f"[{layer_type.upper()}]" if layer_type == 'vector' else ""
            self.file_panel.update_file_label(f"Active: {layer_type_display} {os.path.basename(active_layer['file_path'])}")
        elif hasattr(self, 'label_file'):
            layer_type_display = f"[{layer_type.upper()}]" if layer_type == 'vector' else ""
            self.label_file.setText(f"Active: {layer_type_display} {os.path.basename(active_layer['file_path'])}")

        if layer_type != 'vector':
            self._update_band_display(metadata)

        self._update_ui_for_active_layer()

        self.logger.info(f"Viewer updated for active {layer_type} layer: {active_layer['name']}")
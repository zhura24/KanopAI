"""
Main Window Module - Modular Architecture

This module contains the refactored MainWindow implementation split into logical mixins:
- RasterMixin: Raster file operations and layer management  
- PolygonMixin: Polygon drawing and management
- CentroidMixin: Centroid detection and canopy analysis
- DetectionMixin: ONNX detection operations
- SignalsMixin: Signal connections and event handling
- ViewMixin: Zoom, pan, and coordinate display operations
- SidebarMixin: Sidebar toggle and animation operations

The main window class combines all mixins via multiple inheritance.
"""

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

from .main_window_impl import MainWindow

__all__ = [
    'MainWindow',
    'RasterMixin',
    'PolygonMixin', 
    'CentroidMixin',
    'CentroidUIHandlersMixin',
    'DetectionMixin',
    'SignalsMixin',
    'ViewMixin',
    'SidebarMixin',
    'ChannelMappingMixin',
    'ExportUIMixin',
    'TilePreviewMixin',
    'DisplayControlsMixin',
    'StatusBarMixin',
    'LayerGraphicsMixin',
    'LayerUIMixin',
    'LayerManagementMixin',
    'PolygonStylingMixin',
    'EventHandlersMixin',
]

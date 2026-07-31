"""Utilitas penanganan exception untuk KanopiAI."""

import logging
import functools
from typing import Optional, Callable, Any, TypeVar
from PyQt6.QtCore import QObject, pyqtSignal


T = TypeVar('T')


class DetectionError(Exception):
    """Exception untuk error deteksi."""
    pass


class CoordinateTransformError(Exception):
    """Exception untuk kegagalan transformasi koordinat."""
    pass


class TileLoadError(Exception):
    """Exception untuk kegagalan loading tile."""
    pass


class GeospatialError(Exception):
    """Exception untuk error komputasi geospasial."""
    pass


class ModelInferenceError(Exception):
    """Exception untuk kegagalan inferensi model ML."""
    pass


def safe_execute(
    func: Callable[..., T],
    default: T,
    logger: Optional[logging.Logger] = None,
    error_msg: Optional[str] = None,
    exceptions: tuple = (Exception,)
) -> T:
    """Eksekusi fungsi dengan penanganan exception otomatis."""
    try:
        return func()
    except exceptions as e:
        if logger:
            msg = error_msg or f"Error in {func.__name__}"
            logger.error(f"{msg}: {e}", exc_info=True)
        return default


def log_exceptions(
    default_return: Any = None,
    logger: Optional[logging.Logger] = None,
    error_msg: Optional[str] = None,
    reraise: bool = False
):
    """Decorator untuk logging exception dengan default return value."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                log = logger or logging.getLogger(func.__module__)
                msg = error_msg or f"Exception in {func.__name__}"
                log.error(f"{msg}: {e}", exc_info=True)
                if reraise:
                    raise
                return default_return
        return wrapper
    return decorator


def validate_not_none(*args, error_msg: str = "Unexpected None value"):
    """Validasi bahwa tidak ada argumen yang None."""
    for i, arg in enumerate(args):
        if arg is None:
            raise ValueError(f"{error_msg} at position {i}")


def validate_coordinates(x1: float, y1: float, x2: float, y2: float):
    """Validasi koordinat bounding box."""
    if not all(isinstance(v, (int, float)) for v in [x1, y1, x2, y2]):
        raise CoordinateTransformError("Coordinates must be numeric")
    if x2 <= x1:
        raise CoordinateTransformError(f"Invalid x: x2 ({x2}) <= x1 ({x1})")
    if y2 <= y1:
        raise CoordinateTransformError(f"Invalid y: y2 ({y2}) <= y1 ({y1})")


def safe_division(
    numerator: float,
    denominator: float,
    default: float = 0.0,
    logger: Optional[logging.Logger] = None
) -> float:
    """Pembagian dengan pengecekan zero."""
    if denominator == 0:
        if logger:
            logger.warning(f"Division by zero: {numerator}/{denominator}, returning {default}")
        return default
    return numerator / denominator


def clamp_value(
    value: float,
    min_val: float,
    max_val: float,
    logger: Optional[logging.Logger] = None
) -> float:
    """Batasi nilai ke range valid."""
    if value < min_val or value > max_val:
        if logger:
            logger.debug(f"Clamping {value} to [{min_val}, {max_val}]")
        return max(min_val, min(max_val, value))
    return value


class SafeWorker(QObject):
    """Base class untuk Qt worker dengan penanganan exception built-in."""

    error_occurred = pyqtSignal(str)

    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__()
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def safe_run(self, func: Callable, error_context: str = "Worker execution"):
        """Eksekusi fungsi dengan penanganan error otomatis dan signaling."""
        try:
            return func()
        except Exception as e:
            error_msg = f"{error_context} failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.error_occurred.emit(error_msg)
            return None

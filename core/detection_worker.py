"""Worker thread untuk deteksi ONNX dengan tile processing."""

from typing import Optional, List, Dict, Tuple, Any, Union
from PyQt6.QtCore import QThread, pyqtSignal
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
import logging
from utils.constants import (
    YOLO_PADDING_VALUE,
    YOLO_DEFAULT_CONF_THRESHOLD,
    YOLO_DEFAULT_IOU_THRESHOLD,
    YOLO_TIGHT_IOU_THRESHOLD,
    YOLO_EDGE_WEIGHT_MULTIPLIER,
    YOLO_BORDER_DISTANCE_ALPHA,
    NORMALIZED_COORD_THRESHOLD,
    NORMALIZED_RANGE_THRESHOLD,
    MAX_IMAGE_CHANNELS,
    TRANSPOSED_FEATURE_THRESHOLD,
    TRANSPOSED_DETECTION_THRESHOLD,
    MIN_YOLO_FEATURES,
    MAX_COORD_TRANSFORM_LOGS,
    MAX_SAMPLE_DETECTIONS_LOG,
    EPSILON
)
from utils.coordinate_utils import (
    reverse_letterbox_coordinates,
    ensure_valid_box_ordering,
    clamp_to_bounds,
    tile_to_global_coords,
    compute_edge_metrics
)
from utils.exception_utils import safe_execute
from core.smart_router import SmartRouter

logger = logging.getLogger(__name__)


class DetectionWorker(QThread):
    """Worker thread to run ONNX inference on image tiles.

    Inputs expected in start() via constructor:
      - model_path or existing ort.InferenceSession
      - tiles: list of dicts with keys: x, y, w, h, data (numpy array, C,H,W or H,W or H,W,C)
      - batch_size: int
      - input_size: (h,w) target model input size

    Emits:
      - progress(int) 0..100
      - finished(dict) where dict contains 'detections': list of detection dicts
      - error(str)
    """

    progress = pyqtSignal(int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, model_path: Optional[str] = None, session: Optional[Any] = None, 
                 tiles: Optional[List[Dict[str, Any]]] = None, batch_size: int = 8, 
                 input_size: Tuple[int, int] = (640, 640), 
                 conf_thresh: float = YOLO_DEFAULT_CONF_THRESHOLD, 
                 iou_thresh: float = YOLO_DEFAULT_IOU_THRESHOLD, 
                 model_type: str = 'yolo', qgis_preproc: bool = True, 
                 norm_mean: float = 0.0, norm_std: float = 1.0, 
                 channel_map: Optional[List[int]] = None, 
                 tile_overlap: float = 0,
                 raster_loader: Optional[Any] = None) -> None:
        super().__init__()
        self.model_path = model_path
        self.session = session
        self.tiles = tiles or []
        self.batch_size = int(batch_size)
        self.input_size = tuple(input_size)
        self.conf_thresh = float(conf_thresh)
        self.iou_thresh = float(iou_thresh)
        self.model_type = model_type
        # Preprocessing options
        self.qgis_preproc = bool(qgis_preproc)
        self.norm_mean = float(norm_mean)
        self.norm_std = float(norm_std)
        # channel_map: list of band indices or None
        self.channel_map = channel_map
        # tile overlap in pixels used for seam-aware merging
        self.tile_overlap = float(tile_overlap)
        self.raster_loader = raster_loader
        self._stopped = False

    def stop(self) -> None:
        self._stopped = True

    def _letterbox(self, img: NDArray, target_size: Tuple[int, int]) -> Tuple[NDArray, float, Tuple[int, int]]:
        """Letterbox resize with aspect ratio preservation and robust error handling.

        CRITICAL for coordinate accuracy:
        - Preserves aspect ratio (no distortion)
        - Returns scale and pad metadata for reverse transform
        - Validates all intermediate values

        Args:
            img: numpy array H,W,C or C,H,W
            target_size: (height, width) tuple

        Returns:
            (letterboxed_img, scale, (pad_left, pad_top))
        """
        try:
            # Validate input
            if img is None or img.size == 0:
                raise ValueError("Input image is None or empty")

            if img.ndim == 3 and img.shape[0] <= MAX_IMAGE_CHANNELS:
                img = np.transpose(img, (1, 2, 0))

            h, w = img.shape[:2]
            if h <= 0 or w <= 0:
                raise ValueError(f"Invalid image dimensions: {h}x{w}")

            th, tw = target_size
            if th <= 0 or tw <= 0:
                raise ValueError(f"Invalid target size: {th}x{tw}")

            scale, (left, top), (nw, nh) = self._compute_letterbox_meta(h, w, target_size)

            # Validate new dimensions
            if nh <= 0 or nw <= 0 or nh > th or nw > tw:
                logger.warning(f"Letterbox resize resulted in invalid dimensions: {nw}x{nh}, clamping")
                nh = max(1, min(th, nh))
                nw = max(1, min(tw, nw))

            # Resize with fallback methods
            # Resize with fallback methods
            resized = None
            try:
                import cv2
                # Try standard cv2.resize (often works for n-channels, but depends on version/backend)
                resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
            except Exception as e:
                logger.warning(f"cv2.resize failed: {e}, trying per-channel resize")
                try:
                    import cv2
                    # Multi-channel manual resize loop
                    if img.ndim == 3 and img.shape[2] > 0:
                        chans = []
                        for c in range(img.shape[2]):
                            chans.append(cv2.resize(img[:,:,c], (nw, nh), interpolation=cv2.INTER_LINEAR))
                        resized = np.stack(chans, axis=2)
                    else:
                         raise ValueError("Not a multi-channel image suitable for loop")
                except Exception as e2:
                    logger.warning(f"Per-channel resize failed: {e2}, trying PIL fallback")
                    try:
                        from PIL import Image
                        # PIL only supports restricted modes (1, 3, 4 channels)
                        num_c = img.shape[2] if img.ndim == 3 else 1
                        if num_c <= 4:
                             if img.dtype != np.uint8:
                                # Normalize blindly for PIL if float
                                img_pil = (np.clip(img, 0, 1) * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
                             else:
                                img_pil = img
                             
                             if num_c == 1:
                                 mode = 'L'
                             elif num_c == 3:
                                 mode = 'RGB'
                             elif num_c == 4:
                                 mode = 'RGBA'
                             else:
                                 mode = None # Let PIL guess or fail

                             resized = np.array(Image.fromarray(img_pil, mode=mode).resize((nw, nh)))
                             if img.dtype == np.float32:
                                resized = resized.astype(np.float32) / 255.0
                        else:
                             raise ValueError(f"Too many channels ({num_c}) for PIL")
                    except Exception as e3:
                         logger.error(f"All resize methods failed: {e3}, using nearest-neighbor subsampling")
                         # Simple subsampling as last resort (better than np.resize which corrupts data)
                         try:
                             y_indices = np.linspace(0, h-1, nh).astype(int)
                             x_indices = np.linspace(0, w-1, nw).astype(int)
                             resized = img[y_indices][:, x_indices]
                         except Exception as e4:
                             logger.error(f"Subsampling failed: {e4}, returning zero array")
                             resized = np.zeros((nh, nw, img.shape[2] if img.ndim==3 else 1), dtype=img.dtype)

            pad_h = th - nh
            pad_w = tw - nw
            if pad_h < 0 or pad_w < 0:
                logger.error(f"Negative padding detected: pad_h={pad_h}, pad_w={pad_w}")
                pad_h = max(0, pad_h)
                pad_w = max(0, pad_w)
            bottom = pad_h - top
            right = pad_w - left

            if resized.ndim == 3:
                num_channels = resized.shape[2]
            else:
                num_channels = 1

            # YOLO standard padding value, scaled based on dtype
            try:
                if np.issubdtype(resized.dtype, np.floating):
                    vmax = float(np.nanmax(resized)) if resized.size > 0 else 1.0
                    pad_val = YOLO_PADDING_VALUE if vmax > NORMALIZED_RANGE_THRESHOLD else (YOLO_PADDING_VALUE / 255.0)
                else:
                    pad_val = YOLO_PADDING_VALUE
            except Exception as e:
                logger.debug(f"Failed to determine padding value, using default: {e}")
                pad_val = YOLO_PADDING_VALUE

            if num_channels > 1:
                out = np.full((th, tw, num_channels), pad_val, dtype=resized.dtype)
                if resized.ndim == 2:
                    resized = np.expand_dims(resized, axis=2)
                out[top:top+nh, left:left+nw, :] = resized
            else:
                out = np.full((th, tw), pad_val, dtype=resized.dtype)
                out[top:top+nh, left:left+nw] = resized
                out[top:top+nh, left:left+nw] = resized

            if out.shape[0] != th or out.shape[1] != tw:
                raise ValueError(f"Output shape mismatch: expected {th}x{tw}, got {out.shape[0]}x{out.shape[1]}")

            logger.debug(f"Letterbox: {w}x{h} -> {nw}x{nh} in {tw}x{th}, scale={scale:.3f}, pad=({left},{top})")

            return out, scale, (left, top)

        except Exception as e:
            logger.error(f"Letterbox failed: {e}, returning original image with scale=1.0, pad=(0,0)")
            try:
                import cv2
                th, tw = target_size
                simple_resize = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
                return simple_resize, 1.0, (0, 0)
            except Exception as e:
                logger.error(f"Fallback resize also failed: {e}, returning original image")
                return img, 1.0, (0, 0)

    def _map_to_model_channels(self, img: NDArray, target_channels: int = 8, metadata: Optional[Dict] = None) -> NDArray:
        """Map any band count to target model input channels (default 8) using SmartRouter."""
        if img.ndim == 2:
            # Grayscale - expand to 3D (C, H, W)
            img = np.expand_dims(img, axis=0)
        elif img.shape[2] <= MAX_IMAGE_CHANNELS:
            # HWC -> CHW
            img = np.transpose(img, (2, 0, 1))
        
        C, H, W = img.shape
        
        # Use SmartRouter to get mapping if metadata is available
        if metadata:
            mapping = SmartRouter.get_mapping(metadata)
            # SmartRouter expects (C, H, W) and returns (8, H, W)
            img_mapped = SmartRouter.prepare_8band_data(img, mapping)
            
            # If target is not 8, adjust
            if target_channels != 8:
                return img_mapped[:target_channels]
            return img_mapped
            
        # Fallback to simple padding/dropping if no metadata
        if C == target_channels:
            return img
        elif C > target_channels:
            return img[:target_channels]
        else:
            logger.info(f"Padding {target_channels - C} channels with zeros (input had {C} bands)")
            padded = np.zeros((target_channels, H, W), dtype=img.dtype)
            padded[:C] = img
            return padded

    def _normalize_multispectral(self, img: NDArray) -> NDArray:
        """Apply percentile-based normalization like in training notebook."""
        # Input is (C, H, W)
        img = img.astype(np.float64)
        
        # Handle NoData/NaN
        img[img < -9999] = 0.0  
        img = np.nan_to_num(img, nan=0.0)

        for c in range(img.shape[0]):
            band = img[c]
            # Check if band is all zeros (padding)
            if np.all(band == 0):
                continue
                
            valid = band[band > 0]
            if len(valid) > 100:
                p2, p98 = np.percentile(valid, [2, 98])
                if p98 > p2:
                    img[c] = np.clip((band - p2) / (p98 - p2), 0, 1)
                else:
                    img[c] = 0.0
            else:
                img[c] = 0.0
        return img.astype(np.float32)

    def _preprocess(self, tile):
        if 'data' in tile:
            img = tile['data']
        elif self.raster_loader:
            try:
                # Lazy load from raster loader
                img = self.raster_loader.read_window(
                    int(tile['x']), int(tile['y']), 
                    int(tile['w']), int(tile['h'])
                )
                if img is None:
                    msg = f"RasterLoader returned None for tile at x={tile.get('x')}, y={tile.get('y')}"
                    logger.error(msg)
                    raise ValueError(msg)
            except Exception as e:
                logger.error(f"Error lazy loading tile data: {e}", exc_info=True)
                raise
        else:
             logger.error(f"Tile missing 'data' key and no raster_loader available. Keys: {list(tile.keys())}")
             raise KeyError("Tile missing 'data' and no raster_loader available")

        # Store original shape for metadata
        original_shape = img.shape

        # Detect target channels from model if possible
        target_channels = 8
        if self.session:
            try:
                model_input_shape = self.session.get_inputs()[0].shape
                if len(model_input_shape) == 4:
                    tc = model_input_shape[1]
                    if isinstance(tc, int):
                        target_channels = tc
            except Exception as e:
                logger.debug(f"Could not detect target channels from model: {e}")

        # Map to model channels using SmartRouter logic
        metadata = self.raster_loader.get_metadata() if self.raster_loader else None
        img = self._map_to_model_channels(img, target_channels=target_channels, metadata=metadata)

        # Apply percentile-based normalization (img is now CHW)
        img = self._normalize_multispectral(img)

        # Convert to float32 if needed
        if img.dtype != np.float32:
            img = img.astype(np.float32)

        # For letterbox, we need HWC format
        img_hwc = np.transpose(img, (1, 2, 0))

        if self.qgis_preproc:
            out, scale, pad = self._letterbox(img_hwc, self.input_size)
        else:
            try:
                import cv2
                th, tw = self.input_size
                resized = cv2.resize(img_hwc, (tw, th), interpolation=cv2.INTER_LINEAR)
            except Exception as e:
                logger.debug(f"OpenCV resize failed, trying PIL: {e}")
                resized = img_hwc # Fallback
            out = resized
            scale = 1.0
            pad = (0, 0)
        
        # Normalize (usually 0.0 and 1.0 for YOLO)
        if self.norm_mean != 0.0 or self.norm_std != 1.0:
            out = (out - self.norm_mean) / self.norm_std
            
        # Transpose back to CHW for ONNX (batch, channel, height, width)
        if out.ndim == 3:
            out = np.transpose(out, (2, 0, 1)) # HWC -> CHW
        else: # Handle grayscale images (H,W) -> (1,H,W)
            out = np.expand_dims(out, 0)
            
        return out, scale, pad, original_shape

    def _compute_letterbox_meta(self, h, w, target_size):
        """Compute letterbox metadata deterministically."""
        th, tw = target_size
        try:
            scale = min(th / float(h), tw / float(w)) if h > 0 and w > 0 else 1.0
        except Exception as e:
            logger.debug(f"Failed to compute scale: {e}")
            scale = 1.0
        nh = int(round(h * scale))
        nw = int(round(w * scale))
        pad_h = th - nh
        pad_w = tw - nw
        top = pad_h // 2
        left = pad_w // 2
        return float(scale), (int(left), int(top)), (int(nw), int(nh))

    def _validate_yolov11_model_outputs(self, session):
        """
        Validate ONNX model inputs and outputs for YOLOv11 detection.

        Inspired by QGIS Deepness plugin's Detector.check_loaded_model_outputs().

        Validates:
        - Input shape format [batch, channels, height, width]
        - Output shape format [batch, detections, features] or [batch, features, detections]
        - Auto-detects transposed outputs
        - Verifies compatibility with detection pipeline

        Args:
            session: onnxruntime.InferenceSession

        Returns:
            dict: {'valid': bool, 'reason': str, 'message': str, 'is_transposed': bool}
        """
        try:
            logger.info(f"Model validation - QGIS Deepness-inspired")

            # Validate inputs
            inputs = session.get_inputs()
            if len(inputs) < 1:
                return {
                    'valid': False,
                    'reason': 'No input layers found',
                    'message': '',
                    'is_transposed': False
                }

            main_input = inputs[0]
            input_shape = main_input.shape
            input_name = main_input.name

            logger.info(f"Model input: {input_name}, Shape: {input_shape}")

            # Validate input format: should be [batch, channels, height, width]
            if len(input_shape) == 4:
                batch, channels, height, width = input_shape
                logger.info(f"Input format: [batch, channels, height, width] = [{batch}, {channels}, {height}, {width}]")

                # Verify typical image input
                try:
                    if isinstance(channels, int) and channels not in [1, 3, 4]:
                        logger.warning(f"Unusual channel count: {channels}, expected 1, 3, or 4")
                except (TypeError, AttributeError) as e:
                    logger.debug(f"Could not verify channel count: {e}")
            else:
                logger.warning(f"Unusual input shape dimensions: {len(input_shape)}, expected 4")

            # Validate outputs
            outputs = session.get_outputs()

            # Check 1: Number of output layers
            if len(outputs) < 1:
                return {
                    'valid': False,
                    'reason': 'No output layers found',
                    'message': '',
                    'is_transposed': False
                }

            if len(outputs) > 2:
                logger.warning(f"Model has {len(outputs)} output layers, expected 1-2 for YOLOv11")

            # Check 2: Output shape for first (main) output
            main_output = outputs[0]
            output_shape = main_output.shape
            output_name = main_output.name

            logger.info(f"Model output: {output_name}, Shape: {output_shape}")

            if len(output_shape) != 3:
                return {
                    'valid': False,
                    'reason': f'Output shape dimension is {len(output_shape)}, expected 3D',
                    'message': '',
                    'is_transposed': False
                }

            # Check 3: Detect if output is transposed
            # Standard: (batch, num_detections, features) e.g., (1, 8400, 85)
            # Transposed: (batch, features, num_detections) e.g., (1, 5, 8400)

            batch_size = output_shape[0]
            dim1 = output_shape[1]
            dim2 = output_shape[2]

            # Determine if transposed: if dim1 is small (< 100) and dim2 is large (> 1000)
            is_transposed = False
            if isinstance(dim1, int) and isinstance(dim2, int):
                if dim1 < 100 and dim2 > 1000:
                    # Likely transposed: (batch, features, num_detections)
                    is_transposed = True
                    num_features = dim1
                    num_detections = dim2
                    logger.warning(f"TRANSPOSED format detected: [batch, features, detections] = [{batch_size}, {num_features}, {num_detections}]")
                    logger.warning(f"Will auto-transpose to [batch, detections, features] and convert CXCYWH to XYXY")
                else:
                    # Standard format: (batch, num_detections, features)
                    num_detections = dim1
                    num_features = dim2
                    logger.info(f"Standard format: [batch, detections, features] = [{batch_size}, {num_detections}, {num_features}]")
            else:
                # Dynamic dimensions
                num_detections = 8400  # Typical for 640x640
                num_features = dim2 if isinstance(dim2, int) else 85
                logger.warning(f"Dynamic dimensions detected, assuming standard format")

            # Batch size should be 1 or dynamic (None, 'batch', etc)
            if batch_size not in [None, 1, 'batch', 'N']:
                try:
                    batch_int = int(batch_size)
                    if batch_int > 1:
                        logger.warning(f"Model batch size is {batch_int}, expected 1 or dynamic")
                except (ValueError, TypeError):
                    pass  # Dynamic batch size (string like 'batch')

            # Check 4: Feature count (should be 4 coords + class scores)
            # For 1-class: 5 (4 + 1 class)
            # For N-class: 4 + N
            if isinstance(num_features, int) and num_features < 5:
                return {
                    'valid': False,
                    'reason': f'Feature count is {num_features}, expected >= 5 (4 coords + class scores)',
                    'message': '',
                    'is_transposed': is_transposed
                }

            # Success!
            transpose_note = " [TRANSPOSED - will auto-fix]" if is_transposed else " [Standard]"
            num_classes = num_features - 4 if isinstance(num_features, int) else "N"

            logger.info(f"Model validation PASSED - Format: YOLOv11 xyxy, Classes: {num_classes}, Compatible: YES")

            message = (f"Output: {output_name}, "
                      f"Shape: ({batch_size}, {dim1}, {dim2}){transpose_note}, "
                      f"Detected: {num_detections} detections, {num_features} features")

            logger.info(f"Validation complete: {message}")

            return {
                'valid': True,
                'reason': '',
                'message': message,
                'is_transposed': is_transposed
            }

        except Exception as e:
            logger.error(f"Model validation failed with exception: {e}")
            return {
                'valid': False,
                'reason': f'Validation error: {str(e)}',
                'message': '',
                'is_transposed': False
            }

    def _merge_tile_overlap_detections(self, all_detections, tile_size, overlap_px, iou_threshold=0.45):
        """
        Explicitly merge detections from overlapping tiles.

        Inspired by QGIS Deepness overlap removal strategy.
        Handles detections that appear in multiple tiles due to overlap.

        Strategy:
        1. Group nearby detections (potential duplicates from overlap)
        2. Weight by distance from tile edge (prefer centered detections)
        3. Apply weighted NMS to remove duplicates
        4. Keep highest confidence detection from best tile

        Args:
            all_detections: List of detection dicts with 'box', 'score', 'class', 'edge', 'border_dist'
            tile_size: Tile size in pixels
            overlap_px: Overlap in pixels
            iou_threshold: IoU threshold for considering detections as duplicates

        Returns:
            List of merged detections (duplicates removed)
        """
        if not all_detections or len(all_detections) == 0:
            return []

        try:
            # Sort by confidence (descending) for NMS
            sorted_dets = sorted(all_detections, key=lambda d: d.get('score', 0.0), reverse=True)

            merged = []
            suppressed = set()

            for i, det1 in enumerate(sorted_dets):
                if i in suppressed:
                    continue

                box1 = det1['box']
                score1 = det1.get('score', 0.0)
                edge1 = det1.get('edge', False)
                border_dist1 = det1.get('border_dist', float('inf'))

                # Check against all remaining detections
                for j in range(i + 1, len(sorted_dets)):
                    if j in suppressed:
                        continue

                    det2 = sorted_dets[j]
                    box2 = det2['box']
                    score2 = det2.get('score', 0.0)
                    edge2 = det2.get('edge', False)
                    border_dist2 = det2.get('border_dist', float('inf'))

                    # Only consider same class
                    if det1.get('class', 0) != det2.get('class', 0):
                        continue

                    # Calculate IoU
                    iou = self._calculate_iou(box1, box2)

                    if iou > iou_threshold:
                        # These are likely the same object detected in overlap zone
                        # Decide which one to keep based on:
                        # 1. Edge proximity (prefer non-edge detections)
                        # 2. Distance from border (prefer centered)
                        # 3. Confidence score

                        # Strategy: Keep the detection that is MORE CENTERED in its tile
                        if edge1 and not edge2:
                            # det2 is more centered, suppress det1
                            suppressed.add(i)
                            break  # Move to next det1
                        elif not edge1 and edge2:
                            # det1 is more centered, suppress det2
                            suppressed.add(j)
                        else:
                            # Both edge or both non-edge
                            # Compare border distances (larger = more centered)
                            if border_dist2 > border_dist1:
                                # det2 is more centered
                                suppressed.add(i)
                                break
                            else:
                                # det1 is more centered or equal
                                suppressed.add(j)

                # If not suppressed, keep detection
                if i not in suppressed:
                    merged.append(det1)

            logger.info(f"Overlap merge: {len(all_detections)} -> {len(merged)} detections "
                       f"({len(all_detections) - len(merged)} duplicates removed)")

            return merged

        except Exception as e:
            logger.error(f"Overlap merge failed: {e}, returning all detections")
            return all_detections

    def _calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union between two boxes.

        Args:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]

        Returns:
            float: IoU value [0, 1]
        """
        try:
            x1_1, y1_1, x2_1, y2_1 = box1
            x1_2, y1_2, x2_2, y2_2 = box2

            # Calculate intersection
            x_left = max(x1_1, x1_2)
            y_top = max(y1_1, y1_2)
            x_right = min(x2_1, x2_2)
            y_bottom = min(y2_1, y2_2)

            if x_right < x_left or y_bottom < y_top:
                return 0.0

            intersection = (x_right - x_left) * (y_bottom - y_top)

            # Calculate union
            box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
            box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
            union = box1_area + box2_area - intersection

            if union <= 0:
                return 0.0

            iou = intersection / union
            return iou

        except Exception as e:
            logger.warning(f"IoU calculation failed: {e}")
            return 0.0

    def _validate_fasterrcnn_model_outputs(self, session):
        """
        Validate ONNX model outputs for Faster R-CNN (boxes, labels, scores).

        Validates:
        - 3 output layers: boxes, labels, scores
        - Output shapes
        """
        try:
            logger.info(f"Model validation - Faster R-CNN")

            # Validate outputs
            outputs = session.get_outputs()

            # Check 1: Number of output layers
            if len(outputs) < 3:
                return {
                    'valid': False,
                    'reason': f'Faster R-CNN requires 3 outputs (boxes, labels, scores), found {len(outputs)}',
                    'message': '',
                    'is_transposed': False
                }

            # Check output names/shapes loosely (assuming order: boxes, labels, scores based on typical export)
            # Typically: boxes=(N,4), labels=(N,), scores=(N,)
            out0_shape = outputs[0].shape
            out1_shape = outputs[1].shape
            out2_shape = outputs[2].shape
            
            logger.info(f"Faster R-CNN outputs: {outputs[0].name}={out0_shape}, {outputs[1].name}={out1_shape}, {outputs[2].name}={out2_shape}")

            message = "Faster R-CNN format detected"
            return {
                'valid': True,
                'reason': '',
                'message': message,
                'is_transposed': False
            }

        except Exception as e:
            return {
                'valid': False,
                'reason': f'Validation error: {str(e)}',
                'message': '',
                'is_transposed': False
            }

    def _postprocess_fasterrcnn(self, outputs, scale, pad, tile_x, tile_y, original_tile_shape):
        """Post-process Faster R-CNN outputs (boxes, labels, scores)."""
        dets = []
        try:
            if len(outputs) < 3:
                logger.error("Faster R-CNN output missing layers")
                return dets

            # Unpack outputs (assuming standard torchvision export order: boxes, labels, scores)
            # Note: The order depends on how it was exported. 
            # Often: boxes (float32), labels (int64), scores (float32)
            
            # Heuristic to identify which is which if not strictly ordered?
            # For now assume: 0=boxes, 1=labels, 2=scores OR 0=boxes, 1=scores, 2=labels
            # We will try to identify by shape/type if possible or assume standard.
            # Standard torchvision: features (not returned), boxes, labels, scores? No, usually just boxes, labels, scores.
            
            # Let's assume the standard export from the notebook:
            # return boxes, labels, scores
            boxes = outputs[0]
            labels = outputs[1]
            scores = outputs[2]
            
            # Ensure they are numpy arrays
            if not isinstance(boxes, np.ndarray): boxes = np.array(boxes)
            if not isinstance(labels, np.ndarray): labels = np.array(labels)
            if not isinstance(scores, np.ndarray): scores = np.array(scores)
            
            # Check consistency
            num_dets = boxes.shape[0]
            if len(labels) != num_dets or len(scores) != num_dets:
                logger.warning(f"Faster R-CNN output mismatch: {boxes.shape}, {labels.shape}, {scores.shape}")
                return dets
                
            if num_dets == 0:
                return dets

            # Resize params
            try:
                th, tw = self.input_size
            except:
                th, tw = 640, 640

            for i in range(num_dets):
                score = float(scores[i])
                if score < self.conf_thresh:
                    continue
                
                label = int(labels[i])
                
                # Faster R-CNN boxes are typically absolute pixel coords [0, W] x [0, H] relative to input chart
                # But wait! If the model was exported with dynamic axes or fixed size.
                # The notebook export suggests standard torchvision behavior. 
                # Torchvision returns absolute coordinates relative to the image size passed to the model.
                # In _preprocess, we resized/letterboxed the image to self.input_size (e.g. 640x640).
                # So these boxes are in 640x640 pixel space.
                
                b = boxes[i]
                x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                
                # Reverse letterbox
                x1_local, y1_local, x2_local, y2_local = reverse_letterbox_coordinates(
                    x1, y1, x2, y2, scale, pad
                )
                
                x1_local, y1_local, x2_local, y2_local = ensure_valid_box_ordering(
                    x1_local, y1_local, x2_local, y2_local
                )
                
                if len(original_tile_shape) >= 2:
                    oh, ow = original_tile_shape[0], original_tile_shape[1]
                else:
                    oh, ow = 640, 640 # Should not happen if original_shape is valid
                x1_local, y1_local, x2_local, y2_local = clamp_to_bounds(
                    x1_local, y1_local, x2_local, y2_local, ow, oh
                )
                
                # Tile to global
                X1, Y1, X2, Y2 = tile_to_global_coords(
                    x1_local, y1_local, x2_local, y2_local, tile_x, tile_y
                )
                
                if X2 <= X1 or Y2 <= Y1:
                    continue

                # Edge detection
                edge, border_dist = compute_edge_metrics(
                    x1_local, y1_local, x2_local, y2_local, ow, oh, self.tile_overlap
                )
                
                dets.append({
                    'box': [X1, Y1, X2, Y2],
                    'score': score,
                    'class': label,
                    'tile': (int(tile_x), int(tile_y), int(ow), int(oh)),
                    'edge': edge,
                    'border_dist': border_dist
                })

        except Exception as e:
            logger.error(f"Faster R-CNN postprocess failed: {e}")
        
        return dets

    def _postprocess_yolo(self, outputs, scale, pad, tile_x, tile_y, original_tile_shape):
        """Robust postprocessing for YOLO ONNX output formats with comprehensive error handling.

        Critical for coordinate accuracy:
        - Handles normalized vs absolute coordinates
        - Reverse letterbox transform (scale, pad)
        - Maps tile-local coords to global image coords
        - Edge detection for tile-aware merging

        Returns: list of {'box':[x1,y1,x2,y2], 'score':float, 'class':int, 'edge':bool, 'border_dist':float}
        """
        dets = []
        try:
            if scale is None or scale <= 0:
                logger.error(f"Invalid scale: {scale}, cannot reverse transform coordinates")
                return dets

            if pad is None or len(pad) < 2:
                logger.warning(f"Invalid pad: {pad}, using (0, 0)")
                pad = (0, 0)

            if original_tile_shape is None or len(original_tile_shape) < 2:
                logger.error(f"Invalid tile shape: {original_tile_shape}")
                return dets

            if isinstance(outputs, list):
                arr = None
                for o in outputs:
                    if isinstance(o, np.ndarray):
                        arr = o
                        break
                if arr is None and len(outputs) > 0:
                    arr = np.array(outputs[0])
            else:
                arr = np.array(outputs)

            if arr is None:
                return dets

            try:
                if not hasattr(self, '_raw_output_logged'):
                    logger.info(f"Raw model output - Shape: {arr.shape}, dtype: {arr.dtype}")
                    if arr.size > 0:
                        logger.info(f"Value range: min={np.min(arr):.4f}, max={np.max(arr):.4f}")
                        if arr.ndim == 2 and arr.shape[0] > 0:
                            first_row = arr[0, :min(10, arr.shape[1])]
                            logger.debug(f"First detection sample (10 values): {first_row}")
                    self._raw_output_logged = True
            except Exception as e:
                logger.warning(f"Failed to log raw output: {e}")

            # CRITICAL: Handle transposed YOLOv11 outputs (features-first instead of detections-first)
            if arr.ndim == 2:
                dim0, dim1 = arr.shape
                if dim0 < TRANSPOSED_FEATURE_THRESHOLD and dim1 > TRANSPOSED_DETECTION_THRESHOLD:
                    logger.warning(f"Transposed output detected: {arr.shape} - Auto-transposing to (detections, features)")
                    arr = arr.T
                    logger.info(f"Transposed to format: {arr.shape}")

                    try:
                        if arr.shape[0] > 0:
                            first_row = arr[0, :min(10, arr.shape[1])]
                            logger.debug(f"First detection after transpose: {first_row}")
                    except (IndexError, AttributeError) as e:
                        logger.debug(f"Could not log detection sample: {e}")

                    # CRITICAL: Transposed models use CXCYWH format, convert to XYXY
                    logger.warning(f"Transposed models output CXCYWH format - Converting to XYXY")

                    if arr.shape[1] >= 5:
                        cx = arr[:, 0].copy()
                        cy = arr[:, 1].copy()
                        w = arr[:, 2].copy()
                        h = arr[:, 3].copy()

                        arr[:, 0] = cx - w / 2.0
                        arr[:, 1] = cy - h / 2.0
                        arr[:, 2] = cx + w / 2.0
                        arr[:, 3] = cy + h / 2.0

                        logger.info(f"Converted CXCYWH to XYXY format successfully")
                        try:
                            if arr.shape[0] > 0:
                                first_row_xyxy = arr[0, :min(5, arr.shape[1])]
                                logger.debug(f"First detection after CXCYWH->XYXY: {first_row_xyxy}")
                        except (IndexError, AttributeError) as e:
                            logger.debug(f"Could not log converted detection: {e}")

            def _shape_to_hw(s):
                try:
                    if s is None:
                        return 0, 0
                    if isinstance(s, (list, tuple)):
                        L = len(s)
                        if L == 3:
                            if int(s[0]) <= 4:
                                return int(s[1]), int(s[2])
                            if int(s[2]) <= 4:
                                return int(s[0]), int(s[1])
                            return int(s[0]), int(s[1])
                        elif L == 2:
                            return int(s[0]), int(s[1])
                        else:
                            return int(s[-2]), int(s[-1])
                    else:
                        return int(s[0]), int(s[1])
                except Exception as e:
                    logger.debug(f"Failed to parse output shape from primary method: {e}")
                    try:
                        return int(s[-2]), int(s[-1])
                    except Exception as e2:
                        logger.warning(f"Failed to parse output shape from fallback: {e2}")
                        return 0, 0

            if arr.ndim == 2 and arr.shape[1] >= MIN_YOLO_FEATURES:
                try:
                    coords = arr[:, :4]
                    if coords.size > 0 and np.all(np.max(np.abs(coords), axis=1) <= NORMALIZED_COORD_THRESHOLD):
                        th, tw = self.input_size
                        arr[:, 0] = arr[:, 0] * tw
                        arr[:, 2] = arr[:, 2] * tw
                        arr[:, 1] = arr[:, 1] * th
                        arr[:, 3] = arr[:, 3] * th
                        logger.debug("[NORMALIZE-DETECT] Batch normalized coords detected and scaled to input pixels")
                except Exception as e:
                    logger.debug(f"Failed to scale normalized batch coordinates: {e}")
                for row in arr:
                    try:
                        D = row.shape[0]
                        if D >= 6:
                            x, y, w, h, conf = float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])
                            cls = int(row[5]) if D >= 6 else 0
                            if D > 6:
                                try:
                                    cls = int(np.argmax(row[5:]))
                                    conf = float(conf)
                                except Exception as e:
                                    logger.debug(f"Failed to parse multi-class detection: {e}")
                                    cls = int(cls)

                        elif D == 5:
                            x, y, w, h, conf = float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])
                            cls = 0
                        else:
                            continue

                        if conf < self.conf_thresh:
                            continue

                        # FORCE YOLOv11 xyxy format
                        try:
                            th, tw = self.input_size
                        except Exception as e:
                            logger.warning(f"Failed to unpack input_size tuple: {e}")
                            th, tw = self.input_size[0], self.input_size[1]

                        max_coord = max(abs(x), abs(y), abs(w), abs(h))
                        if max_coord <= NORMALIZED_COORD_THRESHOLD:
                            x *= tw
                            w *= tw
                            y *= th
                            h *= th

                        x1_letterbox, y1_letterbox = x, y
                        x2_letterbox, y2_letterbox = w, h
                        logger.debug(f"[FORMAT FORCE] Forcing YOLOv11 xyxy: x1={x1_letterbox:.2f},y1={y1_letterbox:.2f},x2={x2_letterbox:.2f},y2={y2_letterbox:.2f}")

                        # CRITICAL: Filter padding area detections
                        try:
                            th, tw = self.input_size
                            valid_x_min = pad[0]
                            valid_y_min = pad[1]
                            valid_x_max = tw - pad[0]
                            valid_y_max = th - pad[1]

                            center_x = (x1_letterbox + x2_letterbox) / 2.0
                            center_y = (y1_letterbox + y2_letterbox) / 2.0

                            if center_x < valid_x_min or center_x > valid_x_max or \
                               center_y < valid_y_min or center_y > valid_y_max:
                                logger.debug(f"[PADDING FILTER] Rejecting detection in padding area: "
                                           f"center=({center_x:.1f},{center_y:.1f}), "
                                           f"valid_region=[{valid_x_min:.1f},{valid_y_min:.1f},{valid_x_max:.1f},{valid_y_max:.1f}]")
                                continue
                        except Exception as e:
                            logger.warning(f"Padding filter check failed: {e}")

                        # CRITICAL: Reverse letterbox transform
                        x1_local, y1_local, x2_local, y2_local = reverse_letterbox_coordinates(
                            x1_letterbox, y1_letterbox, x2_letterbox, y2_letterbox, scale, pad
                        )

                        x1_local, y1_local, x2_local, y2_local = ensure_valid_box_ordering(
                            x1_local, y1_local, x2_local, y2_local
                        )

                        oh, ow = _shape_to_hw(original_tile_shape)
                        x1_local, y1_local, x2_local, y2_local = clamp_to_bounds(
                            x1_local, y1_local, x2_local, y2_local, ow, oh
                        )

                        x1, y1, x2, y2 = tile_to_global_coords(
                            x1_local, y1_local, x2_local, y2_local, tile_x, tile_y
                        )

                        if len(dets) < MAX_COORD_TRANSFORM_LOGS:
                            logger.info(f"[COORD TRANSFORM] Tile ({tile_x},{tile_y}): "
                                       f"letterbox=[{x1_letterbox:.1f},{y1_letterbox:.1f},{x2_letterbox:.1f},{y2_letterbox:.1f}] "
                                       f"-> local=[{x1_local:.1f},{y1_local:.1f},{x2_local:.1f},{y2_local:.1f}] "
                                       f"-> global=[{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}] "
                                       f"| scale={scale:.3f}, pad={pad}")

                        if x2 <= x1 or y2 <= y1:
                            logger.warning(f"Invalid box after transform: [{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}], skipping")
                            continue

                        score = float(conf)

                        try:
                            edge, border_dist = compute_edge_metrics(
                                x1_local, y1_local, x2_local, y2_local, ow, oh, self.tile_overlap
                            )
                        except Exception as e:
                            logger.warning(f"Edge detection failed: {e}")
                            border_dist = 0.0
                            edge = False

                        dets.append({
                            'box': [x1, y1, x2, y2],
                            'score': score,
                            'class': int(cls),
                            'tile': (int(tile_x), int(tile_y), int(ow), int(oh)),
                            'edge': edge,
                            'border_dist': border_dist
                        })
                    except Exception as e:
                        logger.warning(f"Failed to process detection row: {e}")
                        continue

            elif isinstance(outputs, list) and len(outputs) >= 3:
                try:
                    boxes = np.array(outputs[0])
                    scores = np.array(outputs[1]).ravel()
                    labels = np.array(outputs[2]).ravel()

                    try:
                        if boxes.ndim == 2 and boxes.shape[0] > 0 and np.all(np.max(np.abs(boxes[:, :4]), axis=1) <= NORMALIZED_COORD_THRESHOLD):
                            th, tw = self.input_size
                            boxes[:, 0] = boxes[:, 0] * tw
                            boxes[:, 2] = boxes[:, 2] * tw
                            boxes[:, 1] = boxes[:, 1] * th
                            boxes[:, 3] = boxes[:, 3] * th
                            logger.debug("[NORMALIZE-DETECT] Batch normalized boxes scaled to input pixels (multi-output)")
                    except Exception as e:
                        logger.warning(f"Failed to scale normalized boxes: {e}", exc_info=True)

                    for i in range(min(len(boxes), len(scores), len(labels))):
                        try:
                            b = boxes[i]
                            s = float(scores[i])
                            cls = int(labels[i])

                            if s < self.conf_thresh:
                                continue

                            x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])

                            if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= NORMALIZED_COORD_THRESHOLD:
                                x1 *= self.input_size[1]
                                x2 *= self.input_size[1]
                                y1 *= self.input_size[0]
                                y2 *= self.input_size[0]

                            x1_local, y1_local, x2_local, y2_local = reverse_letterbox_coordinates(
                                x1, y1, x2, y2, scale, pad
                            )

                            x1_local, y1_local, x2_local, y2_local = ensure_valid_box_ordering(
                                x1_local, y1_local, x2_local, y2_local
                            )

                            oh, ow = _shape_to_hw(original_tile_shape)
                            x1_local, y1_local, x2_local, y2_local = clamp_to_bounds(
                                x1_local, y1_local, x2_local, y2_local, ow, oh
                            )

                            X1, Y1, X2, Y2 = tile_to_global_coords(
                                x1_local, y1_local, x2_local, y2_local, tile_x, tile_y
                            )

                            if X2 <= X1 or Y2 <= Y1:
                                logger.debug(f"Invalid box after transform: [{X1},{Y1},{X2},{Y2}], skipping")
                                continue

                            try:
                                edge, border_dist = compute_edge_metrics(
                                    x1_local, y1_local, x2_local, y2_local,
                                    ow, oh, self.tile_overlap
                                )
                            except Exception as e:
                                logger.warning(f"Edge detection failed: {e}")
                                border_dist = 0.0
                                edge = False

                            dets.append({
                                'box': [X1, Y1, X2, Y2],
                                'score': s,
                                'class': cls,
                                'tile': (int(tile_x), int(tile_y), int(ow), int(oh)),
                                'edge': edge,
                                'border_dist': border_dist
                            })
                        except Exception as e:
                            logger.warning(f"Failed to process detection {i}: {e}")
                            continue
                except Exception as e:
                    logger.error(f"Failed to process multi-output format: {e}")
                    pass

        except Exception as e:
            logger.error(f"Postprocessing failed: {e}")
            pass

        return dets

    def run(self):
        global_dets = []
        total = len(self.tiles)
        processed = 0

        try:
            for i in range(0, total, self.batch_size):
                if self._stopped:
                    break
                batch = self.tiles[i:i+self.batch_size]
                inputs = []
                meta = []
                for t in batch:
                    arr, scale, pad, original_shape = self._preprocess(t)
                    inputs.append(arr)
                    meta.append((t['x'], t['y'], original_shape, scale, pad))
                inp_batch = np.stack(inputs, axis=0)

                sess = self.session
                if sess is None and self.model_path:
                    try:
                        import onnxruntime as ort
                        providers = []
                        try:
                            providers = [p for p in ort.get_available_providers()]
                        except Exception as e:
                            logger.debug(f"Failed to get ONNX providers: {e}")
                            providers = []
                        try:
                            sess = ort.InferenceSession(self.model_path, providers=providers if providers else None)
                        except Exception as e:
                            logger.debug(f"Failed to create session with providers, trying without: {e}")
                            sess = ort.InferenceSession(self.model_path)
                    except Exception as e:
                        self.error.emit(f"Inference failed: onnxruntime not available or session creation failed: {e}")
                        return

                if sess is None:
                    self.error.emit("Inference failed: no ONNX session available")
                    return

                if hasattr(self, 'model_type') and self.model_type == 'fasterrcnn':
                    try:
                        validation_result = self._validate_fasterrcnn_model_outputs(sess)
                        if not validation_result['valid']:
                            self.error.emit(f"Model validation failed: {validation_result['reason']}")
                            logger.error(f"Model validation failed: {validation_result['reason']}")
                            return
                        logger.info(f"Model validation passed: {validation_result['message']}")
                    except Exception as e:
                        logger.warning(f"Model validation check failed: {e}, continuing...")
                else:
                    try:
                        validation_result = self._validate_yolov11_model_outputs(sess)
                        if not validation_result['valid']:
                            self.error.emit(f"Model validation failed: {validation_result['reason']}")
                            logger.error(f"Model validation failed: {validation_result['reason']}")
                            return
                        logger.info(f"Model validation passed: {validation_result['message']}")
                    except Exception as e:
                        logger.warning(f"Model validation check failed with exception: {e}, continuing anyway...")

                try:
                    input_name = sess.get_inputs()[0].name
                    if inp_batch.dtype != np.float32:
                        inp_batch = inp_batch.astype(np.float32)
                except Exception as e:
                    self.error.emit(f"Inference failed: unable to prepare input: {e}")
                    return

                try:
                    model_input_shape = sess.get_inputs()[0].shape
                    model_batch_dim = None
                    try:
                        mb = model_input_shape[0]
                        if mb is None:
                            model_batch_dim = None
                        else:
                            model_batch_dim = int(mb)
                    except Exception as e:
                        logger.debug(f"Failed to parse batch dimension: {e}")
                        model_batch_dim = None
                except Exception as e:
                    logger.debug(f"Failed to get model input shape: {e}")
                    model_input_shape = None
                    model_batch_dim = None

                try:
                    try:
                        providers = sess.get_providers() if hasattr(sess, 'get_providers') else []
                    except Exception as e:
                        logger.debug(f"Failed to get session providers: {e}")
                        providers = []
                    logger.info(f"ONNX session providers: {providers} | model_input_shape: {model_input_shape} | inferred_batch_dim: {model_batch_dim}")
                except Exception as e:
                    logger.debug(f"Failed to log session info: {e}")

                batch_dets = []
                try:
                    if model_batch_dim is None or model_batch_dim <= 0:
                        try:
                            logger.debug(f"Running dynamic batch of size {inp_batch.shape[0]}")
                        except Exception as e:
                            logger.debug(f"Failed to log batch size: {e}")
                        outs = sess.run(None, {input_name: inp_batch})
                        if hasattr(self, 'model_type') and self.model_type == 'fasterrcnn':
                             # Faster R-CNN typically handles one image at a time or returns list of results
                             # We assume outputs are [boxes, labels, scores]
                             for bi in range(inp_batch.shape[0]):
                                 tx, ty, shape, scale, pad = meta[bi]
                                 # We need to slice outputs for this batch item if possible
                                 # If outputs are list of tensors:
                                 # outputs[0] = boxes (N, 4) ? Or (Batch, N, 4)?
                                 # FasterRCNN with batch > 1 usually returns list of dicts or list of tensors per image?
                                 # Standard ONNX export usually fixes batch=1. 
                                 # If batch=1, outputs are just tensors.
                                 dets = self._postprocess_fasterrcnn(outs, scale, pad, tx, ty, shape)
                                 batch_dets.extend(dets)
                        else:
                            first_out = outs[0] if isinstance(outs, list) and len(outs) > 0 else outs
                            if isinstance(first_out, np.ndarray) and first_out.ndim >= 2:
                                for bi in range(inp_batch.shape[0]):
                                    single = first_out[bi] if first_out.ndim == 3 else first_out
                                    tx, ty, shape, scale, pad = meta[bi]
                                    dets = self._postprocess_yolo(single, scale, pad, tx, ty, shape)
                                    batch_dets.extend(dets)
                            else:
                                for (tx, ty, shape, scale, pad) in meta:
                                    dets = self._postprocess_yolo(first_out, scale, pad, tx, ty, shape)
                                    batch_dets.extend(dets)
                    else:
                        try:
                            logger.info(f"Model has fixed batch_dim={model_batch_dim}; splitting input batch of size {inp_batch.shape[0]} into sub-batches")
                        except Exception as e:
                            logger.debug(f"Failed to log batch split info: {e}")
                        total_inp = inp_batch.shape[0]
                        for start_idx in range(0, total_inp, model_batch_dim):
                            end_idx = min(start_idx + model_batch_dim, total_inp)
                            sub_inp = inp_batch[start_idx:end_idx]
                            sub_meta = meta[start_idx:end_idx]
                            try:
                                logger.debug(f"Running sub-batch {start_idx}:{end_idx} size {sub_inp.shape[0]}")
                            except Exception as e:
                                logger.debug(f"Failed to log sub-batch info: {e}")
                            outs = sess.run(None, {input_name: sub_inp})
                            if hasattr(self, 'model_type') and self.model_type == 'fasterrcnn':
                                 for bi in range(sub_inp.shape[0]):
                                     tx, ty, shape, scale, pad = sub_meta[bi]
                                     dets = self._postprocess_fasterrcnn(outs, scale, pad, tx, ty, shape)
                                     batch_dets.extend(dets)
                            else:
                                first_out = outs[0] if isinstance(outs, list) and len(outs) > 0 else outs
                                if isinstance(first_out, np.ndarray) and first_out.ndim >= 2:
                                    # handle per-item outputs
                                    for bi in range(first_out.shape[0]):
                                        single = first_out[bi] if first_out.ndim == 3 else first_out
                                        tx, ty, shape, scale, pad = sub_meta[bi]
                                        dets = self._postprocess_yolo(single, scale, pad, tx, ty, shape)
                                        batch_dets.extend(dets)
                                else:
                                    for (tx, ty, shape, scale, pad) in sub_meta:
                                        dets = self._postprocess_yolo(first_out, scale, pad, tx, ty, shape)
                                        batch_dets.extend(dets)
                except Exception as e:
                    self.error.emit(f"Inference failed: {e}")
                    return

                try:
                    batch_dets = self._nms_all_classes(batch_dets, self.iou_thresh)
                except Exception as e:
                    logger.warning(f"Batch NMS failed: {e}")

                try:
                    self._merge_into_global(global_dets, batch_dets, self.iou_thresh)
                except Exception as e:
                    logger.debug(f"Merge into global failed, extending: {e}")
                    global_dets.extend(batch_dets)

                processed += len(batch)
                self.progress.emit(int((processed/total)*100))

            try:
                final_dets = self._nms_all_classes(global_dets, self.iou_thresh)
                final_dets = self._merge_overlaps(final_dets, self.iou_thresh)
                final_dets = self._stitch_detections(final_dets, self.iou_thresh)
                try:
                    final_dets = self._nms_all_classes(final_dets, YOLO_TIGHT_IOU_THRESHOLD)
                    logger.debug(f"Applied secondary global NMS (iou=0.3), results={len(final_dets)}")
                except Exception as e:
                    logger.debug(f"Secondary NMS failed: {e}")
            except Exception as e:
                logger.warning(f"Final post-processing failed: {e}")
                final_dets = global_dets

            try:
                logger.info(f"Applying explicit overlap merge (Deepness-inspired)...")
                final_dets = self._merge_tile_overlap_detections(
                    final_dets,
                    tile_size=self.input_size[0],
                    overlap_px=self.tile_overlap,
                    iou_threshold=self.iou_thresh
                )
            except Exception as e:
                logger.warning(f"Explicit overlap merge failed: {e}, continuing with existing detections")

            # CRITICAL: Emit complete detection results with box coordinates
            complete_dets = []
            for idx, d in enumerate(final_dets):
                try:
                    score = float(d.get('score', 0.0))
                    if score < self.conf_thresh:
                        continue

                    cls = int(d.get('class', d.get('label', 0)))
                    box = d.get('box', [])

                    if not box or len(box) < 4:
                        logger.warning(f"Detection {idx} missing valid box coordinates, skipping")
                        continue

                    try:
                        box_validated = [float(x) for x in box[:4]]
                        if not all(np.isfinite(box_validated)):
                            logger.warning(f"Detection {idx} has invalid box values: {box_validated}, skipping")
                            continue
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Detection {idx} box conversion failed: {e}, skipping")
                        continue

                    x1, y1, x2, y2 = box_validated
                    if x2 <= x1 or y2 <= y1:
                        logger.warning(f"Detection {idx} has invalid box dimensions: [{x1},{y1},{x2},{y2}], skipping")
                        continue

                    complete_dets.append({
                        'id': int(idx),
                        'class': cls,
                        'score': score,
                        'box': box_validated
                    })
                except Exception as e:
                    logger.warning(f"Failed to process detection {idx}: {e}")
                    continue

            try:
                logger.info(f"WORKER_DETECTIONS count={len(complete_dets)} (with validated coordinates)")
                if complete_dets:
                    sample = complete_dets[:MAX_SAMPLE_DETECTIONS_LOG]
                    for s in sample:
                        logger.info(f"  Sample: class={s['class']}, score={s['score']:.3f}, box={[round(x,2) for x in s['box']]}")
            except Exception as e:
                logger.debug(f"Failed to log detection samples: {e}")

            self.finished.emit({'detections': complete_dets})
        except Exception as e:
            logger.error(f"Detection worker failed: {e}", exc_info=True)
            self.error.emit(str(e))

    def _iou(self, box_a, box_b):
        # boxes are [x1,y1,x2,y2]
        xa1, ya1, xa2, ya2 = box_a
        xb1, yb1, xb2, yb2 = box_b
        inter_x1 = max(xa1, xb1)
        inter_y1 = max(ya1, yb1)
        inter_x2 = min(xa2, xb2)
        inter_y2 = min(ya2, yb2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
        area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union

    def _iou_matrix(self, boxes):
        """Hitung matrix IoU N x N sekaligus pakai numpy broadcasting.
        Hasilnya identik dgn manggil self._iou() utk tiap pasangan (i,j),
        cuma jauh lebih cepat karena gak ada overhead loop + call Python."""
        boxes = np.asarray(boxes, dtype=np.float64)
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        ix1 = np.maximum(x1[:, None], x1[None, :])
        iy1 = np.maximum(y1[:, None], y1[None, :])
        ix2 = np.minimum(x2[:, None], x2[None, :])
        iy2 = np.minimum(y2[:, None], y2[None, :])
        iw = np.maximum(0.0, ix2 - ix1)
        ih = np.maximum(0.0, iy2 - iy1)
        inter = iw * ih
        union = areas[:, None] + areas[None, :] - inter
        iou = np.zeros_like(union)
        valid = union > 0
        iou[valid] = inter[valid] / union[valid]
        return iou

    def _nms(self, dets, iou_thresh=0.45):
        # dets: list of {'box':[x1,y1,x2,y2], 'score':float, 'class':int}
        # NOTE: divectorize pakai numpy (2025), hasil terverifikasi identik
        # dgn implementasi loop-python lama (lihat catatan tim ttg verifikasi).
        if not dets:
            return []
        boxes = np.array([d['box'] for d in dets], dtype=np.float64)
        scores = np.array([d['score'] for d in dets], dtype=np.float64)
        order = np.argsort(-scores, kind='stable')  # descending, stable spt sorted() lama
        mat = self._iou_matrix(boxes)
        n = len(dets)
        suppressed = np.zeros(n, dtype=bool)
        keep_idx = []
        for idx in order:
            if suppressed[idx]:
                continue
            keep_idx.append(idx)
            suppressed[idx] = True
            suppressed |= (mat[idx] >= iou_thresh)
        return [dets[i] for i in keep_idx]

    def _nms_all_classes(self, dets, iou_thresh=0.45):
        by_class = {}
        for d in dets:
            cls = int(d.get('class', 0))
            by_class.setdefault(cls, []).append(d)
        final = []
        for cls, items in by_class.items():
            kept = self._nms(items, iou_thresh=iou_thresh)
            final.extend(kept)
        return final

    def _cluster_and_merge(self, dets, iou_thresh, keep_class_only):
        """Logic bersama utk _merge_overlaps & _stitch_detections (algoritma
        clustering-nya identik, cuma beda field yg disimpan di output).
        Divectorize: IoU matrix dihitung sekali per kelas pakai numpy,
        loop clustering-nya tetap sama persis urutan/kondisinya spt versi lama,
        cuma lookup IoU-nya jadi O(1) dari matrix, bukan hitung ulang tiap pasangan."""
        if not dets:
            return []
        by_class = {}
        for d in dets:
            cls = int(d.get('class', 0))
            by_class.setdefault(cls, []).append(d)

        merged_all = []
        for cls, items in by_class.items():
            n = len(items)
            boxes = np.array([it['box'] for it in items], dtype=np.float64)
            mat = self._iou_matrix(boxes) if n > 0 else np.zeros((0, 0))
            used = np.zeros(n, dtype=bool)
            for i in range(n):
                if used[i]:
                    continue
                used[i] = True
                if i + 1 < n:
                    mask = (~used[i+1:]) & (mat[i, i+1:] >= iou_thresh)
                    matched = np.where(mask)[0] + i + 1
                else:
                    matched = np.array([], dtype=int)
                if len(matched) > 0:
                    used[matched] = True
                    cluster_idx = np.concatenate(([i], matched))
                else:
                    cluster_idx = np.array([i])
                cluster = [items[k] for k in cluster_idx]

                if len(cluster) == 1:
                    c0 = cluster[0]
                    if keep_class_only:
                        merged_all.append({'box': c0['box'], 'score': c0['score'], 'class': cls})
                    else:
                        merged_all.append(c0)
                else:
                    scores = np.array([c['score'] for c in cluster], dtype=np.float64)
                    edges = np.array([1.0 if not c.get('edge', False) else YOLO_EDGE_WEIGHT_MULTIPLIER for c in cluster], dtype=np.float64)
                    weighted = scores * edges
                    weights = weighted / (weighted.sum() + EPSILON)
                    bx = np.array([c['box'] for c in cluster], dtype=np.float64)
                    x1 = float((bx[:, 0] * weights).sum())
                    y1 = float((bx[:, 1] * weights).sum())
                    x2 = float((bx[:, 2] * weights).sum())
                    y2 = float((bx[:, 3] * weights).sum())
                    merged_score = float(scores.max())
                    merged_all.append({'box': [x1, y1, x2, y2], 'score': merged_score, 'class': cls})

        return sorted(merged_all, key=lambda d: d['score'], reverse=True)

    def _merge_overlaps(self, dets, iou_thresh=0.45):
        """Merge overlapping detections by clustering boxes with IoU >= threshold
        and averaging box coordinates weighted by score. Returns new det list.
        """
        return self._cluster_and_merge(dets, iou_thresh, keep_class_only=False)

    def _merge_into_global(self, global_list, batch_list, iou_thresh=0.45):
        """Merge batch_list detections into global_list in-place.

        For each detection in batch_list, check if it overlaps (IoU >= iou_thresh)
        with any existing detection in global_list (same class). If so, fuse them
        by weighted averaging coordinates using scores; otherwise append to global_list.
        """
        if not batch_list:
            return
        alpha = YOLO_BORDER_DISTANCE_ALPHA
        eps = EPSILON
        for b in batch_list:
            merged = False
            b_cls = int(b.get('class', 0))
            for gi, g in enumerate(global_list):
                if int(g.get('class', 0)) != b_cls:
                    continue
                try:
                    iou = self._iou(g['box'], b['box'])
                except Exception as e:
                    logger.debug(f"Failed to compute IOU: {e}")
                    iou = 0.0
                if iou >= iou_thresh:
                    # compute score multipliers using border distance if available
                    s1 = float(g.get('score', 1.0))
                    s2 = float(b.get('score', 1.0))
                    bd1 = float(g.get('border_dist', 0.0)) if g.get('border_dist', None) is not None else (0.0 if g.get('edge', False) else 1.0)
                    bd2 = float(b.get('border_dist', 0.0)) if b.get('border_dist', None) is not None else (0.0 if b.get('edge', False) else 1.0)
                    m1 = 1.0 + alpha * bd1
                    m2 = 1.0 + alpha * bd2
                    w1 = (s1 * m1) / (s1 * m1 + s2 * m2 + eps)
                    w2 = (s2 * m2) / (s1 * m1 + s2 * m2 + eps)
                    bx = [w1 * g['box'][0] + w2 * b['box'][0],
                          w1 * g['box'][1] + w2 * b['box'][1],
                          w1 * g['box'][2] + w2 * b['box'][2],
                          w1 * g['box'][3] + w2 * b['box'][3]]
                    new_score = float(max(s1, s2))
                    # choose tile info from stronger contributor
                    tile_info = g.get('tile') if (s1 * m1) >= (s2 * m2) else b.get('tile')
                    edge_flag = bool(g.get('edge', False) and b.get('edge', False))
                    border_dist = (bd1 * w1 + bd2 * w2)
                    global_list[gi] = {'box': bx, 'score': new_score, 'class': b_cls, 'tile': tile_info, 'edge': edge_flag, 'border_dist': border_dist}
                    merged = True
                    break
            if not merged:
                # ensure border_dist exists on appended detections
                if 'border_dist' not in b:
                    b['border_dist'] = 0.0
                global_list.append(b)

    def _stitch_detections(self, dets, iou_thresh=0.45):
        """Final seam-aware stitching pass. Groups detections by class and IoU,
        then merges clusters using edge-aware weighted averaging so that
        detections near tile borders contribute less and central detections win.
        Returns a new list of merged detections with keys 'box','score','class'.
        """
        return self._cluster_and_merge(dets, iou_thresh, keep_class_only=True)

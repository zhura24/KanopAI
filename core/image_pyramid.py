"""Image pyramid untuk multi-level rendering."""

import numpy as np
from collections import OrderedDict
import cv2


class ImagePyramid:
    """Pyramid cache untuk rendering cepat di berbagai zoom level."""
    def __init__(self, max_cache_size=5):
        self.levels = OrderedDict()
        self.max_cache_size = max_cache_size
        self.base_data = None
        self.base_shape = None

    def set_base_image(self, data):
        self.base_data = data
        if len(data.shape) == 3:
            self.base_shape = (data.shape[1], data.shape[2])
        else:
            self.base_shape = data.shape

        self.levels.clear()
        self.levels[1.0] = data

    def get_level(self, scale):
        if scale >= 1.0:
            return self.base_data

        if scale in self.levels:
            self.levels.move_to_end(scale)
            return self.levels[scale]

        downsampled = self._create_level(scale)

        self.levels[scale] = downsampled
        self.levels.move_to_end(scale)

        if len(self.levels) > self.max_cache_size:
            self.levels.popitem(last=False)

        return downsampled

    def _create_level(self, scale):
        if self.base_data is None:
            return None

        if len(self.base_data.shape) == 3:
            bands, height, width = self.base_data.shape
            new_height = int(height * scale)
            new_width = int(width * scale)

            result = np.zeros((bands, new_height, new_width), dtype=self.base_data.dtype)

            for i in range(bands):
                result[i] = cv2.resize(
                    self.base_data[i],
                    (new_width, new_height),
                    interpolation=cv2.INTER_AREA
                )

            return result
        else:
            height, width = self.base_data.shape
            new_height = int(height * scale)
            new_width = int(width * scale)

            return cv2.resize(
                self.base_data,
                (new_width, new_height),
                interpolation=cv2.INTER_AREA
            )

    def get_optimal_scale(self, target_dimension):
        if self.base_shape is None:
            return 1.0

        max_dim = max(self.base_shape)

        if max_dim <= target_dimension:
            return 1.0

        return target_dimension / max_dim

    def clear_cache(self):
        base = self.levels.get(1.0)
        self.levels.clear()
        if base is not None:
            self.levels[1.0] = base

    def get_memory_usage(self):
        total = 0
        for scale, data in self.levels.items():
            if data is not None:
                total += data.nbytes
        return total / (1024 * 1024)

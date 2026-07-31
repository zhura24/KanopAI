"""
Smart Router for Multispectral Band Mapping
Automatically detects and routes multispectral bands to standard slots.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

class SmartRouter:
    """
    Otak untuk mendeteksi identitas band dan melakukan routing ke slot standar.
    Standard 8-Slot Mapping:
    0: Coastal Blue
    1: Blue
    2: Green
    3: Yellow
    4: Red
    5: Red Edge
    6: NIR 1
    7: NIR 2
    """
    
    # Keyword mapping untuk deteksi otomatis dari metadata
    BAND_KEYWORDS = {
        'coastal': 0,
        'blue': 1,
        'green': 2,
        'yellow': 3,
        'red': 4,
        'edge': 5,
        'nir': 6,
        'near': 6,
        'ir': 7,  # Infrared tambahan
    }

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def get_mapping(metadata: Dict[str, Any]) -> Dict[str, int]:
        """
        Mendeteksi mapping band dari metadata rasterio.
        Returns: Dict mapping 'slot_index': 'source_band_index' (0-based)
        """
        num_bands = metadata.get('bands', 0)
        descriptions = metadata.get('band_descriptions', [])
        
        # Default mapping (RGB as default)
        # Jika RGB, kita taruh di slot yang sesuai (Blue=1, Green=2, Red=4)
        mapping = {}
        
        # Case 1: Detect by descriptions
        if descriptions:
            for i, desc in enumerate(descriptions):
                if not desc: continue
                desc_lower = desc.lower()
                for kw, slot in SmartRouter.BAND_KEYWORDS.items():
                    if kw in desc_lower:
                        mapping[slot] = i
                        break

        # Case 2: Profiling based on band count (if Case 1 failed or incomplete)
        if len(mapping) < min(num_bands, 3):
            if num_bands == 3: # Standard RGB
                mapping = {4: 0, 2: 1, 1: 2} # Red=0, Green=1, Blue=2 -> Slot 4, 2, 1
            elif num_bands == 4: # R-G-B-NIR (Common in Planet/DJI)
                mapping = {4: 0, 2: 1, 1: 2, 6: 3} 
            elif num_bands == 5: # R-G-B-RE-NIR (Common in MicaSense)
                mapping = {4: 0, 2: 1, 1: 2, 5: 3, 6: 4}
            elif num_bands == 7: # Custom 7-band sensor (User's data)
                # Band 1,2,3 -> Blue, Green, Red. Band 6 -> NIR
                mapping = {1: 0, 2: 1, 4: 2, 6: 5} 
            elif num_bands >= 8: # WorldView-3 style
                # Assume standard order if no descriptions
                mapping = {i: i for i in range(min(num_bands, 8))}

        return mapping

    @staticmethod
    def get_display_indices(mapping: Dict[str, int], num_bands: int) -> Tuple[int, int, int]:
        """
        Mendapatkan 3 index band untuk tampilan RGB terbaik.
        """
        # Prioritas: Slot 4(R), 2(G), 1(B)
        r = mapping.get(4, 0)
        g = mapping.get(2, min(1, num_bands-1))
        b = mapping.get(1, min(2, num_bands-1))
        return r, g, b

    @staticmethod
    def prepare_8band_data(data: np.ndarray, mapping: Dict[str, int]) -> np.ndarray:
        """
        Menyiapkan data 8-channel untuk YOLOv11 dengan Zero Padding.
        Input data: (C, H, W)
        Output data: (8, H, W)
        """
        if data.ndim == 2: # Grayscale
            h, w = data.shape
            data = data.reshape(1, h, w)
            
        c, h, w = data.shape
        output = np.zeros((8, h, w), dtype=data.dtype)
        
        for slot, src_idx in mapping.items():
            if slot < 8 and src_idx < c:
                output[slot] = data[src_idx]
                
        return output

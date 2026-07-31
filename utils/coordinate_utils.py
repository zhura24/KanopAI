"""Utilitas transformasi koordinat untuk pipeline deteksi."""

from typing import Tuple


def reverse_letterbox_coordinates(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    scale: float,
    pad: Tuple[int, int]
) -> Tuple[float, float, float, float]:
    """Balikkan transformasi letterbox ke koordinat gambar asli.

    Args:
        x1, y1: Sudut kiri atas dalam ruang letterbox
        x2, y2: Sudut kanan bawah dalam ruang letterbox
        scale: Faktor skala yang diterapkan selama letterbox
        pad: (pad_kiri, pad_atas) padding yang ditambahkan

    Returns:
        (x1, y1, x2, y2) dalam ruang gambar asli
    """
    x1_unpadded = x1 - pad[0]
    y1_unpadded = y1 - pad[1]
    x2_unpadded = x2 - pad[0]
    y2_unpadded = y2 - pad[1]

    x1_local = x1_unpadded / scale
    y1_local = y1_unpadded / scale
    x2_local = x2_unpadded / scale
    y2_local = y2_unpadded / scale
    return x1_local, y1_local, x2_local, y2_local


def ensure_valid_box_ordering(
    x1: float,
    y1: float,
    x2: float,
    y2: float
) -> Tuple[float, float, float, float]:
    """Pastikan koordinat kotak terurut (x1 < x2, y1 < y2)."""
    if x2 < x1:
        x1, x2 = min(x1, x2), max(x1, x2)
    if y2 < y1:
        y1, y2 = min(y1, y2), max(y1, y2)
    return x1, y1, x2, y2


def clamp_to_bounds(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: float,
    height: float
) -> Tuple[float, float, float, float]:
    """Batasi koordinat kotak ke batas gambar."""
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    return x1, y1, x2, y2


def tile_to_global_coords(
    x1_local: float,
    y1_local: float,
    x2_local: float,
    y2_local: float,
    tile_x: int,
    tile_y: int
) -> Tuple[float, float, float, float]:
    """Konversi koordinat lokal tile ke koordinat gambar global."""
    x1 = tile_x + x1_local
    y1 = tile_y + y1_local
    x2 = tile_x + x2_local
    y2 = tile_y + y2_local
    return x1, y1, x2, y2


def compute_box_center(
    x1: float,
    y1: float,
    x2: float,
    y2: float
) -> Tuple[float, float]:
    """Hitung titik tengah kotak bounding."""
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return cx, cy


def compute_border_distance(
    cx: float,
    cy: float,
    width: float,
    height: float,
    normalize: bool = True
) -> float:
    """Hitung jarak minimum dari pusat kotak ke batas gambar."""
    left_dist = cx
    right_dist = width - cx
    top_dist = cy
    bottom_dist = height - cy
    min_dist = float(min(left_dist, right_dist, top_dist, bottom_dist))

    if normalize:
        denom = max(1.0, min(height, width) / 2.0)
        return max(0.0, min(1.0, min_dist / denom))
    return min_dist


def is_edge_detection(
    min_dist: float,
    overlap_threshold: float
) -> bool:
    """Periksa apakah deteksi dekat tepi tile."""
    return bool(min_dist <= overlap_threshold)


def compute_edge_metrics(
    x1_local: float,
    y1_local: float,
    x2_local: float,
    y2_local: float,
    tile_width: float,
    tile_height: float,
    tile_overlap: float = 0.0
) -> Tuple[bool, float]:
    """Hitung metrik deteksi tepi untuk penggabungan tile."""
    cx, cy = compute_box_center(x1_local, y1_local, x2_local, y2_local)
    min_dist = compute_border_distance(cx, cy, tile_width, tile_height, normalize=False)
    border_dist = compute_border_distance(cx, cy, tile_width, tile_height, normalize=True)
    half_overlap = float(tile_overlap) / 2.0
    edge = is_edge_detection(min_dist, half_overlap) if tile_overlap > 0 else False
    return edge, border_dist

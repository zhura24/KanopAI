"""Dialog warning for band mismatch between input raster and model, with forced/manual matching options."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import json
from pathlib import Path

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QFormLayout, QGroupBox, QMessageBox, QWidget,
)
from PyQt6.QtCore import Qt

import rasterio

from core.inference_engine import (
    load_band_stats,
    is_multiref_schema,
    auto_detect_band_mapping_multiref,
)


BAND_DIFF_WARNING_THRESHOLD = 0.05


def assess_band_mismatch(
    raster_path: str,
    band_stats_path: str,
) -> Dict[str, Any]:
    """Analyze whether the raster band layout requires forced/manual matching."""
    stats = load_band_stats(Path(band_stats_path))
    expected = len(stats)
    multiref = is_multiref_schema(stats)

    with rasterio.open(raster_path) as src:
        n_bands = src.count
        if n_bands == expected and not multiref:
            return {
                "needs_dialog": False,
                "n_bands": n_bands,
                "expected": expected,
                "multiref": multiref,
                "preview_lines": [],
                "max_diff": 0.0,
            }

        preview_lines: List[str] = []
        max_diff = 0.0

        if multiref:
            logs: List[str] = []

            def _log(msg: str) -> None:
                logs.append(msg)

            auto_detect_band_mapping_multiref(src, stats, log=_log)
            preview_lines = logs
            for line in logs:
                if "diff=" in line:
                    try:
                        diff_part = line.split("diff=")[1].split(")")[0]
                        max_diff = max(max_diff, float(diff_part))
                    except (IndexError, ValueError):
                        pass
        else:
            preview_lines = [
                f"Raster bands ({n_bands}) do not match model slots ({expected}).",
                "Auto 1-to-1 mapping cannot be used.",
            ]
            max_diff = float(abs(n_bands - expected))

        needs_dialog = (
            (n_bands != expected and not multiref)
            or (multiref and max_diff > BAND_DIFF_WARNING_THRESHOLD)
        )

        return {
            "needs_dialog": needs_dialog,
            "n_bands": n_bands,
            "expected": expected,
            "multiref": multiref,
            "preview_lines": preview_lines,
            "max_diff": max_diff,
        }


class ManualBandMappingDialog(QDialog):
    """Manual mapping dialog: model slot → raster band."""

    def __init__(
        self,
        parent: Any,
        band_stats: dict,
        n_bands: int,
        initial_mapping: Optional[Dict[int, int]] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Manual Band Matching")
        self.setMinimumWidth(420)
        self._band_stats = band_stats
        self._n_bands = n_bands
        self._combos: Dict[int, QComboBox] = {}
        self.result_mapping: Optional[Dict[int, int]] = None

        layout = QVBoxLayout(self)
        info = QLabel(
            "Map each model band slot to an input raster band.\n"
            "Ensure the pairing matches the sensor used during training."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        form_group = QGroupBox("Band Mapping")
        form = QFormLayout(form_group)
        slots = sorted(int(k) for k in band_stats.keys())

        for slot in slots:
            combo = QComboBox()
            for b in range(1, n_bands + 1):
                combo.addItem(f"Band {b}", b)
            if initial_mapping and slot in initial_mapping:
                idx = combo.findData(initial_mapping[slot])
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            elif slot <= n_bands:
                combo.setCurrentIndex(slot - 1)
            elif n_bands > 0:
                combo.setCurrentIndex(n_bands - 1)
            self._combos[slot] = combo
            form.addRow(f"Model Slot {slot}:", combo)

        layout.addWidget(form_group)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_ok = QPushButton("Apply Mapping")
        btn_ok.clicked.connect(self._accept_mapping)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_ok)
        layout.addLayout(btn_row)

    def _accept_mapping(self) -> None:
        mapping: Dict[int, int] = {}
        for slot, combo in self._combos.items():
            mapping[slot] = int(combo.currentData())
        self.result_mapping = mapping
        self.accept()


class BandMismatchDialog(QDialog):
    """Band mismatch warning dialog — choose forced or manual matching."""

    FORCED = "forced"
    MANUAL = "manual"
    CANCEL = "cancel"

    def __init__(
        self,
        parent: Any,
        assessment: Dict[str, Any],
        band_stats_path: str,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Band Mismatch Detected")
        self.setMinimumWidth(480)
        self._assessment = assessment
        self._band_stats_path = band_stats_path
        self.choice: str = self.CANCEL
        self.manual_mapping: Optional[Dict[int, int]] = None
        self.enable_adaptive_fallback: bool = False

        layout = QVBoxLayout(self)

        warn = QLabel(
            "<b>Warning:</b> A significant mismatch between input raster bands and model bands was detected. "
            "Choose a matching method before continuing inference."
        )
        warn.setWordWrap(True)
        layout.addWidget(warn)

        summary = QLabel(
            f"Raster bands: {assessment['n_bands']} | "
            f"Model slots: {assessment['expected']} | "
            f"Max diff: {assessment.get('max_diff', 0):.4f}"
        )
        summary.setStyleSheet("color: #f59e0b; font-weight: bold;")
        layout.addWidget(summary)

        if assessment.get("preview_lines"):
            preview_group = QGroupBox("Auto-Detect Preview")
            preview_layout = QVBoxLayout(preview_group)
            for line in assessment["preview_lines"][:12]:
                preview_layout.addWidget(QLabel(line))
            layout.addWidget(preview_group)

        btn_row = QHBoxLayout()
        btn_forced = QPushButton("Forced Matching")
        btn_forced.setToolTip(
            "Pakai adaptive fallback — normalisasi dihitung dari raster ini "
            "(jaring pengaman, bukan pengganti fine-tuning)."
        )
        btn_forced.clicked.connect(self._choose_forced)

        btn_manual = QPushButton("Manual Matching")
        btn_manual.setToolTip("Tentukan sendiri pasangan band model ↔ band input.")
        btn_manual.clicked.connect(self._choose_manual)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)

        btn_row.addWidget(btn_forced)
        btn_row.addWidget(btn_manual)
        btn_row.addWidget(btn_cancel)
        layout.addLayout(btn_row)

    def _choose_forced(self) -> None:
        self.choice = self.FORCED
        self.enable_adaptive_fallback = True
        self.accept()

    def _choose_manual(self) -> None:
        stats = load_band_stats(Path(self._band_stats_path))
        dlg = ManualBandMappingDialog(
            self,
            stats,
            self._assessment["n_bands"],
        )
        if dlg.exec() != QDialog.DialogCode.Accepted or not dlg.result_mapping:
            return
        self.choice = self.MANUAL
        self.manual_mapping = dlg.result_mapping
        self.accept()


def resolve_band_matching(
    parent: Any,
    raster_path: str,
    band_stats_path: str,
) -> Tuple[Optional[Dict[int, int]], bool, bool]:
    """Return (manual_mapping, enable_adaptive_fallback, proceed).

    proceed=False if the user cancels.
    """
    assessment = assess_band_mismatch(raster_path, band_stats_path)
    if not assessment["needs_dialog"]:
        return None, False, True

    dlg = BandMismatchDialog(parent, assessment, band_stats_path)
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return None, False, False

    if dlg.choice == BandMismatchDialog.FORCED:
        return None, True, True
    if dlg.choice == BandMismatchDialog.MANUAL:
        return dlg.manual_mapping, False, True
    return None, False, False

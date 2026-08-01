"""Dialog untuk metadata koreksi hasil inference."""

from __future__ import annotations

from typing import Any, Optional, Tuple

from PyQt6.QtCore import QDate
from PyQt6.QtWidgets import (
    QDateEdit,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
)


class CorrectionMetadataDialog(QDialog):
    """Collect the reviewer name and correction date before editing/export."""

    def __init__(
        self,
        parent: Any,
        initial_name: str = "",
        initial_date: Optional[QDate] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Correction Metadata")
        self.setMinimumWidth(420)
        self.reviewer_name = initial_name.strip()
        self.correction_date = initial_date or QDate.currentDate()

        layout = QVBoxLayout(self)
        info = QLabel(
            "Enter the person responsible for reviewing this inference and the correction date. "
            "This information will be stored in the corrected Shapefile."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        form = QFormLayout()
        self.name_edit = QLineEdit(self.reviewer_name)
        self.name_edit.setPlaceholderText("Reviewer name")
        self.name_edit.setMaxLength(50)
        form.addRow("Reviewer name:", self.name_edit)

        self.date_edit = QDateEdit(self.correction_date)
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDisplayFormat("yyyy-MM-dd")
        form.addRow("Correction date:", self.date_edit)
        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept_metadata)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _accept_metadata(self) -> None:
        reviewer_name = self.name_edit.text().strip()
        if not reviewer_name:
            QMessageBox.warning(self, "Missing reviewer name", "Please enter the reviewer name.")
            self.name_edit.setFocus()
            return

        self.reviewer_name = reviewer_name
        self.correction_date = self.date_edit.date()
        self.accept()


def request_correction_metadata(
    parent: Any,
    initial_name: str = "",
    initial_date: Optional[QDate] = None,
) -> Optional[Tuple[str, str]]:
    """Show the dialog and return (reviewer name, ISO date) or None on cancel."""
    dialog = CorrectionMetadataDialog(parent, initial_name, initial_date)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    return dialog.reviewer_name, dialog.correction_date.toString("yyyy-MM-dd")

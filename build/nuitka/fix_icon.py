"""
Fix logo icon using PyQt6
"""
from PyQt6.QtGui import QPixmap, QImage, QPainter
from PyQt6.QtCore import Qt, QSize
import os
import sys

logo_path = os.path.join(os.path.dirname(__file__), "logo", "logo.png")

try:
    # Load original
    original = QPixmap(logo_path)
    print(f"Original size: {original.width()}x{original.height()}")
    print(f"Has alpha: {original.hasAlphaChannel()}")

    # Check if square
    if original.width() != original.height():
        print(f"Logo is NOT square! Width: {original.width()}, Height: {original.height()}")
        print("This can cause issues with Windows icons.")

        # Create square version with padding
        max_size = max(original.width(), original.height())
        square = QPixmap(max_size, max_size)
        square.fill(Qt.GlobalColor.transparent)

        # Center the original image
        painter = QPainter(square)
        x = (max_size - original.width()) // 2
        y = (max_size - original.height()) // 2
        painter.drawPixmap(x, y, original)
        painter.end()

        # Save square version
        square_path = os.path.join(os.path.dirname(__file__), "logo", "logo_square.png")
        square.save(square_path, "PNG")
        print(f"\nCreated square version: {square_path}")
        print(f"Square size: {square.width()}x{square.height()}")
    else:
        print("Logo is already square - good!")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

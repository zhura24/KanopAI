"""
Channel Mapping Mixin for MainWindow
Handles channel mapping configuration for multispectral images
"""

import numpy as np
from PyQt6.QtWidgets import (
    QLabel, QComboBox, QHBoxLayout, QWidget
)


class ChannelMappingMixin:
    """Mixin for handling channel mapping configuration"""

    def _on_mapping_mode_changed(self, mode):
        """Handle channel mapping mode change"""
        if mode == "Advanced":
            self.channel_mapping_widget.setVisible(True)
            self.logger.info("Channel mapping mode: Advanced")
        else:
            self.channel_mapping_widget.setVisible(False)
            self.logger.info("Channel mapping mode: Default")
        self._update_mapping_preview()

    def _update_mapping_preview(self):
        """Update the mapping preview label to show current channel mapping"""
        try:
            # Check if we're in advanced mode
            has_radio = hasattr(self, 'radio_map_advanced')
            is_advanced = has_radio and self.radio_map_advanced.isChecked()

            if not is_advanced:
                # Default mode - sequential mapping
                preview_text = "Current Mapping: Default (Sequential)\n"
                preview_text += "Band 1 → R, Band 2 → G, Band 3 → B"
            else:
                # Advanced mode - show custom mapping
                if not hasattr(self, 'channel_mapping_combos') or not self.channel_mapping_combos:
                    preview_text = "Current Mapping: Custom\n"
                    preview_text += "No mapping configured (load model first)"
                else:
                    preview_text = "Current Mapping: Custom\n"
                    mappings = []
                    for idx, combo in enumerate(self.channel_mapping_combos):
                        try:
                            band_text = combo.currentText()
                            channel_names = ["R", "G", "B"]
                            channel_name = channel_names[idx] if idx < len(channel_names) else f"Ch{idx}"
                            mappings.append(f"{band_text} → {channel_name}")
                        except Exception as e:
                            self.logger.debug(f"Failed to get channel mapping text for combo {idx}: {e}")
                            mappings.append(f"Model Ch {idx} → (error)")
                    preview_text += ", ".join(mappings)

            if hasattr(self, 'label_mapping_preview'):
                self.label_mapping_preview.setText(preview_text)
        except Exception as e:
            self.logger.debug(f"Error updating mapping preview: {e}")

    def _get_band_label(self, band_index, total_bands):
        """Get descriptive label for a band based on image type and band index

        Args:
            band_index: 0-based band index
            total_bands: Total number of bands in the image

        Returns:
            Descriptive label string (e.g., "Band 1 (Red)", "Band 4 (NIR)")
        """
        # For 3-band RGB images
        if total_bands == 3:
            labels = ["Band 1 (Red)", "Band 2 (Green)", "Band 3 (Blue)"]
            if band_index < len(labels):
                return labels[band_index]

        # For 4-band images (RGB + NIR/Alpha)
        elif total_bands == 4:
            labels = ["Band 1 (Red)", "Band 2 (Green)", "Band 3 (Blue)", "Band 4 (NIR/Alpha)"]
            if band_index < len(labels):
                return labels[band_index]

        # For multispectral images (typically 7+ bands)
        # Common multispectral band order: Blue, Green, Red, NIR, Red Edge, etc.
        elif total_bands >= 7:
            # Standard multispectral labels (adjust based on your sensor)
            labels = [
                "Band 1 (Blue)",
                "Band 2 (Green)",
                "Band 3 (Red)",
                "Band 4 (NIR)",
                "Band 5 (Red Edge)",
                "Band 6 (Red Edge 2)",
                "Band 7 (Red Edge 3)"
            ]
            if band_index < len(labels):
                return labels[band_index]
            else:
                return f"Band {band_index + 1}"

        # Default for other band counts
        return f"Band {band_index + 1}"

    def _create_channel_mapping_widgets(self):
        """Create channel mapping widgets based on model input channels"""
        # Clear existing widgets
        for i in reversed(range(self.channel_mapping_layout.count())):
            widget = self.channel_mapping_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        self.channel_mapping_combos.clear()

        # If session missing, keep UI but do not populate widgets
        if not hasattr(self, 'detector_model_session') or not self.detector_model_session:
            self.logger.debug("[CHANNEL WIDGETS] Model not loaded, skipping widget creation")
            return

        # Get model input shape from session
        try:
            inputs = self.detector_model_session.get_inputs()
            if not inputs:
                self.logger.warning("[CHANNEL WIDGETS] No model inputs found")
                return
            input_shape = inputs[0].shape
            self.logger.info(f"[CHANNEL WIDGETS] Model input shape: {input_shape}")
        except Exception as e:
            self.logger.error(f"[CHANNEL WIDGETS] Failed to get model input shape: {e}")
            return

        # Determine number of channels from input shape
        # Typical shapes: (batch, channels, height, width) or (batch, height, width, channels)
        if len(input_shape) >= 4:
            # Assume (B, C, H, W) format
            num_channels = input_shape[1] if isinstance(input_shape[1], int) else 3
        else:
            num_channels = 3

        # Get available image bands from ACTIVE LAYER (not self.current_data)
        active_layer = self._get_active_layer()
        if active_layer:
            num_bands = active_layer.get('metadata', {}).get('bands', 3)
        elif self.current_data is not None:
            # current_data may be a RasterLoader (new path) or a numpy array (legacy)
            if hasattr(self.current_data, 'get_metadata'):
                num_bands = self.current_data.get_metadata().get('bands', 3)
            elif hasattr(self.current_data, 'shape'):
                if len(self.current_data.shape) == 3:
                    num_bands = self.current_data.shape[0]
                else:
                    num_bands = 1
            else:
                num_bands = 3
        else:
            num_bands = 3

        # Log detection info
        self.logger.info(f"Channel mapping auto-detection - Model channels: {num_channels}, Image bands: {num_bands}")
        if active_layer:
            self.logger.info(f"  Model shape: {input_shape}, Active layer: {active_layer['name']} ({num_bands} bands)")

        # Create mapping combobox for each model input channel
        for i in range(num_channels):
            row_layout = QHBoxLayout()

            label = QLabel(f"Model Ch {i}:")
            label.setMinimumWidth(80)
            row_layout.addWidget(label)

            combo = QComboBox()
            # Add image bands as options with descriptive labels
            for band_idx in range(num_bands):
                band_label = self._get_band_label(band_idx, num_bands)
                combo.addItem(band_label, band_idx)

            # Set default mapping (sequential)
            if i < num_bands:
                combo.setCurrentIndex(i)
                self.logger.debug(f"  Model Ch {i} -> Image Band {i} (default)")
            else:
                # If model needs more channels than image has, repeat last band
                combo.setCurrentIndex(num_bands - 1)
                self.logger.debug(f"  Model Ch {i} -> Image Band {num_bands - 1} (repeated)")

            # Connect to update preview
            combo.currentIndexChanged.connect(self._update_mapping_preview)

            row_layout.addWidget(combo)
            self.channel_mapping_combos.append(combo)

            container = QWidget()
            container.setLayout(row_layout)
            self.channel_mapping_layout.addWidget(container)

        self.logger.info(f"Channel mapping widgets created successfully")

        # Update mapping preview after creation
        self._update_mapping_preview()

    def _apply_channel_mapping(self, image_data, channel_map):
        """Apply custom channel mapping to image data

        Args:
            image_data: numpy array (C, H, W) or (H, W)
            channel_map: list of band indices to map to model channels

        Returns:
            remapped image data
        """
        if image_data is None or not channel_map:
            return image_data

        # Ensure data is in (C, H, W) format
        if len(image_data.shape) == 2:
            # (H, W) -> (1, H, W)
            image_data = np.expand_dims(image_data, axis=0)
        elif len(image_data.shape) == 3 and image_data.shape[-1] in [1, 3, 4]:
            # (H, W, C) -> (C, H, W)
            image_data = np.transpose(image_data, (2, 0, 1))

        # Apply channel mapping
        remapped = []
        for band_idx in channel_map:
            if band_idx < image_data.shape[0]:
                remapped.append(image_data[band_idx])
            else:
                # If requested band doesn't exist, use first band
                self.logger.warning(f"Band {band_idx} not found, using band 0")
                remapped.append(image_data[0])

        result = np.stack(remapped, axis=0)
        self.logger.info(f"Applied channel mapping | Input: {image_data.shape} -> Output: {result.shape}")
        return result

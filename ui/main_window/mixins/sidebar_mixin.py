"""Sidebar toggle operations mixin."""

from PyQt6.QtCore import QPropertyAnimation, QEasingCurve


class SidebarMixin:
    """Mixin for sidebar toggle animations and operations."""
    
    def toggle_left_sidebar(self):
        """Toggle left sidebar visibility with animation."""
        try:
            current_w = self.left_sidebar_container.maximumWidth()
        except Exception as e:
            self.logger.debug(f"Failed to get sidebar width, using default: {e}")
            current_w = 350 if self.left_sidebar_visible else 40
        
        if self.left_panel_scroll.isVisible():
            # Collapse
            target = 40
            anim = QPropertyAnimation(self.left_sidebar_container, b"maximumWidth")
            anim.setDuration(220)
            anim.setStartValue(current_w)
            anim.setEndValue(target)
            anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
            
            try:
                current_btn_w = self.btn_toggle_left_sidebar.maximumWidth()
            except Exception as e:
                self.logger.debug(f"Failed to get button width, using fallback: {e}")
                current_btn_w = self.btn_toggle_left_sidebar.width()
            btn_anim = QPropertyAnimation(self.btn_toggle_left_sidebar, b"maximumWidth")
            btn_anim.setDuration(220)
            btn_anim.setStartValue(current_btn_w)
            btn_anim.setEndValue(30)
            btn_anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
            
            def _on_finished_collapse():
                try:
                    self.left_panel_scroll.setVisible(False)
                    self.btn_toggle_left_sidebar.setText('❯')
                    try:
                        self.btn_toggle_left_sidebar.setMinimumWidth(30)
                        self.btn_toggle_left_sidebar.setMaximumWidth(30)
                        self.btn_toggle_left_sidebar.setFixedHeight(30)
                    except Exception as e:
                        self.logger.debug(f"Failed to set button dimensions: {e}")
                    self.left_sidebar_visible = False
                    if hasattr(self, 'logger'):
                        self.logger.debug("Left sidebar hidden (animated)")
                except Exception as e:
                    self.logger.error(f"Error in collapse animation callback: {e}")
            
            anim.finished.connect(_on_finished_collapse)
            self._left_sidebar_anim = anim
            self._left_sidebar_btn_anim = btn_anim
            anim.start()
            btn_anim.start()
        else:
            # Expand
            target = 350
            try:
                self.left_panel_scroll.setVisible(True)
            except Exception as e:
                self.logger.warning(f"Failed to set panel visibility: {e}")
            
            anim = QPropertyAnimation(self.left_sidebar_container, b"maximumWidth")
            anim.setDuration(220)
            anim.setStartValue(current_w)
            anim.setEndValue(target)
            anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
            
            try:
                current_btn_w = self.btn_toggle_left_sidebar.maximumWidth()
            except Exception as e:
                self.logger.debug(f"Failed to get button width, using fallback: {e}")
                current_btn_w = self.btn_toggle_left_sidebar.width()
            btn_anim = QPropertyAnimation(self.btn_toggle_left_sidebar, b"maximumWidth")
            btn_anim.setDuration(220)
            btn_anim.setStartValue(current_btn_w)
            btn_anim.setEndValue(target)
            btn_anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
            
            def _on_finished_expand():
                try:
                    self.btn_toggle_left_sidebar.setText('❮')
                    try:
                        self.btn_toggle_left_sidebar.setMinimumWidth(target)
                        self.btn_toggle_left_sidebar.setMaximumWidth(target)
                    except Exception as e:
                        self.logger.debug(f"Failed to set button dimensions: {e}")
                    self.left_sidebar_visible = True
                    if hasattr(self, 'logger'):
                        self.logger.debug("Left sidebar shown (animated)")
                except Exception as e:
                    self.logger.error(f"Error in expand animation callback: {e}")
            
            anim.finished.connect(_on_finished_expand)
            self._left_sidebar_anim = anim
            self._left_sidebar_btn_anim = btn_anim
            anim.start()
            btn_anim.start()
    
    def toggle_right_sidebar(self):
        """Toggle right sidebar visibility with animation."""
        try:
            current_w = self.right_sidebar_container.maximumWidth()
        except Exception as e:
            self.logger.debug(f"Failed to get sidebar width, using default: {e}")
            current_w = 350 if self.right_sidebar_visible else 40
        
        if self.right_panel_scroll.isVisible():
            # Collapse
            target = 40
            anim = QPropertyAnimation(self.right_sidebar_container, b"maximumWidth")
            anim.setDuration(220)
            anim.setStartValue(current_w)
            anim.setEndValue(target)
            anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
            
            try:
                current_btn_w = self.btn_toggle_right_sidebar.maximumWidth()
            except Exception as e:
                self.logger.debug(f"Failed to get button width, using fallback: {e}")
                current_btn_w = self.btn_toggle_right_sidebar.width()
            btn_anim = QPropertyAnimation(self.btn_toggle_right_sidebar, b"maximumWidth")
            btn_anim.setDuration(220)
            btn_anim.setStartValue(current_btn_w)
            btn_anim.setEndValue(30)
            btn_anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
            
            def _on_finished_collapse_right():
                try:
                    self.right_panel_scroll.setVisible(False)
                    self.btn_toggle_right_sidebar.setText('❮')
                    try:
                        self.btn_toggle_right_sidebar.setMinimumWidth(30)
                        self.btn_toggle_right_sidebar.setMaximumWidth(30)
                        self.btn_toggle_right_sidebar.setFixedHeight(30)
                    except Exception as e:
                        self.logger.debug(f"Failed to set button dimensions: {e}")
                    self.right_sidebar_visible = False
                    if hasattr(self, 'logger'):
                        self.logger.debug("Right sidebar hidden (animated)")
                except Exception as e:
                    self.logger.error(f"Error in collapse animation callback: {e}")
            
            anim.finished.connect(_on_finished_collapse_right)
            self._right_sidebar_anim = anim
            self._right_sidebar_btn_anim = btn_anim
            anim.start()
            btn_anim.start()
        else:
            # Expand
            target = 350
            try:
                self.right_panel_scroll.setVisible(True)
            except Exception as e:
                self.logger.warning(f"Failed to set panel visibility: {e}")
            
            anim = QPropertyAnimation(self.right_sidebar_container, b"maximumWidth")
            anim.setDuration(220)
            anim.setStartValue(current_w)
            anim.setEndValue(target)
            anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
            
            try:
                current_btn_w = self.btn_toggle_right_sidebar.maximumWidth()
            except Exception as e:
                self.logger.debug(f"Failed to get button width, using fallback: {e}")
                current_btn_w = self.btn_toggle_right_sidebar.width()
            btn_anim = QPropertyAnimation(self.btn_toggle_right_sidebar, b"maximumWidth")
            btn_anim.setDuration(220)
            btn_anim.setStartValue(current_btn_w)
            btn_anim.setEndValue(target)
            btn_anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
            
            def _on_finished_expand_right():
                try:
                    self.btn_toggle_right_sidebar.setText('❯')
                    try:
                        self.btn_toggle_right_sidebar.setMinimumWidth(target)
                        self.btn_toggle_right_sidebar.setMaximumWidth(target)
                    except Exception as e:
                        self.logger.debug(f"Failed to set button dimensions: {e}")
                    self.right_sidebar_visible = True
                    if hasattr(self, 'logger'):
                        self.logger.debug("Right sidebar shown (animated)")
                except Exception as e:
                    self.logger.error(f"Error in expand animation callback: {e}")
            
            anim.finished.connect(_on_finished_expand_right)
            self._right_sidebar_anim = anim
            self._right_sidebar_btn_anim = btn_anim
            anim.start()
            btn_anim.start()

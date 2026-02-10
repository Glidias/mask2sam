from krita import Krita
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QMessageBox, QRadioButton, QFileDialog, QScrollArea, QApplication
)
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QFont, QImage
from PyQt5.QtCore import Qt, QPoint, QEvent
import json
import os

class PointsCanvas(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.points = []
        self.dragging_index = None
        self.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.setStyleSheet("background-color: #111; border: 1px solid #444;")
        self.setMouseTracking(True)
        self.dialog_ref = parent
        self.zoom_factor = 1.0
        self.original_pixmap = None
        self.grabGesture(Qt.PinchGesture)  # Enable pinch gesture for macOS

    def set_document_and_thumbnail_size(self, doc_width, doc_height, thumb_width, thumb_height):
        self.doc_width = doc_width
        self.doc_height = doc_height
        self.thumb_width = thumb_width
        self.thumb_height = thumb_height
        self.base_scale_x = thumb_width / doc_width if doc_width > 0 else 1.0
        self.base_scale_y = thumb_height / doc_height if doc_height > 0 else 1.0

    def set_background_pixmap(self, pixmap):
        self.original_pixmap = pixmap
        self.apply_zoom()

    def apply_zoom(self):
        if self.original_pixmap:
            scaled = self.original_pixmap.scaled(
                int(self.original_pixmap.width() * self.zoom_factor),
                int(self.original_pixmap.height() * self.zoom_factor),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.setPixmap(scaled)
            self.resize(scaled.size())

    # mousewheel event
    def wheelEvent(self, event):
        # check if shift key modifier is pressed
        if event.modifiers() & Qt.ShiftModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                new_zoom = self.zoom_factor * 1.1
            else:
                new_zoom = self.zoom_factor / 1.1
            if 0.1 <= new_zoom <= 10.0:
                self.zoom_factor = new_zoom
                self.apply_zoom()
            event.accept()

    def screen_to_doc(self, screen_x, screen_y):
        current_scale_x = self.base_scale_x * self.zoom_factor
        current_scale_y = self.base_scale_y * self.zoom_factor
        doc_x = int(screen_x / current_scale_x) if current_scale_x > 0 else screen_x
        doc_y = int(screen_y / current_scale_y) if current_scale_y > 0 else screen_y
        doc_x = max(0, min(doc_x, self.doc_width - 1))
        doc_y = max(0, min(doc_y, self.doc_height - 1))
        return doc_x, doc_y

    def doc_to_screen(self, doc_x, doc_y):
        current_scale_x = self.base_scale_x * self.zoom_factor
        current_scale_y = self.base_scale_y * self.zoom_factor
        screen_x = int(doc_x * current_scale_x)
        screen_y = int(doc_y * current_scale_y)
        return screen_x, screen_y

    def add_point(self, screen_x, screen_y, point_type):
        doc_x, doc_y = self.screen_to_doc(screen_x, screen_y)
        same_type = [p for p in self.points if p['type'] == point_type]
        index = len(same_type)
        self.points.append({'x': doc_x, 'y': doc_y, 'type': point_type, 'index': index})
        self.update()

    def clear_points(self):
        self.points = []
        self.update()

    def get_points_for_export(self):
        pos = [{'x': p['x'], 'y': p['y']} for p in self.points if p['type'] == 'pos']
        neg = [{'x': p['x'], 'y': p['y']} for p in self.points if p['type'] == 'neg']
        return pos, neg

    def load_points_from_export(self, pos_points, neg_points):
        self.points = []
        for p in pos_points:
            self.points.append({'x': p['x'], 'y': p['y'], 'type': 'pos', 'index': len([x for x in self.points if x['type'] == 'pos'])})
        for p in neg_points:
            self.points.append({'x': p['x'], 'y': p['y'], 'type': 'neg', 'index': len([x for x in self.points if x['type'] == 'neg'])})
        self._reindex()
        self.update()

    def _reindex(self):
        pos_idx = 0
        neg_idx = 0
        for p in self.points:
            if p['type'] == 'pos':
                p['index'] = pos_idx
                pos_idx += 1
            else:
                p['index'] = neg_idx
                neg_idx += 1

    def get_point_at(self, screen_x, screen_y, tol=12):
        doc_x, doc_y = self.screen_to_doc(screen_x, screen_y)
        current_scale_x = self.base_scale_x * self.zoom_factor
        doc_tol = tol / current_scale_x if current_scale_x > 0 else tol
        for i, p in enumerate(self.points):
            dx = p['x'] - doc_x
            dy = p['y'] - doc_y
            if dx*dx + dy*dy <= doc_tol*doc_tol:
                return i
        return None

    def delete_point_and_pair(self, idx):
        if idx < 0 or idx >= len(self.points):
            return
        target = self.points[idx]
        t_idx = target['index']
        t_type = target['type']

        pair_i = None
        for i, p in enumerate(self.points):
            if p['index'] == t_idx and p['type'] != t_type:
                pair_i = i
                break

        to_del = sorted([idx, pair_i] if pair_i is not None else [idx], reverse=True)
        for i in to_del:
            del self.points[i]
        self._reindex()
        self.update()

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return

        screen_x, screen_y = event.x(), event.y()
        pt_idx = self.get_point_at(screen_x, screen_y)
        modifiers = event.modifiers()

        if modifiers & Qt.ControlModifier:
            if pt_idx is not None:
                self.delete_point_and_pair(pt_idx)
        # OPPOSITE: Shift only
        elif modifiers == Qt.ShiftModifier:
            dialog = self.dialog_ref
            if dialog and hasattr(dialog, 'current_mode'):
                opposite = 'neg' if dialog.current_mode == 'pos' else 'pos'
                if pt_idx is None:
                    self.add_point(screen_x, screen_y, opposite)
        # REGULAR: No modifiers or other combinations
        else:
            if pt_idx is not None:
                self.dragging_index = pt_idx
            else:
                dialog = self.dialog_ref
                if dialog and hasattr(dialog, 'current_mode'):
                    self.add_point(screen_x, screen_y, dialog.current_mode)
        self.update()

    def mouseMoveEvent(self, event):
        if self.dragging_index is not None and self.dragging_index < len(self.points):
            screen_x, screen_y = event.x(), event.y()
            doc_x, doc_y = self.screen_to_doc(screen_x, screen_y)
            self.points[self.dragging_index]['x'] = doc_x
            self.points[self.dragging_index]['y'] = doc_y
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging_index = None

    def event(self, event):
        if event.type() == QEvent.Gesture:
            return self.gestureEvent(event)
        return super().event(event)

    def gestureEvent(self, event):
        pinch = event.gesture(Qt.PinchGesture)
        if pinch:
            scale_factor = pinch.scaleFactor()
            if abs(scale_factor - 1.0) > 0.01:
                new_zoom = self.zoom_factor * scale_factor
                if 0.1 <= new_zoom <= 10.0:
                    self.zoom_factor = new_zoom
                    self.apply_zoom()
            return True
        return False

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.points:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        font_size = max(8, int(10 * self.zoom_factor))
        font = QFont("Arial", font_size, QFont.Bold)
        painter.setFont(font)

        for p in self.points:
            screen_x, screen_y = self.doc_to_screen(p['x'], p['y'])

            color = QColor(0, 255, 0) if p['type'] == 'pos' else QColor(255, 0, 0)
            pen_width = max(1, int(2 * self.zoom_factor))
            pen = QPen(color, pen_width)
            painter.setPen(pen)
            radius = max(3, int(7 * self.zoom_factor))
            painter.drawEllipse(QPoint(screen_x, screen_y), radius, radius)

            text = str(p['index'])
            text_x = screen_x + int(12 * self.zoom_factor)
            text_y = screen_y + int(5 * self.zoom_factor)

            # White outline
            painter.setPen(QColor(255, 255, 255))
            offsets = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
            for dx, dy in offsets:
                painter.drawText(text_x + dx, text_y + dy, text)

            # Colored fill
            fill_color = QColor(0, 80, 0) if p['type'] == 'pos' else QColor(80, 0, 0)
            painter.setPen(fill_color)
            painter.drawText(text_x, text_y, text)

        painter.end()


class PointsEditorDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Points Editor")
        self.resize(800, 700)
        self.current_mode = 'pos'
        self.last_save_path = None
        self.result_points = None
        self.user_initiated_close = False
        self.setup_ui()
        self.load_background()

        if self.auto_load():
            print("✅ Auto-loaded previously saved points.")

    def setup_ui(self):
        main_layout = QVBoxLayout()

        control_layout = QHBoxLayout()

        self.radio_pos = QRadioButton("Positive (Green)")
        self.radio_neg = QRadioButton("Negative (Red)")
        self.radio_pos.setChecked(True)
        self.radio_pos.toggled.connect(lambda: setattr(self, 'current_mode', 'pos'))
        self.radio_neg.toggled.connect(lambda: setattr(self, 'current_mode', 'neg'))

        save_btn = QPushButton("Save Points...")
        load_btn = QPushButton("Load Points...")
        clear_btn = QPushButton("Clear All")
        self.ok_btn = QPushButton("OK")
        self.close_btn = QPushButton("Close")

        save_btn.clicked.connect(self.save_points)
        load_btn.clicked.connect(self.load_points)
        clear_btn.clicked.connect(lambda: self.canvas.clear_points())
        self.ok_btn.clicked.connect(self.on_ok_clicked)
        self.close_btn.clicked.connect(self.on_close_clicked)

        control_layout.addWidget(self.radio_pos)
        control_layout.addWidget(self.radio_neg)
        control_layout.addWidget(save_btn)
        control_layout.addWidget(load_btn)
        control_layout.addWidget(clear_btn)
        control_layout.addStretch()
        control_layout.addWidget(self.ok_btn)
        control_layout.addWidget(self.close_btn)

        main_layout.addLayout(control_layout)

        self.canvas = PointsCanvas()
        self.canvas.dialog_ref = self

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.canvas)
        scroll_area.setStyleSheet("QScrollArea { border: none; }")

        main_layout.addWidget(scroll_area)
        self.setLayout(main_layout)

    def load_background(self):
        try:
            app = Krita.instance()
            doc = app.activeDocument()
            if not doc:
                QMessageBox.warning(self, "No Document", "Open an image first.")
                return

            orig_w, orig_h = doc.width(), doc.height()
            thumb_qimage = doc.projection(0, 0, orig_w, orig_h)
            if thumb_qimage.isNull():
                raise Exception("Thumbnail is null")

            thumb_w = thumb_qimage.width()
            thumb_h = thumb_qimage.height()

            thumb_qimage = thumb_qimage.convertToFormat(QImage.Format_RGBA8888)
            pixmap = QPixmap.fromImage(thumb_qimage)
            self.canvas.set_background_pixmap(pixmap)

            self.canvas.set_document_and_thumbnail_size(orig_w, orig_h, thumb_w, thumb_h)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Background load failed:\n{str(e)}")

    def get_autosave_path(self):
        try:
            doc = Krita.instance().activeDocument()
            if not doc or not doc.fileName():
                return os.path.expanduser("~/krita_points_autosave.json")
            doc_path = doc.fileName()
            return doc_path + ".points.json"
        except:
            return os.path.expanduser("~/krita_points_autosave.json")

    def auto_save(self):
        try:
            pos_pts, neg_pts = self.canvas.get_points_for_export()
            if not pos_pts and not neg_pts:
                return

            autosave_path = self.get_autosave_path()
            data = {
                "coordinates_positive": pos_pts,
                "coordinates_negative": neg_pts,
                "auto_saved": True,
                "document": Krita.instance().activeDocument().fileName() if Krita.instance().activeDocument() else "unsaved"
            }

            with open(autosave_path, 'w') as f:
                json.dump(data, f)
        except:
            pass

    def auto_load(self):
        try:
            autosave_path = self.get_autosave_path()
            if os.path.exists(autosave_path):
                with open(autosave_path, 'r') as f:
                    data = json.load(f)
                if data.get("auto_saved"):
                    pos_pts = data.get("coordinates_positive", [])
                    neg_pts = data.get("coordinates_negative", [])
                    self.canvas.load_points_from_export(pos_pts, neg_pts)
                    return True
        except:
            pass
        return False

    def on_ok_clicked(self):
        self.user_initiated_close = True
        self.result_points = self.canvas.get_points_for_export()
        self.auto_save()
        self.accept()

    def on_close_clicked(self):
        self.user_initiated_close = True
        self.result_points = None
        self.auto_save()
        self.reject()

    def closeEvent(self, event):
        if not self.user_initiated_close:
            self.result_points = None
        super().closeEvent(event)

    def get_result(self):
        return self.result_points

    def save_points(self):
        pos_pts, neg_pts = self.canvas.get_points_for_export()
        data = {
            "coordinates_positive": pos_pts,
            "coordinates_negative": neg_pts
        }

        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Points", self.last_save_path or "", "JSON Files (*.json)"
        )
        if not filename:
            return
        if not filename.endswith(".json"):
            filename += ".json"

        try:
            with open(filename, 'w') as f:
                json.dump(data, f)
            self.last_save_path = filename
            self.auto_save()
            # QMessageBox.information(self, "Saved", f"Saved {len(pos_pts)} positive and {len(neg_pts)} negative points.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save:\n{str(e)}")

    def load_points(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Points", self.last_save_path or "", "JSON Files (*.json)"
        )
        if not filename:
            return

        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            pos_pts = data.get("coordinates_positive", [])
            neg_pts = data.get("coordinates_negative", [])
            self.canvas.load_points_from_export(pos_pts, neg_pts)
            self.last_save_path = filename
            self.auto_save()
            # QMessageBox.information(self, "Loaded", f"Loaded {len(pos_pts)} positive and {len(neg_pts)} negative points.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load:\n{str(e)}")


# === RUN IN SCRIPTER (uncomment if running directly in Scriper)===
"""
app = Krita.instance()
if not app.activeWindow():
    QMessageBox.critical(None, "Error", "No Krita window!")
else:
    dialog = PointsEditorDialog(app.activeWindow().qwindow())
    result = dialog.exec_()

    points = dialog.get_result()
    if points is not None:
        pos_points, neg_points = points
        print(f"✅ OK clicked! Positive points: {pos_points}")
        print(f"✅ OK clicked! Negative points: {neg_points}")
    else:
        if result == QDialog.Rejected:
            print("CloseOperation: Close button clicked (auto-saved, no points returned)")
        else:
            print("CloseOperation: X button clicked (no auto-save, no points returned)")
"""
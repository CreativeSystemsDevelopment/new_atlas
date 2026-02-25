"""
ATLAS Component Annotator — Native Desktop App (PySide6 / Qt)

Interactive tool for associating electrical schematic components with their text labels.
Uses QGraphicsView for hardware-accelerated canvas with zoom/pan.

Usage:
    python annotator.py <pdf_path>         # first run: preprocess + launch
    python annotator.py                    # subsequent: launch with existing data
"""
import json
import math
import mimetypes
import os
import shutil
import ssl
import sys
import subprocess
import tempfile
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime

# -------------------------------------------------------------------
# Ensure dependencies
# -------------------------------------------------------------------
for _pkg, _imp in [("PyMuPDF", "fitz"), ("PySide6", "PySide6")]:
    try:
        __import__(_imp)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", _pkg])

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene,
    QGraphicsRectItem, QGraphicsLineItem, QGraphicsPixmapItem,
    QGraphicsSimpleTextItem, QToolBar, QLabel, QDockWidget,
    QListWidget, QListWidgetItem, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QStatusBar, QFileDialog, QSizePolicy,
    QMessageBox, QGroupBox, QDialog, QFormLayout, QDialogButtonBox,
    QSpinBox, QProgressDialog, QLineEdit, QRadioButton, QButtonGroup,
    QCheckBox, QInputDialog, QSlider,
)
from PySide6.QtCore import Qt, QRectF, QPointF, QLineF, Signal, QSize, QThread, QTimer
from PySide6.QtGui import (
    QPixmap, QPen, QColor, QBrush, QPainter, QFont, QImage,
    QKeySequence, QAction, QWheelEvent, QShortcut,
)
from PySide6.QtCore import QEvent
import time

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
ANN_DIR  = os.path.join(BASE, "annotations")
MODEL_PATH = os.path.join(DATA_DIR, "master", "best.pt")  # default auto-detect model
# Primary mask output location: put masks beside annotation artifacts.
MASK_DIR  = os.path.join(ANN_DIR, "masks")
# Legacy fallback: existing masks remain readable from the old folder.
MASK_DIR_LEGACY = os.path.join(DATA_DIR, "masks")


def _mask_file_candidates(page_num: int, suffix: str) -> list[str]:
    """Return possible mask file paths (primary first, then legacy)."""
    filename = f"page_{page_num}_{suffix}"
    return [
        os.path.join(MASK_DIR, filename),
        os.path.join(MASK_DIR_LEGACY, filename),
    ]


def _existing_mask_path(page_num: int, suffix: str) -> Optional[str]:
    """Find the first existing mask file path across known folders."""
    for candidate in _mask_file_candidates(page_num, suffix):
        if os.path.exists(candidate):
            return candidate
    return None


def _migrate_legacy_masks() -> None:
    """Move legacy mask files from data/masks into annotations/masks."""
    if not os.path.isdir(MASK_DIR_LEGACY):
        return
    if os.path.abspath(MASK_DIR_LEGACY) == os.path.abspath(MASK_DIR):
        return

    os.makedirs(MASK_DIR, exist_ok=True)

    for root, _dirs, files in os.walk(MASK_DIR_LEGACY):
        rel = os.path.relpath(root, MASK_DIR_LEGACY)
        target_root = MASK_DIR if rel == "." else os.path.join(MASK_DIR, rel)
        os.makedirs(target_root, exist_ok=True)

        for filename in files:
            src = os.path.join(root, filename)
            dst = os.path.join(target_root, filename)
            if os.path.exists(dst):
                continue
            try:
                shutil.move(src, dst)
            except OSError:
                shutil.copy2(src, dst)
                os.remove(src)

    for root, dirnames, _files in os.walk(MASK_DIR_LEGACY, topdown=False):
        if root == MASK_DIR_LEGACY:
            continue
        if not dirnames and not os.listdir(root):
            try:
                os.rmdir(root)
            except OSError:
                pass

    # Best effort: keep one clean marker so we don't repeatedly try to migrate.
    marker = os.path.join(MASK_DIR, ".legacy_migration_complete")
    try:
        with open(marker, "w", encoding="utf-8") as f:
            f.write("migrated")
    except OSError:
        pass

VIEW_DPI = 300  # DPI used for display in the annotator canvas

# Colors
C_BG = QColor("#0d1117")
C_SURFACE = QColor("#161b22")
C_BORDER = QColor("#30363d")
C_TEXT = QColor("#e6edf3")
C_DIM = QColor("#8b949e")
C_COMP = QColor("#22c55e")       # green — component box
C_COMP_FILL = QColor(34, 197, 94, 30)
C_LABEL = QColor("#3b82f6")      # blue — label box
C_LABEL_FILL = QColor(59, 130, 246, 30)
C_LINK = QColor("#f59e0b")  # Amber for continuation
C_HOVER = QColor(255, 255, 255, 40)
C_SELECT = QColor("#00d4ff")     # highlight


# Import QInputDevice for type checking
from PySide6.QtGui import QInputDevice

# ===================================================================
# SchematicView — QGraphicsView with zoom/pan and annotation tools
# ===================================================================
class SchematicView(QGraphicsView):
    """Canvas widget: renders schematic page, handles zoom/pan and annotation drawing."""

    pairCreated = Signal()  # emitted when a new component-label pair is completed
    requestPrevPage = Signal()
    requestNextPage = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setBackgroundBrush(C_BG)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)

        # Enable touch events for stylus/tablet support
        # We disable generic touch events to implement "Stylus Only" mode (palm rejection)
        # self.setAttribute(Qt.WA_AcceptTouchEvents, True)
        # self.setAcceptTouchEvents(True)
        # self.viewport().setAttribute(Qt.WA_AcceptTouchEvents, True)
        
        # Touch gesture state
        self._touch_points = []
        self._touch_pan_start = None
        self._initial_pinch_distance = None
        self._flick_start_pos = None
        self._flick_start_time = None
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Settings (updated by AnnotatorWindow)
        self.snap_padding = 1
        self.stroke_width = 1

        # State
        self._zoom = 1.0
        self._panning = False
        self._pan_start = QPointF()
        self._drawing = False
        self._draw_origin = QPointF()

        # Mode: 'component', 'label', 'continuation', 'cont_page', 'cont_rung', 'wire'
        self.mode = "component"
        self.main_mode = "component"  # Sticky mode: 'component', 'continuation', 'wire'

        # Data
        self.metadata = None
        self.page_idx = 0
        self.page_pixmap_item = None

        # Current pending component rect (scene coords / PDF points)
        self.pending_comp = None  # [x0, y0, x1, y1] in PDF coords
        self._pending_rect_item = None

        # Continuation annotation state
        self.pending_continuation = None  # {symbol:{bbox}, page:None, rung:None}

        # Annotation items on scene
        self.pairs = []           # [{component:{bbox}, label:{text,bbox}}, ...]
        self.wire_labels = []     # [{text, bbox}] - wire labels only
        self._pair_items = []     # graphics items for rendering
        self._wire_items = []     # graphics items for wire labels

        # Temporary draw rect
        self._rubber_rect = None

        # Hover highlight for text
        self._hover_item = None

        # Redo stack
        self._redo_stack = []  # stores undone actions for Ctrl+Y

        # Highlight item for sidebar selection
        self._highlight_item = None

        # Label-adjust mode state (adjustable bbox handles)
        self._label_adjust_bbox = None    # [x0, y0, x1, y1] in PDF coords
        self._label_adjust_text = None    # detected text
        self._label_adjust_items = []     # graphics items (rect + handles)
        self._label_adjust_dragging = None  # None, 'left', 'right', 'top', 'bottom'

        # Mask mode state
        self._mask_mode = False
        self._mask_source_image = None      # QImage of page for pixel reads
        self._mask_overlay_image = None     # QImage for blue overlay
        self._mask_overlay_item = None      # QGraphicsPixmapItem on scene
        self._mask_painting = False         # pen is down and painting
        self._mask_erasing = False          # eraser active (S Pen eraser end)
        self._mask_erase_mode = False       # E key toggle erase mode
        self._mask_active_pair = None       # pair idx currently being masked
        self._mask_bboxes = []              # precomputed [(x0,y0,x1,y1,pair_idx)]
        self._mask_dirty = False
        self._mask_update_counter = 0
        self._mask_saved_pairs = set()      # pair indices with saved masks
        self._mask_brush_radius = 4         # paint brush radius (pixels)
        self._mask_eraser_radius = 6        # eraser radius (pixels)

        self.setMouseTracking(True)

    # ---- Data loading ----
    def load_metadata(self, meta):
        self.metadata = meta

    def load_page(self, idx):
        if not self.metadata or idx < 0 or idx >= len(self.metadata["pages"]):
            return
        self.page_idx = idx
        self.mode = "component"
        self.pending_comp = None
        self.pending_continuation = None
        self._clear_temp_items()

        pg = self.metadata["pages"][idx]

        pixmap = QPixmap(os.path.join(DATA_DIR, "pages", pg["image"]))

        self.scene().clear()
        self._pair_items.clear()
        self._wire_items.clear()
        self.page_pixmap_item = self.scene().addPixmap(pixmap)
        self.scene().setSceneRect(QRectF(pixmap.rect()))

        # Load annotations
        self._load_annotations(idx)
        self._redraw_pairs()
        self._redraw_wires()
        self.fit_view()

        # Reinitialize mask overlay if in mask mode
        if self._mask_mode:
            self._mask_init_for_page()

    def fit_view(self):
        if self.page_pixmap_item:
            self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)
            self._zoom = self.transform().m11()

    # ---- Coordinate helpers ----
    def _scale(self):
        return VIEW_DPI / 72.0

    def scene_to_pdf(self, sx, sy):
        s = self._scale()
        return sx / s, sy / s

    def pdf_to_scene(self, px, py):
        s = self._scale()
        return px * s, py * s

    # ---- Snap algorithms ----
    def snap_component(self, rough):
        """Snap a rough rectangle (PDF coords) to tightly enclosing vector shapes.
        Uses center-containment: only shapes whose center falls inside the
        user's drawn box are included. Filters out wire-like shapes (high
        aspect ratio) to prevent long runs from expanding the result.
        """
        pg = self.metadata["pages"][self.page_idx]
        rx0, ry0, rx1, ry1 = rough
        hits = []
        for sh in pg["shapes"]:
            sx0, sy0, sx1, sy1 = sh["bbox"]
            sw, sh2 = sx1 - sx0, sy1 - sy0
            # Skip huge shapes
            if sw > 120 and sh2 > 120:
                continue
            # Skip wire-like shapes (very elongated, aspect > 8:1)
            if sw > 0.5 and sh2 > 0.5:
                aspect = max(sw, sh2) / max(min(sw, sh2), 0.1)
                if aspect > 8:
                    continue
            # Center-containment: shape center must be inside rough rect
            cx = (sx0 + sx1) / 2
            cy = (sy0 + sy1) / 2
            if rx0 <= cx <= rx1 and ry0 <= cy <= ry1:
                hits.append(sh)
        if not hits:
            return rough
        x0 = min(h["bbox"][0] for h in hits)
        y0 = min(h["bbox"][1] for h in hits)
        x1 = max(h["bbox"][2] for h in hits)
        y1 = max(h["bbox"][3] for h in hits)
        pad = self.snap_padding
        return [x0 - pad, y0 - pad, x1 + pad, y1 + pad]

    def snap_text(self, px, py):
        """Find text block at PDF coords.
        Label mode: Single block, alpha-only preference.
        Wire/Cont mode: Merge adjacent blocks (alphanumeric).
        """
        pg = self.metadata["pages"][self.page_idx]

        # 1. Component Label Mode
        if self.mode == "label":
            best, min_dist = None, 20
            for tb in pg["text_blocks"]:
                text = tb["text"].strip()
                # Skip digit-only blocks — labels must contain letters
                if not any(ch.isalpha() for ch in text):
                    continue
                cx = (tb["bbox"][0] + tb["bbox"][2]) / 2
                cy = (tb["bbox"][1] + tb["bbox"][3]) / 2
                d = math.hypot(px - cx, py - cy)
                if d < min_dist:
                    min_dist = d
                    best = tb
            
            if not best: return None
            
            text = best["text"].strip()
            # Strip digits, keep letters
            letters = "".join(filter(str.isalpha, text))
            if letters and len(letters) < len(text):
                ratio = len(letters) / len(text)
                x0, y0, x1, y1 = best["bbox"]
                new_w = (x1 - x0) * ratio
                return {"text": letters, "bbox": [x0, y0, x0 + new_w, y1]}
            return best

        # 2. Wire/Continuation Mode (Merge neighbors)
        else:
            # Find closest block as primary
            primary, min_dist = None, 20
            for tb in pg["text_blocks"]:
                cx = (tb["bbox"][0] + tb["bbox"][2]) / 2
                cy = (tb["bbox"][1] + tb["bbox"][3]) / 2
                d = math.hypot(px - cx, py - cy)
                if d < min_dist:
                    min_dist = d
                    primary = tb
            
            if not primary: return None

            # Find neighbors on same line
            p_y0, p_y1 = primary["bbox"][1], primary["bbox"][3]
            p_cy = (p_y0 + p_y1) / 2
            
            row_candidates = []
            for tb in pg["text_blocks"]:
                ty0, ty1 = tb["bbox"][1], tb["bbox"][3]
                tcy = (ty0 + ty1) / 2
                if abs(tcy - p_cy) < 1.4:  # Strict line alignment
                    row_candidates.append(tb)
            
            row_candidates.sort(key=lambda b: b["bbox"][0])
            
            try:
                pidx = row_candidates.index(primary)
            except ValueError:
                return primary # Should not happen

            group = [primary]

            # Expand Left
            curr = primary
            for i in range(pidx - 1, -1, -1):
                prev = row_candidates[i]
                if (curr["bbox"][0] - prev["bbox"][2]) < 2.0: # No spaces allowed
                    group.insert(0, prev)
                    curr = prev
                else: 
                    break
            
            # Expand Right
            curr = primary
            for i in range(pidx + 1, len(row_candidates)):
                next_b = row_candidates[i]
                if (next_b["bbox"][0] - curr["bbox"][2]) < 2.0:
                    group.append(next_b)
                    curr = next_b
                else:
                    break
            
            # Merge
            full_text = "".join(b["text"].strip() for b in group)
            x0 = min(b["bbox"][0] for b in group)
            y0 = min(b["bbox"][1] for b in group)
            x1 = max(b["bbox"][2] for b in group)
            y1 = max(b["bbox"][3] for b in group)
            
            return {"text": full_text, "bbox": [x0, y0, x1, y1]}

    # ---- Drawing helpers ----
    def _add_pdf_rect(self, bbox, pen_color, fill_color, width=None, dash=False):
        sx0, sy0 = self.pdf_to_scene(bbox[0], bbox[1])
        sx1, sy1 = self.pdf_to_scene(bbox[2], bbox[3])
        if width is None:
            width = self.stroke_width
        pen = QPen(pen_color, width / self._zoom)
        if dash:
            pen.setStyle(Qt.DashLine)
        item = self.scene().addRect(
            QRectF(QPointF(sx0, sy0), QPointF(sx1, sy1)),
            pen, QBrush(fill_color)
        )
        item.setZValue(10)
        return item

    def _add_link_line(self, bbox1, bbox2):
        c1 = self.pdf_to_scene((bbox1[0]+bbox1[2])/2, (bbox1[1]+bbox1[3])/2)
        c2 = self.pdf_to_scene((bbox2[0]+bbox2[2])/2, (bbox2[1]+bbox2[3])/2)
        pen = QPen(C_LINK, 1.0 / self._zoom)
        pen.setStyle(Qt.DashDotLine)
        item = self.scene().addLine(QLineF(QPointF(*c1), QPointF(*c2)), pen)
        item.setZValue(9)
        return item

    def _add_label_text(self, text, bbox1, bbox2):
        c1 = self.pdf_to_scene((bbox1[0]+bbox1[2])/2, (bbox1[1]+bbox1[3])/2)
        c2 = self.pdf_to_scene((bbox2[0]+bbox2[2])/2, (bbox2[1]+bbox2[3])/2)
        mx, my = (c1[0]+c2[0])/2, (c1[1]+c2[1])/2
        item = QGraphicsSimpleTextItem(text)
        font = QFont("Inter", max(6, int(8 / self._zoom)))
        font.setBold(True)
        item.setFont(font)
        item.setBrush(C_LINK)
        item.setPos(mx, my - 12 / self._zoom)
        item.setZValue(11)
        self.scene().addItem(item)
        return item

    def _redraw_pairs(self):
        for items in self._pair_items:
            for it in items:
                self.scene().removeItem(it)
        self._pair_items.clear()
        for p in self.pairs:
            group = []
            if p.get("type") == "continuation":
                # Continuation annotation - amber theme
                group.append(self._add_pdf_rect(p["symbol"]["bbox"], C_LINK, QColor(245, 158, 11, 30)))
                group.append(self._add_pdf_rect(p["page"]["bbox"], C_LINK, QColor(245, 158, 11, 40)))
                group.append(self._add_pdf_rect(p["rung"]["bbox"], C_LINK, QColor(245, 158, 11, 40)))
                # Dotted line from symbol to page number
                group.append(self._add_link_line(p["symbol"]["bbox"], p["page"]["bbox"]))
                # Label showing page→rung
                page_text = p["page"]["text"]
                rung_text = p["rung"]["text"]
                group.append(self._add_label_text(f"→{page_text}:{rung_text}", p["symbol"]["bbox"], p["page"]["bbox"]))
            else:
                # Regular component-label pair - green/blue theme
                group.append(self._add_pdf_rect(p["component"]["bbox"], C_COMP, C_COMP_FILL))
                group.append(self._add_pdf_rect(p["label"]["bbox"], C_LABEL, C_LABEL_FILL))
                group.append(self._add_link_line(p["component"]["bbox"], p["label"]["bbox"]))
                group.append(self._add_label_text(p["label"]["text"], p["component"]["bbox"], p["label"]["bbox"]))
            self._pair_items.append(group)

    def _redraw_wires(self):
        """Render wire labels in purple."""
        for it in self._wire_items:
            self.scene().removeItem(it)
        self._wire_items.clear()
        for w in self.wire_labels:
            # Purple wire label box
            rect_item = self._add_pdf_rect(w["bbox"], QColor(168, 85, 247), QColor(168, 85, 247, 25))
            self._wire_items.append(rect_item)
            # Label text
            label_item = self._add_label_text(w["text"], w["bbox"], w["bbox"])
            self._wire_items.append(label_item)

    def _clear_temp_items(self):
        if self._pending_rect_item:
            self.scene().removeItem(self._pending_rect_item)
            self._pending_rect_item = None
        if self._rubber_rect:
            self.scene().removeItem(self._rubber_rect)
            self._rubber_rect = None
        if self._hover_item:
            self.scene().removeItem(self._hover_item)
            self._hover_item = None

    # ---- Highlight selected pair on canvas ----
    def _clear_highlight(self):
        if hasattr(self, '_highlight_item') and self._highlight_item:
            self.scene().removeItem(self._highlight_item)
            self._highlight_item = None

    def _draw_highlight(self, bbox):
        self._clear_highlight()
        x0, y0, x1, y1 = bbox
        rect = QGraphicsRectItem(x0, y0, x1 - x0, y1 - y0)
        pen = QPen(QColor(255, 255, 0, 180), 3)
        pen.setStyle(Qt.DashLine)
        rect.setPen(pen)
        rect.setBrush(QColor(255, 255, 0, 40))
        rect.setZValue(999)
        self.scene().addItem(rect)
        self._highlight_item = rect

    # ---- Label-adjust mode: adjustable bbox handles ----
    def _label_adjust_draw(self):
        """Draw the adjustable label bbox and four edge handles."""
        self._label_adjust_clear()
        if not self._label_adjust_bbox:
            return
        x0, y0, x1, y1 = self._label_adjust_bbox
        # Bbox rectangle (dashed blue)
        rect = QGraphicsRectItem(x0, y0, x1 - x0, y1 - y0)
        pen = QPen(C_LABEL, 2.0 / max(self._zoom, 0.1))
        pen.setStyle(Qt.DashLine)
        rect.setPen(pen)
        rect.setBrush(QColor(59, 130, 246, 40))
        rect.setZValue(50)
        self.scene().addItem(rect)
        self._label_adjust_items.append(rect)

        # Handle size in PDF coords (scale-independent 6px)
        hs = 6.0 / max(self._zoom, 0.1)
        cy = (y0 + y1) / 2
        cx = (x0 + x1) / 2

        handle_defs = [
            ('left',   x0 - hs/2, cy - hs/2),
            ('right',  x1 - hs/2, cy - hs/2),
            ('top',    cx - hs/2,  y0 - hs/2),
            ('bottom', cx - hs/2,  y1 - hs/2),
        ]
        for name, hx, hy in handle_defs:
            h = QGraphicsRectItem(hx, hy, hs, hs)
            h.setPen(QPen(QColor(255, 255, 255), 1.0 / max(self._zoom, 0.1)))
            h.setBrush(C_LABEL)
            h.setZValue(51)
            h.setData(0, name)  # store edge name
            self.scene().addItem(h)
            self._label_adjust_items.append(h)

        # Text preview above the bbox
        txt_item = QGraphicsSimpleTextItem(self._label_adjust_text)
        txt_item.setBrush(QColor(255, 255, 255))
        font = QFont("Consolas", max(3, 8 / max(self._zoom, 0.1)))
        txt_item.setFont(font)
        txt_item.setPos(x0, y0 - 12 / max(self._zoom, 0.1))
        txt_item.setZValue(52)
        self.scene().addItem(txt_item)
        self._label_adjust_items.append(txt_item)

    def _label_adjust_clear(self):
        for item in self._label_adjust_items:
            self.scene().removeItem(item)
        self._label_adjust_items.clear()

    def _label_adjust_hit_handle(self, scene_pos):
        """Return the handle edge name if scene_pos is near a handle, else None."""
        hs = 6.0 / max(self._zoom, 0.1)
        margin = hs * 1.5  # generous hit area
        if not self._label_adjust_bbox:
            return None
        x0, y0, x1, y1 = self._label_adjust_bbox
        cy = (y0 + y1) / 2
        cx = (x0 + x1) / 2
        sx, sy = scene_pos.x(), scene_pos.y()

        tests = [
            ('left',   x0, cy),
            ('right',  x1, cy),
            ('top',    cx, y0),
            ('bottom', cx, y1),
        ]
        for name, hx, hy in tests:
            if abs(sx - hx) < margin and abs(sy - hy) < margin:
                return name
        return None

    def _label_adjust_accept(self):
        """Accept the adjusted label bbox and create the pair."""
        if not self._label_adjust_bbox or not self.pending_comp:
            return
        self._redo_stack.clear()
        self.pairs.append({
            "component": {"bbox": list(self.pending_comp)},
            "label": {"text": self._label_adjust_text,
                      "bbox": list(self._label_adjust_bbox)}
        })
        self._label_adjust_clear()
        self._label_adjust_bbox = None
        self._label_adjust_text = None
        self._label_adjust_dragging = None
        self._clear_temp_items()
        self.pending_comp = None
        self.mode = self.main_mode
        self.setCursor(Qt.CrossCursor if self.main_mode == "component" else Qt.PointingHandCursor)
        self._redraw_pairs()
        self._save_annotations()
        self.pairCreated.emit()

    def _label_adjust_cancel(self):
        """Cancel label adjustment, return to label mode."""
        self._label_adjust_clear()
        self._label_adjust_bbox = None
        self._label_adjust_text = None
        self._label_adjust_dragging = None
        self.mode = "label"
        self.setCursor(Qt.PointingHandCursor)
        self.pairCreated.emit()

    # ---- Mouse events ----
    def _is_eraser(self, event):
        """Check if event comes from stylus eraser."""
        dev = event.device()
        # In PySide6, we might need to check pointerType if it's a pointing device
        if hasattr(dev, "pointerType"):
             from PySide6.QtGui import QPointingDevice
             if dev.pointerType() == QPointingDevice.PointerType.Eraser:
                 return True
        return False

    def mouseDoubleClickEvent(self, event):
        if self._is_eraser(event):
            self.undo()
            return
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event):
        # STYLUS ONLY MODE: Ignore TouchScreen inputs (palm rejection)
        if event.device().type() == QInputDevice.DeviceType.TouchScreen:
             return

        pos = self.mapToScene(event.position().toPoint())

        # ---- Mask mode ----
        if self._mask_mode:
            if self._is_eraser(event):
                self._mask_painting = True
                self._mask_erasing = True
                self._mask_erase_at(pos)
                return
            if event.button() == Qt.RightButton:
                self._panning = True
                self._pan_start = event.position()
                self.setCursor(Qt.ClosedHandCursor)
                return
            if event.button() == Qt.LeftButton:
                # E-key erase mode: erase with left click
                if self._mask_erase_mode:
                    self._mask_painting = True
                    self._mask_erasing = True
                    self._mask_erase_at(pos)
                    return
                pair_idx = self._mask_find_pair_at(
                    int(pos.x()), int(pos.y()))
                if pair_idx is None:
                    self._mask_on_click_outside()
                else:
                    self._mask_active_pair = pair_idx
                    self._mask_painting = True
                    self._mask_paint_at(pos)
            return

        if self._is_eraser(event):
            return  # Don't draw with eraser

        # Right-click or Side Button: pan
        if event.button() == Qt.RightButton:
            self._panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            return

        if event.button() != Qt.LeftButton:
            return

        if self.mode == "component" or self.mode == "continuation":
            self._drawing = True
            self._draw_origin = pos
            # create rubber-band rect
            color = C_LINK if self.mode == "continuation" else C_COMP
            pen = QPen(color, 1.0 / self._zoom)
            pen.setStyle(Qt.DashLine)
            self._rubber_rect = self.scene().addRect(QRectF(pos, pos), pen)
            self._rubber_rect.setZValue(20)

        elif self.mode == "label":
            px, py = self.scene_to_pdf(pos.x(), pos.y())
            tb = self.snap_text(px, py)
            if tb:
                self._redo_stack.clear()
                self.pairs.append({
                    "component": {"bbox": list(self.pending_comp)},
                    "label": {"text": tb["text"], "bbox": list(tb["bbox"])}
                })
                self._clear_temp_items()
                self.pending_comp = None
                self.mode = self.main_mode
                self.setCursor(Qt.CrossCursor if self.main_mode == "component" else Qt.PointingHandCursor)
                self._redraw_pairs()
                self._save_annotations()
                self.pairCreated.emit()

        elif self.mode == "cont_page":
            # Click page number in continuation symbol
            px, py = self.scene_to_pdf(pos.x(), pos.y())
            tb = self.snap_text(px, py)
            if tb:
                self.pending_continuation["page"] = {"text": tb["text"], "bbox": list(tb["bbox"])}
                self.mode = "cont_rung"
                self.pairCreated.emit()

        elif self.mode == "cont_rung":
            # Click rung number to complete continuation annotation
            px, py = self.scene_to_pdf(pos.x(), pos.y())
            tb = self.snap_text(px, py)
            if tb:
                self._redo_stack.clear()
                self.pending_continuation["rung"] = {"text": tb["text"], "bbox": list(tb["bbox"])}
                self.pending_continuation["type"] = "continuation"
                self.pairs.append(self.pending_continuation)
                self._clear_temp_items()
                self.pending_continuation = None
                self.mode = self.main_mode
                self.setCursor(Qt.CrossCursor if self.main_mode == "component" else Qt.PointingHandCursor)
                self._redraw_pairs()
                self._save_annotations()
                self.pairCreated.emit()

        elif self.mode == "wire":
            # Quick-click wire label annotation
            px, py = self.scene_to_pdf(pos.x(), pos.y())
            tb = self.snap_text(px, py)
            if tb:
                self._redo_stack.clear()
                self.wire_labels.append({"text": tb["text"], "bbox": list(tb["bbox"])})
                self._redraw_wires()
                self._save_annotations()
                # Stay in wire mode for rapid clicking

    def mouseMoveEvent(self, event):
        if event.device().type() == QInputDevice.DeviceType.TouchScreen:
             return

        pos = self.mapToScene(event.position().toPoint())

        # ---- Mask mode painting ----
        if self._mask_mode and self._mask_painting:
            if self._mask_erasing:
                self._mask_erase_at(pos)
            else:
                self._mask_paint_at(pos)
            return

        if self._panning:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y()))
            return

        # Label-adjust handle dragging
        if self.mode == "label_adjust" and self._label_adjust_dragging:
            bbox = self._label_adjust_bbox
            edge = self._label_adjust_dragging
            if edge == 'left':
                bbox[0] = min(pos.x(), bbox[2] - 2)
            elif edge == 'right':
                bbox[2] = max(pos.x(), bbox[0] + 2)
            elif edge == 'top':
                bbox[1] = min(pos.y(), bbox[3] - 2)
            elif edge == 'bottom':
                bbox[3] = max(pos.y(), bbox[1] + 2)
            self._label_adjust_draw()
            return

        # Label-adjust cursor feedback
        if self.mode == "label_adjust" and self._label_adjust_bbox:
            handle = self._label_adjust_hit_handle(pos)
            if handle in ('left', 'right'):
                self.setCursor(Qt.SizeHorCursor)
            elif handle in ('top', 'bottom'):
                self.setCursor(Qt.SizeVerCursor)
            else:
                self.setCursor(Qt.PointingHandCursor)

        if self._drawing and self._rubber_rect:
            r = QRectF(self._draw_origin, pos).normalized()
            self._rubber_rect.setRect(r)
            return

        # Hover: show text highlight in label mode, continuation modes, or wire mode
        if self.mode in ["label", "cont_page", "cont_rung", "wire"]:
            px, py = self.scene_to_pdf(pos.x(), pos.y())
            tb = self.snap_text(px, py)
            if self._hover_item:
                self.scene().removeItem(self._hover_item)
                self._hover_item = None
            if tb:
                if self.mode in ["cont_page", "cont_rung"]:
                    color = C_LINK
                elif self.mode == "wire":
                    color = QColor(168, 85, 247)  # purple
                else:
                    color = C_LABEL
                self._hover_item = self._add_pdf_rect(tb["bbox"], color, C_HOVER, dash=True)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        # ---- Mask mode release ----
        if self._mask_mode:
            if self._mask_painting:
                self._mask_painting = False
                self._mask_erasing = False
                if self._mask_dirty:
                    self._mask_refresh_display()
            if self._panning:
                self._panning = False
                self.setCursor(Qt.CrossCursor)
            return

        if self._is_eraser(event):
            self.redo()
            return

        # Stop label-adjust handle dragging
        if self.mode == "label_adjust" and self._label_adjust_dragging:
            self._label_adjust_dragging = None
            return

        if self._panning:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            if self.mode in ["wire", "label", "label_adjust", "cont_page", "cont_rung"]:
                 self.setCursor(Qt.PointingHandCursor)
            else:
                 self.setCursor(Qt.CrossCursor)
            return

        if event.button() == Qt.LeftButton and self._drawing:
            self._drawing = False
            pos = self.mapToScene(event.position().toPoint())
            r = QRectF(self._draw_origin, pos).normalized()

            # Remove rubber band
            if self._rubber_rect:
                self.scene().removeItem(self._rubber_rect)
                self._rubber_rect = None

            # Only register if meaningful size
            if r.width() > 3 and r.height() > 3:
                self._redo_stack.clear()
                # Convert to PDF coords and snap
                px0, py0 = self.scene_to_pdf(r.left(), r.top())
                px1, py1 = self.scene_to_pdf(r.right(), r.bottom())
                snapped = self.snap_component([px0, py0, px1, py1])

                if self.mode == "component":
                    self.pending_comp = snapped
                    # Draw snapped rect
                    self._pending_rect_item = self._add_pdf_rect(
                        snapped, C_COMP, C_COMP_FILL
                    )
                    self.mode = "label"
                    self.setCursor(Qt.PointingHandCursor)
                    self.pairCreated.emit()  # updates UI mode display

                elif self.mode == "continuation":
                    # Started continuation annotation - symbol box drawn
                    self.pending_continuation = {"symbol": {"bbox": snapped}}
                    self._pending_rect_item = self._add_pdf_rect(
                        snapped, C_LINK, QColor(245, 158, 11, 30), 1.0
                    )
                    self.mode = "cont_page"
                    self.setCursor(Qt.PointingHandCursor)
                    self.pairCreated.emit()

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self._zoom *= factor
        self._zoom = max(0.1, min(20, self._zoom))
        self.scale(factor, factor)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_D and event.modifiers() & Qt.ControlModifier:
            # Ctrl+D: enter continuation annotation mode
            self._clear_temp_items()
            self.pending_comp = None
            self.pending_continuation = None
            self.mode = "continuation"
            self.main_mode = "continuation"  # Sync main_mode
            self.setCursor(Qt.CrossCursor)
            self.pairCreated.emit()
        elif event.key() == Qt.Key_Escape:
            if self.mode == "label_adjust":
                self._label_adjust_cancel()
                return
            self._clear_temp_items()
            self.pending_comp = None
            self.pending_continuation = None
            self.mode = self.main_mode
            self.setCursor(Qt.CrossCursor if self.main_mode == "component" else Qt.PointingHandCursor)
            self.pairCreated.emit()
        elif event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if self.mode == "label_adjust":
                self._label_adjust_accept()
                return
        elif event.key() == Qt.Key_F:
            self.fit_view()
        elif event.key() == Qt.Key_Z and event.modifiers() & Qt.ControlModifier:
            self.undo()
        elif event.key() == Qt.Key_Y and event.modifiers() & Qt.ControlModifier:
            self.redo()
        elif event.key() == Qt.Key_Delete:
            pass  # handled by main window
        else:
            super().keyPressEvent(event)

    def undo(self):
        if self.pending_comp:
            # Last box was a component box — remove it
            self._redo_stack.append({"type": "comp", "bbox": list(self.pending_comp)})
            self._clear_temp_items()
            self.pending_comp = None
            self.mode = "component"
            self.main_mode = "component"
            self.setCursor(Qt.CrossCursor)
            self.pairCreated.emit()
            return

        # Determine what to undo based on mode priority
        undo_wire = False
        if self.main_mode == "wire":
            if self.wire_labels: undo_wire = True
            elif self.pairs: undo_wire = False
        else:
            if self.pairs: undo_wire = False
            elif self.wire_labels: undo_wire = True
        
        if undo_wire:
             # Last action was a wire label
             last = self.wire_labels.pop()
             self._redo_stack.append({"type": "wire", "data": last})
             self._redraw_wires()
             self._save_annotations()
             # Mode stays wire
             self.pairCreated.emit()
        elif self.pairs:
             # Last box was a label — remove the pair, restore component as pending
             last = self.pairs.pop()
             self._redo_stack.append({"type": "label", "pair": last})
             self.pending_comp = last["component"]["bbox"]
             self._redraw_pairs()
             self._save_annotations()
             self._pending_rect_item = self._add_pdf_rect(
                 self.pending_comp, C_COMP, C_COMP_FILL, 1.0
             )
             self.mode = "label"
             self.main_mode = "component"
             self.setCursor(Qt.PointingHandCursor)
             self.pairCreated.emit()

    def redo(self):
        if self._redo_stack:
            action = self._redo_stack.pop()
            if action["type"] == "comp":
                # Re-place component box
                self.pending_comp = action["bbox"]
                self._pending_rect_item = self._add_pdf_rect(
                    self.pending_comp, C_COMP, C_COMP_FILL, 1.0
                )
                self.mode = "label"
                self.main_mode = "component"
                self.setCursor(Qt.PointingHandCursor)
                self.pairCreated.emit()
            elif action["type"] == "label":
                # Re-complete the pair
                self._clear_temp_items()
                self.pending_comp = None
                self.pairs.append(action["pair"])
                self._redraw_pairs()
                self._save_annotations()
                self.mode = "component"
                self.main_mode = "component"
                self.setCursor(Qt.CrossCursor)
                self.pairCreated.emit()
            elif action["type"] == "wire":
                self.wire_labels.append(action["data"])
                self._redraw_wires()
                self._save_annotations()
                self.pairCreated.emit()

    def zoom_in(self):
        self._zoom *= 1.25
        self.scale(1.25, 1.25)

    def zoom_out(self):
        self._zoom /= 1.25
        self.scale(0.8, 0.8)

    # ---- Mask mode ----
    def _mask_enter(self):
        """Enter mask mode."""
        self._mask_mode = True
        self._mask_init_for_page()
        self.pairCreated.emit()

    def _mask_exit(self):
        """Exit mask mode, save overlay."""
        self._mask_save_overlay()
        self._mask_save_metadata()
        self._mask_mode = False
        self._mask_painting = False
        self._mask_erasing = False
        self._mask_erase_mode = False
        self._mask_active_pair = None
        if self._mask_overlay_item:
            self.scene().removeItem(self._mask_overlay_item)
            self._mask_overlay_item = None
        self._mask_source_image = None
        self._mask_overlay_image = None
        self.setCursor(Qt.CrossCursor)
        self.pairCreated.emit()

    def _mask_init_for_page(self):
        """Set up mask overlay for the current page."""
        self._mask_source_image = self.page_pixmap_item.pixmap().toImage()
        w = self._mask_source_image.width()
        h = self._mask_source_image.height()
        self._mask_overlay_image = QImage(w, h, QImage.Format_ARGB32)
        self._mask_overlay_image.fill(QColor(0, 0, 0, 0))
        self._mask_load_overlay()
        if self._mask_overlay_item:
            self.scene().removeItem(self._mask_overlay_item)
        self._mask_overlay_item = self.scene().addPixmap(
            QPixmap.fromImage(self._mask_overlay_image))
        self._mask_overlay_item.setZValue(5)
        self._mask_cache_bboxes()
        self._mask_load_metadata()
        self._mask_active_pair = None
        self._mask_painting = False
        self._mask_dirty = False
        self._mask_update_counter = 0
        self.setCursor(Qt.CrossCursor)

    def _mask_cache_bboxes(self):
        """Precompute annotated component bboxes in pixel (scene) coords."""
        s = self._scale()
        self._mask_bboxes = []
        for i, p in enumerate(self.pairs):
            if p.get("type") == "continuation":
                continue
            bbox = p["component"]["bbox"]
            self._mask_bboxes.append((
                bbox[0] * s, bbox[1] * s, bbox[2] * s, bbox[3] * s, i
            ))

    def _mask_find_pair_at(self, sx, sy):
        """Return pair index if pixel (sx, sy) is inside a component bbox."""
        for x0, y0, x1, y1, idx in self._mask_bboxes:
            if x0 <= sx <= x1 and y0 <= sy <= y1:
                return idx
        return None

    def _mask_paint_at(self, scene_pos):
        """Paint blue over black pixels near scene_pos within annotated bboxes."""
        sx, sy = int(scene_pos.x()), int(scene_pos.y())
        w = self._mask_source_image.width()
        h = self._mask_source_image.height()
        radius = self._mask_brush_radius
        blue_rgba = QColor(0, 120, 255, 180).rgba()
        changed = False
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx * dx + dy * dy > radius * radius:
                    continue
                px, py = sx + dx, sy + dy
                if px < 0 or py < 0 or px >= w or py >= h:
                    continue
                if self._mask_find_pair_at(px, py) is None:
                    continue
                src = self._mask_source_image.pixel(px, py)
                r = (src >> 16) & 0xFF
                g = (src >> 8) & 0xFF
                b_val = src & 0xFF
                if r < 80 and g < 80 and b_val < 80:
                    self._mask_overlay_image.setPixel(px, py, blue_rgba)
                    changed = True
        if changed:
            self._mask_dirty = True
            self._mask_update_counter += 1
            if self._mask_update_counter % 3 == 0:
                self._mask_refresh_display()

    def _mask_erase_at(self, scene_pos):
        """Erase mask pixels near scene_pos."""
        sx, sy = int(scene_pos.x()), int(scene_pos.y())
        w = self._mask_overlay_image.width()
        h = self._mask_overlay_image.height()
        radius = self._mask_eraser_radius
        transparent = QColor(0, 0, 0, 0).rgba()
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx * dx + dy * dy > radius * radius:
                    continue
                px, py = sx + dx, sy + dy
                if px < 0 or py < 0 or px >= w or py >= h:
                    continue
                self._mask_overlay_image.setPixel(px, py, transparent)
        self._mask_dirty = True
        self._mask_update_counter += 1
        if self._mask_update_counter % 3 == 0:
            self._mask_refresh_display()

    def _mask_refresh_display(self):
        """Update the overlay QGraphicsPixmapItem from the overlay QImage."""
        if self._mask_overlay_item and self._mask_overlay_image:
            self._mask_overlay_item.setPixmap(
                QPixmap.fromImage(self._mask_overlay_image))
            self._mask_dirty = False

    def _mask_on_click_outside(self):
        """Called when user clicks outside annotated bboxes — save current."""
        if self._mask_active_pair is not None:
            self._mask_saved_pairs.add(self._mask_active_pair)
            self._mask_active_pair = None
            self._mask_save_overlay()
            self._mask_save_metadata()
            self.pairCreated.emit()

    def _mask_save_overlay(self):
        """Save mask overlay image to disk."""
        os.makedirs(MASK_DIR, exist_ok=True)
        path = os.path.join(MASK_DIR, f"page_{self.page_idx + 1}_overlay.png")
        if self._mask_overlay_image:
            self._mask_overlay_image.save(path)

    def _mask_load_overlay(self):
        """Load existing mask overlay from disk."""
        path = _existing_mask_path(self.page_idx + 1, f"overlay.png")
        if path and os.path.exists(path):
            loaded = QImage(path)
            if (loaded.width() == self._mask_overlay_image.width() and
                    loaded.height() == self._mask_overlay_image.height()):
                self._mask_overlay_image = loaded.convertToFormat(
                    QImage.Format_ARGB32)

    def _mask_save_metadata(self):
        """Save mask metadata to disk."""
        os.makedirs(MASK_DIR, exist_ok=True)
        path = os.path.join(MASK_DIR, f"page_{self.page_idx + 1}_masks.json")
        info = {}
        for idx in self._mask_saved_pairs:
            if idx < len(self.pairs):
                p = self.pairs[idx]
                if p.get("type") == "continuation":
                    continue
                info[str(idx)] = {
                    "label": p["label"]["text"],
                    "component_bbox": p["component"]["bbox"],
                }
        data = {
            "page": self.page_idx + 1,
            "masked_pairs": sorted(self._mask_saved_pairs),
            "pair_info": info,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _mask_load_metadata(self):
        """Load mask metadata from disk."""
        path = _existing_mask_path(self.page_idx + 1, f"masks.json")
        if path is None:
            self._mask_saved_pairs = set()
            return
        self._mask_saved_pairs = set()
        try:
            with open(path) as f:
                d = json.load(f)
            self._mask_saved_pairs = set(d.get("masked_pairs", []))
        except (json.JSONDecodeError, KeyError, OSError):
            pass

    # ---- Persistence ----
    def viewportEvent(self, event):
        if event.type() == QEvent.TouchBegin:
            self._touch_points = event.points()
            if len(self._touch_points) == 2:
                # Start pinch
                p0 = self._touch_points[0].position()
                p1 = self._touch_points[1].position()
                self._initial_pinch_distance = QLineF(p0, p1).length()
            elif len(self._touch_points) == 1:
                # Start flick detection
                self._flick_start_pos = self._touch_points[0].position()
                self._flick_start_time = time.time()
            return True
        elif event.type() == QEvent.TouchUpdate:
            pts = event.points()
            if len(pts) == 2 and self._initial_pinch_distance:
                # Pinch zoom
                p0 = pts[0].position()
                p1 = pts[1].position()
                dist = QLineF(p0, p1).length()
                scale_factor = dist / self._initial_pinch_distance
                # Dampen zoom sensitivity slightly
                if abs(scale_factor - 1.0) > 0.01:
                    self._zoom *= scale_factor
                    self.scale(scale_factor, scale_factor)
                self._initial_pinch_distance = dist
            return True
        elif event.type() == QEvent.TouchEnd:
            pts = event.points()
            if len(pts) == 1 and self._flick_start_pos:
                # Check flick
                end_pos = pts[0].position()
                dt = time.time() - self._flick_start_time
                dx = end_pos.x() - self._flick_start_pos.x()
                if dt < 0.3 and abs(dx) > 100:
                    # Swipe left/right
                    if dx > 0:
                        self.requestPrevPage.emit()
                    else:
                        self.requestNextPage.emit()
            self._initial_pinch_distance = None
            self._flick_start_pos = None
            return True
        return super().viewportEvent(event)

    def _save_annotations(self):
        os.makedirs(ANN_DIR, exist_ok=True)
        path = os.path.join(ANN_DIR, f"page_{self.page_idx + 1}.json")
        # Only write annotation files for pages that actually have annotations
        if not self.pairs and not self.wire_labels:
            # Remove stale empty file if it exists
            if os.path.exists(path):
                os.remove(path)
            return
        with open(path, "w") as f:
            json.dump({"pairs": self.pairs, "wire_labels": self.wire_labels}, f, indent=2)

    def _load_annotations(self, idx):
        path = os.path.join(ANN_DIR, f"page_{idx + 1}.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
                # Support both old format (list) and new format (dict)
                if isinstance(data, list):
                    self.pairs = data
                    self.wire_labels = []
                else:
                    self.pairs = data.get("pairs", [])
                    self.wire_labels = data.get("wire_labels", [])
        else:
            self.pairs = []
            self.wire_labels = []
        self._redraw_wires()


# ===================================================================
# AnnotatorWindow — Main application window
# ===================================================================
# ===================================================================
# Roboflow upload worker thread
# ===================================================================
class RoboflowUploadWorker(QThread):
    progress_signal = Signal(int, int, str)   # current, total, message
    finished_signal = Signal(list)            # list of error strings

    def __init__(self, api_key, workspace, project_name, meta, ann_dir, pages_dir, selected_pages=None):
        super().__init__()
        self.api_key = api_key
        self.workspace = workspace
        self.project_name = project_name
        self.meta = meta
        self.ann_dir = ann_dir
        self.pages_dir = pages_dir
        self.selected_pages = selected_pages  # list of 1-based page nums, or None for all
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        errors = []
        try:
            import roboflow
            rf = roboflow.Roboflow(api_key=self.api_key)
            ws = rf.workspace(self.workspace)
            # Try to get existing project, create if it doesn't exist
            try:
                project = ws.project(self.project_name)
            except Exception:
                project = ws.create_project(
                    project_name=self.project_name,
                    project_type="object-detection",
                    project_license="private",
                    annotation="atlas-components",
                )
        except Exception as e:
            self.finished_signal.emit([f"Connection/project error: {e}"])
            return

        pages_map = {pg["page"]: pg for pg in self.meta["pages"]}

        if self.selected_pages:
            ann_files = sorted(f"page_{p}.json" for p in self.selected_pages
                               if os.path.exists(os.path.join(self.ann_dir, f"page_{p}.json")))
        else:
            ann_files = sorted(
                f for f in os.listdir(self.ann_dir)
                if f.startswith("page_") and f.endswith(".json") and not f.startswith("_")
            )
        total = len(ann_files)
        if total == 0:
            self.finished_signal.emit(["No annotated pages found in annotations folder."])
            return

        for idx, fname in enumerate(ann_files):
            if self._cancelled:
                errors.append("Upload cancelled by user.")
                break
            try:
                pg_num = int(fname.replace("page_", "").replace(".json", ""))
            except ValueError:
                continue

            pg_meta = pages_map.get(pg_num)
            if not pg_meta:
                errors.append(f"{fname}: page metadata not found")
                continue

            ann_path = os.path.join(self.ann_dir, fname)
            with open(ann_path, encoding="utf-8") as f:
                ann = json.load(f)

            image_path = os.path.join(self.pages_dir, pg_meta["image"])
            if not os.path.exists(image_path):
                errors.append(f"page_{pg_num}: image not found ({image_path})")
                continue

            self.progress_signal.emit(idx, total, f"Uploading page {pg_num} ({idx+1}/{total})…")

            xml_str = self._build_voc_xml(ann, pg_meta)
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".xml", delete=False, mode="w", encoding="utf-8"
                ) as tmp:
                    tmp.write(xml_str)
                    tmp_path = tmp.name

                project.upload(
                    image_path=image_path,
                    annotation_path=tmp_path,
                    split="train",
                    batch_name="atlas-upload",
                    num_retry_uploads=1,
                )
            except Exception as e:
                errors.append(f"page_{pg_num}: {e}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

        self.progress_signal.emit(total, total, "Upload complete")
        self.finished_signal.emit(errors)

    def _build_voc_xml(self, ann, pg_meta):
        """Convert ATLAS annotation JSON to Pascal VOC XML."""
        img_w, img_h = pg_meta["display_size"]
        pdf_w = pg_meta["pdf_width"]
        pdf_h = pg_meta["pdf_height"]
        sx = img_w / pdf_w
        sy = img_h / pdf_h

        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = "atlas"
        ET.SubElement(root, "filename").text = pg_meta["image"]
        sz = ET.SubElement(root, "size")
        ET.SubElement(sz, "width").text = str(img_w)
        ET.SubElement(sz, "height").text = str(img_h)
        ET.SubElement(sz, "depth").text = "3"
        ET.SubElement(root, "segmented").text = "0"

        def add_obj(class_name, bbox):
            x1, y1, x2, y2 = bbox
            xmin = max(0, round(x1 * sx))
            ymin = max(0, round(y1 * sy))
            xmax = min(img_w, round(x2 * sx))
            ymax = min(img_h, round(y2 * sy))
            if xmax <= xmin or ymax <= ymin:
                return
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = class_name
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"
            bb = ET.SubElement(obj, "bndbox")
            ET.SubElement(bb, "xmin").text = str(xmin)
            ET.SubElement(bb, "ymin").text = str(ymin)
            ET.SubElement(bb, "xmax").text = str(xmax)
            ET.SubElement(bb, "ymax").text = str(ymax)

        for pair in ann.get("pairs", []):
            if pair.get("type") == "continuation":
                if "symbol" in pair:
                    add_obj("continuation_symbol", pair["symbol"]["bbox"])
                if "page" in pair:
                    add_obj("continuation_page", pair["page"]["bbox"])
                if "rung" in pair:
                    add_obj("continuation_rung", pair["rung"]["bbox"])
            else:
                if "component" in pair:
                    add_obj("component", pair["component"]["bbox"])
                if "label" in pair:
                    add_obj("component_label", pair["label"]["bbox"])

        for wire in ann.get("wire_labels", []):
            add_obj("wire_label", wire["bbox"])

        return ET.tostring(root, encoding="unicode", xml_declaration=False)


class AnnotatorWindow(QMainWindow):
    def __init__(self, metadata):
        super().__init__()
        self.meta = metadata
        self.setWindowTitle("ATLAS Component Annotator")
        self.resize(1400, 900)
        self.setMinimumSize(1000, 600)
        self._apply_dark_theme()

        # -----------------------------------------------------------------
        # Central content: annotation canvas (direct, no stacked widget)
        # -----------------------------------------------------------------
        self.view = SchematicView()
        self.view.load_metadata(metadata)
        self.view.pairCreated.connect(self._update_sidebar)
        self.view.pairCreated.connect(self._update_mask_sidebar)
        self.view.requestPrevPage.connect(self._prev_page)
        self.view.requestNextPage.connect(self._next_page)
        self.setCentralWidget(self.view)

        # Fingerprinting viewer lives as a separate dialog (never hides the canvas)
        self.fingerprinting = FingerprintingDialog(metadata, parent=self)

        # Toolbar
        self._build_toolbar()

        # Sidebar dock (right)
        self._build_sidebar()

        # Mask sidebar dock (left)
        self._build_mask_sidebar()

        # Status bar
        self.status_label = QLabel("Draw a box around a component")
        self.coord_label = QLabel("")
        self.statusBar().addWidget(self.status_label, 1)
        self.statusBar().addPermanentWidget(self.coord_label)

        # Load settings
        self._settings = self._load_settings()
        self._apply_settings_to_view()

        # Load last page (or page 0 if first run)
        start_page = self._load_last_page()
        self.view.load_page(start_page)
        self._update_sidebar()
        self._update_page_label()

        # Ctrl+S shortcut for manual save
        save_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        save_shortcut.activated.connect(self._manual_save)

    def _show_fingerprinting_page(self):
        """Open the Fingerprinting results viewer as a dialog."""
        self.fingerprinting.show()
        self.fingerprinting.raise_()
        self.fingerprinting.activateWindow()

    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow { background: #0d1117; }
            QToolBar { background: #161b22; border-bottom: 1px solid #30363d;
                       spacing: 8px; padding: 4px 8px; }
            QToolBar QLabel { color: #e6edf3; font-size: 13px; }
            QPushButton { background: #30363d; color: #e6edf3; border: none;
                          padding: 5px 14px; border-radius: 5px; font-size: 12px; }
            QPushButton:hover { background: #484f58; }
            QPushButton:pressed { background: #00d4ff; color: #0d1117; }
            QStatusBar { background: #161b22; border-top: 1px solid #30363d; color: #8b949e; }
            QStatusBar QLabel { color: #8b949e; font-size: 11px; }
            QDockWidget { color: #e6edf3; font-size: 12px; }
            QDockWidget::title { background: #161b22; padding: 6px;
                                  border-bottom: 1px solid #30363d; }
            QListWidget { background: #0d1117; color: #e6edf3; border: none;
                          font-size: 12px; }
            QListWidget::item { padding: 6px 8px; border-bottom: 1px solid #161b22; }
            QListWidget::item:selected { background: #1a2233; color: #00d4ff; }
            QListWidget::item:hover { background: #161b22; }
            QGroupBox { color: #8b949e; border: 1px solid #30363d; border-radius: 5px;
                        margin-top: 8px; padding-top: 14px; font-size: 11px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
        """)

    def _build_toolbar(self):
        tb = QToolBar("Main")
        tb.setMovable(False)
        tb.setIconSize(QSize(16, 16))
        self.addToolBar(tb)

        # Logo
        logo = QLabel("  ⚡ ATLAS  ")
        logo.setStyleSheet("font-weight: 700; font-size: 15px; color: #00d4ff;")
        tb.addWidget(logo)
        tb.addSeparator()

        # Page nav
        self.btn_prev = QPushButton("◀ Prev")
        self.btn_prev.clicked.connect(self._prev_page)
        tb.addWidget(self.btn_prev)

        self.page_label = QLabel(" 1 / 1 ")
        self.page_label.setStyleSheet("min-width: 60px; text-align: center; font-weight: 600;")
        tb.addWidget(self.page_label)

        self.btn_next = QPushButton("Next ▶")
        self.btn_next.clicked.connect(self._next_page)
        tb.addWidget(self.btn_next)

        tb.addSeparator()

        self.btn_fit = QPushButton("⊞ Fit")
        self.btn_fit.clicked.connect(self.view.fit_view)
        tb.addWidget(self.btn_fit)

        tb.addSeparator()

        # Continuation mode button
        btn_cont = QPushButton("⊙ Continuation")
        btn_cont.clicked.connect(self._enter_continuation_mode)
        tb.addWidget(btn_cont)

        # Wire mode button
        btn_wire = QPushButton("⚡ Wire")
        btn_wire.clicked.connect(self._enter_wire_mode)
        tb.addWidget(btn_wire)

        # Component mode button
        btn_comp = QPushButton("◻ Component")
        btn_comp.clicked.connect(self._enter_component_mode)
        tb.addWidget(btn_comp)

        tb.addSeparator()

        # Mode indicator
        self.mode_label = QLabel("  ● COMPONENT  ")
        self.mode_label.setStyleSheet(
            "background: #22c55e; color: #000; font-weight: 700; font-size: 11px;"
            "padding: 3px 10px; border-radius: 10px;"
        )
        tb.addWidget(self.mode_label)

        # Spacer
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(spacer)

        # Undo/Redo buttons
        btn_undo = QPushButton("↶ Undo")
        btn_undo.clicked.connect(self.view.undo)
        tb.addWidget(btn_undo)

        btn_redo = QPushButton("↷ Redo")
        btn_redo.clicked.connect(self.view.redo)
        tb.addWidget(btn_redo)



        tb.addSeparator()

        # Zoom buttons
        btn_zoom_in = QPushButton("➕ In")
        btn_zoom_in.clicked.connect(self.view.zoom_in)
        tb.addWidget(btn_zoom_in)

        btn_zoom_out = QPushButton("➖ Out")
        btn_zoom_out.clicked.connect(self.view.zoom_out)
        tb.addWidget(btn_zoom_out)

        tb.addSeparator()

        # Pair counter
        self.pair_count_label = QLabel("0 pairs")
        self.pair_count_label.setStyleSheet("color: #f59e0b; font-weight: 600;")
        tb.addWidget(self.pair_count_label)

        # --- Second toolbar row for settings + export ---
        self.addToolBarBreak()
        tb2 = QToolBar("Export")
        tb2.setMovable(False)
        tb2.setIconSize(QSize(16, 16))
        self.addToolBar(tb2)

        btn_settings = QPushButton("⚙ Settings")
        btn_settings.clicked.connect(self._open_settings)
        tb2.addWidget(btn_settings)

        tb2.addSeparator()

        btn_export = QPushButton("💾 Export JSON")
        btn_export.clicked.connect(self._export_json)
        tb2.addWidget(btn_export)

        btn_export_img = QPushButton("🖼 Export Image")
        btn_export_img.clicked.connect(self._export_images)
        tb2.addWidget(btn_export_img)

        tb2.addSeparator()

        btn_roboflow = QPushButton("🌐 Upload to Roboflow")
        btn_roboflow.clicked.connect(self._export_to_roboflow)
        tb2.addWidget(btn_roboflow)

        btn_rf_dataset = QPushButton("📦 Export YOLO")
        btn_rf_dataset.clicked.connect(self._export_rf_dataset)
        tb2.addWidget(btn_rf_dataset)

        btn_rf_coco = QPushButton("🎯 Export Roboflow")
        btn_rf_coco.clicked.connect(self._export_roboflow_coco)
        tb2.addWidget(btn_rf_coco)

        btn_comp_yolo = QPushButton("⚡ Export Components")
        btn_comp_yolo.clicked.connect(self._export_component_yolo)
        tb2.addWidget(btn_comp_yolo)

        tb2.addSeparator()

        btn_detect = QPushButton("🤖 Auto-Detect")
        btn_detect.clicked.connect(self._auto_detect)
        btn_detect.setStyleSheet(
            "background: #1a472a; color: #4ade80; font-weight: 700;"
            "padding: 5px 14px; border-radius: 5px; font-size: 12px;"
        )
        tb2.addWidget(btn_detect)

        tb2.addSeparator()

        # Fingerprinting page (results viewer)
        btn_fp = QPushButton("🧬 Fingerprinting")
        btn_fp.clicked.connect(self._show_fingerprinting_page)
        tb2.addWidget(btn_fp)

        btn_test = QPushButton("Test")
        btn_test.clicked.connect(self._test_apply_fingerprinting_results)
        tb2.addWidget(btn_test)

    def _test_apply_fingerprinting_results(self):
        """Backup annotations and overwrite component bboxes from fingerprint results."""
        ok = self.fingerprinting.prompt_and_load_results_if_needed()
        if not ok:
            return

        summary = self.fingerprinting.backup_and_apply_results_to_annotations()
        if not summary:
            return

        # Reload current page to reflect updated boxes.
        try:
            cur_idx = int(self.view.page_idx)
        except Exception:
            cur_idx = 0
        self.view.load_page(cur_idx)
        self._update_sidebar()
        self._update_page_label()

        backup_dir = summary.get("backup_dir", "")
        changed = summary.get("changed_pairs", 0)
        changed_pages = summary.get("changed_pages", 0)
        skipped = summary.get("skipped", 0)
        QMessageBox.information(
            self,
            "Test: Applied Fingerprinting Results",
            f"Backed up annotations to:\n{backup_dir}\n\n"
            f"Updated pairs: {changed}\n"
            f"Pages touched: {changed_pages}\n"
            f"Skipped results: {skipped}",
        )

    def _build_sidebar(self):
        dock = QDockWidget("Associations", self)
        dock.setAllowedAreas(Qt.RightDockWidgetArea)
        dock.setFeatures(QDockWidget.NoDockWidgetFeatures)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        self.pair_list = QListWidget()
        self.pair_list.currentRowChanged.connect(self._on_pair_selected)
        self.pair_list.itemDoubleClicked.connect(self._on_pair_double_clicked)
        layout.addWidget(self.pair_list)

        # Delete button
        btn_del = QPushButton("🗑  Delete Selected")
        btn_del.setStyleSheet("margin: 6px;")
        btn_del.clicked.connect(self._delete_selected)
        layout.addWidget(btn_del)

        # Save button — prominent, always visible
        btn_save = QPushButton("💾  SAVE")
        btn_save.setStyleSheet(
            "margin: 6px; background: #2563eb; color: #fff; font-weight: 800;"
            "font-size: 14px; padding: 8px; border-radius: 6px; border: none;"
        )
        btn_save.clicked.connect(self._manual_save)
        layout.addWidget(btn_save)

        # Save confirmation label
        self.save_confirm_label = QLabel("")
        self.save_confirm_label.setStyleSheet(
            "color: #4ade80; font-size: 11px; font-weight: 600; margin: 0 6px;"
        )
        self.save_confirm_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.save_confirm_label)

        # Instructions
        instructions = QGroupBox("Controls")
        inst_layout = QVBoxLayout(instructions)
        for line in [
            "Left-drag → box component",
            "Left-click → assign label",
            "Ctrl+D → continuation mode",
            "Right-drag → pan",
            "Scroll → zoom",
            "← → → change page",
            "Ctrl+Z → undo",
            "Ctrl+Y → redo",
            "Del → delete selected",
            "Esc → cancel",
            "F → fit to view",
        ]:
            lbl = QLabel(line)
            lbl.setStyleSheet("color: #8b949e; font-size: 11px;")
            inst_layout.addWidget(lbl)
        layout.addWidget(instructions)

        dock.setWidget(container)
        dock.setMinimumWidth(220)
        dock.setMaximumWidth(280)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    # ---- Slots ----
    def _prev_page(self):
        self.view._save_annotations()  # auto-save before leaving page
        if self.view._mask_mode:
            self.view._mask_save_overlay()
            self.view._mask_save_metadata()
        self.view.load_page(self.view.page_idx - 1)
        self._save_last_page(self.view.page_idx)
        self._update_page_label()
        self._update_sidebar()
        self._update_mask_sidebar()

    def _next_page(self):
        self.view._save_annotations()  # auto-save before leaving page
        if self.view._mask_mode:
            self.view._mask_save_overlay()
            self.view._mask_save_metadata()
        self.view.load_page(self.view.page_idx + 1)
        self._save_last_page(self.view.page_idx)
        self._update_page_label()
        self._update_sidebar()
        self._update_mask_sidebar()

    def _save_last_page(self, idx):
        session_path = os.path.join(ANN_DIR, "_session.json")
        os.makedirs(ANN_DIR, exist_ok=True)
        with open(session_path, "w") as f:
            json.dump({"last_page": idx}, f)

    def _load_last_page(self):
        session_path = os.path.join(ANN_DIR, "_session.json")
        try:
            with open(session_path) as f:
                return json.load(f).get("last_page", 0)
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            # No session file — find the first page that has saved annotations
            total = len(self.meta.get("pages", []))
            for i in range(total):
                if os.path.exists(os.path.join(ANN_DIR, f"page_{i + 1}.json")):
                    return i
            return 0

    def closeEvent(self, event):
        """Auto-save current page annotations and persist last page on close."""
        self.view._save_annotations()
        self._save_last_page(self.view.page_idx)
        super().closeEvent(event)

    def _update_page_label(self):
        total = len(self.meta["pages"])
        self.page_label.setText(f" {self.view.page_idx + 1} / {total} ")



    def _enter_component_mode(self):
        """Activate component annotation mode."""
        self.view._clear_temp_items()
        self.view.pending_comp = None
        self.view.pending_continuation = None
        self.view.main_mode = "component"
        self.view.mode = "component"
        self.view.setCursor(Qt.CrossCursor)
        self._update_sidebar()

    def _enter_continuation_mode(self):
        """Activate continuation annotation mode."""
        self.view._clear_temp_items()
        self.view.pending_comp = None
        self.view.pending_continuation = None
        self.view.main_mode = "continuation"
        self.view.mode = "continuation"
        self.view.setCursor(Qt.CrossCursor)
        self._update_sidebar()

    def _enter_wire_mode(self):
        """Activate wire annotation mode."""
        self.view._clear_temp_items()
        self.view.pending_comp = None
        self.view.pending_continuation = None
        self.view.main_mode = "wire"
        self.view.mode = "wire"
        self.view.setCursor(Qt.PointingHandCursor)
        self._update_sidebar()

    def _update_sidebar(self):
        self.pair_list.clear()
        # List regular pairs
        for p in self.view.pairs:
            if p.get("type") == "continuation":
                page_text = p["page"]["text"]
                rung_text = p["rung"]["text"]
                bbox = p["symbol"]["bbox"]
                self.pair_list.addItem(f"🟡 →{page_text}:{rung_text}  ⊙  ({bbox[0]:.0f}, {bbox[1]:.0f})")
            else:
                txt = p["label"]["text"]
                bbox = p["component"]["bbox"]
                self.pair_list.addItem(f"🟢 {txt}  →  ({bbox[0]:.0f}, {bbox[1]:.0f})")
        
        # List wire labels
        for w in self.view.wire_labels:
             txt = w["text"]
             bbox = w["bbox"]
             self.pair_list.addItem(f"⚡ {txt}  (wire)")

        total = len(self.view.pairs) + len(self.view.wire_labels)
        self.pair_count_label.setText(f"{total} item{'s' if total != 1 else ''}")

        # Update mode indicator
        if self.view.mode == "label":
            self.mode_label.setText("  ● LABEL  ")
            self.mode_label.setStyleSheet(
                "background: #3b82f6; color: #fff; font-weight: 700; font-size: 11px;"
                "padding: 3px 10px; border-radius: 10px;"
            )
            self.status_label.setText("Click on the text label for this component")
        elif self.view.mode == "label_adjust":
            self.mode_label.setText("  ● ADJUST  ")
            self.mode_label.setStyleSheet(
                "background: #06b6d4; color: #000; font-weight: 700; font-size: 11px;"
                "padding: 3px 10px; border-radius: 10px;"
            )
            self.status_label.setText("Drag handles to adjust bbox • Enter/click to accept • Esc to cancel")
        elif self.view.mode == "continuation":
            self.mode_label.setText("  ● CONTINUATION  ")
            self.mode_label.setStyleSheet(
                "background: #f59e0b; color: #000; font-weight: 700; font-size: 11px;"
                "padding: 3px 10px; border-radius: 10px;"
            )
            self.status_label.setText("Draw a box around the continuation symbol")
        elif self.view.mode == "cont_page":
            self.mode_label.setText("  ● PAGE #  ")
            self.mode_label.setStyleSheet(
                "background: #f59e0b; color: #000; font-weight: 700; font-size: 11px;"
                "padding: 3px 10px; border-radius: 10px;"
            )
            self.status_label.setText("Click the page number in the symbol")
        elif self.view.mode == "cont_rung":
            self.mode_label.setText("  ● RUNG #  ")
            self.mode_label.setStyleSheet(
                "background: #f59e0b; color: #000; font-weight: 700; font-size: 11px;"
                "padding: 3px 10px; border-radius: 10px;"
            )
            self.status_label.setText("Click the rung number for this continuation")
        elif self.view._mask_mode:
            if self.view._mask_erase_mode:
                self.mode_label.setText("  ● MASK ERASE  ")
                self.mode_label.setStyleSheet(
                    "background: #ff4444; color: #fff; font-weight: 700; font-size: 11px;"
                    "padding: 3px 10px; border-radius: 10px;"
                )
                self.status_label.setText("Erasing mask pixels • E to draw • M to exit")
            else:
                self.mode_label.setText("  ● MASK  ")
                self.mode_label.setStyleSheet(
                    "background: #0088ff; color: #fff; font-weight: 700; font-size: 11px;"
                    "padding: 3px 10px; border-radius: 10px;"
                )
                self.status_label.setText("Draw over black lines • E to erase • M to exit")
        elif self.view.mode == "wire":
            self.mode_label.setText("  ● WIRE  ")
            self.mode_label.setStyleSheet(
                "background: #a855f7; color: #fff; font-weight: 700; font-size: 11px;"
                "padding: 3px 10px; border-radius: 10px;"
            )
            self.status_label.setText("Click wire labels to capture them")
        else:
            self.mode_label.setText("  ● COMPONENT  ")
            self.mode_label.setStyleSheet(
                "background: #22c55e; color: #000; font-weight: 700; font-size: 11px;"
                "padding: 3px 10px; border-radius: 10px;"
            )
            self.status_label.setText("Draw a box around a component (left-click drag)")

    def _on_pair_selected(self, row):
        """Highlight the selected pair/wire on the canvas."""
        self.view._clear_highlight()
        n_pairs = len(self.view.pairs)
        if 0 <= row < n_pairs:
            p = self.view.pairs[row]
            if p.get("type") == "continuation":
                bbox = p["symbol"]["bbox"]
            else:
                bbox = p["component"]["bbox"]
            self.view._draw_highlight(bbox)
        elif n_pairs <= row < n_pairs + len(self.view.wire_labels):
            w = self.view.wire_labels[row - n_pairs]
            self.view._draw_highlight(w["bbox"])

    def _on_pair_double_clicked(self, item):
        """Allow editing the text label of a pair or wire by double-clicking."""
        row = self.pair_list.row(item)
        n_pairs = len(self.view.pairs)
        if 0 <= row < n_pairs:
            p = self.view.pairs[row]
            if p.get("type") == "continuation":
                # Edit page and rung for continuation
                old_page = p["page"]["text"]
                old_rung = p["rung"]["text"]
                new_page, ok1 = QInputDialog.getText(
                    self, "Edit Continuation", "Page number:", text=old_page)
                if not ok1:
                    return
                new_rung, ok2 = QInputDialog.getText(
                    self, "Edit Continuation", "Rung number:", text=old_rung)
                if not ok2:
                    return
                p["page"]["text"] = new_page.strip()
                p["rung"]["text"] = new_rung.strip()
            else:
                old_text = p["label"]["text"]
                new_text, ok = QInputDialog.getText(
                    self, "Edit Label", "Label text:", text=old_text)
                if not ok or not new_text.strip():
                    return
                p["label"]["text"] = new_text.strip()
        elif n_pairs <= row < n_pairs + len(self.view.wire_labels):
            w = self.view.wire_labels[row - n_pairs]
            old_text = w["text"]
            new_text, ok = QInputDialog.getText(
                self, "Edit Wire Label", "Wire label text:", text=old_text)
            if not ok or not new_text.strip():
                return
            w["text"] = new_text.strip()
        else:
            return

        self.view._redraw_pairs()
        self.view._redraw_wires()
        self.view._save_annotations()
        self._update_sidebar()

    def _delete_selected(self):
        row = self.pair_list.currentRow()
        n_pairs = len(self.view.pairs)
        if 0 <= row < n_pairs:
            self.view.pairs.pop(row)
            self.view._redraw_pairs()
            self.view._save_annotations()
            self._update_sidebar()
        elif n_pairs <= row < n_pairs + len(self.view.wire_labels):
            # Delete wire label
            w_idx = row - n_pairs
            self.view.wire_labels.pop(w_idx)
            self.view._redraw_wires()
            self.view._save_annotations()
            self._update_sidebar()

    def _manual_save(self):
        """Explicit save triggered by button or Ctrl+S."""
        self.view._save_annotations()
        import datetime
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        n = len(self.view.pairs) + len(self.view.wire_labels)
        msg = f"✓ Saved {n} items at {ts}"
        self.save_confirm_label.setText(msg)
        self.statusBar().showMessage(msg, 5000)

    # ---- Settings ----
    _SETTINGS_PATH = os.path.join(ANN_DIR, "_settings.json")
    _SETTINGS_DEFAULTS = {"stroke_width": 1, "snap_padding": 1}

    def _load_settings(self):
        if os.path.exists(self._SETTINGS_PATH):
            try:
                with open(self._SETTINGS_PATH) as f:
                    data = json.load(f)
                # Merge with defaults so new keys always exist
                return {**self._SETTINGS_DEFAULTS, **data}
            except Exception:
                pass
        return dict(self._SETTINGS_DEFAULTS)

    def _save_settings(self):
        with open(self._SETTINGS_PATH, "w") as f:
            json.dump(self._settings, f, indent=2)

    def _apply_settings_to_view(self):
        self.view.stroke_width = self._settings["stroke_width"]
        self.view.snap_padding = self._settings["snap_padding"]

    def _open_settings(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Settings")
        dlg.setMinimumWidth(300)
        dlg.setStyleSheet("""
            QDialog { background: #161b22; color: #e6edf3; }
            QLabel { color: #e6edf3; font-size: 13px; }
            QSpinBox { background: #0d1117; color: #e6edf3; border: 1px solid #30363d;
                       padding: 4px 8px; border-radius: 4px; font-size: 13px; min-width: 70px; }
            QSpinBox::up-button, QSpinBox::down-button { width: 18px; background: #30363d; }
            QPushButton { background: #30363d; color: #e6edf3; border: none;
                          padding: 5px 18px; border-radius: 5px; font-size: 12px; }
            QPushButton:hover { background: #484f58; }
            QPushButton[text="OK"] { background: #22c55e; color: #000; font-weight: 700; }
        """)

        form = QFormLayout(dlg)
        form.setContentsMargins(20, 20, 20, 16)
        form.setSpacing(12)

        spin_stroke = QSpinBox()
        spin_stroke.setRange(1, 20)
        spin_stroke.setSuffix(" px")
        spin_stroke.setValue(self._settings["stroke_width"])
        form.addRow("Bounding box stroke:", spin_stroke)

        spin_pad = QSpinBox()
        spin_pad.setRange(0, 50)
        spin_pad.setSuffix(" px (PDF pts)")
        spin_pad.setValue(self._settings["snap_padding"])
        form.addRow("Snap padding:", spin_pad)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)

        if dlg.exec() == QDialog.Accepted:
            self._settings["stroke_width"] = spin_stroke.value()
            self._settings["snap_padding"] = spin_pad.value()
            self._save_settings()
            self._apply_settings_to_view()
            # Redraw current page so stroke change is visible immediately
            self.view._redraw_pairs()
            self.view._redraw_wires()

    # ---- Page Range Selection Dialog ----
    def _select_pages_dialog(self, title="Select Pages", annotated_only=True):
        """
        Show a dialog letting the user choose: all annotated pages,
        current page, or a custom range.  Returns a sorted list of
        1-based page numbers, or None if cancelled.

        If annotated_only=True (default), only pages with saved
        annotation files are eligible.
        """
        total = len(self.meta["pages"])
        current_pg = self.view.page_idx + 1  # 1-based

        # Build set of annotated page numbers
        annotated = set()
        for f in os.listdir(ANN_DIR):
            if f.startswith("page_") and f.endswith(".json") and not f.startswith("_"):
                try:
                    annotated.add(int(f.replace("page_", "").replace(".json", "")))
                except ValueError:
                    pass

        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.setMinimumWidth(400)
        layout = QVBoxLayout(dlg)

        grp = QButtonGroup(dlg)

        rb_all = QRadioButton(
            f"All annotated pages ({len(annotated)} pages: "
            + ", ".join(str(p) for p in sorted(annotated)[:12])
            + (", …" if len(annotated) > 12 else "") + ")"
        )
        rb_all.setChecked(True)
        grp.addButton(rb_all, 0)
        layout.addWidget(rb_all)

        has_ann = current_pg in annotated
        rb_cur = QRadioButton(
            f"Current page only (page {current_pg})"
            + ("" if has_ann or not annotated_only else " — no annotations")
        )
        grp.addButton(rb_cur, 1)
        layout.addWidget(rb_cur)

        rb_range = QRadioButton("Custom range:")
        grp.addButton(rb_range, 2)
        range_edit = QLineEdit()
        range_edit.setPlaceholderText(f"e.g.  7-11, 14  (pages 1–{total})")
        range_edit.setEnabled(False)
        rb_range.toggled.connect(range_edit.setEnabled)
        range_row = QHBoxLayout()
        range_row.addWidget(rb_range)
        range_row.addWidget(range_edit, 1)
        layout.addLayout(range_row)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)

        if dlg.exec() != QDialog.Accepted:
            return None

        choice = grp.checkedId()

        if choice == 0:  # all annotated
            pages = sorted(annotated)
        elif choice == 1:  # current page
            pages = [current_pg]
        else:  # custom range
            pages = self._parse_page_range(range_edit.text(), total)
            if pages is None:
                QMessageBox.warning(
                    self, "Invalid Range",
                    "Could not parse the page range.\n\n"
                    "Use formats like: 7-11  or  7,8,9  or  7-11, 14"
                )
                return None

        if annotated_only:
            pages = [p for p in pages if p in annotated]
            if not pages:
                QMessageBox.warning(
                    self, "No Annotations",
                    "None of the selected pages have saved annotations."
                )
                return None

        return pages

    @staticmethod
    def _parse_page_range(text, total_pages):
        """Parse '7-11, 14' → [7, 8, 9, 10, 11, 14].  Returns None on error."""
        result = set()
        text = text.strip()
        if not text:
            return None
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                try:
                    a, b = part.split("-", 1)
                    a, b = int(a.strip()), int(b.strip())
                except ValueError:
                    return None
                if a < 1 or b > total_pages or a > b:
                    return None
                result.update(range(a, b + 1))
            else:
                try:
                    n = int(part)
                except ValueError:
                    return None
                if n < 1 or n > total_pages:
                    return None
                result.add(n)
        return sorted(result) if result else None

    def _export_json(self):
        selected = self._select_pages_dialog("Export JSON — Select Pages")
        if selected is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Annotations", "atlas_annotations.json", "JSON (*.json)"
        )
        if not path:
            return
        all_ann = {}
        for pg_num in selected:
            fpath = os.path.join(ANN_DIR, f"page_{pg_num}.json")
            if os.path.exists(fpath):
                with open(fpath) as f:
                    all_ann[f"page_{pg_num}"] = json.load(f)
        with open(path, "w") as f:
            json.dump(all_ann, f, indent=2)
        self.statusBar().showMessage(f"Exported {len(all_ann)} page(s) to {path}", 5000)

    def _export_images(self):
        """Export 300 DPI annotated images for selected pages that have saved annotations."""
        selected = self._select_pages_dialog("Export Images — Select Pages")
        if selected is None:
            return
        from PySide6.QtGui import QImage, QImageReader
        QImageReader.setAllocationLimit(0)  # remove 256 MB cap for large master images
        dest_dir = QFileDialog.getExistingDirectory(
            self, "Select Export Destination Folder"
        )
        if not dest_dir:
            return
        selected_set = set(selected)

        EXPORT_DPI = 300
        MASTER_DPI = self.meta.get("master_dpi", 1200)
        scale_factor = EXPORT_DPI / MASTER_DPI  # master → 600 DPI
        # PDF points → 600 DPI pixels
        ann_scale = EXPORT_DPI / 72.0

        def draw_rect(painter, bbox, color_hex, alpha=60, pen_width=4):
            x0, y0, x1, y1 = [v * ann_scale for v in bbox]
            pen = QPen(QColor(color_hex))
            pen.setWidth(pen_width)
            painter.setPen(pen)
            fill = QColor(color_hex)
            fill.setAlpha(alpha)
            painter.setBrush(QBrush(fill))
            painter.drawRect(QRectF(x0, y0, x1 - x0, y1 - y0))

        errors = []
        exported = 0
        pages = self.meta["pages"]
        for i, page_meta in enumerate(pages):
            if (i + 1) not in selected_set:
                continue
            ann_path = os.path.join(ANN_DIR, f"page_{i+1}.json")
            if not os.path.exists(ann_path):
                continue

            # Determine source image and scale factor
            master_img_path = os.path.join(
                DATA_DIR, "master", page_meta.get("master_image", f"page_{i+1}.png")
            )
            if os.path.exists(master_img_path):
                src_path = master_img_path
                sf = scale_factor
            else:
                display_img_path = os.path.join(
                    DATA_DIR, "pages", page_meta.get("image", f"page_{i+1}.png")
                )
                if not os.path.exists(display_img_path):
                    errors.append(f"page_{i+1}: source image not found")
                    continue
                src_path = display_img_path
                sf = EXPORT_DPI / self.meta.get("display_dpi", 300)

            # Load as QImage (safe for off-screen painting)
            img = QImage(src_path)
            if img.isNull():
                errors.append(f"page_{i+1}: failed to load image")
                continue

            # Convert to ARGB32 and scale to target DPI
            img = img.convertToFormat(QImage.Format_ARGB32)
            w = round(img.width() * sf)
            h = round(img.height() * sf)
            canvas = img.scaled(w, h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

            # Draw annotations
            with open(ann_path) as f:
                ann = json.load(f)

            painter = QPainter(canvas)
            painter.setRenderHint(QPainter.Antialiasing)

            pen_width = self._settings["stroke_width"]
            font = QFont("Arial", max(8, round(ann_scale * 3.5)))
            painter.setFont(font)

            for pair in ann.get("pairs", []):
                if pair.get("type") == "continuation":
                    if "symbol" in pair:
                        draw_rect(painter, pair["symbol"]["bbox"], "#f59e0b", alpha=50, pen_width=pen_width)
                    for key in ("page", "rung"):
                        if key in pair:
                            draw_rect(painter, pair[key]["bbox"], "#f59e0b", alpha=30, pen_width=max(2, pen_width - 1))
                else:
                    if "component" in pair:
                        draw_rect(painter, pair["component"]["bbox"], "#22c55e", alpha=50, pen_width=pen_width)
                    if "label" in pair:
                        lbl = pair["label"]
                        draw_rect(painter, lbl["bbox"], "#3b82f6", alpha=50, pen_width=pen_width)
                        x0, y0 = lbl["bbox"][0] * ann_scale, lbl["bbox"][1] * ann_scale
                        painter.setPen(QPen(QColor("#3b82f6")))
                        painter.drawText(QPointF(x0, y0 - 2), lbl.get("text", ""))

            for wire in ann.get("wire_labels", []):
                draw_rect(painter, wire["bbox"], "#f59e0b", alpha=40, pen_width=max(2, pen_width - 1))

            painter.end()

            out_path = os.path.join(dest_dir, f"page_{i+1}_annotated.png")
            ok = canvas.save(out_path, "PNG")
            if ok:
                exported += 1
            else:
                errors.append(f"page_{i+1}: save failed → {out_path}")
            self.statusBar().showMessage(f"Exporting images… {exported} done", 500)
            QApplication.processEvents()

        msg = f"Exported {exported} annotated image(s) at {EXPORT_DPI} DPI\nDestination: {dest_dir}"
        if errors:
            msg += "\n\nErrors:\n" + "\n".join(errors)
        self.statusBar().showMessage(
            f"Exported {exported} annotated image(s) to {dest_dir}", 7000
        )
        QMessageBox.information(self, "Export Complete", msg)

    # ---- Roboflow upload ----
    def _export_to_roboflow(self):
        """Open credentials dialog then stream annotated pages to a Roboflow project."""
        try:
            import roboflow  # noqa: F401
        except ImportError:
            QMessageBox.critical(
                self, "roboflow not found",
                "The 'roboflow' package is not installed in the current Python environment.\n\n"
                "Launch the annotator with the project venv:\n"
                "  .venv\\Scripts\\python.exe tools\\annotator\\annotator.py"
            )
            return
        selected = self._select_pages_dialog("Upload to Roboflow — Select Pages")
        if selected is None:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Upload to Roboflow")
        dlg.setMinimumWidth(420)
        form = QFormLayout(dlg)

        saved_rf = self._settings.get("roboflow", {})

        api_key_edit = QLineEdit(saved_rf.get("api_key", ""))
        api_key_edit.setEchoMode(QLineEdit.Password)
        api_key_edit.setPlaceholderText("Private API key from app.roboflow.com/settings/api")
        form.addRow("API Key:", api_key_edit)

        workspace_edit = QLineEdit(saved_rf.get("workspace", ""))
        workspace_edit.setPlaceholderText("Workspace slug (shown in Roboflow URL)")
        form.addRow("Workspace:", workspace_edit)

        project_edit = QLineEdit(saved_rf.get("project", "atlas-schematics"))
        project_edit.setPlaceholderText("Project name — created automatically if new")
        form.addRow("Project:", project_edit)

        info = QLabel(
            "Classes exported: component · component_label\n"
            "continuation_symbol · continuation_page · continuation_rung · wire_label\n"
            "Format: Pascal VOC XML  |  Dataset type: object-detection"
        )
        info.setStyleSheet("color: #888; font-size: 11px;")
        form.addRow(info)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        form.addRow(buttons)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)

        if dlg.exec() != QDialog.Accepted:
            return

        api_key   = api_key_edit.text().strip()
        workspace = workspace_edit.text().strip()
        project   = project_edit.text().strip()

        if not api_key or not workspace or not project:
            QMessageBox.warning(self, "Missing Fields", "Please fill in all three fields.")
            return

        # Persist non-secret fields
        self._settings["roboflow"] = {
            "api_key": api_key,
            "workspace": workspace,
            "project": project,
        }
        self._save_settings()

        ann_files = sorted(f"page_{p}.json" for p in selected)

        progress = QProgressDialog(
            "Connecting to Roboflow…", "Cancel", 0, len(ann_files), self
        )
        progress.setWindowTitle("Upload to Roboflow")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumWidth(400)
        progress.show()
        QApplication.processEvents()

        self._rf_worker = RoboflowUploadWorker(
            api_key=api_key,
            workspace=workspace,
            project_name=project,
            meta=self.meta,
            ann_dir=ANN_DIR,
            pages_dir=os.path.join(DATA_DIR, "pages"),
            selected_pages=selected,
        )

        def on_progress(cur, total, msg):
            if progress.wasCanceled():
                self._rf_worker.cancel()
            else:
                progress.setValue(cur)
                progress.setLabelText(msg)
                QApplication.processEvents()

        def on_finished(errors):
            progress.setValue(len(ann_files))
            progress.close()
            n_ok = len(ann_files) - len(errors)
            summary = (
                f"Uploaded {n_ok} / {len(ann_files)} page(s) to "
                f"Roboflow project '{project}'."
            )
            if errors:
                display_errors = errors[:10]
                if len(errors) > 10:
                    display_errors.append(f"…and {len(errors) - 10} more.")
                summary += "\n\nErrors:\n" + "\n".join(display_errors)
            self.statusBar().showMessage(summary.split("\n")[0], 8000)
            QMessageBox.information(self, "Roboflow Upload", summary)

        self._rf_worker.progress_signal.connect(on_progress)
        self._rf_worker.finished_signal.connect(on_finished)
        self._rf_worker.start()

    # ---- Export Roboflow dataset (local, YOLOv8 format) ----
    def _export_rf_dataset(self):
        """Export annotated pages as a YOLOv8-format dataset folder ready for Roboflow import."""
        selected = self._select_pages_dialog("Export YOLO Dataset — Select Pages")
        if selected is None:
            return
        # --- config dialog ---
        dlg = QDialog(self)
        dlg.setWindowTitle("Export Roboflow Dataset")
        dlg.setMinimumWidth(400)
        form = QFormLayout(dlg)

        name_edit = QLineEdit(self._settings.get("rf_dataset_name", "atlas-dataset"))
        name_edit.setPlaceholderText("folder name for the dataset")
        form.addRow("Dataset name:", name_edit)

        split_spin = QSpinBox()
        split_spin.setRange(50, 100)
        split_spin.setValue(self._settings.get("rf_train_pct", 80))
        split_spin.setSuffix(" % train")
        form.addRow("Train split:", split_spin)

        info = QLabel(
            "Format: YOLOv8 PyTorch  |  Images: 300 DPI display PNGs\n"
            "Classes: component · component_label · continuation_symbol\n"
            "         continuation_page · continuation_rung · wire_label"
        )
        info.setStyleSheet("color: #888; font-size: 11px;")
        form.addRow(info)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        form.addRow(btns)

        if dlg.exec() != QDialog.Accepted:
            return

        dataset_name = name_edit.text().strip() or "atlas-dataset"
        train_pct    = split_spin.value() / 100.0
        self._settings["rf_dataset_name"] = dataset_name
        self._settings["rf_train_pct"]    = split_spin.value()
        self._save_settings()

        dest_root = QFileDialog.getExistingDirectory(self, "Select Export Destination Folder")
        if not dest_root:
            return

        # --- use selected pages ---
        pages_map = {pg["page"]: pg for pg in self.meta["pages"]}
        ann_files = sorted(f"page_{p}.json" for p in selected)

        # --- train / valid split (deterministic by index) ---
        n_train  = max(1, round(len(ann_files) * train_pct))
        train_set = set(ann_files[:n_train])

        # --- class list (order matters — becomes class ids 0..5) ---
        CLASSES = [
            "component",
            "component_label",
            "continuation_symbol",
            "continuation_page",
            "continuation_rung",
            "wire_label",
        ]
        cls_idx = {c: i for i, c in enumerate(CLASSES)}

        # --- build folder structure ---
        dataset_dir = os.path.join(dest_root, dataset_name)
        for split in ("train", "valid"):
            os.makedirs(os.path.join(dataset_dir, split, "images"), exist_ok=True)
            os.makedirs(os.path.join(dataset_dir, split, "labels"), exist_ok=True)

        # --- data.yaml ---
        yaml_path = os.path.join(dataset_dir, "data.yaml")
        with open(yaml_path, "w") as f:
            f.write(f"train: train/images\n")
            f.write(f"val: valid/images\n")
            f.write(f"nc: {len(CLASSES)}\n")
            names_str = "[" + ", ".join(f"'{c}'" for c in CLASSES) + "]\n"
            f.write(f"names: {names_str}")

        # --- README ---
        with open(os.path.join(dataset_dir, "README.roboflow.txt"), "w") as f:
            f.write(
                f"Roboflow YOLOv8 dataset exported by ATLAS Annotator\n"
                f"Dataset: {dataset_name}\n"
                f"Pages: {len(ann_files)}  |  Train: {n_train}  |  Valid: {len(ann_files)-n_train}\n"
                f"Image DPI: 300  |  Format: YOLOv8 PyTorch\n"
            )

        errors  = []
        exported = 0

        for fname in ann_files:
            try:
                pg_num   = int(fname.replace("page_", "").replace(".json", ""))
                pg_meta  = pages_map.get(pg_num)
                if not pg_meta:
                    errors.append(f"{fname}: page metadata missing"); continue

                img_w, img_h = pg_meta["display_size"]
                pdf_w        = pg_meta["pdf_width"]
                pdf_h        = pg_meta["pdf_height"]
                sx           = img_w / pdf_w   # PDF pts → display pixels
                sy           = img_h / pdf_h

                split     = "train" if fname in train_set else "valid"
                img_name  = pg_meta["image"]                     # e.g. page_8.png
                src_img   = os.path.join(DATA_DIR, "pages", img_name)
                dst_img   = os.path.join(dataset_dir, split, "images", img_name)
                lbl_name  = img_name.replace(".png", ".txt").replace(".jpg", ".txt")
                dst_lbl   = os.path.join(dataset_dir, split, "labels", lbl_name)

                if not os.path.exists(src_img):
                    errors.append(f"{fname}: image not found ({src_img})"); continue

                # copy image
                import shutil
                shutil.copy2(src_img, dst_img)

                # build YOLO label file
                with open(ann_path := os.path.join(ANN_DIR, fname)) as f:
                    ann = json.load(f)

                lines = []

                def yolo_line(class_name, bbox):
                    x0, y0, x1, y1 = bbox
                    # convert PDF pts → pixels
                    px0 = max(0.0, x0 * sx);  px1 = min(img_w, x1 * sx)
                    py0 = max(0.0, y0 * sy);  py1 = min(img_h, y1 * sy)
                    if px1 <= px0 or py1 <= py0:
                        return
                    cx = ((px0 + px1) / 2) / img_w
                    cy = ((py0 + py1) / 2) / img_h
                    bw = (px1 - px0) / img_w
                    bh = (py1 - py0) / img_h
                    lines.append(f"{cls_idx[class_name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                for pair in ann.get("pairs", []):
                    if pair.get("type") == "continuation":
                        if "symbol" in pair: yolo_line("continuation_symbol", pair["symbol"]["bbox"])
                        if "page"   in pair: yolo_line("continuation_page",   pair["page"]["bbox"])
                        if "rung"   in pair: yolo_line("continuation_rung",   pair["rung"]["bbox"])
                    else:
                        if "component" in pair: yolo_line("component",       pair["component"]["bbox"])
                        if "label"     in pair: yolo_line("component_label", pair["label"]["bbox"])
                for wire in ann.get("wire_labels", []):
                    yolo_line("wire_label", wire["bbox"])

                with open(dst_lbl, "w") as f:
                    f.write("\n".join(lines))

                exported += 1
                self.statusBar().showMessage(f"Exporting dataset… {exported}/{len(ann_files)}", 400)
                QApplication.processEvents()

            except Exception as e:
                errors.append(f"{fname}: {e}")

        msg = (
            f"Exported {exported} page(s) to:\n{dataset_dir}\n\n"
            f"Train: {n_train}   Valid: {len(ann_files)-n_train}\n"
            f"Format: YOLOv8 PyTorch  |  data.yaml included"
        )
        if errors:
            msg += "\n\nErrors:\n" + "\n".join(errors[:10])
            if len(errors) > 10:
                msg += f"\n…and {len(errors)-10} more."
        self.statusBar().showMessage(f"Dataset exported → {dataset_dir}", 8000)
        QMessageBox.information(self, "Dataset Export Complete", msg)

    # ---- Auto-detect components using YOLO model ----
    def _auto_detect(self):
        """Run YOLO inference on the current page and populate annotations."""
        # 1. Check ultralytics is available
        try:
            from ultralytics import YOLO as _YOLO
        except ImportError:
            QMessageBox.critical(
                self, "Missing Dependency",
                "ultralytics is not installed.\n\nRun: pip install ultralytics"
            )
            return

        # 2. Resolve model path
        model_path = getattr(self, "_detect_model_path", MODEL_PATH)
        if not os.path.exists(model_path):
            path, _ = QFileDialog.getOpenFileName(
                self, "Select YOLO model (.pt)",
                os.path.join(DATA_DIR, "master"),
                "PyTorch model (*.pt)"
            )
            if not path:
                return
            model_path = path
        self._detect_model_path = model_path

        # 3. Load / cache model
        from PySide6.QtWidgets import QApplication
        if getattr(self, "_yolo_model_path", None) != model_path:
            self.status_label.setText(f"Loading model: {os.path.basename(model_path)} …")
            QApplication.processEvents()
            self._yolo_model = _YOLO(model_path)
            self._yolo_model_path = model_path

        # 4. Current page info
        pg   = self.view.metadata["pages"][self.view.page_idx]
        img_path = os.path.join(DATA_DIR, "pages", pg["image"])
        pdf_w = pg["pdf_width"]
        pdf_h = pg["pdf_height"]
        img_w, img_h = pg["display_size"]
        sx = img_w / pdf_w   # pixels per PDF point
        sy = img_h / pdf_h

        # 5. Run inference
        self.status_label.setText("Running detection …")
        QApplication.processEvents()
        results = self._yolo_model.predict(img_path, conf=0.30, verbose=False)

        # 6. Convert detections → pairs
        new_pairs = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            boxes   = r.boxes.xyxy.cpu().numpy()   # pixel coords in original image space
            cls_ids = r.boxes.cls.int().cpu().numpy()
            for box, cls_id in zip(boxes, cls_ids):
                px0, py0, px1, py1 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                x0 = round(px0 / sx, 2)
                y0 = round(py0 / sy, 2)
                x1 = round(px1 / sx, 2)
                y1 = round(py1 / sy, 2)
                cls_name = r.names[int(cls_id)]
                lbl_cx = (x0 + x1) / 2
                lbl_w  = len(cls_name) * 3.5
                new_pairs.append({
                    "component": {"bbox": [x0, y0, x1, y1]},
                    "label": {
                        "text": cls_name,
                        "bbox": [
                            round(lbl_cx - lbl_w / 2, 2), round(y0 - 5, 2),
                            round(lbl_cx + lbl_w / 2, 2), round(y0 - 1, 2),
                        ],
                    },
                })

        if not new_pairs:
            self.status_label.setText("Auto-detect: no components found.")
            QMessageBox.information(self, "Auto-Detect", "No components detected on this page.")
            return

        # 7. Ask the user what to do with existing annotations
        existing = len(self.view.pairs)
        msg = QMessageBox(self)
        msg.setWindowTitle("Auto-Detect Results")
        msg.setText(
            f"Detected <b>{len(new_pairs)}</b> component(s).\n"
            f"Page currently has <b>{existing}</b> annotation(s)."
        )
        btn_replace = msg.addButton("Replace All", QMessageBox.AcceptRole)
        btn_append  = msg.addButton("Append",      QMessageBox.YesRole)
        msg.addButton("Cancel",                    QMessageBox.RejectRole)
        msg.exec()
        clicked = msg.clickedButton()
        if clicked == btn_replace:
            self.view.pairs = new_pairs
        elif clicked == btn_append:
            self.view.pairs.extend(new_pairs)
        else:
            self.status_label.setText("Auto-detect cancelled.")
            return

        # 8. Redraw and save
        self.view._redraw_pairs()
        self.view._save_annotations()
        self.view.pairCreated.emit()
        total = len(self.view.pairs)
        self.status_label.setText(
            f"Auto-detect complete: {len(new_pairs)} detected → {total} total pairs on this page."
        )

    # ---- Export Component-only YOLO dataset (classes = component marks) ----
    def _export_component_yolo(self):
        """Export component bboxes as a YOLO dataset where each class is the component mark."""
        selected = self._select_pages_dialog("Export Component YOLO — Select Pages")
        if selected is None:
            return

        # --- config dialog ---
        dlg = QDialog(self)
        dlg.setWindowTitle("Export Component YOLO Dataset")
        dlg.setMinimumWidth(400)
        form = QFormLayout(dlg)

        name_edit = QLineEdit(self._settings.get("comp_dataset_name", "atlas-components"))
        name_edit.setPlaceholderText("folder name for the dataset")
        form.addRow("Dataset name:", name_edit)

        split_spin = QSpinBox()
        split_spin.setRange(50, 100)
        split_spin.setValue(self._settings.get("comp_train_pct", 80))
        split_spin.setSuffix(" % train")
        form.addRow("Train split:", split_spin)

        info = QLabel(
            "Format: YOLO  |  Images: 300 DPI display PNGs\n"
            "Classes: one per unique component mark (e.g. CR, MC, ELB …)\n"
            "Full-width characters are normalized to ASCII."
        )
        info.setStyleSheet("color: #888; font-size: 11px;")
        form.addRow(info)

        drive_cb = QCheckBox("Upload to Google Drive (Atlas/train)")
        drive_cb.setChecked(self._settings.get("comp_upload_drive", False))
        form.addRow(drive_cb)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        form.addRow(btns)

        if dlg.exec() != QDialog.Accepted:
            return

        dataset_name = name_edit.text().strip() or "atlas-components"
        train_pct    = split_spin.value() / 100.0
        upload_drive = drive_cb.isChecked()
        self._settings["comp_dataset_name"] = dataset_name
        self._settings["comp_train_pct"]    = split_spin.value()
        self._settings["comp_upload_drive"] = upload_drive
        self._save_settings()

        dest_root = QFileDialog.getExistingDirectory(self, "Select Export Destination Folder")
        if not dest_root:
            return

        pages_map = {pg["page"]: pg for pg in self.meta["pages"]}
        ann_files = sorted(f"page_{p}.json" for p in selected)

        # --- normalize full-width to ASCII ---
        def fw_to_ascii(s):
            out = []
            for ch in s:
                cp = ord(ch)
                if 0xFF01 <= cp <= 0xFF5E:       # fullwidth ! to ~
                    out.append(chr(cp - 0xFEE0))
                elif cp == 0x3000:               # fullwidth space
                    out.append(' ')
                else:
                    out.append(ch)
            return ''.join(out)

        # --- pass 1: collect all unique marks to build class list ---
        all_marks = set()
        for fname in ann_files:
            ann_path = os.path.join(ANN_DIR, fname)
            if not os.path.exists(ann_path):
                continue
            with open(ann_path, encoding='utf-8') as f:
                ann = json.load(f)
            for pair in ann.get("pairs", []):
                if pair.get("type") == "continuation":
                    continue
                label = pair.get("label", {}).get("text", "")
                if label:
                    all_marks.add(fw_to_ascii(label))

        CLASSES = sorted(all_marks)
        cls_idx = {c: i for i, c in enumerate(CLASSES)}

        if not CLASSES:
            QMessageBox.warning(self, "No Components",
                                "No component marks found in the selected pages.")
            return

        # --- train / valid split ---
        n_train  = max(1, round(len(ann_files) * train_pct))
        train_set = set(ann_files[:n_train])

        # --- build folder structure ---
        dataset_dir = os.path.join(dest_root, dataset_name)
        for split in ("train", "valid"):
            os.makedirs(os.path.join(dataset_dir, split, "images"), exist_ok=True)
            os.makedirs(os.path.join(dataset_dir, split, "labels"), exist_ok=True)

        # --- data.yaml ---
        yaml_path = os.path.join(dataset_dir, "data.yaml")
        with open(yaml_path, "w") as f:
            f.write(f"train: train/images\n")
            f.write(f"val: valid/images\n")
            f.write(f"nc: {len(CLASSES)}\n")
            names_str = "[" + ", ".join(f"'{c}'" for c in CLASSES) + "]\n"
            f.write(f"names: {names_str}")

        errors  = []
        exported = 0

        for fname in ann_files:
            try:
                pg_num   = int(fname.replace("page_", "").replace(".json", ""))
                pg_meta  = pages_map.get(pg_num)
                if not pg_meta:
                    errors.append(f"{fname}: page metadata missing"); continue

                img_w, img_h = pg_meta["display_size"]
                pdf_w        = pg_meta["pdf_width"]
                pdf_h        = pg_meta["pdf_height"]
                sx           = img_w / pdf_w
                sy           = img_h / pdf_h

                split     = "train" if fname in train_set else "valid"
                img_name  = pg_meta["image"]
                src_img   = os.path.join(DATA_DIR, "pages", img_name)
                dst_img   = os.path.join(dataset_dir, split, "images", img_name)
                lbl_name  = img_name.replace(".png", ".txt").replace(".jpg", ".txt")
                dst_lbl   = os.path.join(dataset_dir, split, "labels", lbl_name)

                if not os.path.exists(src_img):
                    errors.append(f"{fname}: image not found ({src_img})"); continue

                import shutil
                shutil.copy2(src_img, dst_img)

                with open(os.path.join(ANN_DIR, fname), encoding='utf-8') as f:
                    ann = json.load(f)

                lines = []
                for pair in ann.get("pairs", []):
                    if pair.get("type") == "continuation":
                        continue
                    comp = pair.get("component")
                    label_text = pair.get("label", {}).get("text", "")
                    if not comp or not label_text:
                        continue
                    mark = fw_to_ascii(label_text)
                    if mark not in cls_idx:
                        continue
                    bbox = comp["bbox"]
                    x0, y0, x1, y1 = bbox
                    px0 = max(0.0, x0 * sx);  px1 = min(img_w, x1 * sx)
                    py0 = max(0.0, y0 * sy);  py1 = min(img_h, y1 * sy)
                    if px1 <= px0 or py1 <= py0:
                        continue
                    cx = ((px0 + px1) / 2) / img_w
                    cy = ((py0 + py1) / 2) / img_h
                    bw = (px1 - px0) / img_w
                    bh = (py1 - py0) / img_h
                    lines.append(f"{cls_idx[mark]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                with open(dst_lbl, "w") as f:
                    f.write("\n".join(lines))

                exported += 1
                self.statusBar().showMessage(f"Exporting components… {exported}/{len(ann_files)}", 400)
                QApplication.processEvents()

            except Exception as e:
                errors.append(f"{fname}: {e}")

        msg = (
            f"Exported {exported} page(s) to:\n{dataset_dir}\n\n"
            f"Train: {n_train}   Valid: {len(ann_files)-n_train}\n"
            f"Classes ({len(CLASSES)}): {', '.join(CLASSES)}\n"
            f"Format: YOLO  |  data.yaml included"
        )
        if errors:
            msg += "\n\nErrors:\n" + "\n".join(errors[:10])
            if len(errors) > 10:
                msg += f"\n…and {len(errors)-10} more."
        self.statusBar().showMessage(f"Component dataset exported → {dataset_dir}", 8000)

        # --- optional Drive upload ---
        if upload_drive and exported > 0:
            try:
                result = self._upload_dataset_to_drive(dataset_dir)
                if result == "cancelled":
                    msg += "\n\n⚠ Drive upload cancelled"
                else:
                    msg += "\n\n✓ Uploaded to Google Drive (Atlas/train)"
            except Exception as e:
                msg += f"\n\n✗ Drive upload failed: {e}"

        QMessageBox.information(self, "Component Export Complete", msg)

    # ---- Google Drive upload helper ----
    def _upload_dataset_to_drive(self, local_dataset_dir):
        """Upload a YOLO dataset folder to Google Drive at Atlas/train/.

        Returns 'cancelled' if the user aborted, or 'ok' on success.
        """
        import google.auth
        import google.auth.transport.requests

        creds, project = google.auth.default(
            scopes=['https://www.googleapis.com/auth/drive']
        )
        creds.refresh(google.auth.transport.requests.Request())
        token = creds.token
        hdrs = {
            'Authorization': f'Bearer {token}',
            'x-goog-user-project': project or 'gen-lang-client-0746582623',
        }
        ctx = ssl.create_default_context()

        def _api(endpoint, params=None):
            url = f'https://www.googleapis.com/drive/v3/{endpoint}'
            if params:
                url += '?' + urllib.parse.urlencode(params)
            req = urllib.request.Request(url, headers=hdrs)
            with urllib.request.urlopen(req, context=ctx) as resp:
                return json.loads(resp.read())

        def _find(name, parent_id=None):
            q = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
            if parent_id:
                q += f" and '{parent_id}' in parents"
            return _api('files', {'q': q, 'fields': 'files(id,name)', 'spaces': 'drive'}).get('files', [])

        def _mkdir(name, parent_id=None):
            meta = {'name': name, 'mimeType': 'application/vnd.google-apps.folder'}
            if parent_id:
                meta['parents'] = [parent_id]
            body = json.dumps(meta).encode()
            req = urllib.request.Request(
                'https://www.googleapis.com/drive/v3/files',
                data=body, headers={**hdrs, 'Content-Type': 'application/json'}, method='POST')
            with urllib.request.urlopen(req, context=ctx) as resp:
                return json.loads(resp.read())

        def _delete(fid):
            req = urllib.request.Request(
                f'https://www.googleapis.com/drive/v3/files/{fid}',
                headers=hdrs, method='DELETE')
            urllib.request.urlopen(req, context=ctx)

        def _children(folder_id):
            items, pt = [], None
            while True:
                params = {'q': f"'{folder_id}' in parents and trashed=false",
                          'fields': 'nextPageToken,files(id,name,mimeType)',
                          'pageSize': 100, 'spaces': 'drive'}
                if pt:
                    params['pageToken'] = pt
                r = _api('files', params)
                items.extend(r.get('files', []))
                pt = r.get('nextPageToken')
                if not pt:
                    break
            return items

        def _clear(folder_id):
            for ch in _children(folder_id):
                if ch['mimeType'] == 'application/vnd.google-apps.folder':
                    _clear(ch['id'])
                _delete(ch['id'])

        def _ensure(name, parent_id):
            existing = _find(name, parent_id)
            return existing[0]['id'] if existing else _mkdir(name, parent_id)['id']

        def _upload(local_path, parent_id):
            fname = os.path.basename(local_path)
            mime = mimetypes.guess_type(local_path)[0] or 'application/octet-stream'
            meta = json.dumps({'name': fname, 'parents': [parent_id]}).encode()
            with open(local_path, 'rb') as f:
                data = f.read()
            boundary = b'----UpBnd12345'
            body = (b'--' + boundary + b'\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n'
                    + meta + b'\r\n--' + boundary + b'\r\nContent-Type: '
                    + mime.encode() + b'\r\n\r\n' + data + b'\r\n--' + boundary + b'--')
            req = urllib.request.Request(
                'https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart',
                data=body,
                headers={**hdrs,
                         'Content-Type': f'multipart/related; boundary={boundary.decode()}',
                         'Content-Length': str(len(body))},
                method='POST')
            with urllib.request.urlopen(req, context=ctx) as resp:
                return json.loads(resp.read())

        # Find or create Atlas/train
        atlas = _find('Atlas')
        atlas_id = atlas[0]['id'] if atlas else _mkdir('Atlas')['id']
        train = _find('train', atlas_id)
        if train:
            train_id = train[0]['id']
            _clear(train_id)
        else:
            train_id = _mkdir('train', atlas_id)['id']

        # Walk and upload with progress dialog
        n_files = sum(len(files) for _, _, files in os.walk(local_dataset_dir))
        progress = QProgressDialog(
            "Preparing upload…", "Cancel", 0, n_files, self)
        progress.setWindowTitle("Uploading to Google Drive")
        progress.setMinimumDuration(0)
        progress.setWindowModality(Qt.WindowModal)
        progress.setValue(0)
        QApplication.processEvents()

        uploaded = 0
        cancelled = False
        for root, dirs, files in os.walk(local_dataset_dir):
            rel = os.path.relpath(root, local_dataset_dir)
            pid = train_id
            if rel != '.':
                for part in rel.split(os.sep):
                    pid = _ensure(part, pid)
            for fname in sorted(files):
                if progress.wasCanceled():
                    cancelled = True
                    break
                rel_display = os.path.join(rel, fname) if rel != '.' else fname
                progress.setLabelText(
                    f"Uploading {uploaded + 1}/{n_files}\n{rel_display}")
                progress.setValue(uploaded)
                QApplication.processEvents()
                _upload(os.path.join(root, fname), pid)
                uploaded += 1
            if cancelled:
                break

        progress.setValue(n_files)
        progress.close()

        if cancelled:
            self.statusBar().showMessage(
                f"Drive upload cancelled ({uploaded}/{n_files} files sent)", 5000)
            return "cancelled"

        self.statusBar().showMessage("Drive upload complete!", 5000)
        return "ok"

    # ---- Export Roboflow dataset (COCO JSON ZIP, native Roboflow import format) ----
    def _export_roboflow_coco(self):
        """
        Export annotated pages as a Roboflow-native COCO JSON dataset ZIP.

        Structure (drag-and-drop directly into a Roboflow project):
          <name>.zip/
            train/
              *.png
              _annotations.coco.json   <- COCO format, all train images
            valid/
              *.png
              _annotations.coco.json

        COCO bbox = [x_min, y_min, width, height] in absolute pixels.
        """
        selected = self._select_pages_dialog("Export Roboflow COCO — Select Pages")
        if selected is None:
            return
        # --- config dialog ---
        dlg = QDialog(self)
        dlg.setWindowTitle("Export Roboflow Dataset (COCO JSON)")
        dlg.setMinimumWidth(420)
        form = QFormLayout(dlg)

        name_edit = QLineEdit(self._settings.get("rf_coco_name", "atlas-roboflow"))
        form.addRow("Dataset name:", name_edit)

        split_spin = QSpinBox()
        split_spin.setRange(50, 100)
        split_spin.setValue(self._settings.get("rf_coco_train_pct", 80))
        split_spin.setSuffix(" % train")
        form.addRow("Train split:", split_spin)

        info = QLabel(
            "Format: COCO JSON (Roboflow native import)\n"
            "Output: <name>.zip with train/ and valid/ splits\n"
            "Annotations: _annotations.coco.json per split\n"
            "Bbox: absolute pixels  |  Images: 300 DPI PNGs\n"
            "Classes: component · component_label · continuation_symbol\n"
            "         continuation_page · continuation_rung · wire_label"
        )
        info.setStyleSheet("color: #888; font-size: 11px;")
        form.addRow(info)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        form.addRow(btns)

        if dlg.exec() != QDialog.Accepted:
            return

        dataset_name = name_edit.text().strip() or "atlas-roboflow"
        train_pct    = split_spin.value() / 100.0
        self._settings["rf_coco_name"]      = dataset_name
        self._settings["rf_coco_train_pct"] = split_spin.value()
        self._save_settings()

        dest_dir = QFileDialog.getExistingDirectory(self, "Select Export Destination Folder")
        if not dest_dir:
            return

        # --- use selected pages ---
        pages_map = {pg["page"]: pg for pg in self.meta["pages"]}
        ann_files = sorted(f"page_{p}.json" for p in selected)

        n_train   = max(1, round(len(ann_files) * train_pct))
        train_set = set(ann_files[:n_train])

        # --- COCO categories (0-based ids, matching Roboflow convention) ---
        CLASSES = [
            "component",
            "component_label",
            "continuation_symbol",
            "continuation_page",
            "continuation_rung",
            "wire_label",
        ]
        categories = [
            {"id": i, "name": name, "supercategory": "atlas"}
            for i, name in enumerate(CLASSES)
        ]
        cls_idx = {c: i for i, c in enumerate(CLASSES)}

        import zipfile, datetime, shutil
        zip_path = os.path.join(dest_dir, dataset_name + ".zip")
        import_date = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S+00:00")

        # Accumulators per split
        splits = {"train": {"images": [], "annotations": []},
                  "valid": {"images": [], "annotations": []}}
        img_id  = 0
        ann_id  = 0
        errors  = []
        exported = 0

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for fname in ann_files:
                try:
                    pg_num  = int(fname.replace("page_", "").replace(".json", ""))
                    pg_meta = pages_map.get(pg_num)
                    if not pg_meta:
                        errors.append(f"{fname}: page metadata missing"); continue

                    img_w, img_h = pg_meta["display_size"]
                    pdf_w        = pg_meta["pdf_width"]
                    pdf_h        = pg_meta["pdf_height"]
                    sx           = img_w / pdf_w
                    sy           = img_h / pdf_h

                    split    = "train" if fname in train_set else "valid"
                    img_name = pg_meta["image"]
                    src_img  = os.path.join(DATA_DIR, "pages", img_name)

                    if not os.path.exists(src_img):
                        errors.append(f"{fname}: image not found"); continue

                    # Add image file to zip
                    zf.write(src_img, f"{split}/{img_name}")

                    # COCO images entry
                    splits[split]["images"].append({
                        "id":            img_id,
                        "license":       1,
                        "file_name":     img_name,
                        "height":        img_h,
                        "width":         img_w,
                        "date_captured": import_date,
                    })

                    # Build annotations for this image
                    with open(os.path.join(ANN_DIR, fname)) as f:
                        ann = json.load(f)

                    def add_ann(class_name, bbox):
                        nonlocal ann_id
                        x0, y0, x1, y1 = bbox
                        px0 = max(0.0, x0 * sx);  px1 = min(img_w, x1 * sx)
                        py0 = max(0.0, y0 * sy);  py1 = min(img_h, y1 * sy)
                        bw  = px1 - px0;          bh  = py1 - py0
                        if bw <= 0 or bh <= 0:
                            return
                        splits[split]["annotations"].append({
                            "id":          ann_id,
                            "image_id":    img_id,
                            "category_id": cls_idx[class_name],
                            "bbox":        [round(px0, 2), round(py0, 2),
                                            round(bw,  2), round(bh,  2)],
                            "area":        round(bw * bh, 2),
                            "segmentation": [],
                            "iscrowd":     0,
                        })
                        ann_id += 1

                    for pair in ann.get("pairs", []):
                        if pair.get("type") == "continuation":
                            if "symbol" in pair: add_ann("continuation_symbol", pair["symbol"]["bbox"])
                            if "page"   in pair: add_ann("continuation_page",   pair["page"]["bbox"])
                            if "rung"   in pair: add_ann("continuation_rung",   pair["rung"]["bbox"])
                        else:
                            if "component" in pair: add_ann("component",       pair["component"]["bbox"])
                            if "label"     in pair: add_ann("component_label", pair["label"]["bbox"])
                    for wire in ann.get("wire_labels", []):
                        add_ann("wire_label", wire["bbox"])

                    img_id += 1
                    exported += 1
                    self.statusBar().showMessage(
                        f"Building Roboflow dataset… {exported}/{len(ann_files)}", 400
                    )
                    QApplication.processEvents()

                except Exception as e:
                    errors.append(f"{fname}: {e}")

            # Write _annotations.coco.json for each split
            for split, data in splits.items():
                if not data["images"]:
                    continue
                coco_doc = {
                    "info": {
                        "year":        str(datetime.datetime.utcnow().year),
                        "version":     "1",
                        "description": f"ATLAS Annotator export — {dataset_name}",
                        "contributor": "ATLAS Annotator",
                        "url":         "",
                        "date_created": import_date,
                    },
                    "licenses": [{"id": 1, "url": "", "name": "Private"}],
                    "categories": categories,
                    "images":      data["images"],
                    "annotations": data["annotations"],
                }
                zf.writestr(
                    f"{split}/_annotations.coco.json",
                    json.dumps(coco_doc, indent=2)
                )

        n_train_actual = len(splits["train"]["images"])
        n_valid_actual = len(splits["valid"]["images"])
        msg = (
            f"Exported {exported} page(s) to:\n{zip_path}\n\n"
            f"Train: {n_train_actual}   Valid: {n_valid_actual}\n"
            f"Total annotations: {ann_id}\n"
            f"Format: COCO JSON  |  Drag-and-drop into Roboflow project"
        )
        if errors:
            msg += "\n\nErrors:\n" + "\n".join(errors[:10])
        self.statusBar().showMessage(f"Roboflow dataset → {zip_path}", 8000)
        QMessageBox.information(self, "Roboflow Export Complete", msg)

    # ---- Key events ----
    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            return
        if event.key() == Qt.Key_Left:
            self._prev_page()
        elif event.key() == Qt.Key_Right:
            self._next_page()
        elif event.key() == Qt.Key_Delete:
            self._delete_selected()
        elif event.key() == Qt.Key_F:
            self.view.fit_view()
        elif event.key() == Qt.Key_Escape:
            if self.view._mask_mode:
                self._enter_mask_mode()  # exit mask mode
            else:
                self.view.keyPressEvent(event)
            self._update_sidebar()
        elif event.text().lower() == "m":
             self._enter_mask_mode()
        elif event.text().lower() == "e":
             self._toggle_mask_erase()
        elif event.text().lower() == "w":
             if self.view._mask_mode:
                 self._enter_mask_mode()  # exit mask mode first
             self._enter_wire_mode()
        elif event.text().lower() == "c":
             if self.view._mask_mode:
                 self._enter_mask_mode()  # exit mask mode first
             self._enter_component_mode()
        elif event.key() == Qt.Key_D and event.modifiers() & Qt.ControlModifier:
             if self.view._mask_mode:
                 self._enter_mask_mode()  # exit mask mode first
             self._enter_continuation_mode()
        else:
            super().keyPressEvent(event)

    # ---- Mask mode management ----
    def _enter_mask_mode(self):
        """Toggle mask mode on/off."""
        if self.view._mask_mode:
            self.view._mask_exit()
            self.mask_dock.hide()
        else:
            self.view._mask_enter()
            self.mask_dock.show()
        self._update_sidebar()
        self._update_mask_sidebar()

    def _toggle_mask_erase(self):
        """Toggle erase mode within mask mode (E key)."""
        if not self.view._mask_mode:
            return
        self.view._mask_erase_mode = not self.view._mask_erase_mode
        self._update_sidebar()

    def _build_mask_sidebar(self):
        """Build the left sidebar showing mask progress per component."""
        dock = QDockWidget("Mask Progress", self)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea)
        dock.setFeatures(QDockWidget.NoDockWidgetFeatures)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # ---- Brush size slider ----
        brush_group = QGroupBox("Brush")
        bg_lay = QHBoxLayout(brush_group)
        bg_lay.setContentsMargins(4, 2, 4, 2)
        self.mask_brush_label = QLabel("4")
        self.mask_brush_label.setFixedWidth(20)
        self.mask_brush_slider = QSlider(Qt.Horizontal)
        self.mask_brush_slider.setRange(1, 20)
        self.mask_brush_slider.setValue(4)
        self.mask_brush_slider.valueChanged.connect(self._on_brush_size_changed)
        bg_lay.addWidget(QLabel("1"))
        bg_lay.addWidget(self.mask_brush_slider)
        bg_lay.addWidget(self.mask_brush_label)
        layout.addWidget(brush_group)

        # ---- Eraser size slider ----
        eraser_group = QGroupBox("Eraser")
        eg_lay = QHBoxLayout(eraser_group)
        eg_lay.setContentsMargins(4, 2, 4, 2)
        self.mask_eraser_label = QLabel("6")
        self.mask_eraser_label.setFixedWidth(20)
        self.mask_eraser_slider = QSlider(Qt.Horizontal)
        self.mask_eraser_slider.setRange(1, 30)
        self.mask_eraser_slider.setValue(6)
        self.mask_eraser_slider.valueChanged.connect(self._on_eraser_size_changed)
        eg_lay.addWidget(QLabel("1"))
        eg_lay.addWidget(self.mask_eraser_slider)
        eg_lay.addWidget(self.mask_eraser_label)
        layout.addWidget(eraser_group)

        # ---- Unique labels list ----
        lbl_header = QLabel("Labels")
        lbl_header.setStyleSheet("font-weight: 700; font-size: 11px; margin-top: 4px;")
        layout.addWidget(lbl_header)
        self.mask_label_list = QListWidget()
        self.mask_label_list.setStyleSheet("font-size: 11px;")
        layout.addWidget(self.mask_label_list, 1)

        # ---- Per-instance list ----
        inst_header = QLabel("Instances")
        inst_header.setStyleSheet("font-weight: 700; font-size: 11px; margin-top: 4px;")
        layout.addWidget(inst_header)
        self.mask_list = QListWidget()
        self.mask_list.setStyleSheet("font-size: 11px;")
        layout.addWidget(self.mask_list, 1)

        dock.setWidget(container)
        dock.setMinimumWidth(200)
        dock.setMaximumWidth(280)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        self.mask_dock = dock
        dock.hide()  # shown only in mask mode

    def _on_brush_size_changed(self, val):
        self.view._mask_brush_radius = val
        self.mask_brush_label.setText(str(val))

    def _on_eraser_size_changed(self, val):
        self.view._mask_eraser_radius = val
        self.mask_eraser_label.setText(str(val))

    def _update_mask_sidebar(self):
        """Update the left mask progress sidebar."""
        self.mask_list.clear()
        self.mask_label_list.clear()
        if not self.view._mask_mode:
            return
        masked = self.view._mask_saved_pairs

        # Build label -> pair indices map
        from collections import OrderedDict
        label_map = OrderedDict()
        for i, p in enumerate(self.view.pairs):
            if p.get("type") == "continuation":
                continue
            lbl = p["label"]["text"]
            label_map.setdefault(lbl, []).append(i)

        # Instances list — one per unique class, checked if ANY instance is masked
        for lbl, indices in label_map.items():
            any_done = any(i in masked for i in indices)
            item = QListWidgetItem()
            if any_done:
                item.setText(f"\u2705 {lbl}")
                item.setForeground(QColor(80, 200, 80))
            else:
                item.setText(f"\u2b1c {lbl}")
                item.setForeground(QColor(200, 200, 200))
            self.mask_list.addItem(item)

        # Labels list — checked when ALL instances of that class are masked
        for lbl, indices in label_map.items():
            all_done = all(i in masked for i in indices)
            done_count = sum(1 for i in indices if i in masked)
            total = len(indices)
            item = QListWidgetItem()
            if all_done:
                item.setText(f"\u2705 {lbl}  ({done_count}/{total})")
                item.setForeground(QColor(80, 200, 80))
            else:
                item.setText(f"\u2b1c {lbl}  ({done_count}/{total})")
                item.setForeground(QColor(200, 200, 200))
            self.mask_label_list.addItem(item)


# ===================================================================
# Main entry point
# ===================================================================
def main():
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else None
    metadata_path = os.path.join(DATA_DIR, "metadata.json")
    _migrate_legacy_masks()

    # Preprocess if needed
    if pdf_path and not os.path.exists(metadata_path):
        print(f"\nPreprocessing: {pdf_path}")
        subprocess.check_call([
            sys.executable, os.path.join(BASE, "preprocess.py"), pdf_path
        ])
    elif not os.path.exists(metadata_path):
        print("Error: No preprocessed data found.")
        print("  Usage: python annotator.py <path_to_schematic.pdf>")
        sys.exit(1)

    os.makedirs(ANN_DIR, exist_ok=True)

    with open(metadata_path) as f:
        metadata = json.load(f)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # consistent cross-platform look
    window = AnnotatorWindow(metadata)
    window.show()
    sys.exit(app.exec())

# ===================================================================
# Fingerprinting Page (Viewer)
# ===================================================================


class FingerprintImageView(QGraphicsView):
    """Compact image viewer for fingerprint outputs with explicit pan/zoom."""

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setBackgroundBrush(C_BG)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setMouseTracking(True)

        self._zoom = 1.0
        self._panning = False
        self._pan_start = QPointF()

    def _zoom_by(self, factor: float):
        new_zoom = self._zoom * factor
        new_zoom = max(0.08, min(25.0, new_zoom))
        if new_zoom == self._zoom:
            return
        actual = new_zoom / self._zoom
        self.scale(actual, actual)
        self._zoom = new_zoom

    def fit_to_content(self):
        rect = self.scene().sceneRect()
        if rect.isNull() or rect.width() <= 1 or rect.height() <= 1:
            return
        self.fitInView(rect, Qt.KeepAspectRatio)
        self._zoom = self.transform().m11()

    def zoom_in(self):
        self._zoom_by(1.15)

    def zoom_out(self):
        self._zoom_by(1 / 1.15)

    def reset_zoom(self):
        self.resetTransform()
        self._zoom = 1.0

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self._panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x())
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y())
            )
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton and self._panning:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.zoom_in()
        else:
            self.zoom_out()


class FingerprintingDialog(QDialog):
    """Standalone dialog for viewing fingerprint search results.

    Opens as a separate window so it never hides/invalidates the
    annotation canvas underneath.

    Expected results JSON format
    ----------------------------
    List[dict], each dict has:
      - page: int
      - pair: int (optional)
      - label: str (optional)
      - score: float (optional)
      - bbox_pdf: [x0,y0,x1,y1]
      - fingerprint_vis: str (optional, relative to tools/annotator/data)
    """

    def __init__(self, metadata, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fingerprinting Results")
        self.resize(1200, 800)
        self.meta = metadata
        self._scale = float(metadata.get("scale", VIEW_DPI / 72.0))

        self._results_path = None
        self._by_page = {}  # page_num -> list[dict]
        self._pages = []

        # Apply same dark theme as main window
        self.setStyleSheet("""
            QDialog { background: #0d1117; }
            QPushButton { background: #30363d; color: #e6edf3; border: none;
                          padding: 5px 14px; border-radius: 5px; font-size: 12px; }
            QPushButton:hover { background: #484f58; }
            QLabel { color: #e6edf3; }
            QListWidget { background: #0d1117; color: #e6edf3; border: none;
                          font-size: 12px; }
            QListWidget::item { padding: 6px 8px; border-bottom: 1px solid #161b22; }
            QListWidget::item:selected { background: #1a2233; color: #00d4ff; }
            QListWidget::item:hover { background: #161b22; }
        """)

        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # Left panel
        left = QVBoxLayout()
        left.setSpacing(8)
        root.addLayout(left, 0)

        top_row = QHBoxLayout()
        btn_load = QPushButton("Load Results JSON")
        btn_load.clicked.connect(self._load_results_clicked)
        top_row.addWidget(btn_load)
        left.addLayout(top_row)

        self.path_label = QLabel("No results loaded")
        self.path_label.setStyleSheet("color: #8b949e; font-size: 11px;")
        left.addWidget(self.path_label)

        # --- Confidence slider + Search button ---
        conf_row = QHBoxLayout()
        conf_row.setSpacing(6)
        conf_lbl = QLabel("Confidence:")
        conf_lbl.setStyleSheet("font-size: 12px;")
        conf_row.addWidget(conf_lbl)

        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(50, 99)       # 0.50 – 0.99
        self.conf_slider.setValue(85)            # default 0.85
        self.conf_slider.setTickPosition(QSlider.TicksBelow)
        self.conf_slider.setTickInterval(5)
        self.conf_slider.setSingleStep(1)
        self.conf_slider.valueChanged.connect(self._on_conf_changed)
        conf_row.addWidget(self.conf_slider, 1)

        self.conf_value_lbl = QLabel("0.85")
        self.conf_value_lbl.setFixedWidth(36)
        self.conf_value_lbl.setStyleSheet("font-size: 12px; font-weight: bold;")
        conf_row.addWidget(self.conf_value_lbl)
        left.addLayout(conf_row)

        btn_search = QPushButton("Search")
        btn_search.setStyleSheet(
            "QPushButton { background: #238636; font-weight: bold; }"
            "QPushButton:hover { background: #2ea043; }"
        )
        btn_search.clicked.connect(self._run_search)
        left.addWidget(btn_search)

        self.search_status = QLabel("")
        self.search_status.setStyleSheet("color: #8b949e; font-size: 11px;")
        self.search_status.setWordWrap(True)
        left.addWidget(self.search_status)

        self.page_list = QListWidget()
        self.page_list.currentRowChanged.connect(self._on_page_selected)
        self.page_list.itemClicked.connect(lambda _item: self._render_page_with_boxes(self._current_page)
                                            if self._current_page is not None else None)
        left.addWidget(self.page_list, 1)

        self.det_list = QListWidget()
        self.det_list.currentRowChanged.connect(self._on_detection_selected)
        left.addWidget(self.det_list, 1)

        class_header = QLabel("All Components (Masked)")
        class_header.setStyleSheet("font-weight: 700; font-size: 11px; margin-top: 4px;")
        left.addWidget(class_header)

        self.component_class_list = QListWidget()
        self.component_class_list.setStyleSheet("font-size: 11px;")
        left.addWidget(self.component_class_list, 1)

        # Right panel: image preview + navigation and zoom controls.
        right_panel = QVBoxLayout()
        right_panel.setSpacing(6)
        root.addLayout(right_panel, 1)

        view_controls = QHBoxLayout()
        self.btn_fp_prev = QPushButton("◀ Prev Page")
        self.btn_fp_prev.clicked.connect(self._prev_fp_page)
        self.btn_fp_prev.setEnabled(False)
        view_controls.addWidget(self.btn_fp_prev)

        self.fp_page_label = QLabel("No pages")
        self.fp_page_label.setMinimumWidth(120)
        view_controls.addWidget(self.fp_page_label)

        self.btn_fp_next = QPushButton("Next Page ▶")
        self.btn_fp_next.clicked.connect(self._next_fp_page)
        self.btn_fp_next.setEnabled(False)
        view_controls.addWidget(self.btn_fp_next)

        view_controls.addStretch()

        self.btn_fp_zoom_in = QPushButton("＋")
        self.btn_fp_zoom_in.clicked.connect(self._zoom_in_fp_page)
        self.btn_fp_zoom_in.setEnabled(False)
        view_controls.addWidget(self.btn_fp_zoom_in)

        self.btn_fp_zoom_out = QPushButton("－")
        self.btn_fp_zoom_out.clicked.connect(self._zoom_out_fp_page)
        self.btn_fp_zoom_out.setEnabled(False)
        view_controls.addWidget(self.btn_fp_zoom_out)

        self.btn_fp_fit = QPushButton("Fit")
        self.btn_fp_fit.clicked.connect(self._fit_fp_view)
        self.btn_fp_fit.setEnabled(False)
        view_controls.addWidget(self.btn_fp_fit)

        self.btn_fp_zoom_100 = QPushButton("100%")
        self.btn_fp_zoom_100.clicked.connect(self._fp_zoom_100)
        self.btn_fp_zoom_100.setEnabled(False)
        view_controls.addWidget(self.btn_fp_zoom_100)

        right_panel.addLayout(view_controls)

        self.scene = QGraphicsScene()
        self.gview = FingerprintImageView(self.scene, self)
        right_panel.addWidget(self.gview, 1)

        self._last_pixmap = QPixmap()
        self._current_page = None
        self._current_page_dets = []
        self._auto_loaded = False

    def showEvent(self, event):
        super().showEvent(event)
        self._fit_in_view_deferred()
        self._refresh_component_class_list()
        # Auto-load the most recent results JSON the first time we open.
        if not self._auto_loaded:
            self._auto_loaded = True
            self._auto_load_latest_results()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_in_view_deferred()

    def _fit_in_view(self):
        self.gview.fit_to_content()

    def _fit_in_view_deferred(self):
        QTimer.singleShot(0, self._fit_in_view)

    def _prev_fp_page(self):
        row = self.page_list.currentRow()
        if row > 0:
            self.page_list.setCurrentRow(row - 1)

    def _next_fp_page(self):
        row = self.page_list.currentRow()
        if row < len(self._pages) - 1:
            self.page_list.setCurrentRow(row + 1)

    def _zoom_in_fp_page(self):
        if self.gview:
            self.gview.zoom_in()

    def _zoom_out_fp_page(self):
        if self.gview:
            self.gview.zoom_out()

    def _fit_fp_view(self):
        self.gview.fit_to_content()

    def _fp_zoom_100(self):
        if self.gview:
            self.gview.reset_zoom()
            if self._current_page is not None and not self._last_pixmap.isNull():
                self._show_pixmap(self._last_pixmap, auto_fit=False)
            elif self._current_page is not None:
                self._render_page_with_boxes(self._current_page)

    def _update_fp_page_controls(self):
        total = len(self._pages)
        if not total:
            self.fp_page_label.setText("No pages")
            self.btn_fp_prev.setEnabled(False)
            self.btn_fp_next.setEnabled(False)
            self.btn_fp_zoom_in.setEnabled(False)
            self.btn_fp_zoom_out.setEnabled(False)
            self.btn_fp_fit.setEnabled(False)
            self.btn_fp_zoom_100.setEnabled(False)
            return

        row = self.page_list.currentRow()
        if row < 0:
            row = 0
            self.page_list.setCurrentRow(0)
        self.fp_page_label.setText(f"Page {self._pages[row]} ({row + 1}/{total})")
        self.btn_fp_prev.setEnabled(row > 0)
        self.btn_fp_next.setEnabled(row < total - 1)
        self.btn_fp_zoom_in.setEnabled(True)
        self.btn_fp_zoom_out.setEnabled(True)
        self.btn_fp_fit.setEnabled(True)
        self.btn_fp_zoom_100.setEnabled(True)

    def _show_pixmap(self, pix: QPixmap, auto_fit: bool = True):
        self.scene.clear()
        self.scene.addPixmap(pix)
        self.scene.setSceneRect(QRectF(pix.rect()))
        self._last_pixmap = pix
        if auto_fit:
            self._fit_in_view_deferred()

    @staticmethod
    def _find_latest_results_json() -> str | None:
        """Recursively find the most recently modified fingerprint results JSON."""
        candidates = []
        seen = set()
        for mask_dir in (MASK_DIR, MASK_DIR_LEGACY):
            if not os.path.isdir(mask_dir):
                continue
            for dirpath, _dirs, files in os.walk(mask_dir):
                for f in files:
                    if f == "fingerprint_distance_results.json":
                        full = os.path.join(dirpath, f)
                        if full in seen:
                            continue
                        seen.add(full)
                        candidates.append((os.path.getmtime(full), full))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][1]

    def _auto_load_latest_results(self):
        path = self._find_latest_results_json()
        if path:
            self.load_results(path)

    @staticmethod
    def _scan_annotation_and_mask_classes() -> tuple[set[str], set[str]]:
        all_classes: set[str] = set()
        masked_classes: set[str] = set()

        # Gather all component classes from all annotation pages.
        if os.path.isdir(ANN_DIR):
            for name in os.listdir(ANN_DIR):
                if not (name.startswith("page_") and name.endswith(".json")):
                    continue
                path = os.path.join(ANN_DIR, name)
                try:
                    with open(path, encoding="utf-8") as fh:
                        data = json.load(fh)
                except Exception:
                    continue
                for pair in data.get("pairs", []):
                    try:
                        lbl = (pair.get("label", {}).get("text") or "").strip()
                    except Exception:
                        lbl = ""
                    if lbl:
                        all_classes.add(lbl)

        # Gather masked classes from all saved mask JSON files.
        for mask_dir in (MASK_DIR, MASK_DIR_LEGACY):
            if not os.path.isdir(mask_dir):
                continue
            for name in os.listdir(mask_dir):
                if not (name.startswith("page_") and name.endswith("_masks.json")):
                    continue
                path = os.path.join(mask_dir, name)
                try:
                    with open(path, encoding="utf-8") as fh:
                        data = json.load(fh)
                except Exception:
                    continue

                pair_info = data.get("pair_info", {}) or {}
                masked_pairs = data.get("masked_pairs", []) or []

                # Prefer explicit masked pair indices.
                for idx in masked_pairs:
                    info = pair_info.get(str(idx), {})
                    lbl = (info.get("label") or "").strip() if isinstance(info, dict) else ""
                    if lbl:
                        masked_classes.add(lbl)

                # Fallback for older files lacking masked_pairs.
                if not masked_pairs:
                    for info in pair_info.values():
                        if not isinstance(info, dict):
                            continue
                        lbl = (info.get("label") or "").strip()
                        if lbl:
                            masked_classes.add(lbl)

        return all_classes, masked_classes

    def _refresh_component_class_list(self):
        all_classes, masked_classes = self._scan_annotation_and_mask_classes()
        self.component_class_list.clear()

        for lbl in sorted(all_classes, key=lambda s: s.lower()):
            item = QListWidgetItem()
            if lbl in masked_classes:
                item.setText(f"\u2705 {lbl}")
                item.setForeground(QColor(80, 200, 80))
            else:
                item.setText(f"\u2b1c {lbl}")
                item.setForeground(QColor(200, 200, 200))
            self.component_class_list.addItem(item)

    def _on_conf_changed(self, value: int):
        self.conf_value_lbl.setText(f"{value / 100:.2f}")

    def _run_search(self):
        """Run fingerprint_distance_search.py with the current confidence slider value."""
        confidence = self.conf_slider.value() / 100.0
        script = os.path.join(BASE, "fingerprint_distance_search.py")
        python = sys.executable

        self.search_status.setText(f"Searching (confidence {confidence:.2f})…")
        self.search_status.repaint()
        QApplication.processEvents()

        cmd = [
            python, script,
            "--template-page", "7",
            "--template-pair", "10",
            "--pages", "7", "12",
            "--n-points", "500",
            "--min-match", f"{confidence:.2f}",
            "--rel-tol", "0.04",
            "--seeds", "3",
            "--draw",
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300,
                cwd=os.path.dirname(script),
            )
            if result.returncode != 0:
                self.search_status.setText(f"Error (exit {result.returncode})")
                QMessageBox.warning(
                    self, "Search Failed",
                    f"Process exited with code {result.returncode}\n\n"
                    f"stderr:\n{result.stderr[:2000]}",
                )
                return
            # Count detections from stdout
            lines = result.stdout.strip().split("\n")
            total_line = [l for l in lines if "Total detections" in l]
            summary = total_line[-1] if total_line else lines[-1] if lines else ""
            self.search_status.setText(summary.strip())
        except subprocess.TimeoutExpired:
            self.search_status.setText("Search timed out (300s)")
            return
        except Exception as e:
            self.search_status.setText(f"Error: {e}")
            return

        # Auto-load the new results
        path = self._find_latest_results_json()
        if path:
            self.load_results(path)

    def _load_results_clicked(self):
        start_dir = MASK_DIR if os.path.isdir(MASK_DIR) else MASK_DIR_LEGACY
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Fingerprint Results JSON",
            start_dir,
            "JSON Files (*.json)",
        )
        if not path:
            return
        self.load_results(path)

    def prompt_and_load_results_if_needed(self) -> bool:
        """Ensure results are loaded; prompt for JSON if not."""
        if self._by_page:
            return True
        start_dir = MASK_DIR if os.path.isdir(MASK_DIR) else MASK_DIR_LEGACY
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Fingerprint Results JSON",
            start_dir,
            "JSON Files (*.json)",
        )
        if not path:
            return False
        self.load_results(path)
        return bool(self._by_page)

    def backup_and_apply_results_to_annotations(self) -> dict:
        """Make a backup copy of annotations, then apply fingerprint bboxes.

        We interpret each result row as:
          - page: 1-based page number
          - pair: index into page_X.json['pairs']
          - bbox_pdf: new component bbox in PDF coords

        If multiple results exist for the same (page, pair), we keep the highest
        score (if provided) and apply that bbox.
        """
        if not self._by_page:
            QMessageBox.warning(self, "No Results", "Load a results JSON first.")
            return {}

        # 1) Backup annotations
        backup_root = os.path.join(BASE, "annotations_backups")
        ts = datetime.now().strftime("backup_%Y%m%d_%H%M%S")
        backup_dir = os.path.join(backup_root, ts)
        os.makedirs(backup_dir, exist_ok=True)
        if not os.path.isdir(ANN_DIR):
            QMessageBox.warning(self, "Missing Annotations", f"Not found:\n{ANN_DIR}")
            return {}
        for name in os.listdir(ANN_DIR):
            if not name.lower().endswith(".json"):
                continue
            src = os.path.join(ANN_DIR, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(backup_dir, name))

        # 2) Collapse to best result per (page, pair)
        best: dict[tuple[int, int], dict] = {}
        skipped = 0
        for page, dets in self._by_page.items():
            for det in dets:
                try:
                    pnum = int(det.get("page", page))
                    pair_idx = det.get("pair")
                    if pair_idx is None:
                        skipped += 1
                        continue
                    pair_idx = int(pair_idx)
                    bbox = det.get("bbox_pdf")
                    if not bbox or len(bbox) != 4:
                        skipped += 1
                        continue
                    score = det.get("score")
                    score_val = float(score) if score is not None else -1.0
                except Exception:
                    skipped += 1
                    continue

                key = (pnum, pair_idx)
                prev = best.get(key)
                if prev is None or score_val > prev["_score"]:
                    best[key] = {
                        "bbox": [float(v) for v in bbox],
                        "_score": score_val,
                    }

        # 3) Apply updates into page_N.json
        pages_touched = set()
        changed_pairs = 0

        # Group by page for fewer file writes
        by_page_key: dict[int, list[tuple[int, list[float]]]] = {}
        for (pnum, pair_idx), payload in best.items():
            by_page_key.setdefault(pnum, []).append((pair_idx, payload["bbox"]))

        for pnum, updates in by_page_key.items():
            ann_path = os.path.join(ANN_DIR, f"page_{pnum}.json")
            if not os.path.exists(ann_path):
                skipped += len(updates)
                continue
            try:
                with open(ann_path, encoding="utf-8") as fh:
                    ann = json.load(fh)
            except Exception:
                skipped += len(updates)
                continue

            pairs = ann.get("pairs")
            if not isinstance(pairs, list):
                skipped += len(updates)
                continue

            changed_this_page = 0
            for pair_idx, bbox in updates:
                if pair_idx < 0 or pair_idx >= len(pairs):
                    skipped += 1
                    continue
                pair_obj = pairs[pair_idx]
                if not isinstance(pair_obj, dict):
                    skipped += 1
                    continue
                comp = pair_obj.get("component")
                if not isinstance(comp, dict):
                    skipped += 1
                    continue
                old_bbox = comp.get("bbox")
                comp["bbox"] = bbox
                if old_bbox != bbox:
                    changed_pairs += 1
                    changed_this_page += 1

            if changed_this_page:
                pages_touched.add(pnum)
                try:
                    with open(ann_path, "w", encoding="utf-8") as fh:
                        json.dump(ann, fh, indent=2, ensure_ascii=False)
                except Exception:
                    # If write fails, we still have backup.
                    pass

        return {
            "backup_dir": backup_dir,
            "changed_pairs": changed_pairs,
            "changed_pages": len(pages_touched),
            "skipped": skipped,
        }

    def load_results(self, path: str):
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as e:
            QMessageBox.warning(self, "Load Failed", f"Could not load JSON:\n{e}")
            return

        if not isinstance(data, list):
            QMessageBox.warning(self, "Invalid Format", "Results JSON must be a list")
            return

        by_page = {}
        for r in data:
            if not isinstance(r, dict):
                continue
            page = r.get("page")
            bbox = r.get("bbox_pdf")
            if page is None or bbox is None:
                continue
            by_page.setdefault(int(page), []).append(r)

        self._results_path = path
        self._by_page = by_page
        self._pages = sorted(by_page.keys())

        self.path_label.setText(os.path.basename(path))
        self._refresh_component_class_list()
        self.page_list.clear()
        self.det_list.clear()
        self.scene.clear()
        self._current_page = None
        self._current_page_dets = []

        for pg in self._pages:
            self.page_list.addItem(f"Page {pg}  ({len(self._by_page[pg])})")

        if self._pages:
            self.page_list.setCurrentRow(0)
        self._update_fp_page_controls()

    def _on_page_selected(self, row: int):
        if row < 0 or row >= len(self._pages):
            return
        pg = self._pages[row]
        self._current_page = pg
        self._current_page_dets = self._by_page.get(pg, [])
        self._populate_detection_list(pg)
        self._update_fp_page_controls()
        self._render_page_with_boxes(pg)

    def _populate_detection_list(self, page_num: int):
        self.det_list.clear()
        dets = self._by_page.get(page_num, [])
        # Sort: higher score first if present.
        def k(d):
            s = d.get("score")
            try:
                return -float(s)
            except Exception:
                return 0.0

        dets_sorted = sorted(dets, key=k)
        self._current_page_dets = dets_sorted
        for d in dets_sorted:
            pair = d.get("pair", "?")
            lbl = d.get("label", "")
            score = d.get("score")
            if score is None:
                self.det_list.addItem(f"pair {pair}  {lbl}")
            else:
                self.det_list.addItem(f"pair {pair}  {lbl}  {float(score):.2f}")

        # Important UX detail: do NOT auto-select a detection.
        # Selecting a detection shows its fingerprint_vis, which would otherwise
        # immediately replace the page-with-boxes view and feel like images "disappear".
        self.det_list.setCurrentRow(-1)

    def _on_detection_selected(self, row: int):
        if row < 0 or row >= len(self._current_page_dets):
            return
        det = self._current_page_dets[row]
        vis_rel = det.get("fingerprint_vis")
        if vis_rel:
            # fingerprint_vis is relative to tools/annotator/data
            vis_path = os.path.join(DATA_DIR, vis_rel)
            if os.path.exists(vis_path):
                self._render_image(vis_path)
                return

        # Fallback: show the page with boxes.
        if self._current_page is not None:
            self._render_page_with_boxes(self._current_page)

    def _render_image(self, img_path: str):
        base = QImage(img_path)
        if base.isNull():
            QMessageBox.warning(self, "Load Failed", f"Could not load:\n{img_path}")
            return
        self._show_pixmap(QPixmap.fromImage(base))

    def _render_page_with_boxes(self, page_num: int):
        img_path = os.path.join(DATA_DIR, "pages", f"page_{page_num}.png")
        if not os.path.exists(img_path):
            QMessageBox.warning(self, "Missing Page Image", f"Not found:\n{img_path}")
            return

        base = QImage(img_path)
        if base.isNull():
            QMessageBox.warning(self, "Load Failed", f"Could not load:\n{img_path}")
            return

        annotated = base.convertToFormat(QImage.Format_ARGB32)
        painter = QPainter(annotated)

        # First pass: composite each detection's fingerprint starburst crop
        # onto the page at the correct bbox position.
        for r in self._by_page.get(page_num, []):
            bbox = r.get("bbox_pdf")
            vis_rel = r.get("fingerprint_vis")
            if not bbox or len(bbox) != 4 or not vis_rel:
                continue
            vis_path = os.path.join(DATA_DIR, vis_rel)
            if not os.path.exists(vis_path):
                continue
            crop = QImage(vis_path)
            if crop.isNull():
                continue
            x0, y0, x1, y1 = [float(v) for v in bbox]
            px0 = int(round(x0 * self._scale))
            py0 = int(round(y0 * self._scale))
            px1 = int(round(x1 * self._scale))
            py1 = int(round(y1 * self._scale))
            target_w = px1 - px0
            target_h = py1 - py0
            if target_w > 0 and target_h > 0:
                # Scale crop to match the bbox size on the full page image
                scaled = crop.scaled(target_w, target_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
                painter.drawImage(px0, py0, scaled)

        # Second pass: draw green bounding boxes and score labels on top
        pen = QPen(QColor(0, 255, 0, 255))
        pen.setWidth(3)
        painter.setPen(pen)
        painter.setFont(QFont("Arial", 10, QFont.Bold))

        for r in self._by_page.get(page_num, []):
            bbox = r.get("bbox_pdf")
            if not bbox or len(bbox) != 4:
                continue
            x0, y0, x1, y1 = [float(v) for v in bbox]
            px0 = int(round(x0 * self._scale))
            py0 = int(round(y0 * self._scale))
            px1 = int(round(x1 * self._scale))
            py1 = int(round(y1 * self._scale))
            painter.drawRect(px0, py0, px1 - px0, py1 - py0)

            score = r.get("score")
            lbl = r.get("label", "")
            txt = f"{lbl} {float(score):.2f}".strip() if score is not None else lbl
            if txt:
                painter.drawText(px0, max(0, py0 - 4), txt)

        painter.end()

        self._show_pixmap(QPixmap.fromImage(annotated))


if __name__ == "__main__":
    main()

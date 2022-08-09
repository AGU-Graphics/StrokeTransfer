# MIT License
#
# Copyright (c) 2022  Hideki Todo, Kunihiko Kobayashi, Jin Katsuragi, Haruna Shimotahira, Shizuo Kaji, Yonghao Yue
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import copy
import json
import os
import re
import sys

import numpy as np
import PyQt5
from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtGui import QIcon, QImage, QPixmap, qGray, QKeySequence, QColor
from PyQt5.QtWidgets import (QApplication, QFileDialog, QGridLayout,
                             QHBoxLayout, QLabel, QListWidget, QMainWindow,
                             QPushButton, QSizePolicy, QSlider, QSpinBox,
                             QToolButton, QVBoxLayout, QWidget, QShortcut, QColorDialog)

from util.tool import interpolation


class InputAnnotation():
    """ Data class for input annotation data.

    Attributes:
        name: label data shown in the list widget UI.
        positions: list of (2, ) np.array vector data for annotated positions (polyline).
        width: float parameter value for the width.
    """

    def __init__(self):
        self.name = ""
        self.positions = []
        self.width = 0.05


class FloatSlider(QWidget):
    """ Float slider UI which supports value range (min, max) and default value.

    Attributes:
        name: label of the data.
        vmin: minimum value that can be controlled on the slider.
        vmax: maximum value that can be controlled on the slider.

    Examples:
        FloadSlider can be used similarly to QSlider.

        # Create GUI
        slider = FloatSlider(name="Width", vmin=0.01, vmax=0.1, v=0.03)
        slider.valueChanged.connect(slider_change)

        # Get value.
        v = slider.value()
    """
    valueChanged = pyqtSignal(float)

    def __init__(self, name="", vmin=0.0, vmax=1.0, v=0.1):
        """
        Args:
            name: label of the data.
            vmin: minimum value that can be controlled on the slider.
            vmax: maximum value that can be controlled on the slider.
            v: default value shown in the initial state.
        """
        super(FloatSlider, self).__init__()
        self.name = name
        self.vmin = vmin
        self.vmax = vmax

        # Create slider.
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(self.slider_pos(v))
        self.slider.setFocusPolicy(Qt.StrongFocus)
        self.slider.valueChanged.connect(self.slider_change)

        # Create data label.
        self.label = QLabel(self.label_txt())
        layout = QGridLayout()
        layout.addWidget(self.label, 0, 0)
        layout.addWidget(self.slider, 0, 1)
        self.setLayout(layout)

    def label_txt(self):
        return f"{self.name}={self.value():.3f}: "

    def value(self):
        t = self.slider.value() / 100.0
        return self.vmin + t * (self.vmax - self.vmin)

    def set_value(self, v):
        p = self.slider_pos(v)
        self.slider.setValue(p)

    def slider_pos(self, v):
        t = (v - self.vmin) / (self.vmax - self.vmin)
        return int(100 * t)

    def slider_change(self, val):
        v = self.value()
        self.label.setText(self.label_txt())
        self.valueChanged.emit(v)


def gray_icon(icon_file):
    """ Make gray icon for toggle mode of ToolButton.

    Args:
        icon_file: input icon file name.

    Returns:
        gray_icon(QIcon): converted gray icon of the input icon file.
    """
    src = QPixmap(icon_file)
    if isinstance(src, QPixmap):
        src = src.toImage()
    dest = QImage(src.size(), QImage.Format_ARGB32)
    widthRange = range(src.width())
    for y in range(src.height()):
        for x in widthRange:
            pixel = src.pixelColor(x, y)
            alpha = pixel.alpha()
            if alpha < 255:
                alpha //= 3
            gray = qGray(src.pixel(x, y))
            pixel.setRgb(gray, gray, gray, alpha)
            dest.setPixelColor(x, y, pixel)
    return QIcon(QPixmap.fromImage(dest))


class ToolButton(QToolButton):
    """ Tool button UI which supports toggle mode.

    Examples:
        ToolButton supports toggle mode for visibility control.
        This tool button is clickable on is_visible=False state with "Disabled" look.

        # Usual tool button usage.
        open_image_button = ToolButton("icons/open_image.png", "Open\nImage", parent)

        # Toggle mode.
        view_line_button = ToolButton("icons/view_line.png", "Show\nAnnotation", parent, toggle_mode=True)
    """

    def __init__(self, icon_file, tool_label, parent=None, toggle_mode=True):
        """
        Args:
            icon_file:  input icon file name.
            tool_label: text label for tool button.
            parent: parent UI element.
            toggle_mode: if True, the tool button is used for visibility control.
        """
        super(ToolButton, self).__init__(parent)

        self.on_icon = QIcon(icon_file)
        self.toggle_mode = toggle_mode

        if toggle_mode:
            self.is_visible = True
            self.off_icon = gray_icon(icon_file)

        self.setIcon(self.on_icon)
        self.setText(tool_label)

        self.on_style_sheet = "QToolButton {background-color: #eee; color : #333;  font-size: 14px; font-family: Arial;}"
        self.off_style_sheet = "QToolButton {background-color: #eee; color : #999;  font-size: 14px; font-family: Arial;}"

        self.setStyleSheet(self.on_style_sheet)

    def mousePressEvent(self, event):
        if not self.toggle_mode:
            return QToolButton.mousePressEvent(self, event)

        if self.is_visible:
            self.setIcon(self.off_icon)
            self.setStyleSheet(self.off_style_sheet)
            self.is_visible = False
        else:
            self.setIcon(self.on_icon)
            self.setStyleSheet(self.on_style_sheet)
            self.is_visible = True

        return QToolButton.mousePressEvent(self, event)


class ZoomWidget(QSpinBox):
    def __init__(self, value=100):
        super(ZoomWidget, self).__init__()


class Canvas(QWidget):
    """ Canvas ui to make annotations.
    """
    double_clicked = pyqtSignal()

    def __init__(self):
        super(Canvas, self).__init__()

        # initial value
        self.annotation = InputAnnotation()
        self.painter = QtGui.QPainter()
        self.pixmap = QtGui.QPixmap()
        self.scale = 1.0
        self.color_list = [QtGui.QColor(0, 255, 0, 128), QtGui.QColor(255, 0, 0, 128),
                           QtGui.QColor(0, 0, 255, 128), QtGui.QColor(0, 255, 255, 128),
                           QtGui.QColor(255, 255, 0, 200)]

        self.selected_annotation_id = -1
        self.annotation_list = []
        self.image_max = 1
        self.filedata = []
        self.tracking_annotation = []
        self.visible_annotation = True
        self.visible_orientation = True

        # Visualization settings for interpolated orientations.
        self.vector_length = 0.016
        self.triangle_length = 2
        self.triangle_width = 0.005

        self.canvas_layout = QGridLayout()
        self.setLayout(self.canvas_layout)

        self.setMouseTracking(True)

    def load_image(self, filepath):
        img = QtGui.QImage()

        if not img.load(filepath):
            return False

        # QImage -> QPixmap
        self.pixmap = QtGui.QPixmap.fromImage(img)
        self.calc_image_max(filepath)

    def paintEvent(self, event):
        if not self.pixmap:
            return super(Canvas, self).paintEvent(event)

        painter = self.painter

        painter.begin(self)

        painter.scale(self.scale, self.scale)

        painter.translate(self.offsetToCenterForQt())

        painter.drawPixmap(0, 0, self.pixmap)

        if len(self.annotation.positions) > 1:
            self.paint(painter)

        if len(self.tracking_annotation) > 1:
            self.paint_tracking_annotation(painter)
            self.tracking_annotation.clear()

        if self.visible_orientation:
            if (len(self.annotation_list) > 0):
                self.paint_orientations(painter)

        if self.visible_annotation:
            self.paint_annotations(painter)

        painter.end()

    def offsetToCenter(self):
        scale = self.scale
        area = super(Canvas, self).size()
        w, h = self.pixmap.width() * scale, self.pixmap.height() * scale
        aw, ah = area.width(), area.height()
        x = (aw - w) / (2 * scale) if aw > w else 0
        y = (ah - h) / (2 * scale) if ah > h else 0

        return x, y

    def offsetToCenterForQt(self):
        x, y = self.offsetToCenter()
        return QtCore.QPoint(int(x), int(y))

    def offsetToCenterForNp(self):
        x, y = self.offsetToCenter()
        return np.array([int(x), int(y)]).astype(np.float32)

    def mousePressEvent(self, event):
        if self.hasMouseTracking():
            point = np.array([event.localPos().x(), event.localPos().y()]).astype(np.float32)
            pos = self.transformPos(point)
            pos[0] = self.changeto01(pos[0])
            pos[1] = self.changeto01(pos[1])
            self.selected_annotation_id = -1

            if event.button() == QtCore.Qt.LeftButton:
                if len(self.annotation.positions) > 0:
                    if str(pos) != str(self.annotation.positions[-1]):
                        self.append_position(pos)
                else:
                    self.append_position(pos)
                self.repaint()

        if event.button() == QtCore.Qt.RightButton:
            if self.hasMouseTracking():
                self.setMouseTracking(False)
                self.tracking_annotation.clear()
                self.repaint()
            else:
                self.setMouseTracking(True)

    def mouseDoubleClickEvent(self, event):
        self.double_clicked.emit()

    def mouseMoveEvent(self, event):
        if len(self.annotation.positions) > 0:
            point = np.array([event.localPos().x(), event.localPos().y()]).astype(np.float32)
            pos = self.transformPos(point)
            pos[0] = self.changeto01(pos[0])
            pos[1] = self.changeto01(pos[1])
            self.tracking_annotation.append(self.annotation.positions[-1])
            self.tracking_annotation.append(pos)
            self.repaint()

    def paint(self, painter):
        pen = QtGui.QPen(self.color_list[0])
        pen.setWidth(max(1, int(round(self.image_max * self.annotation.width))))
        painter.setPen(pen)
        if self.annotation.positions:
            self.paint_annotation(self.annotation.positions, painter)

    def paint_annotations(self, painter):
        x = 0
        if self.annotation_list:
            for l in self.annotation_list:
                painter.setPen(self.setPen(x, l))
                x += 1
                self.paint_annotation(l.positions, painter)

    def paint_annotation(self, list, painter):
        drawlist = []

        for l in list:
            drawlist.append(self.changeToQPoint(l))
        startpoint = drawlist[0]
        for ln in range(1, len(drawlist)):
            painter.drawLine(self.changefrom01(startpoint), self.changefrom01(drawlist[ln]))
            startpoint = drawlist[ln]

    def paint_tracking_annotation(self, painter):
        pen = QtGui.QPen(self.color_list[3])

        pen.setWidth(max(1, int(round(self.image_max * self.annotation.width))))
        painter.setPen(pen)
        self.paint_annotation(self.tracking_annotation, painter)

    def setPen(self, select, i):
        pen = QtGui.QPen(self.color_list[1])
        pen2 = QtGui.QPen(self.color_list[2])

        pen.setWidth(max(1, int(round(self.image_max * i.width))))
        pen2.setWidth(max(1, int(round(self.image_max * i.width))))

        if select != self.selected_annotation_id:
            return pen
        else:
            return pen2

    def transformPos(self, point):
        return point / self.scale - self.offsetToCenterForNp()

    def append_position(self, point):
        self.annotation.positions.append(point)

    def insert_annotation(self, xy, width):
        self.selected_annotation_id = -1
        for n in xy:
            self.annotation.positions.append(copy.deepcopy(self.changeFromQPoint(n)))
        self.annotation.width = width
        self.annotation_list.append(self.annotation)
        self.annotation = InputAnnotation()
        self.repaint()

    def append_annotation(self, st):
        self.annotation.name = st
        co = copy.deepcopy(self.annotation)
        self.annotation_list.append(co)
        width = self.annotation.width
        self.annotation = InputAnnotation()
        self.annotation.width = width

    def delete_annotation(self):
        self.annotation_list[self.selected_annotation_id].positions.clear()
        self.annotation_list.pop(self.selected_annotation_id)
        self.selected_annotation_id = -1

    def select_annotation(self, x):
        self.selected_annotation_id = x

    def changeto01(self, point):
        return point / self.image_max

    def changefrom01(self, point):
        return point * self.image_max

    def calc_image_max(self, filepath):
        img = Image.open(filepath)
        width = img.width
        height = img.height
        self.image_max = max(width, height)

    def changeFromQPoint(self, q):
        n = re.findall(r"\d+", str(q))

        x = float(n[1] + "." + n[2])
        y = float(n[3] + "." + n[4])
        return np.array([x, y]).astype(np.float32)

    def changeToQPoint(self, numarray):
        # numarray -> QPointF
        return PyQt5.QtCore.QPointF(numarray[0], numarray[1])

    def paint_orientations(self, painter):
        p, u = interpolation.interpolate_vector_field_from_gui(self.annotation_list)
        pen = QtGui.QPen(self.color_list[4])

        pen.setWidth(max(1, int(round(self.image_max * 0.0025))))
        painter.setPen(pen)
        self.paint_vector(self.painter, p, u)

    def paint_vector(self, painter, p, u):

        v = self.calc_normal_vector(u)
        for i in range(0, len(p)):
            a = self.changefrom01(self.changeToQPoint(p[i]))
            b = self.changefrom01(self.changeToQPoint(p[i] + (self.vector_length * u[i])))
            half = p[i] + (self.vector_length * u[i] / self.triangle_length)
            painter.drawLine(a, b)
            self.paint_arrow(painter, b, v[i], half)
            i += 1

    def calc_normal_vector(self, u):
        transform = np.array([[0, -1], [1, 0]])
        v = []
        for vec in u:
            v.append(transform.dot(vec))
        return v

    def paint_arrow(self, painter, b, v, half):
        arrow1 = b
        arrow2 = self.changefrom01(self.changeToQPoint(half + (self.triangle_width * v)))
        arrow3 = self.changefrom01(self.changeToQPoint(half - (self.triangle_width * v)))
        painter.drawPolygon(arrow1, arrow2, arrow3)


class MainWindow(QMainWindow):
    """ Main window of annotation tool.
    """
    def __init__(self, exemplar_file=None):
        super().__init__()

        self.out_file = ""
        if exemplar_file is not None:
            out_dir = os.path.dirname(exemplar_file)
            self.out_file = os.path.join(out_dir, "annotation.json")

        self.annotation_count = 0
        self.colors = []
        self.image = QtGui.QImage()
        self.canvas = Canvas()
        self.zoomWidget = ZoomWidget()
        self.zoomWidget.valueChanged.connect(self.paintCanvas)
        self.canvas.double_clicked.connect(self.append_annotation)

        # Left blocks for tool buttons.
        self.box_child1 = QVBoxLayout()
        self.box_child1.setAlignment(Qt.AlignTop)

        self.open_image_button = self.make_button("Open\nImage", "icons/open_image.png", self.open_image_file)
        self.visible_annotation_button = self.make_button("Show\nAnnotation", "icons/view_line.png",
                                                          self.change_visible_annotation,
                                                          toggle_mode=True)
        self.visible_orientation_button = self.make_button("Show\nVectorField", "icons/view_vector.png",
                                                           self.change_visible_orientation,
                                                           toggle_mode=True)
        self.open_annotation_button = self.make_button("Open\nAnnotation", "icons/open_annotation.png",
                                                       self.open_annotation)
        self.save_annotation_button = self.make_button("Save\nAnnotation", "icons/save_image.png", self.save_annotation)

        self.box_child1.setSpacing(0)
        self.box_child1.setContentsMargins(0, 0, 0, 0)
        self.box_child1.addWidget(self.open_image_button)
        self.box_child1.addWidget(self.open_annotation_button)
        self.box_child1.addWidget(self.visible_annotation_button)
        self.box_child1.addWidget(self.visible_orientation_button)
        self.box_child1.addWidget(self.save_annotation_button)
        self.box_child1.addStretch()

        # -----------------------------------------------------------------------
        # Right blocks for width slider, color settings, annotation list widget.
        self.box_child2 = QVBoxLayout()
        self.width_slider = FloatSlider("Width", 0.01, 0.1, 0.03)
        self.width_slider.valueChanged.connect(self.slider_change)
        self.box_child2.addWidget(self.width_slider)

        self.slider_change()

        self.b2child_c1 = QHBoxLayout()
        self.b2child_c2 = QHBoxLayout()

        self.color1 = QPushButton()
        self.color1.clicked.connect(self.color_change1)
        self.color1.setIconSize(QSize(30, 30))

        self.color2 = QPushButton()
        self.color2.clicked.connect(self.color_change2)
        self.color2.setIconSize(QSize(30, 30))

        self.color3 = QPushButton()
        self.color3.clicked.connect(self.color_change3)
        self.color3.setIconSize(QSize(30, 30))

        self.color4 = QPushButton()
        self.color4.clicked.connect(self.color_change4)
        self.color4.setIconSize(QSize(30, 30))

        self.color5 = QPushButton()
        self.color5.clicked.connect(self.color_change5)
        self.color5.setIconSize(QSize(30, 30))

        self.b2child_c1.addWidget(self.color1)
        self.b2child_c1.addWidget(self.color2)
        self.b2child_c1.addWidget(self.color3)
        self.b2child_c1.addWidget(self.color4)
        self.b2child_c1.addWidget(self.color5)

        self.icon1 = QPushButton()
        self.icon1.clicked.connect(self.color_change1)
        self.icon1.setIcon(QIcon("icons/color1.png"))
        self.icon1.setIconSize(QSize(30, 30))

        self.icon2 = QPushButton()
        self.icon2.clicked.connect(self.color_change2)
        self.icon2.setIcon(QIcon("icons/color2.png"))
        self.icon2.setIconSize(QSize(30, 30))

        self.icon3 = QPushButton()
        self.icon3.clicked.connect(self.color_change3)
        self.icon3.setIcon(QIcon("icons/color3.png"))
        self.icon3.setIconSize(QSize(30, 30))

        self.icon4 = QPushButton()
        self.icon4.clicked.connect(self.color_change4)
        self.icon4.setIcon(QIcon("icons/color4.png"))
        self.icon4.setIconSize(QSize(30, 30))

        self.icon5 = QPushButton()
        self.icon5.clicked.connect(self.color_change5)
        self.icon5.setIcon(QIcon("icons/color5.png"))
        self.icon5.setIconSize(QSize(30, 30))

        self.b2child_c2.addWidget(self.icon1)
        self.b2child_c2.addWidget(self.icon2)
        self.b2child_c2.addWidget(self.icon3)
        self.b2child_c2.addWidget(self.icon4)
        self.b2child_c2.addWidget(self.icon5)

        self.box_child2.addLayout(self.b2child_c2)
        self.box_child2.addLayout(self.b2child_c1)

        self.annotation_list_ui = QListWidget()

        self.annotation_list_ui.clicked.connect(self.annotation_list_selected)
        self.box_child2.addWidget(self.annotation_list_ui)
        self.b2child = QHBoxLayout()

        self.add_button = QPushButton(self)
        self.add_button.clicked.connect(self.append_annotation)
        self.add_button.setIcon(QIcon("icons/add_button.png"))
        self.add_button.setIconSize(QSize(50, 50))
        self.add_button.setFlat(False)
        self.b2child.addWidget(self.add_button)

        self.delete_button = QPushButton(self)
        self.delete_button.clicked.connect(self.delete_annotation)
        self.delete_button.setIcon(QIcon("icons/delete_button.png"))
        self.delete_button.setIconSize(QSize(50, 50))
        self.delete_button.setFlat(False)
        self.b2child.addWidget(self.delete_button)

        self.box_child2.addLayout(self.b2child)
        # -----------------------------------------------------------------------
        # Layout settings.

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.width_slider.setFixedWidth(250)
        self.annotation_list_ui.setFixedWidth(250)
        self.annotation_list_ui.setStyleSheet(
            "QListWidget {padding-bottom: 3px; color : #333;  font-size: 18px; font-family: Arial;}")

        # ------------------------------------------------------------------------
        self.hbox = QHBoxLayout()
        self.box_child1.setSpacing(0)

        self.hbox.addLayout(self.box_child1)
        self.hbox.addWidget(self.canvas)
        self.hbox.addLayout(self.box_child2)
        self.hbox.setSpacing(0)
        self.hbox.setContentsMargins(0, 0, 0, 0)

        self.vbox = QVBoxLayout()
        self.vbox.addLayout(self.hbox)
        self.vbox.setSpacing(0)

        self.main_widget = QWidget()
        self.main_widget.setLayout(self.vbox)
        self.setCentralWidget(self.main_widget)

        # Short cut keys.
        self.open_image_sc = self.make_shortcut("Ctrl+I", self.open_image_file)
        self.open_annotation_sc = self.make_shortcut("Ctrl+O", self.open_annotation)
        self.open_save_sc = self.make_shortcut("Ctrl+S", self.save_annotation)
        self.enter_sc = self.make_shortcut("Return", self.append_annotation)
        self.delete_sc = self.make_shortcut("Delete", self.delete_annotation)
        self.quit_sc = self.make_shortcut("Ctrl+Q", QApplication.instance().quit)

        self.screen_shot_sc = self.make_shortcut("Ctrl+G", self.screen_shot_widgets)

        self.captured_widgets = {"open_image_button": self.open_image_button,
                                 "visible_annotation_button": self.visible_annotation_button,
                                 "visible_orientation_button": self.visible_orientation_button,
                                 "open_annotation_button": self.open_annotation_button,
                                 "save_annotation_button": self.save_annotation_button,
                                 "canvas": self.canvas,
                                 "width_slider": self.width_slider,
                                 "annotation_list": self.annotation_list_ui,
                                 "add_button": self.add_button,
                                 "delete_button": self.delete_button}

        # Window size settings.
        self.setGeometry(50, 50, 930, 550)
        self.setWindowTitle("Annotation Tool")
        self.show()
        self.load_color_setting_file()

        if exemplar_file is not None:
            self.load_image(exemplar_file)

    def make_button(self, tool_txt, tool_icon, tool_action, toggle_mode=False):
        button_group = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        tool_button = ToolButton(tool_icon, tool_txt, self, toggle_mode)
        tool_button.clicked.connect(tool_action)

        tool_button.setIconSize(QSize(64, 64))
        tool_button.setFixedSize(QSize(110, 110))
        tool_button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)

        layout.addWidget(tool_button)

        button_group.setLayout(layout)
        return button_group

    def open_annotation(self):
        filepath = QFileDialog.getOpenFileName(self, "open file", "", "Json (*.json)")[0]
        if filepath:
            self.load_annotation(filepath)

    def load_annotation(self, filepath):
        if not self.canvas.pixmap.isNull():
            self.canvas.annotation_list.clear()
            self.canvas.annotation = InputAnnotation()

            with open(filepath, "r") as f:
                read = f.read()
                json_load = json.loads(read)

            inputdata = []

            for list in json_load:
                for x, y in zip(list["x"], list["y"]):
                    inputdata.append(PyQt5.QtCore.QPointF(x, y))
                self.canvas.insert_annotation(copy.deepcopy(inputdata), list["width"])
                inputdata.clear()

            self.annotation_count = 0
            for ln in self.canvas.annotation_list:
                ln.name = str(self.annotation_count) + ":annotation"
                self.annotation_count += 1

            self.annotation_list_ui.clear()
            count = 0

            for i in self.canvas.annotation_list:
                self.annotation_list_ui.insertItem(count, i.name)
                count += 1

    def change_visible_annotation(self):
        if self.canvas.visible_annotation:
            self.canvas.visible_annotation = False
        else:
            self.canvas.visible_annotation = True
        self.canvas.repaint()

    def change_visible_orientation(self):
        if self.canvas.visible_orientation:
            self.canvas.visible_orientation = False
        else:
            self.canvas.visible_orientation = True
        self.canvas.repaint()

    def save_annotation(self):
        out_file = QFileDialog.getSaveFileName(self, "open file", self.out_file, "Json (*.json)")[0]
        if out_file:
            self.write_annotation(out_file)
            self.out_file = out_file

    def write_annotation(self, filepath):
        x = []
        y = []

        if self.canvas.annotation_list:
            for line in self.canvas.annotation_list:
                for node in line.positions:
                    x.append(float(node[0]))
                    y.append(float(node[1]))

                data = {"x": copy.deepcopy(x), "y": copy.deepcopy(y), "width": copy.deepcopy(line.width)}
                x.clear()
                y.clear()

                self.canvas.filedata.append(copy.deepcopy(data))
            with open(filepath, "w") as f:
                json.dump(self.canvas.filedata, f)

    def append_annotation(self):
        if len(self.canvas.annotation.positions) > 1:

            labelname = str(self.annotation_count) + ":annotation"
            self.annotation_count += 1
            self.canvas.append_annotation(labelname)

            self.annotation_list_ui.clear()
            count = 0

            for i in self.canvas.annotation_list:
                self.annotation_list_ui.insertItem(count, i.name)
                count += 1
            self.canvas.setMouseTracking(True)
            self.canvas.repaint()

    def delete_annotation(self):
        if self.canvas.selected_annotation_id >= 0:
            self.annotation_list_ui.takeItem(self.canvas.selected_annotation_id)
            self.canvas.delete_annotation()
            self.canvas.repaint()

    def slider_change(self):
        if self.canvas.selected_annotation_id >= 0:
            x = self.canvas.annotation_list[self.canvas.selected_annotation_id]
        else:
            x = self.canvas.annotation

        x.width = self.width_slider.value()
        self.canvas.repaint()

    def color_change1(self):
        self.color_change(0)

    def color_change2(self):
        self.color_change(1)

    def color_change3(self):
        self.color_change(2)

    def color_change4(self):
        self.color_change(3)

    def color_change5(self):
        self.color_change(4)

    def color_change(self, num):
        color_old = self.canvas.color_list[num]
        color = QColorDialog.getColor(color_old)
        if not color.isValid():
            return
        rgb = color.getRgb()
        self.button_update(num, rgb)
        self.update_color_setting_file()

    def button_update(self, num, rgb):
        # r, g, b = self.img.split()
        rc = rgb[0]
        gc = rgb[1]
        bc = rgb[2]
        alpha = 255

        if num != 4:
            alpha = 128

        self.canvas.color_list[num].setRgb(rc, gc, bc, alpha)

        if self.canvas.pixmap:
            self.canvas.repaint()

        buttoncolor = "background-color:rgb(" + str(rc) + "," + str(gc) + "," + str(bc) + ")"
        if num == 0:
            self.color1.setStyleSheet(buttoncolor)
        elif num == 1:
            self.color2.setStyleSheet(buttoncolor)
        elif num == 2:
            self.color3.setStyleSheet(buttoncolor)
        elif num == 3:
            self.color4.setStyleSheet(buttoncolor)
        else:
            self.color5.setStyleSheet(buttoncolor)

    def load_color_setting_file(self):
        with open("colors/colors.json", "r") as f:
            read = f.read()
            json_load = json.loads(read)
        for list in json_load:
            for c in list["color"]:
                self.colors.append(copy.deepcopy(c))
        for i in range(0, 5):
            self.update_color(i)

    def update_color(self, num):
        rgb = self.colors[num]
        self.button_update(num, rgb)

    def update_color_setting_file(self):
        output = []
        data = []
        for c in self.canvas.color_list:
            colors = copy.deepcopy(c.getRgb())
            output.append(copy.deepcopy(colors))
        d = {"color": output}
        data.append(d)

        with open("colors/colors.json", "w") as f:
            json.dump(data, f)

    def annotation_list_selected(self):
        self.canvas.select_annotation(self.annotation_list_ui.currentRow())
        x = self.canvas.annotation_list[self.canvas.selected_annotation_id]
        self.width_slider.set_value(x.width)
        self.canvas.repaint()

    def open_image_file(self):
        img_file = QFileDialog.getOpenFileName(self, "open file", "", "Images (*.jpeg *.jpg *.png *.bmp)")[0]
        self.load_image(img_file)

    def load_image(self, image_file):
        self.filepath = image_file
        if os.path.exists(self.filepath):
            self.canvas.load_image(self.filepath)
            self.paintCanvas()

    def make_shortcut(self, key, action):
        sc = QShortcut(QKeySequence(key), self)
        sc.activated.connect(action)
        return sc

    def screen_shot_widgets(self):
        def screen_shot_widget(widget, ui_name):
            ui_pixmap = widget.grab()
            ui_pixmap.save(f"screen_shots/{ui_name}.png", "png")

        for ui_name in self.captured_widgets.keys():
            widget = self.captured_widgets[ui_name]
            screen_shot_widget(widget, ui_name)

    def paintCanvas(self):
        self.canvas.scale = self.scaleFitWindow()
        self.canvas.update()

    def scaleFitWindow(self):
        e = 2.0
        w1 = self.canvas.width() - e
        h1 = self.canvas.height() - e
        a1 = w1 / h1

        w2 = self.canvas.pixmap.width()
        h2 = self.canvas.pixmap.height()
        a2 = w2 / h2

        return w1 / w2 if a2 >= a1 else h1 / h2

    def adjustScale(self):
        value = self.scaleFitWindow()
        value = int(100 * value)
        self.zoomWidget.setValue(value)

    def resizeEvent(self, event):
        if self.canvas and not self.canvas.pixmap.isNull():
            self.adjustScale()
        super(MainWindow, self)


def main(exemplar_file=None):
    app = QApplication(sys.argv)
    win = MainWindow(exemplar_file=exemplar_file)
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

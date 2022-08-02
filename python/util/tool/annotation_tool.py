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
from PyQt5.QtGui import QIcon, QImage, QPixmap, qGray
from PyQt5.QtWidgets import (QApplication, QFileDialog, QGridLayout,
                             QHBoxLayout, QLabel, QListWidget, QMainWindow,
                             QPushButton, QSizePolicy, QSlider, QSpinBox,
                             QToolButton, QVBoxLayout, QWidget)

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
    double_clicked = pyqtSignal()

    def __init__(self):
        super(Canvas, self).__init__()

        # 初期値
        self.annotation = InputAnnotation()
        self.painter = QtGui.QPainter()
        self.pixmap = QtGui.QPixmap()
        self.scale = 1.0
        self.rectangle_color = [QtGui.QColor(0, 255, 0, 128), QtGui.QColor(255, 0, 0, 128),
                                QtGui.QColor(0, 0, 255, 128), QtGui.QColor(0, 255, 255, 128),
                                QtGui.QColor(255, 255, 0, 200)]

        self.selected_annotation_id = -1
        self.annotation_list = []
        self.image_max = 1
        self.filedata = []
        self.trackingline = []
        self.visible_annotation = True
        self.visible_orientation = True

        # Visualization settings for interpolated orientations.
        self.vector_length = 0.016
        self.triangle_length = 2
        self.triangle_width = 0.005

        self.canvas_layout = QGridLayout()
        self.setLayout(self.canvas_layout)

        self.setMouseTracking(True)

    def openImage(self, filepath):
        img = QtGui.QImage()

        if not img.load(filepath):
            return False

        # QImage -> QPixmap
        self.pixmap = QtGui.QPixmap.fromImage(img)
        self.calcimage(filepath)

    def paintEvent(self, event):
        if not self.pixmap:
            return super(Canvas, self).paintEvent(event)

        p = self.painter

        p.begin(self)

        p.scale(self.scale, self.scale)

        p.translate(self.offsetToCenterForQt())

        p.drawPixmap(0, 0, self.pixmap)

        if len(self.annotation.positions) > 1:
            self.paint(p)

        if len(self.trackingline) > 1:
            self.trackingpaint(p)
            self.trackingline.clear()

        if self.visible_orientation:
            if (len(self.annotation_list) > 0):
                self.callinterpolation(p)

        if self.visible_annotation:
            self.paintallline(p)

        p.end()

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
                        self.add2nodes(pos)
                else:
                    self.add2nodes(pos)
                self.repaint()

        if event.button() == QtCore.Qt.RightButton:
            if self.hasMouseTracking():
                self.setMouseTracking(False)
                self.trackingline.clear()
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
            self.trackingline.append(self.annotation.positions[-1])
            self.trackingline.append(pos)
            self.repaint()

    def paint(self, painter):
        pen = QtGui.QPen(self.rectangle_color[0])
        pen.setWidth(max(1, int(round(self.image_max * self.annotation.width))))
        painter.setPen(pen)
        if self.annotation.positions:
            self.paintline(self.annotation.positions, painter)

    def paintallline(self, painter):
        x = 0
        if self.annotation_list:
            for l in self.annotation_list:
                painter.setPen(self.setPen(x, l))
                x += 1
                self.paintline(l.positions, painter)

    def paintline(self, list, painter):
        drawlist = []

        for l in list:
            drawlist.append(self.changeToQPoint(l))
        startpoint = drawlist[0]
        for ln in range(1, len(drawlist)):
            painter.drawLine(self.changefrom01(startpoint), self.changefrom01(drawlist[ln]))
            startpoint = drawlist[ln]

    def trackingpaint(self, painter):
        pen = QtGui.QPen(self.rectangle_color[3])

        pen.setWidth(max(1, int(round(self.image_max * self.annotation.width))))
        painter.setPen(pen)
        self.paintline(self.trackingline, painter)

    def setPen(self, select, i):
        pen = QtGui.QPen(self.rectangle_color[1])  # 折れ線全体の色
        pen2 = QtGui.QPen(self.rectangle_color[2])  # 選択した線の色

        pen.setWidth(max(1, int(round(self.image_max * i.width))))
        pen2.setWidth(max(1, int(round(self.image_max * i.width))))

        if select != self.selected_annotation_id:
            return pen
        else:
            return pen2

    def transformPos(self, point):
        return point / self.scale - self.offsetToCenterForNp()

    def add2nodes(self, point):
        self.annotation.positions.append(point)

    def insertAnnotation(self, xy, width):
        self.selected_annotation_id = -1
        for n in xy:
            self.annotation.positions.append(copy.deepcopy(self.changeFromQPoint(n)))
        self.annotation.width = width
        self.annotation_list.append(self.annotation)
        self.annotation = InputAnnotation()
        self.repaint()

    def saveline(self, st):
        self.annotation.name = st
        co = copy.deepcopy(self.annotation)
        self.annotation_list.append(co)
        width = self.annotation.width
        self.annotation = InputAnnotation()
        self.annotation.width = width

    def deleteline(self):
        self.annotation_list[self.selected_annotation_id].positions.clear()
        self.annotation_list.pop(self.selected_annotation_id)
        self.selected_annotation_id = -1

    def selectLine(self, x):
        self.selected_annotation_id = x

    def changeto01(self, point):
        return point / self.image_max

    def changefrom01(self, point):
        return point * self.image_max

    def calcimage(self, filepath):
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

    def callinterpolation(self, painter):
        p, u = interpolation.interpolate_vector_field_from_lines(self.annotation_list)
        pen = QtGui.QPen(self.rectangle_color[4])

        pen.setWidth(max(1, int(round(self.image_max * 0.0025))))
        painter.setPen(pen)
        self.paint_vec(self.painter, p, u)

    def paint_vec(self, painter, p, u):

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

    def __init__(self, exemplar_file=None, out_file=None):
        super().__init__()

        self.out_file = out_file

        self.annotation_count = 0
        self.colors = []
        self.image = QtGui.QImage()
        self.canvas = Canvas()
        self.zoomWidget = ZoomWidget()
        self.zoomWidget.valueChanged.connect(self.paintCanvas)
        self.canvas.double_clicked.connect(self.addline)

        # Left blocks for tool buttons.
        self.box_child1 = QVBoxLayout()
        self.box_child1.setAlignment(Qt.AlignTop)

        self.open_image = self.make_button("Open\nImage", "icons/open_image.png", self.openFile)
        self.view_line = self.make_button("Show\nAnnotation", "icons/view_line.png", self.changevisible,
                                          toggle_mode=True)
        self.view_line_vec = self.make_button("Show\nVectorField", "icons/view_vector.png", self.changevisible_vec,
                                              toggle_mode=True)
        self.open_annotation = self.make_button("Open\nAnnotation", "icons/open_annotation.png", self.openAnnotation)
        self.save_button = self.make_button("Save\nAnnotation", "icons/save_image.png", self.saveAnnotation)

        self.box_child1.setSpacing(0)
        self.box_child1.setContentsMargins(0, 0, 0, 0)
        self.box_child1.addWidget(self.open_image)
        self.box_child1.addWidget(self.open_annotation)
        self.box_child1.addWidget(self.view_line)
        self.box_child1.addWidget(self.view_line_vec)
        self.box_child1.addWidget(self.save_button)
        self.box_child1.addStretch()

        # -----------------------------------------------------------------------
        # Right blocks for width slider, color settings, annotation list widget.
        self.box_child2 = QVBoxLayout()
        self.slider = FloatSlider("Width", 0.01, 0.1, 0.03)
        self.slider.valueChanged.connect(self.slider_change)
        self.box_child2.addWidget(self.slider)

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

        self.listwidget = QListWidget()

        self.listwidget.clicked.connect(self.annotationlist)
        self.box_child2.addWidget(self.listwidget)
        self.b2child = QHBoxLayout()

        self.b2add = QPushButton(self)
        self.b2add.clicked.connect(self.addline)
        self.b2add.setIcon(QIcon("icons/add_button.png"))
        self.b2add.setIconSize(QSize(50, 50))
        self.b2add.setFlat(False)
        self.b2child.addWidget(self.b2add)

        self.b2del = QPushButton(self)
        self.b2del.clicked.connect(self.deleteline)
        self.b2del.setIcon(QIcon("icons/delete_button.png"))
        self.b2del.setIconSize(QSize(50, 50))
        self.b2del.setFlat(False)
        self.b2child.addWidget(self.b2del)

        self.box_child2.addLayout(self.b2child)
        # -----------------------------------------------------------------------
        # Layout settings.

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.slider.setFixedWidth(250)
        self.listwidget.setFixedWidth(250)
        self.listwidget.setStyleSheet(
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

        # Window size settings.
        self.setGeometry(50, 50, 930, 550)
        self.setWindowTitle('Annotation Tool')
        self.show()
        self.readColorFile()

        if exemplar_file is not None:
            self.openImage(exemplar_file)

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

    def openAnnotation(self):
        filepath = QFileDialog.getOpenFileName(self, 'open file', '', 'Json (*.json)')[0]
        if filepath:
            self.readAnnotation(filepath)

    def readAnnotation(self, filepath):
        if not self.canvas.pixmap.isNull():
            self.canvas.annotation_list.clear()
            self.canvas.annotation = InputAnnotation()

            with open(filepath, 'r') as f:
                read = f.read()
                json_load = json.loads(read)

            insertlistx = []
            insertlisty = []
            inputdata = []

            for list in json_load:
                for x in list["x"]:
                    insertlistx.append(copy.deepcopy(x))
                for y in list["y"]:
                    insertlisty.append(copy.deepcopy(y))
                for i in range(0, len(insertlistx)):
                    inputdata.append(PyQt5.QtCore.QPointF(copy.deepcopy(insertlistx[i]), copy.deepcopy(insertlisty[i])))
                self.canvas.insertAnnotation(copy.deepcopy(inputdata), list["width"])
                inputdata.clear()
                insertlistx.clear()
                insertlisty.clear()

            self.annotation_count = 0
            for ln in self.canvas.annotation_list:
                ln.name = str(self.annotation_count) + ":annotation"
                self.annotation_count += 1

            self.listwidget.clear()
            count = 0

            for i in self.canvas.annotation_list:
                self.listwidget.insertItem(count, i.name)
                count += 1

    def changevisible(self):
        if self.canvas.visible_annotation:
            self.canvas.visible_annotation = False
        else:
            self.canvas.visible_annotation = True
        self.canvas.repaint()

    def changevisible_vec(self):
        if self.canvas.visible_orientation:
            self.canvas.visible_orientation = False
        else:
            self.canvas.visible_orientation = True
        self.canvas.repaint()

    def saveAnnotation(self):
        filepath = QFileDialog.getSaveFileName(self, 'open file', '', 'Json (*.json)')[0]
        if filepath:
            self.change2json(filepath)

    def change2json(self, filepath):
        x = []
        y = []

        if self.canvas.annotation_list:
            for line in self.canvas.annotation_list:
                for node in line.positions:
                    x.append(float(node[0]))
                    y.append(float(node[1]))

                data = {'x': copy.deepcopy(x), 'y': copy.deepcopy(y), 'width': copy.deepcopy(line.width)}
                x.clear()
                y.clear()
                # Canvasに保存
                self.canvas.filedata.append(copy.deepcopy(data))
            self.saveToFile(filepath)

    def saveToFile(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.canvas.filedata, f)
        print("save")

    def addline(self):
        if len(self.canvas.annotation.positions) > 1:

            labelname = str(self.annotation_count) + ":annotation"
            self.annotation_count += 1
            self.canvas.saveline(labelname)

            self.listwidget.clear()
            count = 0

            for i in self.canvas.annotation_list:
                self.listwidget.insertItem(count, i.name)
                count += 1
            self.canvas.setMouseTracking(True)
            self.canvas.repaint()

    def deleteline(self):
        if self.canvas.selected_annotation_id >= 0:
            self.listwidget.takeItem(self.canvas.selected_annotation_id)
            self.canvas.deleteline()
            self.canvas.repaint()

    def slider_change(self):
        if self.canvas.selected_annotation_id >= 0:
            x = self.canvas.annotation_list[self.canvas.selected_annotation_id]
        else:
            x = self.canvas.annotation

        x.width = self.slider.value()
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
        color = QtWidgets.QColorDialog.getColor()
        rgb = color.getRgb()
        self.button_update(num, rgb)
        self.appdateColorFile()

    def button_update(self, num, rgb):

        # r, g, b = self.img.split()
        rc = rgb[0]
        gc = rgb[1]
        bc = rgb[2]
        alpha = 255

        if num != 4:
            alpha = 128

        self.canvas.rectangle_color[num].setRgb(rc, gc, bc, alpha)

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

    def readColorFile(self):
        with open("colors/colors.json", 'r') as f:
            read = f.read()
            json_load = json.loads(read)
        for list in json_load:
            for c in list["color"]:
                self.colors.append(copy.deepcopy(c))
        for i in range(0, 5):
            self.updateColor(i)

    def updateColor(self, num):
        rgb = self.colors[num]
        self.button_update(num, rgb)

    def appdateColorFile(self):
        output = []
        data = []
        for c in self.canvas.rectangle_color:
            colors = copy.deepcopy(c.getRgb())
            output.append(copy.deepcopy(colors))
        d = {'color': output}
        data.append(d)

        with open("colors/colors.json", 'w') as f:
            json.dump(data, f)
        print("color appdate")

    def annotationlist(self):
        self.canvas.selectLine(self.listwidget.currentRow())
        x = self.canvas.annotation_list[self.canvas.selected_annotation_id]
        self.slider.set_value(x.width)
        self.canvas.repaint()

    def openFile(self):
        img_file = QFileDialog.getOpenFileName(self, 'open file', '', 'Images (*.jpeg *.jpg *.png *.bmp)')[0]
        self.openImage(img_file)

    def openImage(self, image_file):
        self.filepath = image_file
        if os.path.exists(self.filepath):
            self.canvas.openImage(self.filepath)
            self.paintCanvas()

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


def main(exemplar_file=None, out_file=None):
    app = QApplication(sys.argv)
    win = MainWindow(exemplar_file=exemplar_file, out_file=out_file)
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

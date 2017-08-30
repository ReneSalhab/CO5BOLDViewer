# -*- coding: utf-8 -*-
"""
Created on Apr 29 19:06 2017

:author: René Georg Salhab
"""

import time
import math
import bisect
import numpy as np
import numexpr as ne
from collections import OrderedDict
from scipy import interpolate as ip
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSlot

import matplotlib.pyplot as plt
import matplotlib.colors as cl
import matplotlib.colorbar as clbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import mdis
import subclasses as sc


class BasicWindow(QtWidgets.QMainWindow):
    def __init__(self):
        self.version = "0.9.5"
        super(BasicWindow, self).__init__()

        self.centralWidget = QtWidgets.QWidget(self)

        QtWidgets.QToolTip.setFont(QtGui.QFont('SansSerif', 10))

        self.initializeParams()
        self.setGridLayout()
        self.statusBar().showMessage("ready")

        self.show()

    def initializeParams(self):

        # --- Initial time-index ---

        self.timind = 0

        # --- Timeline place-holder ---

        self.time = np.zeros((2, 3))

        # --- position in cube ---

        self.x1ind = 0
        self.x2ind = 0
        self.x3ind = 0

        # --- tau-limits ---

        self.minTau = None
        self.maxTau = None
        self.numTau = None
        self.tauRange = None

        # --- Axes of plot ---

        self.xc1 = None
        self.xc2 = None
        self.xc3 = None

        # --- Data-array for plotting ---

        self.data = None

        # --- Arbitrary parameters ---

        self.boxind = -1

        self.direction = 0
        self.dim = 0
        self.sameNorm = False

        self.minNorm = np.finfo(np.float32).min
        self.maxNorm = np.finfo(np.float32).max

        # --- functions for post-processing data ---

        self.postfunc = {'----': "data", '| |': "abs(data)", 'log10': "log10(data)", 'log10(| |)': "log10(abs(data))"}

        # --- Available Colormaps (handle invert versions with checkbox) ---

        self.cmaps = [c for c in plt.colormaps() if not c.endswith('_r')]
        self.cmaps.sort()

        # --- Message Box ---

        self.msgBox = QtWidgets.QMessageBox()

        # --- eos- and opta-file-control variables ---

        self.par = False
        self.eos = False
        self.opa = False
        self.modelfile = None

        self.eosname = False
        self.opaname = False

        # --- plot-control ---

        self.plot = False
        self.tauheight = None  # 2-tuple. First element is domain of 1D-plot, second element is height of tau=1-surface

    def setGridLayout(self):
        # --- Main Layout with splitter ---
        # --- (enables automatic resizing of widgets when window is resized)

        maingrid = QtWidgets.QHBoxLayout(self.centralWidget)

        # --- Splitter for dynamic seperation of control elements section and plot-box

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        maingrid.addWidget(self.splitter)

        # --- Layout with control elements ---

        self.controlgrid = QtWidgets.QVBoxLayout()

        # --- Widget consisting of layout for control elements ---
        # --- (enables adding of control-elements-layout to splitter)

        controlwid = QtWidgets.QWidget(self.centralWidget)
        controlwid.setLayout(self.controlgrid)

        # ---------------------------------------------------------------------
        # ----------------- Groupbox with time components ---------------------
        # ---------------------------------------------------------------------

        timeGroup = QtWidgets.QGroupBox("Time parameters", self.centralWidget)
        timeLayout = QtWidgets.QGridLayout(timeGroup)
        timeGroup.setLayout(timeLayout)

        # --- Sliders and buttons for time selection ---

        self.timeSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self.centralWidget)
        self.timeSlider.setMinimum(0)
        self.timeSlider.setMaximum(100)
        self.timeSlider.setDisabled(True)
        self.timeSlider.valueChanged.connect(self.SliderChange)
        self.timeSlider.setObjectName("time-Slider")

        self.prevTimeBtn = QtWidgets.QPushButton("Prev")
        self.prevTimeBtn.setDisabled(True)
        self.prevTimeBtn.clicked.connect(self.timeBtnClick)
        self.prevTimeBtn.setObjectName("prev-time-Button")

        self.nextTimeBtn = QtWidgets.QPushButton("Next")
        self.nextTimeBtn.setDisabled(True)
        self.nextTimeBtn.clicked.connect(self.timeBtnClick)
        self.nextTimeBtn.setObjectName("next-time-Button")

        # --- Label for time-slider

        timeTitle = QtWidgets.QLabel("Time step:")

        self.currentTimeEdit = QtWidgets.QLineEdit(str(self.timeSlider.value()))
        self.currentTimeEdit.setAlignment(QtCore.Qt.AlignCenter)
        self.currentTimeEdit.setMaximumWidth(40)
        self.currentTimeEdit.setMinimumWidth(40)
        self.currentTimeEdit.textChanged.connect(self.currentEditChange)
        self.currentTimeEdit.setObjectName("current-time-Edit")

        currentTimeTitle = QtWidgets.QLabel("t [s]:")

        self.actualTimeLabel = QtWidgets.QLabel(str(self.timeSlider.value()))
        self.actualTimeLabel.setMaximumWidth(200)
        self.actualTimeLabel.setMinimumWidth(200)
        self.actualTimeLabel.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.actualTimeLabel.setObjectName("actual-time-Label")

        currentFileTitle = QtWidgets.QLabel("File:")

        self.currentFileLabel = QtWidgets.QLabel("")
        self.currentFileLabel.setMaximumWidth(200)
        self.currentFileLabel.setMinimumWidth(200)
        self.currentFileLabel.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.currentFileLabel.setObjectName("current-file-Label")

        timeLayout.addWidget(timeTitle, 0, 0)
        timeLayout.addWidget(self.timeSlider, 0, 1, 1, 2)
        timeLayout.addWidget(self.currentTimeEdit, 0, 3)
        timeLayout.addWidget(currentTimeTitle, 0, 4)
        timeLayout.addWidget(self.actualTimeLabel, 0, 5)

        timeLayout.addWidget(self.prevTimeBtn, 1, 1)
        timeLayout.addWidget(self.nextTimeBtn, 1, 2)
        timeLayout.addWidget(currentFileTitle, 1, 4)
        timeLayout.addWidget(self.currentFileLabel, 1, 5)

        # ---------------------------------------------------------------------
        # ---------------- Groupbox with position components ------------------
        # ---------------------------------------------------------------------

        posGroup = QtWidgets.QGroupBox("Position parameters", self.centralWidget)
        posLayout = QtWidgets.QGridLayout(posGroup)
        posGroup.setLayout(posLayout)

        # --- ComboBox for projection plane selection ---

        planeLabel = QtWidgets.QLabel("Projection plane:")
        self.planeCombo = QtWidgets.QComboBox(self.centralWidget)
        self.planeCombo.clear()
        self.planeCombo.setDisabled(True)
        self.planeCombo.activated.connect(self.planeCheck)
        self.planeCombo.setObjectName("plane-Combo")
        self.planeCombo.addItems(["xy", "xz", "yz"])

        # --- Cross-hair activation components ---

        crossLabel = QtWidgets.QLabel("cross-hair:")
        self.crossCheck = QtWidgets.QCheckBox(self.centralWidget)
        self.crossCheck.setDisabled(True)

        # --- Sliders for spatial directions ---

        self.x1Slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self.centralWidget)
        self.x1Slider.setMinimum(0)
        self.x1Slider.setMaximum(100)
        self.x1Slider.setDisabled(True)
        self.x1Slider.valueChanged.connect(self.SliderChange)
        self.x1Slider.setObjectName("x1-Slider")

        self.x2Slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self.centralWidget)
        self.x2Slider.setMinimum(0)
        self.x2Slider.setMaximum(100)
        self.x2Slider.setDisabled(True)
        self.x2Slider.valueChanged.connect(self.SliderChange)
        self.x2Slider.setObjectName("x2-Slider")

        self.x3Slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self.centralWidget)
        self.x3Slider.setMinimum(0)
        self.x3Slider.setMaximum(100)
        self.x3Slider.setDisabled(True)
        self.x3Slider.valueChanged.connect(self.SliderChange)
        self.x3Slider.setObjectName("x3-Slider")

        # --- Labels for sliders of spatial directions ---

        x1SliderTitle = QtWidgets.QLabel("x-position:")
        currentX1Title = QtWidgets.QLabel("ix:")
        actualX1Title = QtWidgets.QLabel("x [km]:")

        self.currentX1Edit = QtWidgets.QLineEdit(str(self.x1Slider.value()))
        self.currentX1Edit.setMaximumWidth(40)
        self.currentX1Edit.setMinimumWidth(40)
        self.currentX1Edit.textChanged.connect(self.currentEditChange)
        self.currentX1Edit.setObjectName("current-x-Edit")

        self.actualX1Label = QtWidgets.QLabel(str(0))
        self.actualX1Label.setAlignment(QtCore.Qt.AlignCenter)
        self.actualX1Label.setMaximumWidth(55)
        self.actualX1Label.setMinimumWidth(55)

        x2SliderTitle = QtWidgets.QLabel("y-position:")
        currentX2Title = QtWidgets.QLabel("iy:")
        actualX2Title = QtWidgets.QLabel("y [km]:")

        self.currentX2Edit = QtWidgets.QLineEdit(str(self.x2Slider.value()))
        self.currentX2Edit.setMaximumWidth(40)
        self.currentX2Edit.setMinimumWidth(40)
        self.currentX2Edit.textChanged.connect(self.currentEditChange)
        self.currentX2Edit.setObjectName("current-y-Edit")

        self.actualX2Label = QtWidgets.QLabel(str(0))
        self.actualX2Label.setAlignment(QtCore.Qt.AlignCenter)
        self.actualX2Label.setMaximumWidth(55)
        self.actualX2Label.setMinimumWidth(55)

        self.x3Title = QtWidgets.QLabel("z-position:")
        self.currentX3Title = QtWidgets.QLabel("iz:")
        self.actualX3Title = QtWidgets.QLabel("z [km]:")
        self.x3Combo = QtWidgets.QComboBox(self.centralWidget)
        self.x3Combo.addItems(["z-position:", u"\u03C4-position:"])
        self.x3Combo.activated.connect(self.x3ComboChange)
        self.x3Combo.hide()

        self.currentX3Edit = QtWidgets.QLineEdit(str(self.x3Slider.value()))
        self.currentX3Edit.setMaximumWidth(40)
        self.currentX3Edit.setMinimumWidth(40)
        self.currentX3Edit.textChanged.connect(self.currentEditChange)
        self.currentX3Edit.setObjectName("current-z-Edit")

        self.actualX3Label = QtWidgets.QLabel(str(0))
        self.actualX3Label.setAlignment(QtCore.Qt.AlignCenter)
        self.actualX3Label.setMaximumWidth(55)
        self.actualX3Label.setMinimumWidth(55)

        # --- Box regarding optical height ---
        # ------------------------------------

        self.tauGroup = QtWidgets.QGroupBox("Optical height parameters", posGroup)
        tauLayout = QtWidgets.QHBoxLayout(self.tauGroup)
        self.tauGroup.setLayout(tauLayout)
        self.tauGroup.hide()

        self.minTauLabel = QtWidgets.QLabel(u"Min. log10(\u03C4):")
        self.minTauLabel.setAlignment(QtCore.Qt.AlignHCenter)

        self.numTauLabel = QtWidgets.QLabel(u"number of elements:")
        self.numTauLabel.setAlignment(QtCore.Qt.AlignHCenter)

        self.maxTauLabel = QtWidgets.QLabel(u"Max. log10(\u03C4):")
        self.maxTauLabel.setAlignment(QtCore.Qt.AlignHCenter)

        self.minTauEdit = QtWidgets.QLineEdit("-4")
        self.minTauEdit.setMaximumWidth(40)
        self.minTauEdit.setMinimumWidth(40)
        self.minTauEdit.textChanged.connect(self.tauRangeChange)
        self.minTauEdit.setObjectName("min-tau-Edit")
        self.minTauEdit.setAlignment(QtCore.Qt.AlignRight)

        self.numTauEdit = QtWidgets.QLineEdit("150")
        self.numTauEdit.setMaximumWidth(40)
        self.numTauEdit.setMinimumWidth(40)
        self.numTauEdit.textChanged.connect(self.tauRangeChange)
        self.numTauEdit.setObjectName("num-tau-Edit")
        self.numTauEdit.setAlignment(QtCore.Qt.AlignRight)

        self.maxTauEdit = QtWidgets.QLineEdit("5")
        self.maxTauEdit.setMaximumWidth(40)
        self.maxTauEdit.setMinimumWidth(40)
        self.maxTauEdit.textChanged.connect(self.tauRangeChange)
        self.maxTauEdit.setObjectName("max-tau-Edit")
        self.maxTauEdit.setAlignment(QtCore.Qt.AlignRight)

        # --- Buttons ---

        self.minTauMBtn = QtWidgets.QPushButton("-1")
        self.minTauMBtn.setMaximumWidth(18)
        self.minTauMBtn.setMinimumWidth(18)
        self.minTauMBtn.setMaximumHeight(20)
        self.minTauMBtn.setMinimumHeight(20)
        self.minTauMBtn.clicked.connect(self.tauBtnClick)
        self.minTauMBtn.setObjectName("minus-min-tau-Button")

        self.minTauPBtn = QtWidgets.QPushButton("+1")
        self.minTauPBtn.setMaximumWidth(18)
        self.minTauPBtn.setMinimumWidth(18)
        self.minTauPBtn.setMaximumWidth(20)
        self.minTauPBtn.setMinimumWidth(20)
        self.minTauPBtn.clicked.connect(self.tauBtnClick)
        self.minTauPBtn.setObjectName("plus-min-tau-Button")

        self.numTauTMBtn = QtWidgets.QPushButton("-10")
        self.numTauTMBtn.setMaximumWidth(20)
        self.numTauTMBtn.setMinimumWidth(20)
        self.numTauTMBtn.setMaximumWidth(25)
        self.numTauTMBtn.setMinimumWidth(25)
        self.numTauTMBtn.clicked.connect(self.tauBtnClick)
        self.numTauTMBtn.setObjectName("minus10-num-tau-Button")

        self.numTauMBtn = QtWidgets.QPushButton("-1")
        self.numTauMBtn.setMaximumWidth(18)
        self.numTauMBtn.setMinimumWidth(18)
        self.numTauMBtn.setMaximumWidth(20)
        self.numTauMBtn.setMinimumWidth(20)
        self.numTauMBtn.clicked.connect(self.tauBtnClick)
        self.numTauMBtn.setObjectName("minus-num-tau-Button")

        self.numTauTPBtn = QtWidgets.QPushButton("+10")
        self.numTauTPBtn.setMaximumWidth(20)
        self.numTauTPBtn.setMinimumWidth(20)
        self.numTauTPBtn.setMaximumWidth(25)
        self.numTauTPBtn.setMinimumWidth(25)
        self.numTauTPBtn.clicked.connect(self.tauBtnClick)
        self.numTauTPBtn.setObjectName("plus10-num-tau-Button")

        self.numTauPBtn = QtWidgets.QPushButton("+1")
        self.numTauPBtn.setMaximumWidth(18)
        self.numTauPBtn.setMinimumWidth(18)
        self.numTauPBtn.setMaximumWidth(20)
        self.numTauPBtn.setMinimumWidth(20)
        self.numTauPBtn.clicked.connect(self.tauBtnClick)
        self.numTauPBtn.setObjectName("plus-num-tau-Button")

        self.maxTauMBtn = QtWidgets.QPushButton("-1")
        self.maxTauMBtn.setMaximumWidth(18)
        self.maxTauMBtn.setMinimumWidth(18)
        self.maxTauMBtn.setMaximumWidth(20)
        self.maxTauMBtn.setMinimumWidth(20)
        self.maxTauMBtn.clicked.connect(self.tauBtnClick)
        self.maxTauMBtn.setObjectName("minus-max-tau-Button")

        self.maxTauPBtn = QtWidgets.QPushButton("+1")
        self.maxTauPBtn.setMaximumWidth(18)
        self.maxTauPBtn.setMinimumWidth(18)
        self.maxTauPBtn.setMaximumWidth(20)
        self.maxTauPBtn.setMinimumWidth(20)
        self.maxTauPBtn.clicked.connect(self.tauBtnClick)
        self.maxTauPBtn.setObjectName("plus-max-tau-Button")

        # --- Tau box configuration ---

        minTauGroup = QtWidgets.QGroupBox("", self.tauGroup)
        minTauGroup.setStyleSheet("QGroupBox {border: 0px;}")
        minTauLayout = QtWidgets.QGridLayout(minTauGroup)
        minTauLayout.addWidget(self.minTauLabel, 0, 0, 1, 5)
        minTauLayout.addWidget(QtWidgets.QWidget(), 1, 0)
        minTauLayout.addWidget(self.minTauMBtn, 1, 2)
        minTauLayout.addWidget(self.minTauEdit, 1, 3)
        minTauLayout.addWidget(self.minTauPBtn, 1, 4)
        minTauLayout.addWidget(QtWidgets.QWidget(), 1, 5)
        tauLayout.addWidget(minTauGroup)

        numTauGroup = QtWidgets.QGroupBox("", self.tauGroup)
        numTauGroup.setStyleSheet("QGroupBox {border: 0px;}")
        numTauLayout = QtWidgets.QGridLayout(numTauGroup)
        numTauLayout.addWidget(self.numTauLabel, 0, 0, 1, 5)
        numTauLayout.addWidget(self.numTauTMBtn, 1, 0)
        numTauLayout.addWidget(self.numTauMBtn, 1, 1)
        numTauLayout.addWidget(self.numTauEdit, 1, 2)
        numTauLayout.addWidget(self.numTauPBtn, 1, 3)
        numTauLayout.addWidget(self.numTauTPBtn, 1, 4)
        tauLayout.addWidget(numTauGroup)

        maxTauGroup = QtWidgets.QGroupBox("", self.tauGroup)
        maxTauGroup.setStyleSheet("QGroupBox {border: 0px;}")
        maxTauLayout = QtWidgets.QGridLayout(maxTauGroup)
        maxTauLayout.addWidget(self.maxTauLabel, 0, 0, 1, 5)
        maxTauLayout.addWidget(QtWidgets.QWidget(), 1, 0)
        maxTauLayout.addWidget(self.maxTauMBtn, 1, 1)
        maxTauLayout.addWidget(self.maxTauEdit, 1, 2)
        maxTauLayout.addWidget(self.maxTauPBtn, 1, 3)
        maxTauLayout.addWidget(QtWidgets.QWidget(), 1, 4)
        tauLayout.addWidget(maxTauGroup)

        # --- Position box configuration ---

        posLayout.addWidget(planeLabel, 0, 0)
        posLayout.addWidget(self.planeCombo, 0, 1)

        posLayout.addWidget(crossLabel, 0, 2, 1, 2)
        posLayout.addWidget(self.crossCheck, 0, 4)

        posLayout.addWidget(x1SliderTitle, 1, 0)
        posLayout.addWidget(self.x1Slider, 1, 1)
        posLayout.addWidget(currentX1Title, 1, 2)
        posLayout.addWidget(self.currentX1Edit, 1, 3)
        posLayout.addWidget(actualX1Title, 1, 4)
        posLayout.addWidget(self.actualX1Label, 1, 5)

        posLayout.addWidget(x2SliderTitle, 2, 0)
        posLayout.addWidget(self.x2Slider, 2, 1)
        posLayout.addWidget(currentX2Title, 2, 2)
        posLayout.addWidget(self.currentX2Edit, 2, 3)
        posLayout.addWidget(actualX2Title, 2, 4)
        posLayout.addWidget(self.actualX2Label, 2, 5)

        posLayout.addWidget(self.x3Title, 3, 0)
        posLayout.addWidget(self.x3Combo, 3, 0)
        posLayout.addWidget(self.x3Slider, 3, 1)
        posLayout.addWidget(self.currentX3Title, 3, 2)
        posLayout.addWidget(self.currentX3Edit, 3, 3)
        posLayout.addWidget(self.actualX3Title, 3, 4)
        posLayout.addWidget(self.actualX3Label, 3, 5)

        posLayout.addWidget(self.tauGroup, 4, 0, 1, 6)

        # ---------------------------------------------------------------------
        # -------------- Groupbox with data specific widgets ------------------
        # ---------------------------------------------------------------------

        dataParamsGroup = QtWidgets.QGroupBox("Data specific and presentation parameters", self.centralWidget)
        dataParamsLayout = QtWidgets.QGridLayout(dataParamsGroup)
        dataParamsGroup.setLayout(dataParamsLayout)

        # --- ComboBox for quantity selection ---

        self.quantityCombo = QtWidgets.QComboBox(self.centralWidget)
        self.quantityCombo.clear()
        self.quantityCombo.setDisabled(True)
        self.quantityCombo.setObjectName("quantity-Combo")
        self.quantityCombo.activated.connect(self.quantityChange)

        quantityLabel = QtWidgets.QLabel("Quantity:")

        # --- ComboBox for colormap selection ---

        self.cmCombo = QtWidgets.QComboBox(self.centralWidget)
        self.cmCombo.clear()
        self.cmCombo.setDisabled(True)
        self.cmCombo.activated.connect(self.invertCM)
        self.cmCombo.addItems(self.cmaps)

        # --- default colormap (not available in older matplotlib versions)

        cmi = self.cmaps.index("inferno")

        if cmi < 0:
            self.cmCombo.setCurrentIndex(self.cmaps.index("jet"))
        else:
            self.cmCombo.setCurrentIndex(cmi)
        self.cmCombo.inv = ""
        self.cmCombo.currentCmap = self.cmCombo.currentText() + self.cmCombo.inv
        self.cmCombo.setObjectName("colormap-Combo")

        self.cmInvert = QtWidgets.QCheckBox("Invert CM")
        self.cmInvert.setDisabled(True)
        self.cmInvert.stateChanged.connect(self.invertCM)

        # --- Colorbar ---

        colorfig = plt.figure()

        self.colorcanvas = FigureCanvas(colorfig)
        self.colorcanvas.setMinimumHeight(20)
        self.colorcanvas.setMaximumHeight(20)

        colorax = colorfig.add_axes([0, 0, 1, 1])
        norm = cl.Normalize(0, 1)
        self.colorbar = clbar.ColorbarBase(colorax, orientation="horizontal", norm=norm)
        self.colorbar.set_ticks([0])

        colorbarLabel = QtWidgets.QLabel("Data range:")

        # --- Normalization parameter widgets ---

        self.normCheck = QtWidgets.QCheckBox("Normalize over time")
        self.normCheck.stateChanged.connect(self.normCheckChange)
        self.normCheck.setDisabled(True)

        normMinTitle = QtWidgets.QLabel("Min:")
        self.normMinEdit = QtWidgets.QLineEdit("{dat:13.4g}".format(dat=0))
        self.normMinEdit.setDisabled(True)
        self.normMinEdit.textChanged.connect(self.normChange)
        self.normMinEdit.setObjectName("norm-min-Edit")

        normMaxTitle = QtWidgets.QLabel("Max:")
        self.normMaxEdit = QtWidgets.QLineEdit("{dat:13.4g}".format(dat=100))
        self.normMaxEdit.setDisabled(True)
        self.normMaxEdit.textChanged.connect(self.normChange)
        self.normMaxEdit.setObjectName("norm-max-Edit")

        normMeanTitle = QtWidgets.QLabel("Mean:")
        self.normMeanLabel = QtWidgets.QLabel("{dat:13.4g}".format(dat=50))
        self.normMeanLabel.setDisabled(True)

        unitTitle = QtWidgets.QLabel("Unit:")
        self.unitLabel = QtWidgets.QLabel("")

        # --- ComboBox with math-selection ---

        self.funcCombo = QtWidgets.QComboBox(self.centralWidget)
        self.funcCombo.clear()
        self.funcCombo.setDisabled(True)
        self.funcCombo.activated.connect(self.funcComboChange)
        self.funcCombo.addItems(self.postfunc.keys())
        self.funcCombo.setCurrentText("----")
        self.funcCombo.setObjectName("math-Combo")

        # --- Radiobuttons for 2D-3D-selection ---

        oneDTitle = QtWidgets.QLabel("1D:")
        self.oneDRadio = QtWidgets.QRadioButton(self.centralWidget)
        self.oneDRadio.setChecked(True)
        self.oneDRadio.setDisabled(True)
        self.oneDRadio.setObjectName("1DRadio")
        self.oneDRadio.toggled.connect(self.plotDimensionChange)

        twoDTitle = QtWidgets.QLabel("2D:")
        self.twoDRadio = QtWidgets.QRadioButton(self.centralWidget)
        self.twoDRadio.setChecked(True)
        self.twoDRadio.setDisabled(True)
        self.twoDRadio.setObjectName("2DRadio")
        self.twoDRadio.toggled.connect(self.plotDimensionChange)

        threeDTitle = QtWidgets.QLabel("3D:")
        threeDTitle.setDisabled(True)
        self.threeDRadio = QtWidgets.QRadioButton(self.centralWidget)
        self.threeDRadio.setDisabled(True)
        self.threeDRadio.setObjectName("3DRadio")
        self.threeDRadio.toggled.connect(self.plotDimensionChange)

        tauUnityTitle = QtWidgets.QLabel("tau=1:")
        self.tauUnityCheck = QtWidgets.QCheckBox(self.centralWidget)
        self.tauUnityCheck.setDisabled(True)
        self.tauUnityCheck.setObjectName("tauUnityCheck")
        self.tauUnityCheck.stateChanged.connect(self.tauUnityChange)

        keepPlotRangeTitle = QtWidgets.QLabel("keep plot-range:")
        self.keepPlotRangeCheck = QtWidgets.QCheckBox(self.centralWidget)
        self.keepPlotRangeCheck.setDisabled(True)
        self.keepPlotRangeCheck.setObjectName("keepPlotRangeCheck")

        # --- Setup of data-presentation-layout ---

        dataParamsLayout.addWidget(quantityLabel, 0, 0)
        dataParamsLayout.addWidget(self.quantityCombo, 0, 1, 1, 3)
        dataParamsLayout.addWidget(self.normCheck, 0, 4, 1, 2)
        dataParamsLayout.addWidget(self.funcCombo, 0, 6, 1, 1)

        dataParamsLayout.addWidget(colorbarLabel, 1, 0)
        dataParamsLayout.addWidget(self.colorcanvas, 1, 1, 1, 4)
        dataParamsLayout.addWidget(self.cmCombo, 1, 5)
        dataParamsLayout.addWidget(self.cmInvert, 1, 6)

        dataParamsLayout.addWidget(normMinTitle, 2, 1)
        dataParamsLayout.addWidget(normMeanTitle, 2, 2)
        dataParamsLayout.addWidget(normMaxTitle, 2, 4)
        dataParamsLayout.addWidget(unitTitle, 2, 5)

        dataParamsLayout.addWidget(self.normMinEdit, 3, 1)
        dataParamsLayout.addWidget(self.normMeanLabel, 3, 2, 1, 2)
        dataParamsLayout.addWidget(self.normMaxEdit, 3, 4)
        dataParamsLayout.addWidget(self.unitLabel, 3, 5)

        dataParamsLayout.addWidget(oneDTitle, 4, 0)
        dataParamsLayout.addWidget(self.oneDRadio, 4, 1)

        dataParamsLayout.addWidget(twoDTitle, 4, 2)
        dataParamsLayout.addWidget(self.twoDRadio, 4, 3)

        dataParamsLayout.addWidget(threeDTitle, 4, 4)
        dataParamsLayout.addWidget(self.threeDRadio, 4, 5)

        dataParamsLayout.addWidget(tauUnityTitle, 5, 0)
        dataParamsLayout.addWidget(self.tauUnityCheck, 5, 1)

        dataParamsLayout.addWidget(keepPlotRangeTitle, 5, 2)
        dataParamsLayout.addWidget(self.keepPlotRangeCheck, 5, 3)

        # ---------------------------------------------------------------------
        # ------------- Groupbox with vector plot parameters ------------------
        # ---------------------------------------------------------------------

        vectorPlotGroup = QtWidgets.QGroupBox("Vector plot parameters", self.centralWidget)
        vectorPlotLayout = QtWidgets.QGridLayout(vectorPlotGroup)
        vectorPlotGroup.setLayout(vectorPlotLayout)

        # --- Checkbox for activation of vector-plot ---

        vpLabel = QtWidgets.QLabel("Vector-plot:")

        self.vpCheck = QtWidgets.QCheckBox(self.centralWidget)
        self.vpCheck.setObjectName("vp-Check")
        self.vpCheck.setDisabled(True)
        self.vpCheck.stateChanged.connect(self.vectorSetup)

        # --- Radiobuttons for vector-field selection ---

        vpVelLabel = QtWidgets.QLabel("Velocity field")

        self.vpVelRadio = QtWidgets.QRadioButton(self.centralWidget)
        self.vpVelRadio.setObjectName("vp-velRadio")
        self.vpVelRadio.setChecked(True)
        self.vpVelRadio.setDisabled(True)
        self.vpVelRadio.toggled.connect(self.vectorSetup)

        vpMagLabel = QtWidgets.QLabel("Magnetic field")

        self.vpMagRadio = QtWidgets.QRadioButton(self.centralWidget)
        self.vpMagRadio.setObjectName("vp-magRadio")
        self.vpMagRadio.setDisabled(True)
        self.vpMagRadio.toggled.connect(self.vectorSetup)

        vpScaleLabel = QtWidgets.QLabel("Scale:")
        self.vpScaleEdit = QtWidgets.QLineEdit("{dat:5.2g}".format(dat=1.e-7))
        self.vpScaleEdit.setMinimumWidth(55)
        self.vpScaleEdit.setMaximumWidth(55)
        self.vpScaleEdit.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.vpScaleEdit.setDisabled(True)
        self.vpScaleEdit.textChanged.connect(self.generalPlotRoutine)

        vpXIncLabel = QtWidgets.QLabel("x-increment:")
        self.vpXIncEdit = QtWidgets.QLineEdit("{dat}".format(dat=4))
        self.vpXIncEdit.setMinimumWidth(55)
        self.vpXIncEdit.setMaximumWidth(55)
        self.vpXIncEdit.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.vpXIncEdit.setDisabled(True)
        self.vpXIncEdit.textChanged.connect(self.generalPlotRoutine)

        vpYIncLabel = QtWidgets.QLabel("y-increment:")
        self.vpYIncEdit = QtWidgets.QLineEdit("{dat}".format(dat=4))
        self.vpYIncEdit.setMinimumWidth(55)
        self.vpYIncEdit.setMaximumWidth(55)
        self.vpYIncEdit.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.vpYIncEdit.setDisabled(True)
        self.vpYIncEdit.textChanged.connect(self.generalPlotRoutine)

        vpAlphaLabel = QtWidgets.QLabel("Vector-opacity:")
        self.vpAlphaEdit = QtWidgets.QLineEdit("{dat}".format(dat=1))
        self.vpAlphaEdit.setMinimumWidth(55)
        self.vpAlphaEdit.setMaximumWidth(55)
        self.vpAlphaEdit.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.vpAlphaEdit.setDisabled(True)
        self.vpAlphaEdit.textChanged.connect(self.generalPlotRoutine)

        # --- Setup of vector-plot-layout ---

        vectorPlotLayout.addWidget(vpLabel, 0, 0)
        vectorPlotLayout.addWidget(self.vpCheck, 0, 1)
        vectorPlotLayout.addWidget(QtWidgets.QLabel("\t\t\t"), 0, 2)
        vectorPlotLayout.addWidget(vpScaleLabel, 0, 3)
        vectorPlotLayout.addWidget(self.vpScaleEdit, 0, 4)

        vectorPlotLayout.addWidget(vpVelLabel, 1, 0)
        vectorPlotLayout.addWidget(self.vpVelRadio, 1, 1)
        vectorPlotLayout.addWidget(vpXIncLabel, 1, 3)
        vectorPlotLayout.addWidget(self.vpXIncEdit, 1, 4)

        vectorPlotLayout.addWidget(vpMagLabel, 2, 0)
        vectorPlotLayout.addWidget(self.vpMagRadio, 2, 1)
        vectorPlotLayout.addWidget(vpYIncLabel, 2, 3)
        vectorPlotLayout.addWidget(self.vpYIncEdit, 2, 4)

        vectorPlotLayout.addWidget(vpAlphaLabel, 3, 3)
        vectorPlotLayout.addWidget(self.vpAlphaEdit, 3, 4)

        # --- Add layout with control-elements

        self.splitter.addWidget(controlwid)

        # --- Fill up left layout with groups ---

        self.controlgrid.addWidget(timeGroup)
        self.controlgrid.addWidget(posGroup)
        self.controlgrid.addWidget(dataParamsGroup)
        self.controlgrid.addWidget(vectorPlotGroup)

        self.centralWidget.setLayout(maingrid)
        self.setCentralWidget(self.centralWidget)

    def initialLoad(self):
        start = time.time()

        # --- Initiate axes and cell-sizes ---

        if self.fileType == "cobold" or self.fileType == "mean":
            self.xc1 = self.modelfile[0].dataset[0].box[0]["xc1"].data.squeeze()*1.e-5
            self.xc2 = self.modelfile[0].dataset[0].box[0]["xc2"].data.squeeze()*1.e-5
            self.xc3 = self.modelfile[0].dataset[0].box[0]["xc3"].data.squeeze()*1.e-5

            self.xb1 = self.modelfile[0].dataset[0].box[0]["xb1"].data.squeeze()*1.e-5
            self.xb2 = self.modelfile[0].dataset[0].box[0]["xb2"].data.squeeze()*1.e-5
            self.xb3 = self.modelfile[0].dataset[0].box[0]["xb3"].data.squeeze()*1.e-5

            self.x3Title.hide()
            if not self.opa or not self.eos:
                self.x3Combo.setCurrentIndex(0)
                self.x3Combo.setDisabled(True)
            elif self.opa and self.eos and self.x3Combo.currentIndex() == 1:
                rho = self.modelfile[self.modelind].dataset[self.dsind].box[0]["rho"].data
                ei = self.modelfile[self.modelind].dataset[self.dsind].box[0]["ei"].data

                P, T = self.Eos.PandT(rho, ei)

                if self.par and 'c_radhtautop' in self.parFile.keys():
                    tau = self.Opa.tau(rho, axis=0, T=T, P=P, zb=self.xb3*1.e5,
                                       radhtautop=self.parFile['c_radhtautop'].data)
                else:
                    tau = self.Opa.tau(rho, axis=0, T=T, P=P, zb=self.xb3*1.e5)
                self.minTauEdit.setText(str(tau[:-1].min()))
                self.maxTauEdit.setText(str(tau[:-1].max()))
            if self.x3Combo.currentIndex() == 0:
                self.currentX3Title.setText("iz:")
                self.actualX3Title.setText("z [km]:")
            else:
                self.currentX3Title.setText(u"i\u03C4:")
                self.actualX3Title.setText(u"\u03C4        :")
            self.x3Combo.show()

            self.dx = np.diff(self.xb1).mean()
            self.dy = np.diff(self.xb2).mean()
            self.dz = np.diff(self.xb3).mean()

            # --- check if grid is equi-distant ---

            self.constGrid = np.diff(self.xb3).std() < 0.01

            # --- initiate time-array ---

            if len(self.modelfile):
                self.time = []
                for i in range(len(self.modelfile)):
                    for j in range(len(self.modelfile[i].dataset)):
                        self.time.append([self.modelfile[i].dataset[j]["modeltime"].data, i, j])
        else:
            self.x3Combo.hide()
            self.x3Title.show()
            if self.fileType == "profile":
                self.x3Title.setText(u"\u03BB-position:")
                self.currentX3Title.setText(u"i\u03BB:")
                self.actualX3Title.setText(u"\u03BB     :")
                shape = np.array(self.modelfile[0]['I'].shape)
            elif self.fileType == "nicole":
                self.x3Title.setText("z-position:")
                self.currentX3Title.setText("iz:")
                self.actualX3Title.setText("z [km]:")
                shape = np.array(self.modelfile[0]['tau'].shape)
            self.xc1 = np.arange(0, shape[0])
            self.xc2 = np.arange(0, shape[1])
            self.xc3 = np.arange(0, shape[2])

            self.xb1 = np.arange(0, shape[0] + 1)
            self.xb2 = np.arange(0, shape[1] + 1)
            self.xb3 = np.arange(0, shape[2] + 1)

            self.dx = 1
            self.dy = 1
            self.dz = 1

            self.constGrid = True

            if len(self.modelfile):
                self.time = []
                for i in range(len(self.modelfile)):
                    self.time.append([i, i, 0])

        self.time = np.array(self.time)

        # --- determine axis-boundaries ---

        self.timlen = len(self.time[:, 0])

        self.x1min = self.xc1.min()
        self.x1max = self.xc1.max()

        self.x2min = self.xc2.min()
        self.x2max = self.xc2.max()

        self.x3min = self.xc3.min()
        self.x3max = self.xc3.max()

        self.timemin = self.time[:, 0].min()
        self.timemax = self.time[:, 0].max()

        self.boxind = -1

        for i, quantity in enumerate(self.quantityList):
            if self.quantityCombo.currentText() in quantity.keys():
                self.boxind = i
                break

        self.typeind = self.quantityList[self.boxind][self.quantityCombo.currentText()]

        # determine slider boundaries and de-/activate temporal elements

        if self.timlen > 1:
            self.timeSlider.setDisabled(False)
            self.timeSlider.setMaximum(self.timlen-1)

            self.prevTimeBtn.setDisabled(False)
            self.nextTimeBtn.setDisabled(False)
        else:
            self.timeSlider.setDisabled(True)
            self.timeSlider.setMaximum(self.timlen-1)

            self.prevTimeBtn.setDisabled(True)
            self.nextTimeBtn.setDisabled(True)

        if self.timind >= self.timeSlider.maximum():
            self.timind = self.timeSlider.maximum() - 1

        self.x1Slider.setMaximum(len(self.xc1)-1)
        self.x2Slider.setMaximum(len(self.xc2)-1)
        self.x3Slider.setMaximum(len(self.xc3)-1)

        self.normMinEdit.setDisabled(False)
        self.normMaxEdit.setDisabled(False)

        # de-/activate vector-plot elements

        if self.fileType == "cobold" or self.fileType == "nicole":
            self.vpAlphaEdit.setDisabled(False)
            self.vpCheck.setDisabled(False)
            self.vpMagRadio.setDisabled(False)
            self.vpScaleEdit.setDisabled(False)
            self.vpVelRadio.setDisabled(False)
            self.vpXIncEdit.setDisabled(False)
            self.vpYIncEdit.setDisabled(False)
        else:
            self.vpAlphaEdit.setDisabled(True)
            self.vpCheck.setDisabled(True)
            self.vpMagRadio.setDisabled(True)
            self.vpScaleEdit.setDisabled(True)
            self.vpVelRadio.setDisabled(True)
            self.vpXIncEdit.setDisabled(True)
            self.vpYIncEdit.setDisabled(True)

        self.funcCombo.setDisabled(False)
        self.crossCheck.setDisabled(False)

        # --------------------------------------
        # ---- update parameters of widgets ----
        # --------------------------------------

        self.currentTimeEdit.setText(str(self.timind).rjust(4))

        self.modelind = 0
        self.dsind = 0

        self.plot = False

        self.actualTimeLabel.setText("{dat:10.1f}".format(dat=self.time[self.timind, 0]))

        self.currentFileLabel.setText(self.fname[self.modelind].split("/")[-1])

        self.x1ind = self.x1Slider.value()
        self.currentX1Edit.setText(str(self.x1ind).rjust(10))
        self.actualX1Label.setText("{:13.1f}".format(self.xc1[self.x1ind]).rjust(13))

        self.x2ind = self.x2Slider.value()
        self.currentX2Edit.setText(str(self.x2ind).rjust(10))
        self.actualX2Label.setText("{:13.1f}".format(self.xc2[self.x2ind]).rjust(13))

        self.x3ind = self.x3Slider.value()
        self.currentX3Edit.setText(str(self.x3ind).rjust(10))
        self.actualX3Label.setText("{:13.1f}".format(self.xc3[self.x3ind]).rjust(13))

        self.colorbar.set_cmap(self.cmCombo.currentCmap)
        self.colorbar.draw_all()
        self.colorcanvas.draw()

        if self.fileType == "cobold" or self.fileType == "mean":
            shape = np.array(self.modelfile[self.modelind].dataset[self.dsind].box[self.boxind][self.typeind].data.shape)
        else:
            self.tauUnityCheck.setDisabled(True)
        self.dim = shape.squeeze().size

        # in dependency of the data´s dimension activate, or de-activate the different GUI-elements
        if self.dim == 3:
            self.planeCombo.setDisabled(False)
            self.cmCombo.setDisabled(False)

            self.x1Slider.setDisabled(False)
            self.x2Slider.setDisabled(False)
            self.x3Slider.setDisabled(False)

            self.currentX1Edit.setDisabled(False)
            self.currentX2Edit.setDisabled(False)
            self.currentX3Edit.setDisabled(False)

            self.actualX1Label.setDisabled(False)
            self.actualX2Label.setDisabled(False)
            self.actualX3Label.setDisabled(False)

            self.oneDRadio.setDisabled(False)
            self.twoDRadio.setDisabled(False)
            # self.threeDRadio.setDisabled(False)

            if self.opa and self.eos:
                self.tauUnityCheck.setDisabled(False)
        if self.dim == 2:
            self.planeCombo.setDisabled(True)
            self.cmCombo.setDisabled(False)

            self.x1Slider.setDisabled(True)
            self.x2Slider.setDisabled(True)
            self.x3Slider.setDisabled(True)

            self.currentX1Edit.setDisabled(True)
            self.currentX2Edit.setDisabled(True)
            self.currentX3Edit.setDisabled(True)

            self.actualX1Label.setDisabled(True)
            self.actualX2Label.setDisabled(True)
            self.actualX3Label.setDisabled(True)

            self.oneDRadio.setDisabled(True)
            self.twoDRadio.setDisabled(True)
            self.threeDRadio.setDisabled(True)

            self.tauUnityCheck.setDisabled(True)

            self.direction = np.where(shape == 1)[0]
            if self.direction == 0:
                self.x3ind = 0
            elif self.direction == 1:
                self.x2ind = 0
            elif self.direction == 2:
                self.x1ind == 0
        elif self.dim == 1:
            self.planeCombo.setDisabled(True)
            self.cmCombo.setDisabled(True)

            self.x1Slider.setDisabled(True)
            self.x2Slider.setDisabled(True)
            self.x3Slider.setDisabled(True)

            self.currentX1Edit.setDisabled(True)
            self.currentX2Edit.setDisabled(True)
            self.currentX3Edit.setDisabled(True)

            self.actualX1Label.setDisabled(True)
            self.actualX2Label.setDisabled(True)
            self.actualX3Label.setDisabled(True)

            self.oneDRadio.setDisabled(True)
            self.twoDRadio.setDisabled(True)
            self.threeDRadio.setDisabled(True)

            self.tauUnityCheck.setDisabled(True)

            self.direction = bisect.bisect(shape, 2)
            if self.direction == 0:
                self.x1ind = 0
                self.x2ind = 0
            elif self.direction == 1:
                self.x1ind = 0
                self.x3ind = 0
            elif self.direction == 2:
                self.x2ind = 0
                self.x3ind = 0
        self.normCheck.setDisabled(False)

        if self.opa and self.eos and self.x3Combo.currentIndex() == 1:
            rho = self.modelfile[self.modelind].dataset[self.dsind].box[0]["rho"].data
            ei = self.modelfile[self.modelind].dataset[self.dsind].box[0]["ei"].data
            P, T = self.Eos.PandT(rho, ei)

            if self.par and 'c_radhtautop' in self.parFile.keys():
                tau = self.Opa.tau(rho, axis=0, T=T, P=P, zb=self.xb3 * 1.e5,
                                   radhtautop=self.parFile['c_radhtautop'].data)
            else:
                tau = self.Opa.tau(rho, axis=0, T=T, P=P, zb=self.xb3 * 1.e5)
            self.data = self.setPlotData(self.modelind, self.dsind, tau=tau)
        else:
            self.data = self.setPlotData(self.modelind, self.dsind)

        self.unitLabel.setText(self.unit)
        if self.normCheck.checkState() == QtCore.Qt.Checked:
            self.getTotalMinMax()

        self.cmInvert.setDisabled(False)
        self.keepPlotRangeCheck.setDisabled(False)
        self.quantityCombo.setDisabled(False)

        if self.cmInvert.checkState() == QtCore.Qt.Checked:
            self.invertCM(True)
        else:
            self.invertCM(False)

        self.planeCheck()
        print("Time needed for initial load:", time.time()-start)

    # -------------
    # --- Slots ---
    # -------------

    @pyqtSlot()
    def tauUnityChange(self):
        if self.tauUnityCheck.isChecked():
            if self.eos and self.opa:
                rho = self.modelfile[self.modelind].dataset[self.dsind].box[0]["rho"].data
                ei = self.modelfile[self.modelind].dataset[self.dsind].box[0]["ei"].data
                P, T = self.Eos.PandT(rho, ei)

                if self.par and 'c_radhtautop' in self.parFile.keys():
                    tau = self.Opa.tau(rho, axis=0, T=T, P=P, zb=self.xb3*1.e5,
                                       radhtautop=self.parFile['c_radhtautop'].data)
                else:
                    tau = self.Opa.tau(rho, axis=0, T=T, P=P, zb=self.xb3 * 1.e5)
                self.tauheight = self.Opa.height(self.xc3, 1.0, axis=0, tau=tau).T
            else:
                self.msgBox.setText("EOS or opacity-file not loaded!\nEOS: {0}\topacity: {1}".format(self.eos, self.opa))
                pass
        else:
            self.tauheight = None
        self.planeCheck()

    @pyqtSlot()
    def tauRangeChange(self):
        sender = self.sender()
        try:
            self.minTau = float(self.minTauEdit.text())
            self.maxTau = float(self.maxTauEdit.text())
            self.numTau = int(self.numTauEdit.text())
            if self.numTau < 1:
                self.numTau = 1
                self.numTauEdit.setText(str(self.numTau))

            if sender.objectName() == "num-tau-Edit":
                if self.x3Slider.value() >= self.numTau:
                    self.x3Slider.setValue(self.numTau - 1)
                self.x3Slider.setMaximum(self.numTau - 1)
        except ValueError:
            print("{0} of {1} is an invalid input.".format(sender.text(), sender.objectName()))
            pass

        if self.minTau >= self.maxTau:
            print("Min. tau has to be smaller than max. tau.")
            pass

        if self.plot:
            self.tauRange = np.logspace(self.minTau, self.maxTau, self.numTau)[::-1]
            self.data = self.setPlotData(self.modelind, self.dsind)
            self.planeCheck()

    @pyqtSlot()
    def getTotalMinMax(self):
        if self.dim == 3:
            self.globBound = []

            xmin, ymin, zmin, xmax, ymax, zmax = ([] for _ in range(6))

            if self.fileType == "cobold" or self.fileType == "mean":
                for i, mod in enumerate(self.modelfile):
                    for j, dat in enumerate(mod.dataset):
                        data = self.setPlotData(i, j)
                        xmin.append(data.min(axis=(0, 1)))
                        xmax.append(data.max(axis=(0, 1)))

                        ymin.append(data.min(axis=(0, 2)))
                        ymax.append(data.max(axis=(0, 2)))

                        zmin.append(data.min(axis=(1, 2)))
                        zmax.append(data.max(axis=(1, 2)))
            else:
                for i, mod in enumerate(self.modelfile):
                    data = self.setPlotData(i, 0)
                    xmin.append(data.min(axis=(0, 1)))
                    xmax.append(data.max(axis=(0, 1)))

                    ymin.append(data.min(axis=(0, 2)))
                    ymax.append(data.max(axis=(0, 2)))

                    zmin.append(data.min(axis=(1, 2)))
                    zmax.append(data.max(axis=(1, 2)))

            self.globBound.append([np.array(zmin).min(axis=0), np.array(zmax).max(axis=0)])
            self.globBound.append([np.array(ymin).min(axis=0), np.array(ymax).max(axis=0)])
            self.globBound.append([np.array(xmin).min(axis=0), np.array(xmax).max(axis=0)])
        else:
            min = []
            max = []
            if self.fileType == "cobold" or self.fileType == "mean":
                for i, mod in enumerate(self.modelfile):
                    for j, dat in enumerate(mod.dataset):
                        data = self.setPlotData(i, j)
                        min.append(data.min())
                        max.append(data.max())
            else:
                for i, mod in enumerate(self.modelfile):
                    data = self.setPlotData(i, 0)
                    min.append(data.min())
                    max.append(data.max())
            self.globBound = [min, max]

    @pyqtSlot()
    def normCheckChange(self, state):
        if state == QtCore.Qt.Checked:
            self.getTotalMinMax()

    @pyqtSlot()
    def invertCM(self, state):
        if state == QtCore.Qt.Checked:
            self.cmCombo.inv = "_r"

        else:
            self.cmCombo.inv = ""
        self.cmCombo.currentCmap = self.cmCombo.currentText() + self.cmCombo.inv
        self.plotBox.colorChange(self.cmCombo.currentCmap)
        self.colorbar.set_cmap(self.cmCombo.currentCmap)

        self.colorbar.draw_all()
        self.colorcanvas.draw()

    def setPlotData(self, mod, dat, tau=None):
        self.statusBar().showMessage("Initialize arrays...")
        start = time.time()
        clight = 2.998e10
        const = 4.0 * np.pi

        ver = np.version.version
        # self.constGrid = False

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

        if self.fileType == "cobold":
            if self.quantityCombo.currentText() == "Velocity, horizontal":
                v1 = self.modelfile[mod].dataset[dat].box[0]["v1"].data
                v2 = self.modelfile[mod].dataset[dat].box[0]["v2"].data

                data = ne.evaluate("sqrt(v1**2+v2**2)")
                self.unit = "cm/s"
            elif  self.quantityCombo.currentText() == "Velocity, absolute":
                v1 = self.modelfile[mod].dataset[dat].box[0]["v1"].data
                v2 = self.modelfile[mod].dataset[dat].box[0]["v2"].data
                v3 = self.modelfile[mod].dataset[dat].box[0]["v3"].data

                data = ne.evaluate("sqrt(v1**2+v2**2+v3**2)")
                self.unit = "cm/s"
            elif self.quantityCombo.currentText() == "Kinetic energy":
                v1 = self.modelfile[mod].dataset[dat].box[0]["v1"].data
                v2 = self.modelfile[mod].dataset[dat].box[0]["v2"].data
                v3 = self.modelfile[mod].dataset[dat].box[0]["v3"].data
                rho = self.modelfile[mod].dataset[dat].box[0]["rho"].data

                data = ne.evaluate("0.5*rho*(v1**2+v2**2+v3**2)")
                self.unit = "erg/cm^3"
            elif self.quantityCombo.currentText() == "Momentum":
                v1 = self.modelfile[mod].dataset[dat].box[0]["v1"].data
                v2 = self.modelfile[mod].dataset[dat].box[0]["v2"].data
                v3 = self.modelfile[mod].dataset[dat].box[0]["v3"].data
                rho = self.modelfile[mod].dataset[dat].box[0]["rho"].data

                data = ne.evaluate("rho*sqrt(v1**2+v2**2+v3**2)")
                self.unit = "g/(cm^2 * s)"
            elif self.quantityCombo.currentText() == "Vert. mass flux (Rho*V3)":
                v3 = self.modelfile[mod].dataset[dat].box[0]["v3"].data
                rho = self.modelfile[mod].dataset[dat].box[0]["rho"].data

                data = ne.evaluate("rho*v3")
                self.unit = "g/(cm^2 * s)"
            elif self.quantityCombo.currentText() == "Magnetic field Bx":
                bb1 = self.modelfile[mod].dataset[dat].box[0]["bb1"].data

                data = ip.interp1d(self.xb1, bb1, copy=False, assume_sorted=True)(self.xc1)*math.sqrt(const)
                self.unit = "G"
            elif self.quantityCombo.currentText() == "Magnetic field By":
                bb2 = self.modelfile[mod].dataset[dat].box[0]["bb2"].data

                data = ip.interp1d(self.xb2, bb2, axis=1, copy=False, assume_sorted=True)(self.xc2)*math.sqrt(const)
                self.unit = "G"

            elif self.quantityCombo.currentText() == "Magnetic field Bz":
                bb3 = self.modelfile[mod].dataset[dat].box[0]["bb3"].data

                data = ip.interp1d(self.xb3, bb3, axis=0, copy=False, assume_sorted=True)(self.xc3)*math.sqrt(const)
                self.unit = "G"
            elif self.quantityCombo.currentText() == "Magnetic field Bh (horizontal)":
                bb1 = self.modelfile[mod].dataset[dat].box[0]["bb1"].data
                bb2 = self.modelfile[mod].dataset[dat].box[0]["bb2"].data

                bc1 = ip.interp1d(self.xb1, bb1, copy=False, assume_sorted=True)(self.xc1)
                bc2 = ip.interp1d(self.xb2, bb2, axis=1, copy=False, assume_sorted=True)(self.xc2)

                data = ne.evaluate("sqrt((bc1**2.0+bc2**2.0)*const)")
                self.unit = "G"
            elif self.quantityCombo.currentText() == "Magnetic f.abs.|B|, unsigned":
                bb1 = self.modelfile[mod].dataset[dat].box[0]["bb1"].data
                bb2 = self.modelfile[mod].dataset[dat].box[0]["bb2"].data
                bb3 = self.modelfile[mod].dataset[dat].box[0]["bb3"].data

                bc1 = ip.interp1d(self.xb1, bb1, copy=False, assume_sorted=True)(self.xc1)
                bc2 = ip.interp1d(self.xb2, bb2, axis=1, copy=False, assume_sorted=True)(self.xc2)
                bc3 = ip.interp1d(self.xb3, bb3, axis=0, copy=False, assume_sorted=True)(self.xc3)

                data = ne.evaluate("sqrt((bc1*bc1+bc2*bc2+bc3*bc3)*const)")
                self.unit = "G"
            elif self.quantityCombo.currentText() == "Magnetic field B^2, signed":
                bb1 = self.modelfile[mod].dataset[dat].box[0]["bb1"].data
                bb2 = self.modelfile[mod].dataset[dat].box[0]["bb2"].data
                bb3 = self.modelfile[mod].dataset[dat].box[0]["bb3"].data

                sn = np.ones((bb3.shape[0]-1, bb3.shape[1], bb3.shape[2]))
                sm = np.zeros(sn.shape)
                sm.fill(-1.0)

                bc1 = ip.interp1d(self.xb1, bb1, copy=False, assume_sorted=True)(self.xc1)

                sn = np.where(bc1 < 0.0, -1.0, sn)
                data = ne.evaluate("sn*bc1**2")

                bc2 = ip.interp1d(self.xb2, bb2, axis=1, copy=False, assume_sorted=True)(self.xc2)

                sn.fill(1.0)
                sn = np.where(bc2 < 0.0, -1.0, sn)
                data += ne.evaluate("sn*bc2**2")

                bc3 = ip.interp1d(self.xb3, bb3, axis=0, copy=False, assume_sorted=True)(self.xc3)

                sn.fill(1.0)
                sn = np.where(bc3 < 0.0, -1.0, sn)
                data += ne.evaluate("sn*bc3**2")

                data *= const
                self.unit = "G^2"
            elif self.quantityCombo.currentText() == "Vert. magnetic flux Bz*Az":
                A = np.diff(self.xb1) * np.diff(self.xb2)
                bb3 = self.modelfile[mod].dataset[dat].box[0]["bb3"].data

                data = ip.interp1d(self.xb3, bb3, axis=0, copy=False, assume_sorted=True)(self.xc3)*A*math.sqrt(const)
                self.unit = "G*km^2"
            elif self.quantityCombo.currentText() == "Vert. magnetic gradient Bz/dz":
                bb3 = self.modelfile[mod].dataset[dat].box[0]["bb3"].data
                dz = np.diff(self.xb3)

                data = math.sqrt(const) * np.diff(bb3, axis=0) / dz[:, np.newaxis, np.newaxis]
                self.unit = "G/km"
            elif self.quantityCombo.currentText() == "Magnetic energy":
                bb1 = self.modelfile[mod].dataset[dat].box[0]["bb1"].data
                bb2 = self.modelfile[mod].dataset[dat].box[0]["bb2"].data
                bb3 = self.modelfile[mod].dataset[dat].box[0]["bb3"].data

                bc1 = ip.interp1d(self.xb1, bb1, copy=False, assume_sorted=True)(self.xc1)
                bc2 = ip.interp1d(self.xb2, bb2, axis=1, copy=False, assume_sorted=True)(self.xc2)
                bc3 = ip.interp1d(self.xb3, bb3, axis=0, copy=False, assume_sorted=True)(self.xc3)

                data = ne.evaluate("(bc1**2+bc2**2+bc3**2)/2")
                self.unit = "G^2"
            elif self.quantityCombo.currentText() == "Divergence of B":
                bb1 = self.modelfile[mod].dataset[dat].box[0]["bb1"].data
                bb2 = self.modelfile[mod].dataset[dat].box[0]["bb2"].data
                bb3 = self.modelfile[mod].dataset[dat].box[0]["bb3"].data

                bc1 = ip.interp1d(self.xb1, bb1, copy=False, assume_sorted=True)(self.xc1)
                bc2 = ip.interp1d(self.xb2, bb2, axis=1, copy=False, assume_sorted=True)(self.xc2)
                bc3 = ip.interp1d(self.xb3, bb3, axis=0, copy=False, assume_sorted=True)(self.xc3)

                if ver > '1.11.0':
                    dbxdx = np.gradient(bc1, self.dx, axis=-1)
                    dbydy = np.gradient(bc2, self.dy, axis=1)
                    dbzdz = np.gradient(bc3, self.dz, axis=0)
                else:
                    _, _, dbxdx = np.gradient(bc1, self.dz, self.dy, self.dx)
                    _, dbydy, _ = np.gradient(bc2, self.dz, self.dy, self.dx)
                    dbzdz, _, _ = np.gradient(bc3, self.dz, self.dy, self.dx)

                data = ne.evaluate("(dbxdx + dbydy + dbzdz) * sqrt(const)")
                self.unit = "G/km"

            elif self.quantityCombo.currentText() == "Alfven speed":
                rho = self.modelfile[mod].dataset[dat].box[0]["rho"].data

                bb1 = self.modelfile[mod].dataset[dat].box[0]["bb1"].data
                bb2 = self.modelfile[mod].dataset[dat].box[0]["bb2"].data
                bb3 = self.modelfile[mod].dataset[dat].box[0]["bb3"].data

                bc1 = ip.interp1d(self.xb1, bb1, copy=False, assume_sorted=True)(self.xc1)
                bc2 = ip.interp1d(self.xb2, bb2, axis=1, copy=False, assume_sorted=True)(self.xc2)
                bc3 = ip.interp1d(self.xb3, bb3, axis=0, copy=False, assume_sorted=True)(self.xc3)

                data = ne.evaluate("sqrt((bc1**2+bc2**2+bc3**2)/rho)")
                self.unit = "cm/s"
            elif self.quantityCombo.currentText() == "Electric current density jx":
                bb2 = self.modelfile[mod].dataset[dat].box[0]["bb2"].data
                bb3 = self.modelfile[mod].dataset[dat].box[0]["bb3"].data

                bc2 = ip.interp1d(self.xb2, bb2, axis=1, copy=False, assume_sorted=True)(self.xc2)
                bc3 = ip.interp1d(self.xb3, bb3, axis=0, copy=False, assume_sorted=True)(self.xc3)

                if self.constGrid:
                    if ver > '1.11.0':
                        dbzdy = np.gradient(bc3, self.dy, axis=1)
                        dbydz = np.gradient(bc2, self.dz, axis=0)
                    else:
                        _, dbzdy, _ = np.gradient(bc3, self.dz, self.dy, self.dx)
                        dbydz, _, _ = np.gradient(bc2, self.dz, self.dy, self.dx)
                else:
                    dbzdy = sc.Deriv(bc3, self.xc2, self.xb2, 1)
                    dbydz = sc.Deriv(bc2, self.xc3, self.xb3, 0)

                data = ne.evaluate("clight*(dbzdy-dbydz)/sqrt(const)")
                self.unit = "G/m"
            elif self.quantityCombo.currentText() == "Electric current density jy":
                bb1 = self.modelfile[mod].dataset[dat].box[0]["bb1"].data
                bb3 = self.modelfile[mod].dataset[dat].box[0]["bb3"].data

                bc1 = ip.interp1d(self.xb1, bb1, copy=False, assume_sorted=True)(self.xc1)
                bc3 = ip.interp1d(self.xb3, bb3, axis=0, copy=False, assume_sorted=True)(self.xc3)

                if self.constGrid:
                    if ver > '1.11.0':
                        dbxdz = np.gradient(bc1, self.dz, axis=0)
                        dbzdx = np.gradient(bc3, self.dx, axis=-1)
                    else:
                        dbxdz, _, _ = np.gradient(bc1, self.dz, self.dy, self.dx)
                        _, _, dbzdx = np.gradient(bc3, self.dz, self.dy, self.dx)
                else:
                    dbxdz=sc.Deriv(bc1, self.xc3, self.xb3, 0)
                    dbzdx=sc.Deriv(bc3, self.xc1, self.xb1)

                data = ne.evaluate("clight*(dbxdz-dbzdx)/sqrt(const)")
                self.unit = "G/m"
            elif self.quantityCombo.currentText() == "Electric current density jz":
                bb1 = self.modelfile[mod].dataset[dat].box[0]["bb1"].data
                bb2 = self.modelfile[mod].dataset[dat].box[0]["bb2"].data

                bc1 = ip.interp1d(self.xb1, bb1, copy=False, assume_sorted=True)(self.xc1)
                bc2 = ip.interp1d(self.xb2, bb2, axis=1, copy=False, assume_sorted=True)(self.xc2)

                if self.constGrid:
                    if ver > '1.11.0':
                        dbydx = np.gradient(bc2, self.dx, axis=-1)
                        dbxdy = np.gradient(bc1, self.dy, axis=1)
                    else:
                        _, _, dbydx = np.gradient(bc2, self.dz, self.dy, self.dx)
                        _, dbxdy, _ = np.gradient(bc1, self.dz, self.dy, self.dx)
                else:
                    dbydx = sc.Deriv(bc2, self.xc1, self.xb1)
                    dbxdy = sc.Deriv(bc1, self.xc2, self.xb2, 1)

                data = ne.evaluate("clight*(dbydx-dbxdy)/sqrt(const)")
                self.unit = "G/m"
            elif self.quantityCombo.currentText() == "Electric current density |j|":
                bb1 = self.modelfile[mod].dataset[dat].box[0]["bb1"].data
                bb2 = self.modelfile[mod].dataset[dat].box[0]["bb2"].data
                bb3 = self.modelfile[mod].dataset[dat].box[0]["bb3"].data

                bc1 = ip.interp1d(self.xb1, bb1, copy=False, assume_sorted=True)(self.xc1)
                bc2 = ip.interp1d(self.xb2, bb2, axis=1, copy=False, assume_sorted=True)(self.xc2)
                bc3 = ip.interp1d(self.xb3, bb3, axis=0, copy=False, assume_sorted=True)(self.xc3)

                if self.constGrid:
                    if ver > '1.11.0':
                        dbxdz, dbxdy = np.gradient(bc1, self.dz, self.dy, axis=(0, 1))
                        dbydz, dbydx = np.gradient(bc2, self.dz, self.dx, axis=(0, -1))
                        dbzdy, dbzdx = np.gradient(bc3, self.dy, self.dx, axis=(1, -1))
                    else:
                        dbxdz, dbxdy, _ = np.gradient(bc1, self.dz, self.dy, self.dx)
                        dbydz, _, dbydx = np.gradient(bc2, self.dz, self.dy, self.dx)
                        _, dbzdy, dbzdx = np.gradient(bc3, self.dz, self.dy, self.dx)
                else:
                    dbzdy=sc.Deriv(bc3, self.xc2, self.xb2, 1)
                    dbydz=sc.Deriv(bc2, self.xc3, self.xb3, 0)

                    dbxdz=sc.Deriv(bc1, self.xc3, self.xb3, 0)
                    dbzdx=sc.Deriv(bc3, self.xc1, self.xb1)

                    dbydx=sc.Deriv(bc2, self.xc1, self.xb1)
                    dbxdy=sc.Deriv(bc1, self.xc2, self.xb2, 1)

                data = ne.evaluate("clight*sqrt(((dbzdy-dbydz)**2+(dbxdz-dbzdx)**2+(dbydx-dbxdy)**2)/const)")
                self.unit = "G/m"
            elif self.quantityCombo.currentText() in ["Entropy", "Pressure", "Temperature"]:
                rho = self.modelfile[mod].dataset[dat].box[0]["rho"].data
                ei = self.modelfile[mod].dataset[dat].box[0]["ei"].data

                data = self.Eos.STP(rho, ei, quantity=self.quantityCombo.currentText())
                self.unit = self.Eos.unit(quantity=self.quantityCombo.currentText())
            elif self.quantityCombo.currentText() == "Plasma beta":
                bb1 = self.modelfile[mod].dataset[dat].box[0]["bb1"].data
                bb2 = self.modelfile[mod].dataset[dat].box[0]["bb2"].data
                bb3 = self.modelfile[mod].dataset[dat].box[0]["bb3"].data

                bc1 = ip.interp1d(self.xb1, bb1, copy=False, assume_sorted=True)(self.xc1)
                bc2 = ip.interp1d(self.xb2, bb2, axis=1, copy=False, assume_sorted=True)(self.xc2)
                bc3 = ip.interp1d(self.xb3, bb3, axis=0, copy=False, assume_sorted=True)(self.xc3)

                rho = self.modelfile[mod].dataset[dat].box[0]["rho"].data
                ei = self.modelfile[mod].dataset[dat].box[0]["ei"].data

                P = self.Eos.STP(rho, ei)

                data = ne.evaluate("2.0*P/(bc1**2+bc2**2+bc3**2)")
                self.unit = ""
            elif self.quantityCombo.currentText() == "Sound velocity":
                rho = self.modelfile[mod].dataset[dat].box[0]["rho"].data
                ei = self.modelfile[mod].dataset[dat].box[0]["ei"].data

                P, dPdrho, dPde = self.Eos.Pall(rho, ei)

                data = ne.evaluate("sqrt(P*dPde/(rho**2.0)+dPdrho)")
                self.unit = "cm/s"
            elif self.quantityCombo.currentText() == "c_s / c_A":
                rho = self.modelfile[mod].dataset[dat].box[0]["rho"].data
                ei = self.modelfile[mod].dataset[dat].box[0]["ei"].data

                P, dPdrho, dPde = self.Eos.Pall(rho, ei)

                bb1 = self.modelfile[mod].dataset[dat].box[0]["bb1"].data
                bb2 = self.modelfile[mod].dataset[dat].box[0]["bb2"].data
                bb3 = self.modelfile[mod].dataset[dat].box[0]["bb3"].data

                bc1 = ip.interp1d(self.xb1, bb1, copy=False, assume_sorted=True)(self.xc1)
                bc2 = ip.interp1d(self.xb2, bb2, axis=1, copy=False, assume_sorted=True)(self.xc2)
                bc3 = ip.interp1d(self.xb3, bb3, axis=0, copy=False, assume_sorted=True)(self.xc3)

                data = ne.evaluate("sqrt(P*dPde/(rho**2)+dPdrho)/sqrt((bc1**2+bc2**2+bc3**2)/rho)")
                self.unit = ""
            elif self.quantityCombo.currentText() == "Mean molecular weight":
                rho = self.modelfile[mod].dataset[dat].box[0]["rho"].data
                ei = self.modelfile[mod].dataset[dat].box[0]["ei"].data

                P, T = self.Eos.PandT(rho, ei)
                R = 8.314e7

                data = ne.evaluate("R*rho*T/P")

                self.unit = ""
            elif self.quantityCombo.currentText() == "Mach Number":
                rho = self.modelfile[mod].dataset[dat].box[0]["rho"].data
                ei = self.modelfile[mod].dataset[dat].box[0]["ei"].data

                P, dPdrho, dPde = self.Eos.Pall(rho, ei)

                v1 = self.modelfile[mod].dataset[dat].box[0]["v1"].data
                v2 = self.modelfile[mod].dataset[dat].box[0]["v2"].data
                v3 = self.modelfile[mod].dataset[dat].box[0]["v3"].data

                data = ne.evaluate("sqrt((v1**2+v2**2+v3**2)/(P*dPde/(rho**2.0)+dPdrho))")
                self.unit = ""
            elif self.quantityCombo.currentText() == "Adiabatic coefficient G1":
                rho = self.modelfile[mod].dataset[dat].box[0]["rho"].data
                ei = self.modelfile[mod].dataset[dat].box[0]["ei"].data

                P, dPdrho, dPde = self.Eos.Pall(rho, ei)

                data = ne.evaluate("dPdrho*rho/P+dPde/rho")
                self.unit = ""
            elif self.quantityCombo.currentText() == "Adiabatic coefficient G3":
                rho = self.modelfile[mod].dataset[dat].box[0]["rho"].data
                ei = self.modelfile[mod].dataset[dat].box[0]["ei"].data

                P, dPdrho, dPde = self.Eos.Pall(rho, ei)

                data = ne.evaluate("dPde/rho+1.0")
                self.unit = ""
            elif self.quantityCombo.currentText() == "Opacity":
                rho = self.modelfile[mod].dataset[dat].box[0]["rho"].data
                ei = self.modelfile[mod].dataset[dat].box[0]["ei"].data

                P, T = self.Eos.PandT(rho, ei)
                data = self.Opa.kappa(T, P)

                self.unit = "1/cm"
            elif self.quantityCombo.currentText() == "Optical depth":
                rho = self.modelfile[mod].dataset[dat].box[0]["rho"].data
                ei = self.modelfile[mod].dataset[dat].box[0]["ei"].data

                P, T = self.Eos.PandT(rho, ei)

                if self.par  and 'c_radhtautop' in self.parFile.keys():
                    data = self.Opa.tau(rho, axis=0, T=T, P=P, zb=self.xb3*1.e5,
                                        radhtautop=self.parFile['c_radhtautop'].data)
                else:
                    data = self.Opa.tau(rho, axis=0, T=T, P=P, zb=self.xb3*1.e5)

                self.unit = ""
            else:
                data = self.modelfile[mod].dataset[dat].box[self.boxind][self.typeind].data.squeeze()
                self.unit = self.modelfile[mod].dataset[dat].box[self.boxind][self.typeind].params["u"]
        elif self.fileType == "mean":
            data = self.modelfile[mod].dataset[dat].box[self.boxind][self.typeind].data.squeeze()
            self.unit = self.modelfile[mod].dataset[dat].box[self.boxind][self.typeind].params["u"]
        else:
            data = self.modelfile[mod][self.typeind].squeeze().T
            self.unit = self.modelfile[mod].unit(self.typeind)
        self.dim = data.ndim

        if self.x3Combo.currentIndex() == 1:
            if tau is not None:
                data = self.Opa.quant_at_tau(data, self.tauRange, axis=0, tau=tau)
            else:
                rho = self.modelfile[mod].dataset[dat].box[0]["rho"].data
                ei = self.modelfile[mod].dataset[dat].box[0]["ei"].data

                P, T = self.Eos.PandT(rho, ei)

                if self.par and 'c_radhtautop' in self.parFile.keys():
                    data = self.Opa.quant_at_tau(data, self.tauRange, axis=0, rho=rho, T=T, P=P, zb=self.xb3*1.e5,
                                             radhtautop=self.parFile['c_radhtautop'].data)
                else:
                    data = self.Opa.quant_at_tau(data, self.tauRange, axis=0, rho=rho, T=T, P=P, zb=self.xb3 * 1.e5)

        QtWidgets.QApplication.restoreOverrideCursor()
        text = "time needed for evaluation: {0:5.3g} s".format(time.time()-start)
        self.statusBar().showMessage(text)
        return ne.evaluate(self.postfunc[self.funcCombo.currentText()], local_dict={'data': data})

    @pyqtSlot()
    def planeCheck(self):
        sender = self.sender()
        if sender.objectName() == "plane-Combo":
            self.pos = None
        # do not plot when changing min norm value (as plotted after changing max value)
        self.plot = False
        if self.dim == 3:
            if self.normCheck.checkState() == QtCore.Qt.Checked:
                if self.planeCombo.currentText() == "xy":
                    self.normMinEdit.setText("{dat:16.5g}".format(dat=self.globBound[0][0][self.x3ind]))
                    # plot if max norm value is not changed
                    self.plot = True
                    self.normMaxEdit.setText("{dat:16.5g}".format(dat=self.globBound[0][1][self.x3ind]))
                elif self.planeCombo.currentText() == "xz":
                    self.normMinEdit.setText("{dat:16.5g}".format(dat=self.globBound[1][0][self.x2ind]))
                    self.plot = True
                    self.normMaxEdit.setText("{dat:16.5g}".format(dat=self.globBound[1][1][self.x2ind]))
                elif self.planeCombo.currentText() == "yz":
                    self.normMinEdit.setText("{dat:16.5g}".format(dat=self.globBound[2][0][self.x1ind]))
                    self.plot = True
                    self.normMaxEdit.setText("{dat:16.5g}".format(dat=self.globBound[2][1][self.x1ind]))
                else:
                    self.msgBox.setText("Plane not identified.")
                    self.msgBox.exec_()
                if self.sameNorm:
                    self.generalPlotRoutine()
            else:
                if self.planeCombo.currentText() == "xy":
                    self.normMinEdit.setText("{dat:16.5g}".format(dat=self.data[self.x3ind].min()))
                    self.plot = True
                    self.normMaxEdit.setText("{dat:16.5g}".format(dat=self.data[self.x3ind].max()))
                elif self.planeCombo.currentText() == "xz":
                    self.normMinEdit.setText("{dat:16.5g}".format(dat=self.data[:, self.x2ind].min()))
                    self.plot = True
                    self.normMaxEdit.setText("{dat:16.5g}".format(dat=self.data[:, self.x2ind].max()))
                elif self.planeCombo.currentText() == "yz":
                    self.normMinEdit.setText("{dat:16.5g}".format(dat=self.data[:, :, self.x1ind].min()))
                    self.plot = True
                    self.normMaxEdit.setText("{dat:16.5g}".format(dat=self.data[:, :, self.x1ind].max()))
                else:
                    self.msgBox.setText("Plane not identified.")
                    self.msgBox.exec_()

        else:
            if self.normCheck.checkState() == QtCore.Qt.Checked:
                self.normMinEdit.setText("{dat:16.5g}".format(dat=self.globBound[0]))
                self.plot = True
                self.normMaxEdit.setText("{dat:16.5g}".format(dat=self.globBound[1]))
                if self.sameNorm:
                    self.generalPlotRoutine()
            else:
                self.normMinEdit.setText("{dat:16.5g}".format(dat=self.data.min()))
                self.plot = True
                self.normMaxEdit.setText("{dat:16.5g}".format(dat=self.data.max()))

        self.sameNorm = False

    @pyqtSlot()
    def funcComboChange(self):
        self.data = self.setPlotData(self.modelind, self.dsind)
        self.planeCheck()

    @pyqtSlot()
    def x3ComboChange(self):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

        if self.x3Combo.currentIndex() == 0:
            self.currentX3Title.setText("iz:")
            self.actualX3Title.setText("z [km]:")
            self.tauGroup.hide()

            self.x3Slider.setMaximum(self.xc3.size - 1)
            self.data = self.setPlotData(self.modelind, self.dsind)
            self.tauUnityCheck.setDisabled(False)
        else:
            self.currentX3Title.setText(u"i\u03C4:")
            self.actualX3Title.setText(u"\u03C4        :")

            self.tauUnityCheck.setChecked(False)
            self.tauUnityCheck.setDisabled(True)

            rho = self.modelfile[self.modelind].dataset[self.dsind].box[0]["rho"].data
            ei = self.modelfile[self.modelind].dataset[self.dsind].box[0]["ei"].data

            P, T = self.Eos.PandT(rho, ei)

            if self.par and 'c_radhtautop' in self.parFile.keys():
                tau = self.Opa.tau(rho, axis=0, T=T, P=P, zb=self.xb3*1.e5, radhtautop=self.parFile['c_radhtautop'].data)
            else:
                tau = self.Opa.tau(rho, axis=0, T=T, P=P, zb=self.xb3*1.e5)
            tau[tau < 0] *= -1
            taul = np.log10(tau)

            self.minTau = np.around(taul.max(axis=(1, 2)).min(), decimals=2)
            self.maxTau = np.around(taul.min(axis=(1, 2)).max(), decimals=2)

            try:
                self.numTau = int(self.numTauEdit.text())
            except ValueError:
                print("{0} of {1} is invalid input.".format(self.numTauEdit.text(), self.numTauEdit.objectName()))
                pass

            self.tauRange = np.logspace(self.minTau, self.maxTau, self.numTau)[::-1]
            self.x3Slider.setMaximum(self.numTau - 1)

            self.plot = False
            self.minTauEdit.setText(str(self.minTau))
            self.plot = True
            self.maxTauEdit.setText(str(self.maxTau))

            self.data = self.setPlotData(self.modelind, self.dsind, tau=tau)
            self.tauGroup.show()
        self.planeCheck()
        QtWidgets.QApplication.restoreOverrideCursor()

    @pyqtSlot()
    def SliderChange(self):
        sender = self.sender()

        if sender.objectName() == "time-Slider":
            self.timind = self.timeSlider.value()
            self.actualTimeLabel.setText("{dat:10.1f}".format(dat=self.time[self.timind, 0]))
            self.currentTimeEdit.setText(str(self.timind))

            self.modelind = int(self.time[self.timind, 1])
            self.dsind = int(self.time[self.timind, 2])
            self.currentFileLabel.setText(self.fname[self.modelind].split("/")[-1])

            if self.fileType == "cobold" or self.fileType == "mean":
                self.xc1 = self.modelfile[self.modelind].dataset[self.dsind].box[0]["xc1"].data.squeeze() * 1.e-5
                self.xc2 = self.modelfile[self.modelind].dataset[self.dsind].box[0]["xc2"].data.squeeze() * 1.e-5
                self.xc3 = self.modelfile[self.modelind].dataset[self.dsind].box[0]["xc3"].data.squeeze() * 1.e-5

                self.xb1 = self.modelfile[self.modelind].dataset[self.dsind].box[0]["xb1"].data.squeeze() * 1.e-5
                self.xb2 = self.modelfile[self.modelind].dataset[self.dsind].box[0]["xb2"].data.squeeze() * 1.e-5
                self.xb3 = self.modelfile[self.modelind].dataset[self.dsind].box[0]["xb3"].data.squeeze() * 1.e-5
            else:
                if self.fileType == "profile":
                    shape = np.array(self.modelfile[self.modelind]['I'].shape)
                elif self.fileType == "nicole":
                    shape = np.array(self.modelfile[self.modelind]['tau'].shape)

                self.xc1 = np.arange(0, shape[0])
                self.xc2 = np.arange(0, shape[1])
                self.xc3 = np.arange(0, shape[2])

            self.x1min = self.xc1.min()
            self.x1max = self.xc1.max()

            self.x2min = self.xc2.min()
            self.x2max = self.xc2.max()

            self.x3min = self.xc3.min()
            self.x3max = self.xc3.max()

            self.x1Slider.setMaximum(len(self.xc1) - 1)
            self.x2Slider.setMaximum(len(self.xc2) - 1)
            self.x3Slider.setMaximum(len(self.xc3) - 1)

            if self.tauUnityCheck.isChecked():
                self.tauUnityChange()

            self.data = self.setPlotData(self.modelind, self.dsind)
            self.sameNorm = True

        elif sender.objectName() == "x1-Slider":
            self.plot = False
            self.x1ind = self.x1Slider.value()
            self.currentX1Edit.setText(str(self.x1ind).rjust(10))
            self.actualX1Label.setText("{:10.1f}".format(self.xc1[self.x1ind]).rjust(13))

        elif sender.objectName() == "x2-Slider":
            self.plot = False
            self.x2ind = self.x2Slider.value()
            self.currentX2Edit.setText(str(self.x2ind).rjust(10))
            self.actualX2Label.setText("{:10.1f}".format(self.xc2[self.x2ind]).rjust(13))

        elif sender.objectName() == "x3-Slider":
            self.plot = False
            self.x3ind = self.x3Slider.value()
            self.currentX3Edit.setText(str(self.x3ind).rjust(10))
            if self.x3Combo.currentIndex() == 0:
                self.actualX3Label.setText("{:10.1f}".format(self.xc3[self.x3ind]).rjust(0))
            else:
                self.actualX3Label.setText("{:10.3g}".format(self.tauRange[self.x3ind]).rjust(0))

        self.planeCheck()

    @pyqtSlot()
    def normChange(self):
        if self.planeCombo.currentText() == "xy":
            self.normMeanLabel.setText("{dat:13.4g}".format(dat=self.data[self.x3ind].mean()))
        elif self.planeCombo.currentText() == "xz":
            self.normMeanLabel.setText("{dat:13.4g}".format(dat=self.data[:, self.x2ind].mean()))
        elif self.planeCombo.currentText() == "yz":
            self.normMeanLabel.setText("{dat:13.4g}".format(dat=self.data[:, :, self.x1ind].mean()))

        # catch 'nan' in Min/Max fields:
        try:
            self.minNorm = float(self.normMinEdit.text())
            if self.minNorm > self.maxNorm:
                raise ValueError('minNorm must be <= maxNorm')
        except ValueError:
            self.minNorm = np.finfo(np.float32).min
        try:
            self.maxNorm = float(self.normMaxEdit.text())
            if self.minNorm > self.maxNorm:
                raise ValueError('minNorm must be <= maxNorm')
        except ValueError:
            self.maxNorm = np.finfo(np.float32).max

        if self.plot:
            self.generalPlotRoutine()

    @pyqtSlot()
    def currentEditChange(self):
        sender = self.sender()

        try:
            if sender.objectName() == "current-time-Edit":
                if int(self.currentTimeEdit.text()) > self.timlen:
                    self.currentTimeEdit.setText(str(self.timlen-1))
                elif int(self.currentTimeEdit.text()) < 0:
                    self.currentTimeEdit.setText(str(0))

                self.timind = int(self.currentTimeEdit.text())
                self.timeSlider.setValue(self.timind)

            elif sender.objectName() == "current-x-Edit":
                if int(self.currentX1Edit.text()) > len(self.xc1):
                    self.currentX1Edit.setText(str(len(self.xc1)-1))
                elif int(self.currentX1Edit.text()) < 0:
                    self.currentX1Edit.setText(str(0))

                self.x1ind = int(self.currentX1Edit.text())
                self.x1Slider.setValue(self.x1ind)

            elif sender.objectName() == "current-y-Edit":
                if int(self.currentX2Edit.text()) > len(self.xc2):
                    self.currentX2Edit.setText(str(len(self.xc2)-1))
                elif int(self.currentX2Edit.text()) < 0:
                    self.currentX2Edit.setText(str(0))

                self.x2ind = int(self.currentX2Edit.text())
                self.x2Slider.setValue(self.x2ind)

            elif sender.objectName() == "current-z-Edit":
                if self.x3Combo.currentIndex() == 0:
                    length = len(self.xc3)
                else:
                    length = self.numTau
                if int(self.currentX3Edit.text()) > length:
                    self.currentX3Edit.setText(str(length-1))
                elif int(self.currentX2Edit.text()) < 0:
                    self.currentX3Edit.setText(str(0))

                self.x3ind = int(self.currentX3Edit.text())
                self.x3Slider.setValue(self.x3ind)

            self.statusBar().showMessage("")
        except:
            self.statusBar().showMessage("Invalid input in currentEditChange!")

    @pyqtSlot()
    def timeBtnClick(self):

        sender = self.sender()

        if sender.objectName() == "next-time-Button" and self.timind < (self.timlen - 1):
            self.timind += 1
            self.sameNorm = True
        elif sender.objectName() == "prev-time-Button" and self.timind > 0:
            self.timind -= 1
            self.sameNorm = True
        else:
            self.statusBar().showMessage("Out of range.")
            return

        self.timeSlider.setValue(self.timind)

    @pyqtSlot()
    def tauBtnClick(self):
        sender = self.sender()
        # "minus-max-tau-Button"
        # "plus-max-tau-Button"

        try:
            if sender.objectName() == "minus-min-tau-Button":
                self.minTauEdit.setText(str(float(self.minTauEdit.text()) - 1))
            elif sender.objectName() == "plus-min-tau-Button":
                self.minTauEdit.setText(str(float(self.minTauEdit.text()) + 1))
            elif sender.objectName() == "minus10-num-tau-Button":
                self.numTauEdit.setText(str(int(self.numTauEdit.text()) - 10))
            elif sender.objectName() == "minus-num-tau-Button":
                self.numTauEdit.setText(str(int(self.numTauEdit.text()) - 1))
            elif sender.objectName() == "plus-num-tau-Button":
                self.numTauEdit.setText(str(int(self.numTauEdit.text()) + 1))
            elif sender.objectName() == "plus10-num-tau-Button":
                self.numTauEdit.setText(str(int(self.numTauEdit.text()) + 10))
            elif sender.objectName() == "minus-max-tau-Button":
                self.maxTauEdit.setText(str(float(self.maxTauEdit.text()) - 1))
            elif sender.objectName() == "plus-max-tau-Button":
                self.maxTauEdit.setText(str(float(self.maxTauEdit.text()) + 1))
        except ValueError:
            print("{0} of {1} is an invalid input.".format(sender.text(), sender.objectName()))

    # -----------------------------------------------
    # --- Change of 2D- to 3D-plot and vice versa ---
    # -----------------------------------------------

    @pyqtSlot()
    def plotDimensionChange(self):
        sender = self.sender()

        if sender.objectName() == "2DRadio":
            self.planeCheck()
            # self.threeDPlotBox.hide()
            self.plotBox.show()
        elif sender.objectName() == "3DRadio":
            pass
            # self.threeDPlotBox.Plot(self.data)
            # self.plotBox.hide()
            # self.threeDPlotBox.show()

    # ----------------------------------------
    # --- Change of quantity via combo box ---
    # ----------------------------------------

    @pyqtSlot()
    def quantityChange(self):

        self.boxind = -1

        for i in range(len(self.quantityList)):
            if self.quantityCombo.currentText() in self.quantityList[i].keys():
                self.boxind = i
                break

        # --- Get the index for quantity ---

        self.typeind = self.quantityList[self.boxind][self.quantityCombo.currentText()]

        # ---------------------------------------------------------------------
        # --- get new globally minimal and maximal values for normalization ---
        # ---------------------------------------------------------------------

        if self.normCheck.checkState() == QtCore.Qt.Checked:
            self.getTotalMinMax()

        # --------------------------------------
        # --- update parameters from widgets ---

        self.data = self.setPlotData(self.modelind, self.dsind)

        # self.dim = 3 - self.data.shape.count(1)

        if self.dim == 3:
            self.planeCombo.setDisabled(False)
            self.cmCombo.setDisabled(False)

            self.x1Slider.setDisabled(False)
            self.x2Slider.setDisabled(False)
            self.x3Slider.setDisabled(False)

            self.currentX1Edit.setDisabled(False)
            self.currentX2Edit.setDisabled(False)
            self.currentX3Edit.setDisabled(False)

            self.actualX1Label.setDisabled(False)
            self.actualX2Label.setDisabled(False)
            self.actualX3Label.setDisabled(False)
        if self.dim == 2:
            self.planeCombo.setDisabled(True)
            self.cmCombo.setDisabled(False)

            self.x1Slider.setDisabled(True)
            self.x2Slider.setDisabled(True)
            self.x3Slider.setDisabled(True)

            self.currentX1Edit.setDisabled(True)
            self.currentX2Edit.setDisabled(True)
            self.currentX3Edit.setDisabled(True)

            self.actualX1Label.setDisabled(True)
            self.actualX2Label.setDisabled(True)
            self.actualX3Label.setDisabled(True)

            if self.fileType == "cobold" or self.fileType == "mean":
                self.direction = self.modelfile[self.modelind].dataset[self.dsind].box[self.boxind][self.typeind].\
                    shape.index(1)
            else:
                self.direction = self.modelfile[self.modelind][self.typeind].T.shape.index(1)
        elif self.dim == 1:
            self.planeCombo.setDisabled(True)
            self.cmCombo.setDisabled(True)

            self.x1Slider.setDisabled(True)
            self.x2Slider.setDisabled(True)
            self.x3Slider.setDisabled(True)

            self.currentX1Edit.setDisabled(True)
            self.currentX2Edit.setDisabled(True)
            self.currentX3Edit.setDisabled(True)

            self.actualX1Label.setDisabled(True)
            self.actualX2Label.setDisabled(True)
            self.actualX3Label.setDisabled(True)

            if self.fileType == "cobold" or self.fileType == "mean":
                self.direction = bisect.bisect(self.modelfile[self.modelind].dataset[self.dsind].box[self.boxind]
                                               [self.typeind].shape, 2)
            else:
                self.direction = bisect.bisect(self.modelfile[self.modelind][self.typeind].T.shape, 2)

        self.unitLabel.setText(self.unit)

        self.planeCheck()

    # ----

    @pyqtSlot()
    def vectorSetup(self):
        if self.vpCheck.isChecked():
            self.vpMagRadio.setDisabled(False)
            self.vpVelRadio.setDisabled(False)

            if self.vpVelRadio.isChecked():
                self.vecunit = r'$\frac{cm}{s}$'
                self.u = self.modelfile[self.modelind].dataset[self.dsind].box[0]["v1"].data
                self.v = self.modelfile[self.modelind].dataset[self.dsind].box[0]["v2"].data
                self.w = self.modelfile[self.modelind].dataset[self.dsind].box[0]["v3"].data
            elif self.vpMagRadio.isChecked():
                self.vecunit = 'G'
                const = math.sqrt(4 * np.pi)
                x1 = self.modelfile[0].dataset[0].box[0]["xb1"].data.squeeze()*1.e-5
                x2 = self.modelfile[0].dataset[0].box[0]["xb2"].data.squeeze()*1.e-5
                x3 = self.modelfile[0].dataset[0].box[0]["xb3"].data.squeeze()*1.e-5

                bb1 = self.modelfile[self.modelind].dataset[self.dsind].box[0]["bb1"].data*const
                bb2 = self.modelfile[self.modelind].dataset[self.dsind].box[0]["bb2"].data*const
                bb3 = self.modelfile[self.modelind].dataset[self.dsind].box[0]["bb3"].data*const

                self.u = ip.interp1d(x1, bb1, copy=False, assume_sorted=True)(self.xc1)
                self.v = ip.interp1d(x2, bb2, axis=1, copy=False, assume_sorted=True)(self.xc2)
                self.w = ip.interp1d(x3, bb3, axis=0, copy=False, assume_sorted=True)(self.xc3)
        else:
            self.vpMagRadio.setDisabled(True)
            self.vpVelRadio.setDisabled(True)

        self.generalPlotRoutine()

    @pyqtSlot()
    def generalPlotRoutine(self):
        # "virtual" method
        pass


class ModelSaveDialog(BasicWindow):
    def __init__(self, modelfile):

        # --- convert imput parameters to global parameters

        self.modelfile = modelfile

    def saveEvent():
        pass


class MultiPlotWind(BasicWindow):
    def __init__(self, fname, modelfile, fileType, eos=None, opa=None):
        super(MultiPlotWind, self).__init__()

        self.fname = fname
        self.modelfile = modelfile
        self.fileType = fileType
        if eos is not None:
            self.eos = True
            self.Eos = eos
        if opa is not None:
            self.opa = True
            self.Opa = opa

        self.setWindowTitle("CO5BOLDViewer {} - Multi Plot".format(self.version))
        self.setGeometry(100, 100, 1000, 700)

        self.addWidgets()
        self.plotDim = 2
        self.plotWinds = {}
        self.plotWindsN = 0

        self.initLoad()

        self.show()

    def initLoad(self):
        if self.fileType == "mean":

            # --- content from .mean file ---
            # --- Components depict box number from filestructure (see manual of CO5BOLD)

            OneDQuants = OrderedDict([("Avg. density ", "rho_xmean"),
                                      ("Avg. specific internal energy ", "ei_xmean"),
                                      ("Avg. internal energy per volume ", "rhoei_xmean"),
                                      ("Squared avg. density", "rho_xmean2"),
                                      ("Avg. velocity (x-component)", "v1_xmean"),
                                      ("Avg. velocity (y-component)", "v2_xmean"),
                                      ("Avg. velocity (z-component)", "v3_xmean"),
                                      ("Avg. squared velocity (x-component)", "v1_xmean2"),
                                      ("Avg. squared velocity (y-component)", "v2_xmean2"),
                                      ("Avg. squared velocity (z-component)", "v3_xmean2"),
                                      ("Avg. mass flux (x-component)", "rhov1_xmean"),
                                      ("Avg. mass flux (y-component)", "rhov2_xmean"),
                                      ("Avg. mass flux (z-component)", "rhov3_xmean"),
                                      ("Avg. magnetic field (x-component)", "bc1_xmean"),
                                      ("Avg. magnetic field (y-component)", "bc2_xmean"),
                                      ("Avg. magnetic field (z-component)", "bc3_xmean"),
                                      ("Absolute avg. magnetic field (x-component)", "bc1_xabsmean"),
                                      ("Absolute avg. magnetic field (y-component)", "bc2_xabsmean"),
                                      ("Absolute avg. magnetic field (z-component)", "bc3_xabsmean"),
                                      ("Avg. squared magnetic field (x-component)", "bc1_xmean2"),
                                      ("Avg. squared magnetic field (y-component)", "bc2_xmean2"),
                                      ("Avg. squared magnetic field (z-component)", "bc3_xmean2")]),

            self.quantityList = [OrderedDict([("Bolometric intensity", "intb3_r"), ("Intensity (bin 1)", "int01b3_r"),
                                              ("Intensity (bin 2)", "int02b3_r"), ("Intensity (bin 3)", "int03b3_r"),
                                              ("Intensity (bin 4)", "int04b3_r"), ("Intensity (bin 5)", "int05b3_r")])]
        elif self.fileType == "cobold":

            # --- content from .full or .end file (has one box per dataset) ---
            # --- First list component: Data from file
            # --- Second list component: Data from post computed arrays
            # --- Third list component: Post computed MHD data, if present

            self.quantityList = [OrderedDict([("Density", "rho"), ("Internal energy", "ei"),
                                              ("Velocity (x-component)", "v1"), ("Velocity (y-component)", "v2"),
                                              ("Velocity (z-component)", "v3"), ("Velocity, absolute", "vabs"),
                                              ("Velocity, horizontal", "vhor"), ("Kinetic energy", "kinEn"),
                                              ("Momentum", "momentum"), ("Vert. mass flux (Rho*V3)", "massfl")])]

            # If another file is loaded, set the indicator to an "uncertain" state, i.e. it is not clear, if the
            # specific file corresponds to the recently loaded model

            mhd = False
            for mod in self.modelfile:
                if "bb1" in mod.dataset[0].box[0]:
                    mhd = True
                    break

            if mhd:
                self.quantityList.append(OrderedDict([("Magnetic field Bx", "bc1"), ("Magnetic field By", "bc2"),
                                                      ("Magnetic field Bz", "bc3"), ("Divergence of B", "divB"),
                                                      ("Magnetic field Bh (horizontal)", "bh"),
                                                      ("Magnetic f.abs.|B|, unsigned", "absb"),
                                                      ("Magnetic field B^2, signed", "bsq"),
                                                      ("Vert. magnetic flux Bz*Az", "bfl"),
                                                      ("Vert. magnetic gradient Bz/dz", "bgrad"),
                                                      ("Magnetic energy", "bener"),# ("Magnetic potential Phi", "phi"),
                                                      ("Alfven speed", "ca"), ("Electric current density jx", "jx"),
                                                      ("Electric current density jy", "jy"),
                                                      ("Electric current density jz", "jz"),
                                                      ("Electric current density |j|", "jabs")]))
        elif self.fileType == "profile":
            self.eos = False
            self.opa = False

            self.quantityList = [OrderedDict([("Stokes I", "I"), ("Stokes Q", "Q"), ("Stokes U", "U"),
                                              ("Stokes V", "V")])]
        elif self.fileType == "nicole":
            self.eos = False
            self.opa = False

            self.quantityList = [OrderedDict([("Geometrical height", 'z'), ("log10(Optical depth)", 'tau'),
                                              ("Temperature", 'T'), ("Pressure", 'P'), ("Density", 'rho'),
                                              ("Electron Pressure", "el_p"), ("LOS-Velocity", 'v_los'),
                                              ("Microturbulence", 'v_mic'), ("Longitudinal Magnetic Field", 'b_long'),
                                              ("Transverse Magnetic Field (x-component)", 'b_x'),
                                              ("Transverse Magnetic Field (y-component)", 'b_y'),
                                              ("Local Magnetic Field (x-component)", 'b_local_x'),
                                              ("Local Magnetic Field (y-component)", 'b_local_y'),
                                              ("Local Magnetic Field (z-component)", 'b_local_z'),
                                              ("Local Velocity Field (x-component)", 'v_local_x'),
                                              ("Local Velocity Field (y-component)", 'v_local_y'),
                                              ("Local Velocity Field (z-component)", 'v_local_z')
                                              ])]
        else:
            self.msgBox.setText("File-type is unknown.")
            self.msgBox.exec_()

        # --- Fourth list component: EOS (and opacity) table
        # interpolated data, if already loaded

        if self.eos and (self.fileType == "mean" or self.fileType == "cobold"):
            self.quantityList.append(OrderedDict([("Temperature", "temp"), ("Entropy", "entr"), ("Pressure", "press"),
                                                  ("Adiabatic coefficient G1", "gamma1"), ("Mach Number", "mach"),
                                                  ("Adiabatic coefficient G3", "gamma3"), ("Sound velocity", "c_s"),
                                                  ("Mean molecular weight", "mu"), ("Plasma beta", "beta"),
                                                  ("c_s / c_A", "csca")]))

        if self.opa and (self.fileType == "mean" or self.fileType == "cobold"):
            self.quantityList[-1]["Opacity"] = "opa"
            self.quantityList[-1]["Optical depth"] = "optdep"

        if not self.modelfile[0].closed:
            self.quantityCombo.clear()

            for type in self.quantityList:
                self.quantityCombo.addItems(type.keys())
            self.initialLoad()

        QtWidgets.QApplication.restoreOverrideCursor()

    def addWidgets(self):
        # BasicWindow consists of all elements, but plot-element. Therefore, layout is already set. Only plot-area has
        # to be defined

        # ---------------------------------------------------------------------
        # ---------------------------- Plot window ----------------------------
        # ---------------------------------------------------------------------

        self.plotArea = QtWidgets.QMdiArea(self.centralWidget)

        # ---------------------------------------------------------------------
        # -------------- Groupbox with file-state indicators ------------------
        # ---------------------------------------------------------------------

        addPlotGroup = QtWidgets.QGroupBox("Add Plot", self.centralWidget)
        addPlotLayout = QtWidgets.QHBoxLayout(addPlotGroup)
        addPlotGroup.setLayout(addPlotLayout)

        self.addPlotBtn = QtWidgets.QPushButton("Add Plot")
        self.addPlotBtn.clicked.connect(self.addPlotBtnClick)
        self.addPlotBtn.setObjectName("plus-max-tau-Button")

        addPlotLayout.addWidget(self.addPlotBtn)

        # self.threeDPlotBox = sc.PlotWidget3D(self.centralWidget)

        # --- Add plot-widget to inhereted splitter

        self.splitter.addWidget(self.plotArea)

        # --- Add aditional groups to control panel ---

        self.controlgrid.addWidget(addPlotGroup)

    @pyqtSlot()
    def plotDimensionChange(self):
        sender = self.sender()

        if sender.objectName() == "1DRadio":
            self.plotDim = 1
        elif sender.objectName() == "2DRadio":
            self.plotDim = 2
        elif sender.objectName() == "3DRadio":
            self.plotDim = 3

    def invertCM(self, state):
        if state == QtCore.Qt.Checked:
            self.cmCombo.inv = "_r"

        else:
            self.cmCombo.inv = ""
        self.cmCombo.currentCmap = self.cmCombo.currentText() + self.cmCombo.inv
        # self.plotBox.colorChange(self.cmCombo.currentCmap)
        self.colorbar.set_cmap(self.cmCombo.currentCmap)

        self.colorbar.draw_all()
        self.colorcanvas.draw()

    @pyqtSlot()
    def addPlotBtnClick(self):
        if self.planeCombo.currentText() == "xy":
            axis = 0
        elif self.planeCombo.currentText() == "xz":
            axis = 1
        elif self.planeCombo.currentText() == "yz":
            axis = 2
        else:
            raise ValueError("Axis with unknown value")

        if self.plotDim < 3:
            ind = "subWind"+str(self.plotWindsN)
            self.plotWindsN += 1

            sub = mdis.MdiSubWindow(self.plotArea)
            sub.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
            self.plotWinds[ind] = mdis.MDIPlotWidget(self.data, parent=sub, dimension=self.plotDim, axis=axis)
            sub.setWidget(self.plotWinds[ind])
            sub.setObjectName(ind)
            sub.closed.connect(self.closedSubWindow)

            if self.x3Combo.currentIndex() == 0:
                title = self.quantityCombo.currentText() + " z"
            else:
                title = self.quantityCombo.currentText() + " tau"
            sub.setWindowTitle(title)
            self.plotArea.addSubWindow(sub)
            sub.show()
        self.generalPlotRoutine()

    @pyqtSlot(str)
    def closedSubWindow(self, name):
        self.plotWinds.pop(name, None)

    @pyqtSlot()
    def generalPlotRoutine(self):
        sender = self.sender()
        if sender.objectName() == "quantity-Combo":
            pass
        for plot in self.plotWinds:
            if self.crossCheck.isChecked():
                pos = self.pos
            else:
                pos = None
            if self.keepPlotRangeCheck.isChecked():
                if self.plotWinds[plot].dim == 1:
                    window = np.array(self.plotWinds[plot].ax.get_xlim())
                else:
                    window = np.array([self.plotWinds[plot].ax.get_xlim(), self.plotWinds[plot].ax.get_ylim()])
            else:
                window = None

            if self.planeCombo.currentText() == "xy":
                ind = self.x3ind
                limits = np.array([[self.x1min, self.x1max], [self.x2min, self.x2max]])
            elif self.planeCombo.currentText() == "xz":
                ind = self.x2ind
                if self.x3Combo.currentIndex() == 0:
                    limits = np.array([[self.x1min, self.x1max], [self.x3min, self.x3max]])
                else:
                    limits = np.array([[self.x1min, self.x1max],
                                       [float(self.maxTauEdit.text()), float(self.minTauEdit.text())]])

            elif self.planeCombo.currentText() == "yz":
                ind = self.x1ind
                if self.x3Combo.currentIndex() == 0:
                    limits = np.array([[self.x1min, self.x1max], [self.x3min, self.x3max]])
                else:
                    limits = np.array([[self.x1min, self.x1max],
                                       [float(self.maxTauEdit.text()), float(self.minTauEdit.text())]])

            if self.tauUnityCheck.isChecked() and self.plotWinds[plot].dim == 2:
                self.plotWinds[plot].Plot(ind=ind, limits=limits, window=window, cmap=self.cmCombo.currentCmap, pos=pos,
                                          tauUnity=(self.xc1, self.tauheight[self.x2ind]))
            else:
                self.plotWinds[plot].Plot(ind=ind, limits=limits, window=window, cmap=self.cmCombo.currentCmap, pos=pos)

            if self.vpCheck.isChecked():
                try:
                    self.plotWinds[plot].vectorPlot(self.xc1, self.xc3, self.u[:, self.x2ind], self.w[:, self.x2ind],
                                                    xinc=int(self.vpXIncEdit.text()), yinc=int(self.vpYIncEdit.text()),
                                                    scale=float(self.vpScaleEdit.text()),
                                                    alpha=float(self.vpAlphaEdit.text()))
                except ValueError:
                    pass

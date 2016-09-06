# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 09:18:20 2014

@author: Rene
"""

import numpy as np
import numexpr as ne
from scipy import interpolate

#from PySide import QtCore, QtGui
from PyQt4 import QtCore, QtGui

#import matplotlib as mpl
#mpl.rcParams['backend.qt4']='PySide'
#plt = mpl.pyplot
#cm = mpl.cm
#cl = mpl.colors
#FigureCanvas = mpl.backends.backend_qt4agg.FigureCanvas
#NavigationToolbar = mpl.backends.backend_qt4agg.NavigationToolbar2QT
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as cl
from matplotlib.backends.backend_qt4agg \
import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg \
import NavigationToolbar2QT as NavigationToolbar

#import pyqtgraph as pg

from astropy.io import fits
import h5py
        
# ---------------

import rangeslider
#import rebin

class showImageSaveDialog(QtGui.QMainWindow):

    def __init__(self, inputfile=None, data=None, currenttime=0, currentquant="",
                 time=None, currentx1=0, x1=None, currentx2=0, x2=None,
                 currentx3=0, x3=None, currentcm=0, currenttype=0):

         # --- convert imput parameters to global parameters

         self.inputfile = inputfile
         self.data = data
         self.currentquant = currentquant

         self.currenttime = currenttime
         self.time = time

         self.currentx1 = currentx1
         self.x1 = x1

         self.currentx2 = currentx2
         self.x2 = x2

         self.currentx3 = currentx3
         self.x3 = x3

         self.currentcm = currentcm
         self.currenttype = currenttype

         self.maxwidth = 40

         self.initData = True

         # --- fill list with all available colormaps

         self.cmaps = [m for m in cm.datad if not m.endswith("_r")]
         self.cmaps.sort()

         if ".mean" in self.inputfile[0].name:
            self.meanfile = True
            # --- content from .mean file ---
            # --- Components depict box number from filestructure (see manual
            # --- of Co5bold)

            self.dataTypeList = ({"Bolometric intensity":"intb3_r",
                                  "Intensity (bin 1)": "int01b3_r",
                                  "Intensity (bin 2)": "int02b3_r",
                                  "Intensity (bin 3)": "int03b3_r",
                                  "Intensity (bin 4)": "int04b3_r",
                                  "Intensity (bin 5)": "int05b3_r"},
                                {"Avg. density (x1)": "rho_xmean",
                                 "Squared avg. density (x1)": "rho_xmean2"})
         elif ".full" or ".end" in self.inputfile[0].name:
            self.meanfile = False
            # --- content from .full or .end file (has one box per dataset) ---
            # --- First tuple component: Data from file
            # --- Second tuple component: Data from post computed arrays

            self.dataTypeList = ({"Density":"rho","Internal energy": "ei",
                                  "Velocity x-component": "v1",
                                  "Velocity y-component": "v2",
                                  "Velocity z-component": "v3"},
                                 {"Velocity, absolute": "vabs",
                                  "Velocity, horizontal": "vhor",
                                  "Kinetic energy": "kinEn",
                                  "Momentum": "momentum",
                                  "Vert. mass flux (Rho*V3)": "massfl",
                                  "Temperature": "temp","Entropy": "entr",
                                  "Pressure": "press",
                                  "Adiabatic coefficient G1": "gamma1",
                                  "Adiabatic coefficient G3": "gamma3",
                                  "Sound velocity": "c_s","Mach Number": "mach",
                                  "Mean molecular weight": "mu",
                                  "Opacity": "opa","Optical depth": "optdep",
                                  "Magnetic field Bx": "bc1",
                                  "Magnetic field By": "bc2",
                                  "Magnetic field Bz": "bc3",
                                  "Magnetic field Bh (horizontal)": "bh",
                                  "Magnetic f.abs.|B|, unsigned": "absb",
                                  "Magnetic field B^2, signed": "bsq",
                                  "Magnetic energy": "bener",
                                  "Magnetic potential Phi": "phi",
                                  "Electric current density jx": "jx",
                                  "Electric current density jy": "jy",
                                  "Electric current density jz": "jz",
                                  "Electric current density |j|": "jabs",
                                  "Alfven speed": "ca","Plasma beta": "beta",
                                  "c_s / c_A": "csca"})

         super(showImageSaveDialog, self).__init__()

         self.image_save_window = None

         self.__uiInit()

    # -----------------------------------------------------------------
    # ---------------------- Initialize Layout ------------------------
    # -----------------------------------------------------------------

    def __uiInit(self):

        if self.image_save_window is None:

            self.image_save_window = self

            self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

            self.setGeometry(100,600,800,300)
            self.setWindowTitle("Save Image")

            centralWidget = QtGui.QWidget(self)

            mainGrid = QtGui.QGridLayout(centralWidget)

            # -----------------------------------------------------------------
            # ---------- Groupbox with save configuration elements ------------
            # -----------------------------------------------------------------

            saveConfGroup = QtGui.QGroupBox("Save-settings", centralWidget)

            # --- Sub-layouts

            saveConfLayout = QtGui.QVBoxLayout()

            # --- Widgets

            self.currentImageRadio = QtGui.QRadioButton("Save current image")
            self.currentImageRadio.setToolTip("Save current setup of mainwindow as image file.")
            self.currentImageRadio.setObjectName("currentImage")
            self.currentImageRadio.clicked.connect(self.toggleLayout)
            self.currentImageRadio.setChecked(True)

            self.singleImageRadio = QtGui.QRadioButton("Save selected image")
            self.singleImageRadio.setToolTip("Save a singe image selected by sliders in the plot configuration box.")
            self.singleImageRadio.setObjectName("singleImage")
            self.singleImageRadio.clicked.connect(self.toggleLayout)

            self.rangeImageRadio = QtGui.QRadioButton("Save image sequence")
            self.rangeImageRadio.setToolTip("Save a sequence of images selected by sliders in the plot configuration box.")
            self.rangeImageRadio.setObjectName("rangeImage")
            self.rangeImageRadio.clicked.connect(self.toggleLayout)

            self.imageFormatCombo = QtGui.QComboBox(centralWidget)
            self.imageFormatCombo.addItem("Encapsulated Postscript (.eps)")
            self.imageFormatCombo.addItem("Joint Photographics Experts Group (.jpg, .jpeg)")
            self.imageFormatCombo.addItem("LaTeX PGF Figure (.pgf)")
            self.imageFormatCombo.addItem("Portable Document Format (.pdf)")
            self.imageFormatCombo.addItem("Portable Network Graphics (.png)")
            self.imageFormatCombo.addItem("Postscript (.ps)")
            self.imageFormatCombo.addItem("Raw RGBA bitmap (.raw, .rgba)")
            self.imageFormatCombo.addItem("Scalable Vector Graphics (.svg, .svgz)")
            self.imageFormatCombo.addItem("Raw RGBA bitmap (.raw, .rgba)")
            self.imageFormatCombo.addItem("Tagged Image File Format (.tif, .tiff)")

            # --- Adding elements

            saveConfLayout.addWidget(self.currentImageRadio)
            saveConfLayout.addWidget(self.singleImageRadio)
            saveConfLayout.addWidget(self.rangeImageRadio)
            saveConfLayout.addWidget(self.imageFormatCombo)

            saveConfGroup.setLayout(saveConfLayout)

            # -----------------------------------------------------------------
            # ---------- Groupbox with plot configuration elements ------------
            # -----------------------------------------------------------------

            plotConfGroup = QtGui.QGroupBox("Plot-settings", centralWidget)

            self.plotConfLayout = QtGui.QGridLayout(plotConfGroup)

            # --- Sub-layouts

            self.plotConfLayoutCurrent = QtGui.QGridLayout()
            self.plotConfLayoutSingle = QtGui.QGridLayout()
            self.plotConfLayoutRange = QtGui.QGridLayout()

            self.plotConfCurrentWid = QtGui.QWidget()
            self.plotConfSingleWid = QtGui.QWidget()
            self.plotConfRangeWid = QtGui.QWidget()

            self.plotConfCurrentWid.setLayout(self.plotConfLayoutCurrent)
            self.plotConfSingleWid.setLayout(self.plotConfLayoutSingle)
            self.plotConfRangeWid.setLayout(self.plotConfLayoutRange)

            # --- Widgets for single data selection ---

            # --- Time widgets

            timeLabel = QtGui.QLabel("Timestep:")
            self.timeSlider = QtGui.QSlider(QtCore.Qt.Horizontal,
                                            self.plotConfSingleWid)
            self.timeSlider.setMinimum(0)
            self.timeSlider.setMaximum(len(self.time)-1)
            self.timeSlider.setSliderPosition(self.currenttime)
            self.timeSlider.setObjectName("timeSlider")
            self.timeSlider.valueChanged.connect(self.sliderChange)

            self.timeEdit = QtGui.QLineEdit(str(self.timeSlider.value()),
                                            self.plotConfSingleWid)
            self.timeEdit.setObjectName("timeEdit")
            self.timeEdit.textChanged.connect(self.editChange)

            if self.time is not None:
                self.actualTimeLabel = QtGui.QLabel("{:10.2f}".format(self.time[self.
                                          timeSlider.value()]) + " s", self.plotConfSingleWid)
            else:
                self.actualTimeLabel = QtGui.QLabel("0 s", self.plotConfSingleWid)

            # --- Spatial widgets

            x1Label = QtGui.QLabel("x1:")
            self.x1Slider = QtGui.QSlider(QtCore.Qt.Horizontal,
                                          self.plotConfSingleWid)
            self.x1Slider.setMinimum(0)
            self.x1Slider.setMaximum(len(self.x1)-1)
            self.x1Slider.setSliderPosition(self.currentx1)
            self.x1Slider.setObjectName("x1Slider")
            self.x1Slider.valueChanged.connect(self.sliderChange)

            self.x1Edit = QtGui.QLineEdit(str(self.x1Slider.value()),
                                          self.plotConfSingleWid)
            self.x1Edit.setObjectName("x1Edit")
            self.x1Edit.textChanged.connect(self.editChange)

            if self.x1 is not None:
                self.actualx1Label = QtGui.QLabel("{:10.2f}".format(self.x1[self.
                                          x1Slider.value()]) + " km", self.plotConfSingleWid)
            else:
                self.actualx1Label = QtGui.QLabel("0 km", self.plotConfSingleWid)

            x2Label = QtGui.QLabel("x2:")

            self.x2Slider = QtGui.QSlider(QtCore.Qt.Horizontal,
                                          self.plotConfSingleWid)
            self.x2Slider.setMinimum(0)
            self.x2Slider.setMaximum(len(self.x2)-1)
            self.x2Slider.setSliderPosition(self.currentx2)
            self.x2Slider.setObjectName("x2Slider")
            self.x2Slider.valueChanged.connect(self.sliderChange)

            self.x2Edit = QtGui.QLineEdit(str(self.x2Slider.value()),
                                          self.plotConfSingleWid)
            self.x2Edit.setObjectName("x2Edit")
            self.x2Edit.textChanged.connect(self.editChange)

            if self.x2 is not None:
                self.actualx2Label = QtGui.QLabel("{:10.2f}".format(self.x2[self.
                                          x2Slider.value()]) + " km", self.plotConfSingleWid)
            else:
                self.actualx2Label = QtGui.QLabel("0 km", self.plotConfSingleWid)

            x3Label = QtGui.QLabel("x3:", centralWidget)

            self.x3Slider = QtGui.QSlider(QtCore.Qt.Horizontal,
                                          self.plotConfSingleWid)
            self.x3Slider.setMinimum(0)
            self.x3Slider.setMaximum(len(self.x3)-1)
            self.x3Slider.setSliderPosition(self.currentx3)
            self.x3Slider.setObjectName("x3Slider")
            self.x3Slider.valueChanged.connect(self.sliderChange)

            self.x3Edit = QtGui.QLineEdit(str(self.x3Slider.value()),
                                          self.plotConfSingleWid)
            self.x3Edit.setObjectName("x3Edit")
            self.x3Edit.textChanged.connect(self.editChange)

            if self.x3 is not None:
                self.actualx3Label = QtGui.QLabel("{:10.2f}".format(self.x3[self.
                                                  x3Slider.value()]) + " km",
                                                  self.plotConfSingleWid)
            else:
                self.actualx3Label = QtGui.QLabel("0 km", self.plotConfSingleWid)

            # --- Widgets for single data range ---

            timeRangeLabel = QtGui.QLabel("Timestep:")

            self.timeRangeSlider = rangeslider.RangeSlider(QtCore.Qt.Horizontal,
                                                           self.plotConfRangeWid)
            self.timeRangeSlider.setMinimum(0)
            self.timeRangeSlider.setMaximum(len(self.time)-1)
            self.timeRangeSlider.setLow(self.currenttime)
            self.timeRangeSlider.setHigh(self.currenttime)
            self.timeRangeSlider.setObjectName("timeRangeSlider")
            self.timeRangeSlider.sliderMoved.connect(self.sliderChange)

            self.timeEditLow = QtGui.QLineEdit(str(self.timeRangeSlider.low()),
                                               self.plotConfRangeWid)
            self.timeEditLow.setObjectName("timeEditLow")
            self.timeEditLow.setMaximumWidth(self.maxwidth)
            self.timeEditLow.textChanged.connect(self.editChange)

            self.timeEditHigh = QtGui.QLineEdit(str(self.timeRangeSlider.high()),
                                                self.plotConfRangeWid)
            self.timeEditHigh.setObjectName("timeEditHigh")
            self.timeEditHigh.setMaximumWidth(self.maxwidth)
            self.timeEditHigh.textChanged.connect(self.editChange)

            if self.time is not None:
                self.actualTimeLabelLow = QtGui.QLabel("{:10.2f}".format(self.
                                                       time[self.
                                                       timeRangeSlider.low()]) + " s",
                                                       self.plotConfRangeWid)
                self.actualTimeLabelHigh = QtGui.QLabel("{:10.2f}".format(self.
                                                        time[self.timeRangeSlider.
                                                        high()]) + " s",
                                                        self.plotConfRangeWid)
            else:
                self.actualTimeLabelLow = QtGui.QLabel("0 s",
                                                       self.plotConfRangeWid)
                self.actualTimeLabelHigh = QtGui.QLabel("0 s",
                                                        self.plotConfRangeWid)

            x1RangeLabel = QtGui.QLabel("x1:")

            self.x1RangeSlider = rangeslider.RangeSlider(QtCore.Qt.Horizontal,
                                                         self.plotConfRangeWid)
            self.x1RangeSlider.setMinimum(0)
            self.x1RangeSlider.setMaximum(len(self.x1)-1)
            self.x1RangeSlider.setLow(self.currentx1)
            self.x1RangeSlider.setHigh(self.currentx1)
            self.x1RangeSlider.setObjectName("x1RangeSlider")
            self.x1RangeSlider.sliderMoved.connect(self.sliderChange)

            self.x1EditLow = QtGui.QLineEdit(str(self.x1RangeSlider.low()),
                                             self.plotConfRangeWid)
            self.x1EditLow.setObjectName("x1EditLow")
            self.x1EditLow.setMaximumWidth(self.maxwidth)
            self.x1EditLow.textChanged.connect(self.editChange)

            self.x1EditHigh = QtGui.QLineEdit(str(self.x1RangeSlider.high()),
                                              self.plotConfRangeWid)
            self.x1EditHigh.setObjectName("x1EditHigh")
            self.x1EditHigh.setMaximumWidth(self.maxwidth)
            self.x1EditHigh.textChanged.connect(self.editChange)

            if self.x1 is not None:
                self.actualx1LabelLow = QtGui.QLabel("{:10.2f}".format(self.x1[self.
                                          x1RangeSlider.low()]) + " km",
                                          self.plotConfRangeWid)
                self.actualx1LabelHigh = QtGui.QLabel("{:10.2f}".format(self.x1[self.
                                          x1RangeSlider.high()]) + " km",
                                           self.plotConfRangeWid)
            else:
                self.actualx1LabelLow = QtGui.QLabel("0 km",
                                                       self.plotConfRangeWid)
                self.actualx1LabelHigh = QtGui.QLabel("0 km",
                                                        self.plotConfRangeWid)

            x2RangeLabel = QtGui.QLabel("x2:")

            self.x2RangeSlider = rangeslider.RangeSlider(QtCore.Qt.Horizontal,
                                                         self.plotConfRangeWid)
            self.x2RangeSlider.setMinimum(0)
            self.x2RangeSlider.setMaximum(len(self.x2)-1)
            self.x2RangeSlider.setSliderPosition(self.currentx2)
            self.x2RangeSlider.setObjectName("x2RangeSlider")
            self.x2RangeSlider.sliderMoved.connect(self.sliderChange)

            self.x2EditLow = QtGui.QLineEdit(str(self.x2RangeSlider.low()),
                                             self.plotConfRangeWid)
            self.x2EditLow.setObjectName("x2EditLow")
            self.x2EditLow.setMaximumWidth(self.maxwidth)
            self.x2EditLow.textChanged.connect(self.editChange)

            self.x2EditHigh = QtGui.QLineEdit(str(self.x2RangeSlider.high()),
                                              self.plotConfRangeWid)
            self.x2EditHigh.setObjectName("x2EditHigh")
            self.x2EditHigh.setMaximumWidth(self.maxwidth)
            self.x2EditHigh.textChanged.connect(self.editChange)

            if self.x2 is not None:
                self.actualx2LabelLow = QtGui.QLabel("{:10.2f}".format(self.x2[self.
                                          x2RangeSlider.low()]) + " km",
                                          self.plotConfRangeWid)
                self.actualx2LabelHigh = QtGui.QLabel("{:10.2f}".format(self.x2[self.
                                          x2RangeSlider.high()]) + " km",
                                          self.plotConfRangeWid)
            else:
                self.actualx2LabelLow = QtGui.QLabel("0 km",
                                                     self.plotConfRangeWid)
                self.actualx2LabelHigh = QtGui.QLabel("0 km",
                                                      self.plotConfRangeWid)

            x3RangeLabel = QtGui.QLabel("x3:")

            self.x3RangeSlider = rangeslider.RangeSlider(QtCore.Qt.Horizontal,
                                                         self.plotConfRangeWid)
            self.x3RangeSlider.setMinimum(0)
            self.x3RangeSlider.setMaximum(len(self.x3)-1)
            self.x3RangeSlider.setSliderPosition(self.currentx3)
            self.x3RangeSlider.setObjectName("x3RangeSlider")
            self.x3RangeSlider.sliderMoved.connect(self.sliderChange)

            self.x3EditLow = QtGui.QLineEdit(str(self.x3RangeSlider.low()),
                                             self.plotConfRangeWid)
            self.x3EditLow.setObjectName("x3EditLow")
            self.x3EditLow.setMaximumWidth(self.maxwidth)
            self.x3EditLow.textChanged.connect(self.editChange)

            self.x3EditHigh = QtGui.QLineEdit(str(self.x3RangeSlider.high()),
                                              self.plotConfRangeWid)
            self.x3EditHigh.setObjectName("x3EditHigh")
            self.x3EditHigh.setMaximumWidth(self.maxwidth)
            self.x3EditHigh.textChanged.connect(self.editChange)

            if self.x3 is not None:
                self.actualx3LabelLow = QtGui.QLabel("{:10.2f}".format(self.x3[self.
                                          x3RangeSlider.low()]) + " km",
                                          self.plotConfRangeWid)
                self.actualx3LabelHigh = QtGui.QLabel("{:10.2f}".format(self.x3[self.
                                          x3RangeSlider.high()]) + " km",
                                          self.plotConfRangeWid)
            else:
                self.actualx3LabelLow = QtGui.QLabel("0 km",
                                                     self.plotConfRangeWid)
                self.actualx3LabelHigh = QtGui.QLabel("0 km",
                                                      self.plotConfRangeWid)

            # --- save or cancel widgets ---

            saveCancelWid = QtGui.QWidget(centralWidget)
            saveCancelLayout = QtGui.QHBoxLayout()
            saveButton = QtGui.QPushButton("Save")

            cancelButton = QtGui.QPushButton("Cancel")
            cancelButton.clicked.connect(self.close)

            saveCancelLayout.addStretch(1)
            saveCancelLayout.addWidget(saveButton)
            saveCancelLayout.addWidget(cancelButton)

            saveCancelWid.setLayout(saveCancelLayout)

            # -----------------------------------------------------------------
            # ---------------- Adding elements to plot layout -----------------
            # -----------------------------------------------------------------

            # --- Single selection layout ---

            # --- Time elements

            self.plotConfLayoutSingle.addWidget(timeLabel,0,0)
            self.plotConfLayoutSingle.addWidget(self.timeSlider,0,1)
            self.plotConfLayoutSingle.addWidget(self.timeEdit,0,2)
            self.plotConfLayoutSingle.addWidget(self.actualTimeLabel,0,3)

            # --- Spatial elements

            self.plotConfLayoutSingle.addWidget(x1Label,1,0)
            self.plotConfLayoutSingle.addWidget(self.x1Slider,1,1)
            self.plotConfLayoutSingle.addWidget(self.x1Edit,1,2)
            self.plotConfLayoutSingle.addWidget(self.actualx1Label,1,3)

            self.plotConfLayoutSingle.addWidget(x2Label,2,0)
            self.plotConfLayoutSingle.addWidget(self.x2Slider,2,1)
            self.plotConfLayoutSingle.addWidget(self.x2Edit,2,2)
            self.plotConfLayoutSingle.addWidget(self.actualx2Label,2,3)

            self.plotConfLayoutSingle.addWidget(x3Label,3,0)
            self.plotConfLayoutSingle.addWidget(self.x3Slider,3,1)
            self.plotConfLayoutSingle.addWidget(self.x3Edit,3,2)
            self.plotConfLayoutSingle.addWidget(self.actualx3Label,3,3)

            # --- Data setting group

            self.dataSettingGroup = QtGui.QGroupBox("Data selection and costumization")
            dataSettingLayout = QtGui.QGridLayout()
            self.dataSettingGroup.setLayout(dataSettingLayout)

            dataTypeLabel = QtGui.QLabel("Data type: ")
            self.dataTypeCombo = QtGui.QComboBox(self.dataSettingGroup)
            for i in range(len(self.dataTypeList)):
                    self.dataTypeCombo.addItems(sorted(self.dataTypeList[i].keys()))
            self.dataTypeCombo.setCurrentIndex(self.currenttype)

            cmLabel = QtGui.QLabel("Colormap: ")
            self.cmCombo = QtGui.QComboBox(self.dataSettingGroup)
            self.cmCombo.clear()
            self.cmCombo.addItems(self.cmaps)
            self.cmCombo.setCurrentIndex(self.currentcm)
            self.cmCombo.setObjectName("colormap-Combo")

            dataSettingLayout.addWidget(dataTypeLabel,0,0)
            dataSettingLayout.addWidget(self.dataTypeCombo,0,1)

            dataSettingLayout.addWidget(cmLabel,1,0)
            dataSettingLayout.addWidget(self.cmCombo,1,1)

            # --------------------------------------

            # --- Range selection layout ---

            # --- Time elements

            self.plotConfLayoutRange.addWidget(timeRangeLabel,0,0)
            self.plotConfLayoutRange.addWidget(self.timeEditLow,0,1)
            self.plotConfLayoutRange.addWidget(self.actualTimeLabelLow,0,2)
            self.plotConfLayoutRange.addWidget(self.timeRangeSlider,0,3)
            self.plotConfLayoutRange.addWidget(self.timeEditHigh,0,4)
            self.plotConfLayoutRange.addWidget(self.actualTimeLabelHigh,0,5)

            # --- Spatial elements

            self.plotConfLayoutRange.addWidget(x1RangeLabel,1,0)
            self.plotConfLayoutRange.addWidget(self.x1EditLow,1,1)
            self.plotConfLayoutRange.addWidget(self.actualx1LabelLow,1,2)
            self.plotConfLayoutRange.addWidget(self.x1RangeSlider,1,3)
            self.plotConfLayoutRange.addWidget(self.x1EditHigh,1,4)
            self.plotConfLayoutRange.addWidget(self.actualx1LabelHigh,1,5)

            self.plotConfLayoutRange.addWidget(x2RangeLabel,2,0)
            self.plotConfLayoutRange.addWidget(self.x2EditLow,2,1)
            self.plotConfLayoutRange.addWidget(self.actualx2LabelLow,2,2)
            self.plotConfLayoutRange.addWidget(self.x2RangeSlider,2,3)
            self.plotConfLayoutRange.addWidget(self.x2EditHigh,2,4)
            self.plotConfLayoutRange.addWidget(self.actualx2LabelHigh,2,5)

            self.plotConfLayoutRange.addWidget(x3RangeLabel,3,0)
            self.plotConfLayoutRange.addWidget(self.x3EditLow,3,1)
            self.plotConfLayoutRange.addWidget(self.actualx3LabelLow,3,2)
            self.plotConfLayoutRange.addWidget(self.x3RangeSlider,3,3)
            self.plotConfLayoutRange.addWidget(self.x3EditHigh,3,4)
            self.plotConfLayoutRange.addWidget(self.actualx3LabelHigh,3,5)

            # -----------------------------------------------------------------

            self.plotConfLayout.addWidget(self.plotConfCurrentWid,0,0)
            self.plotConfLayout.addWidget(self.plotConfSingleWid,0,0)
            self.plotConfLayout.addWidget(self.plotConfRangeWid,0,0)
            self.plotConfLayout.addWidget(self.dataSettingGroup,1,0)

            plotConfGroup.setLayout(self.plotConfLayout)

            mainGrid.addWidget(saveConfGroup,0,0)
            mainGrid.addWidget(plotConfGroup,0,1)
            mainGrid.addWidget(saveCancelWid,1,1)

            self.plotConfCurrentWid.setVisible(True)
            self.plotConfRangeWid.setVisible(False)
            self.plotConfSingleWid.setVisible(False)
            self.dataSettingGroup.setVisible(False)

            centralWidget.setLayout(mainGrid)

            self.setCentralWidget(centralWidget)

        else:
                self = self.image_save_window

        self.show()
        self.activateWindow()

    def toggleLayout(self):

        sender = self.sender()

        if sender.objectName() == "currentImage":
            self.plotConfCurrentWid.setVisible(True)
            self.plotConfRangeWid.setVisible(False)
            self.plotConfSingleWid.setVisible(False)
            self.dataSettingGroup.setVisible(False)
        elif sender.objectName() == "singleImage":
            self.plotConfCurrentWid.setVisible(False)
            self.plotConfSingleWid.setVisible(True)
            self.plotConfRangeWid.setVisible(False)
            self.dataSettingGroup.setVisible(True)
        elif sender.objectName() == "rangeImage":
            self.plotConfCurrentWid.setVisible(False)
            self.plotConfSingleWid.setVisible(False)
            self.plotConfRangeWid.setVisible(True)
            self.dataSettingGroup.setVisible(True)

    def editChange(self):

        sender = self.sender()

        if sender.objectName() == "x1Edit":
            self.x1Slider.setValue(int(self.x1Edit.text()))
            self.actualx1Label.setText(str(self.x1[self.x1Slider.value()]) + " km")
        elif sender.objectName() == "x2Edit":
            self.x2Slider.setValue(int(self.x2Edit.text()))
            self.actualx2Label.setText(str(self.x2[self.x2Slider.value()]) + " km")
        elif sender.objectName() == "x3Edit":
            self.x3Slider.setValue(int(self.x3Edit.text()))
            self.actualx3Label.setText(str(self.x3[self.x3Slider.value()]) + " km")
        elif sender.objectName() == "timeEdit":
            self.timeSlider.setValue(int(self.timeEdit.text()))
            self.actualTimeLabel.setText(str(self.time[self.timeSlider.value()]) + " s")
        elif sender.objectName() == "x1EditLow":
            self.x1RangeSlider.setLow(int(self.x1EditLow.text()))
            self.actualx1LabelLow.setText(str(self.x1[self.x1RangeSlider.low()]) + " km")
        elif sender.objectName() == "x1EditHigh":
            self.x1RangeSlider.setHigh(int(self.x1EditHigh.text()))
            self.actualx1LabelHigh.setText(str(self.x1[self.x1RangeSlider.high()]) + " km")
        elif sender.objectName() == "x2EditLow":
            self.x2RangeSlider.setLow(int(self.x2EditLow.text()))
            self.actualx2LabelLow.setText(str(self.x2[self.x2RangeSlider.low()]) + " km")
        elif sender.objectName() == "x2EditHigh":
            self.x2RangeSlider.setHigh(int(self.x2EditHigh.text()))
            self.actualx2LabelHigh.setText(str(self.x2[self.x2RangeSlider.high()]) + " km")
        elif sender.objectName() == "x3EditLow":
            self.x3RangeSlider.setLow(int(self.x3EditLow.text()))
            self.actualx3LabelLow.setText(str(self.x3[self.x3RangeSlider.low()]) + " km")
        elif sender.objectName() == "x3EditHigh":
            self.x3RangeSlider.setHigh(int(self.x3EditHigh.text()))
            self.actualx3LabelHigh.setText(str(self.x3[self.x3RangeSlider.high()]) + " km")
        elif sender.objectName() == "timeEditLow":
            self.timeRangeSlider.setLow(int(self.timeEditLow.text()))
            self.actualTimeLabelLow.setText(str(self.time[self.timeRangeSlider.low()]) + " km")
        elif sender.objectName() == "timeEditHigh":
            self.timeRangeSlider.setHigh(int(self.timeEditHigh.text()))
            self.actualTimeLabelHigh.setText(str(self.time[self.timeRangeSlider.high()]) + " km")

    def sliderChange(self):

        sender = self.sender()

        if sender.objectName() == "x1Slider":
            self.x1Edit.setText(str(self.x1Slider.value()))
            self.actualx1Label.setText(str(self.x1[self.x1Slider.value()]) + " km")
        elif sender.objectName() == "x2Slider":
            self.x2Edit.setText(str(self.x2Slider.value()))
            self.actualx2Label.setText(str(self.x2[self.x2Slider.value()]) + " km")
        elif sender.objectName() == "x3Slider":
            self.x3Edit.setText(str(self.x3Slider.value()))
            self.actualx3Label.setText(str(self.x3[self.x3Slider.value()]) + " km")
        elif sender.objectName() == "timeSlider":
            self.timeEdit.setText(str(self.timeSlider.value()))
            self.actualTimeLabel.setText(str(self.time[self.timeSlider.value()]) + " s")
        elif sender.objectName() == "x1RangeSlider":
            self.x1EditLow.setText(str(self.x1RangeSlider.low()))
            self.x1EditHigh.setText(str(self.x1RangeSlider.high()))
            self.actualx1LabelLow.setText(str(self.x1[self.x1RangeSlider.low()]) + " km")
            self.actualx1LabelHigh.setText(str(self.x1[self.x1RangeSlider.high()]) + " km")
        elif sender.objectName() == "x2RangeSlider":
            self.x2EditLow.setText(str(self.x2RangeSlider.low()))
            self.x2EditHigh.setText(str(self.x2RangeSlider.high()))
            self.actualx2LabelLow.setText(str(self.x2[self.x2RangeSlider.low()]) + " km")
            self.actualx2LabelHigh.setText(str(self.x2[self.x2RangeSlider.high()]) + " km")
        elif sender.objectName() == "x3RangeSlider":
            self.x3EditLow.setText(str(self.x3RangeSlider.low()))
            self.x3EditHigh.setText(str(self.x3RangeSlider.high()))
            self.actualx3LabelLow.setText(str(self.x3[self.x3RangeSlider.low()]) + " km")
            self.actualx3LabelHigh.setText(str(self.x3[self.x3RangeSlider.high()]) + " km")
        elif sender.objectName() == "timeRangeSlider":
            self.timeEditLow.setText(str(self.timeRangeSlider.low()))
            self.timeEditHigh.setText(str(self.timeRangeSlider.high()))
            self.actualTimeLabelLow.setText(str(self.time[self.timeRangeSlider.low()]) + " s")
            self.actualTimeLabelHigh.setText(str(self.time[self.timeRangeSlider.high()]) + " s")    

    def setPlotData(self):
        
        self.statusBar().showMessage("Initialize arrays...")
        
        if not self.meanfile:
            if self.dataTypeCombo.currentText() == "Velocity, horizontal":
                v1 = self.modelfile.dataset[self.timind].box[0]["v1"].data
                v2 = self.modelfile.dataset[self.timind].box[0]["v2"].data

                self.data = ne.evaluate("sqrt(v1**2+v2**2)")
                self.unit = "cm/s"

            elif  self.dataTypeCombo.currentText() == "Velocity, absolute":
                v1 = self.modelfile.dataset[self.timind].box[0]["v1"].data
                v2 = self.modelfile.dataset[self.timind].box[0]["v2"].data
                v3 = self.modelfile.dataset[self.timind].box[0]["v3"].data

                self.data = ne.evaluate("sqrt(v1**2+v2**2+v3**2)")
                self.unit = "cm/s"

            elif self.dataTypeCombo.currentText() == "Kinetic energy":
                v1 = self.modelfile.dataset[self.timind].box[0]["v1"].data
                v2 = self.modelfile.dataset[self.timind].box[0]["v2"].data
                v3 = self.modelfile.dataset[self.timind].box[0]["v3"].data
                rho = self.modelfile.dataset[self.timind].box[0]["rho"].data

                self.data = ne.evaluate("0.5*rho*(v1**2+v2**2+v3**2)")
                self.unit = "erg/cm^3"
            
            elif self.dataTypeCombo.currentText() == "Momentum":
                v1 = self.modelfile.dataset[self.timind].box[0]["v1"].data
                v2 = self.modelfile.dataset[self.timind].box[0]["v2"].data
                v3 = self.modelfile.dataset[self.timind].box[0]["v3"].data
                rho = self.modelfile.dataset[self.timind].box[0]["rho"].data

                self.data = ne.evaluate("rho*sqrt(v1**2+v2**2+v3**2)")
                self.unit = "g/(cm^2 * s)"

            elif self.dataTypeCombo.currentText() == "Vert. mass flux (Rho*V3)":
                v3 = self.modelfile.dataset[self.timind].box[0]["v3"].data
                rho = self.modelfile.dataset[self.timind].box[0]["rho"].data

                self.data = ne.evaluate("rho*v3")
                self.unit = "g/(cm^2 * s)"

            elif self.dataTypeCombo.currentText() == "Magnetic field Bx":
                x1 = self.modelfile.dataset[0].box[0]["xb1"].data.squeeze()*1.e-5

                bb1 = self.modelfile.dataset[self.timind].box[0]["bb1"].data

                self.data = interpolate.interp1d(x1, bb1, axis=0, copy=False)(self.x1)
                self.data *= np.sqrt(4.0 * np.pi)
                self.unit = "G"

            elif self.dataTypeCombo.currentText() == "Magnetic field By":
                x2 = self.modelfile.dataset[0].box[0]["xb2"].data.squeeze()*1.e-5
                                        
                bb2 = self.modelfile.dataset[self.timind].box[0]["bb2"].data

                self.data = interpolate.interp1d(x2, bb2, axis=0, copy=False)(self.x2)
                self.data *= np.sqrt(4.0 * np.pi)
                self.unit = "G"

            elif self.dataTypeCombo.currentText() == "Magnetic field Bz":
                x3 = self.modelfile.dataset[0].box[0]["xb3"].data.squeeze()*1.e-5

                bb3 = self.modelfile.dataset[self.timind].box[0]["bb3"].data

                self.data = interpolate.interp1d(x3, bb3, axis=0, copy=False)(self.x3)
                self.data *= np.sqrt(4.0 * np.pi)
                self.unit = "G"

            elif self.dataTypeCombo.currentText() == "Magnetic field Bh (horizontal)":
                x1 = self.modelfile.dataset[0].box[0]["xb1"].data.squeeze()*1.e-5
                x2 = self.modelfile.dataset[0].box[0]["xb2"].data.squeeze()*1.e-5

                bb1 = self.modelfile.dataset[self.timind].box[0]["bb1"].data
                bb2 = self.modelfile.dataset[self.timind].box[0]["bb2"].data

                bc1 = interpolate.interp1d(x1, bb1, copy=False)(self.x1)
                bc2 = interpolate.interp1d(x2, bb2, axis=1, copy=False)(self.x2)

                self.data = ne.evaluate("sqrt(bc1**2+bc2**2)")
                self.data *= np.sqrt(4.0 * np.pi)
                self.unit = "G"

            elif self.dataTypeCombo.currentText() == "Magnetic f.abs.|B|, unsigned":
                
                x1 = self.modelfile.dataset[0].box[0]["xb1"].data.squeeze()*1.e-5
                x2 = self.modelfile.dataset[0].box[0]["xb2"].data.squeeze()*1.e-5
                x3 = self.modelfile.dataset[0].box[0]["xb3"].data.squeeze()*1.e-5

                bb1 = self.modelfile.dataset[self.timind].box[0]["bb1"].data
                bb2 = self.modelfile.dataset[self.timind].box[0]["bb2"].data
                bb3 = self.modelfile.dataset[self.timind].box[0]["bb3"].data

                bc1 = interpolate.interp1d(x1, bb1, copy=False)(self.x1)
                bc2 = interpolate.interp1d(x2, bb2, axis=1, copy=False)(self.x2)
                bc3 = interpolate.interp1d(x3, bb3, axis=0, copy=False)(self.x3)

                self.data = ne.evaluate("sqrt(bc1**2+bc2**2+bc3**2)")
                self.data *= np.sqrt(4.0 * np.pi)
                self.unit = "G"

            elif self.dataTypeCombo.currentText() == "Magnetic field B^2, signed":
                x1 = self.modelfile.dataset[0].box[0]["xb1"].data.squeeze()*1.e-5
                x2 = self.modelfile.dataset[0].box[0]["xb2"].data.squeeze()*1.e-5
                x3 = self.modelfile.dataset[0].box[0]["xb3"].data.squeeze()*1.e-5

                bb1 = self.modelfile.dataset[self.timind].box[0]["bb1"].data
                bb2 = self.modelfile.dataset[self.timind].box[0]["bb2"].data
                bb3 = self.modelfile.dataset[self.timind].box[0]["bb3"].data
                
                bc1 = interpolate.interp1d(x1, bb1, copy=False)(self.x1)
                bc2 = interpolate.interp1d(x2, bb2, axis=1, copy=False)(self.x2)
                bc3 = interpolate.interp1d(x3, bb3, axis=0, copy=False)(self.x3)

                sn = np.ones(bc1.shape)

                sm = np.ones(bc1.shape)
                sm[:,:,:] = -1.0

                sn = np.where(bc1 < 0.0, -1.0, sn)
                self.data = ne.evaluate("sn*bc1**2")

                sn[:,:,:] = 1.0
                sn = np.where(bc2 < 0.0, -1.0, sn)
                self.data += ne.evaluate("sn*bc2**2")

                sn[:,:,:] = 1.0
                sn = np.where(bc3 < 0.0, -1.0, sn)

                self.data += ne.evaluate("sn*bc3**2")
                self.data *= 4.0 * np.pi
                self.unit = "G^2"

            elif self.dataTypeCombo.currentText() == "Magnetic energy":
                x1 = self.modelfile.dataset[0].box[0]["xb1"].data.squeeze()*1.e-5
                x2 = self.modelfile.dataset[0].box[0]["xb2"].data.squeeze()*1.e-5
                x3 = self.modelfile.dataset[0].box[0]["xb3"].data.squeeze()*1.e-5

                bb1 = self.modelfile.dataset[self.timind].box[0]["bb1"].data
                bb2 = self.modelfile.dataset[self.timind].box[0]["bb2"].data
                bb3 = self.modelfile.dataset[self.timind].box[0]["bb3"].data

                bc1 = interpolate.interp1d(x1, bb1, copy=False)(self.x1)
                bc2 = interpolate.interp1d(x2, bb2, axis=1, copy=False)(self.x2)
                bc3 = interpolate.interp1d(x3, bb3, axis=0, copy=False)(self.x3)

                self.data = ne.evaluate("bc1**2+bc2**2+bc3**2") / 2.0
                self.unit = "G^2"

#            elif self.dataTypeCombo.currentText() == "Magnetic potential Phi":
#                
#                
#                x3 = self.modelfile.dataset[0].box[0]["xb3"].data.flatten()*1.e-5
#                                        
#                
#                bb2 = self.modelfile.dataset[self.timind].box[0]["bb2"].data
#                bb3 = self.modelfile.dataset[self.timind].box[0]["bb3"].data
#                
#                bc1 = self.modelfile.dataset[self.timind].box[0]["rho"].data
#                bc2 = self.modelfile.dataset[self.timind].box[0]["rho"].data
#                bc3 = self.modelfile.dataset[self.timind].box[0]["rho"].data
#                
#                for i in range(len(self.x3)):
#                    for j in range(len(self.x2)):
#                        bc1[i,j,:] = np.interp(self.x1, x1, bb1[i,j,:])
#                    
#                    for j in range(len(self.x1)):
#                        bc2[i,:,j] = np.interp(self.x2, x2, bb2[i,:,j])
#                
#                for i in range(len(self.x2)):
#                    for j in range(len(self.x1)):
#                        bc3[:,i,j] = np.interp(self.x3, x3, bb3[:,i,j])
#                
#                if self.planeCombo.currentText() == "xy":
#                    x1 = self.modelfile.dataset[0].box[0]["xb1"].data.flatten()*1.e-5
#                    x2 = self.modelfile.dataset[0].box[0]["xb2"].data.flatten()*1.e-5
#                                        
#                    dx= x1[1:] - x1[:len(x1)-2]
#                    dy= x2[1:] - x2[:len(x1)-2]
#                    
#                    bbx = self.modelfile.dataset[self.timind].box[0]["bb1"].data
#                    for i in range(len(self.x3)):
#                        for j in range(len(self.x2)):
#                            bbx[i,j,:] = np.interp(self.x1, x1, bb[i,j,:])
#                    
#                elif self.planeCombo.currentText() == "xz":    
#                
#                elif self.planeCombo.currentText() == "yz":
#                    
#                self.data = ne.evaluate("bc1**2+bc2**2+bc3**2") / 2.0
#                
#                self.unit = "G"
            
            elif self.dataTypeCombo.currentText() == "Alfven speed":

                x1 = self.modelfile.dataset[0].box[0]["xb1"].data.squeeze()*1.e-5
                x2 = self.modelfile.dataset[0].box[0]["xb2"].data.squeeze()*1.e-5
                x3 = self.modelfile.dataset[0].box[0]["xb3"].data.squeeze()*1.e-5
                rho = self.modelfile.dataset[self.timind].box[0]["rho"].data

                bb1 = self.modelfile.dataset[self.timind].box[0]["bb1"].data
                bb2 = self.modelfile.dataset[self.timind].box[0]["bb2"].data
                bb3 = self.modelfile.dataset[self.timind].box[0]["bb3"].data

                bc1 = interpolate.interp1d(x1, bb1, copy=False)(self.x1)
                bc2 = interpolate.interp1d(x2, bb2, axis=1, copy=False)(self.x2)
                bc3 = interpolate.interp1d(x3, bb3, axis=0, copy=False)(self.x3)

                self.data = ne.evaluate("sqrt(bc1**2+bc2**2+bc3**2)")
                self.data /= np.sqrt(rho)
                self.unit = "G"

            elif self.dataTypeCombo.currentText() == "Entropy":
                    if self.noeos:
                        self.msgBox.setText("No eos-file available.")
                        self.msgBox.exec_()
                    else:
                        self.eos_interpolation("c1")
            elif self.dataTypeCombo.currentText() == "Pressure":
                    if self.noeos:
                        self.msgBox.setText("No eos-file available.")
                        self.msgBox.exec_()
                    else:
                        self.eos_interpolation("c2")
            elif self.dataTypeCombo.currentText() == "Temperature":
                    if self.noeos:
                        self.msgBox.setText("No eos-file available.")
                        self.msgBox.exec_()
                    else:
                        self.eos_interpolation("c3")
            else:
                self.data = self.modelfile.dataset[self.timind].\
                                box[self.typelistind][self.dataind].data
                self.unit = self.modelfile.dataset[self.timind].\
                        box[self.typelistind][self.dataind].params["u"]
        else: 
            self.data = self.modelfile.dataset[self.timind].\
                                box[self.typelistind][self.dataind].data
            self.unit = self.modelfile.dataset[self.timind].\
                        box[self.typelistind][self.dataind].params["u"]
        self.statusBar().showMessage("")

    def saveEvent():
        pass

# ----------------------------------
# --- The Matplotlib-Plot-Widget ---
# ----------------------------------

class PlotWidget(FigureCanvas):
    def __init__(self, parent=None):

        self.msgBox = QtGui.QMessageBox()

        self.fig = plt.Figure(figsize=(8,6), dpi=100)

        FigureCanvas.__init__(self, self.fig)

        self.setParent(parent)
        
        self.ax = self.fig.add_subplot(111)
#        self.ax = self.fig.add_axes(aspect="equal",extent=[0, 10, 0, 10])
        
        self.fig.tight_layout()

        self.toolbar = NavigationToolbar(self, self)
        
        x1 = np.linspace(0,np.pi,100)
        x2 = np.linspace(0,np.pi,100)
        data=[[x1[i] * x2[j] for i in range(len(x1))]for j in range(len(x2))]

        self.image = self.ax.imshow(data, interpolation="bilinear", origin="bottom")

    def plotFig(self, data, x1min, x1max, x2min=None, x2max=None, vmin=None,
                vmax=None, parent=None, dim=2, cmap = "jet"):
        self.ax.cla()

        if dim > 1:
            self.image = self.ax.imshow(data, interpolation="bilinear",
                           origin="bottom", extent=(x1min, x1max, x2min, x2max),
                           norm=cl.Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
        else:
            x = np.linspace(x1min, x1max, len(data))
            self.plot = self.ax.plot(x, data)

        self.draw()
    
    def lP(self, xv, yv, x1min, x1max, x2min, x2max):
        pass
#        print("plot")

#        self.xl = self.ax.axvline(x=xv,xmin=x1min,xmax=x1max)
#        self.yl = self.ax.axhline(y=yv,ymin=x2min,ymax=x2max)

#        self.ax.plot((x1min,yv),(x1max,yv),'r-')
#        self.ax.plot((xv,x2min),(xv,x2max),'r-')

#        self.draw()

    def vectorPlot(self, x, y, u, v, xinc=1, yinc=1, scale=1.0, alpha=1.0,
                   unit=''):
        x, y = np.meshgrid(x,y)
        self.vector = self.ax.quiver(x[::xinc,::yinc], y[::xinc,::yinc],
                                     u[::xinc,::yinc], v[::xinc,::yinc],
                                     scale=1.0/scale, alpha=alpha, edgecolor='k',
                                     facecolor='white', linewidth=0.5)

        self.draw()

    def colorChange(self, cmap="jet"):
        self.image.set_cmap(cmap)
        self.draw()

    def updatePlot(self,data,vmin,vmax):
        self.image.set_clim(vmin, vmax)
        self.image.set_array(data)
        self.draw()

# ----------------------------------------------------------------------------
# --------------------------------- Functions --------------------------------       
# ----------------------------------------------------------------------------

def saveFits(filename, modelfile, datatype, data, time, pos, plane):
    
    if plane == "xy":
        dataHDU = fits.PrimaryHDU(data[pos[2],:,:])
        dataHDU.header["z-pos"] = pos[2]
        
        x1 = modelfile.dataset[0].box[0]["xc1"].data.flatten()
        x2 = modelfile.dataset[0].box[0]["xc2"].data.flatten()
        
        col1 = fits.Column(name='x-axis', format='E', array=x1)
        col2 = fits.Column(name='y-axis', format='E', array=x2)
    elif plane == "xz":
        dataHDU = fits.PrimaryHDU(data[:,pos[1],:])
        dataHDU.header["y-pos"] = pos[1]
        
        x1 = modelfile.dataset[0].box[0]["xc1"].data.flatten()
        x2 = modelfile.dataset[0].box[0]["xc3"].data.flatten()
        
        col1 = fits.Column(name='x-axis', format='E', array=x1)
        col2 = fits.Column(name='z-axis', format='E', array=x2)
    elif plane == "yz":
        dataHDU = fits.PrimaryHDU(data[:,:,pos[0]])
        dataHDU.header["x-pos"] = pos[0]
        
        x1 = modelfile.dataset[0].box[0]["xc2"].data.flatten()
        x2 = modelfile.dataset[0].box[0]["xc3"].data.flatten()
        
        col1 = fits.Column(name='y-axis', format='E', array=x1)
        col2 = fits.Column(name='z-axis', format='E', array=x2)
   
    dataHDU.header["plane"] = plane
    dataHDU.header["time"] = time
    dataHDU.header["datatype"] = datatype
    
    cols = fits.ColDefs([col1,col2])
    
    tbhdu = fits.new_table(cols)
    
    HDUlist = fits.HDUList([dataHDU,tbhdu])
    
    HDUlist.writeto(filename)
    
def saveHD5(filename, modelfile, datatype, data, time, pos, plane):
    
    HD5file = h5py.File(filename, "w")
    
    datagroup = HD5file.create_group(datatype)
    
    datagroup["x"] = modelfile.dataset[0].box[0]["xc1"].data.flatten()
    datagroup["y"] = modelfile.dataset[0].box[0]["xc2"].data.flatten()
    datagroup["z"] = modelfile.dataset[0].box[0]["xc3"].data.flatten()
    
    datagroup["time"] = time
    datagroup["position"] = (datagroup["x"][pos[0]],datagroup["y"][pos[1]],
                             datagroup["z"][pos[2]])
    
    datagroup["plane"] = plane
    
    if plane == "xy":
        datagroup["data"] = data[pos[2],:,:]
        
        datagroup["data"].dims[0].label = "x"
        datagroup["data"].dims[1].label = "y"
        
        datagroup['data'].dims.create_scale(datagroup['x'])
        datagroup['data'].dims.create_scale(datagroup['y'])
        
        datagroup['data'].dims[0].attach_scale(datagroup['x'])
        datagroup['data'].dims[1].attach_scale(datagroup['y'])
        
    elif plane == "xz":
        datagroup["data"] = data[:,pos[1],:]
        
        datagroup["data"].dims[0].label = "x"
        datagroup["data"].dims[1].label = "z"
        
        datagroup['data'].dims.create_scale(datagroup['x'])
        datagroup['data'].dims.create_scale(datagroup['z'])
        
        datagroup['data'].dims[0].attach_scale(datagroup['x'])
        datagroup['data'].dims[1].attach_scale(datagroup['z'])
        
    elif plane == "yz":
        datagroup["data"] = data[:,:,pos[0]]
        
        datagroup["data"].dims[0].label = "y"
        datagroup["data"].dims[1].label = "z"
        
        datagroup['data'].dims.create_scale(datagroup['y'])
        datagroup['data'].dims.create_scale(datagroup['z'])
        
        datagroup['data'].dims[0].attach_scale(datagroup['y'])
        datagroup['data'].dims[1].attach_scale(datagroup['z'])
        
    HD5file.close()

def partialderiv(qc, vc, vb, iv):
    """
    qc: 3D array containing the variable to be differentiated, 
        values cell-centered\n
    vc: differentiate with respect to this axis, values cell-centered\n
    vb: differentiate with respect to this axis, values at cell boundaries\n
    iv: index of axis:
        dQ / dx : iv=1\n
        dQ / dy : iv=2\n
        dQ / dz : iv=3\n
    """
    vc1 = vc.flatten()
    vb1 = vb.flatten()
    
    vb1 = np.clip(vb1, vc1.min(), vc1.max())
    
    dv = np.diff(vb1)
    
    if iv == 1:
        qb = interpolate.interp1d(vc1, qc, copy=False)(vb1)
        deriv= np.diff(qb)/dv[np.newaxis,np.newaxis,:]
                
    elif iv == 2:
        qb = interpolate.interp1d(vc1, qc, axis=1, copy=False)(vb1)
        deriv= np.diff(qb,axis=1)/dv[np.newaxis,:,np.newaxis]
                
    elif iv == 3:
        qb = interpolate.interp1d(vc1, qc, axis=0, copy=False)(vb1)
        deriv=np.diff(qb,axis=0)/dv[:,np.newaxis,np.newaxis]
    return deriv
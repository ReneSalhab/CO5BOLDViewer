# -*- coding: utf-8 -*-
"""
Created on Tue Nov 05 10:12:33 2013

@author: René Georg Salhab
"""

import os
import time
import math
import bisect
import numpy as np
import numexpr as ne
from collections import OrderedDict
from scipy import interpolate as ip
from PyQt5 import QtCore, QtGui, QtWidgets

import matplotlib.pyplot as plt
import matplotlib.colors as cl
import matplotlib.colorbar as clbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import uio
from opta import Opac
import subclasses as sc
import windows as wind
from par import ParFile
from eosinter import EosInter


# noinspection PyUnresolvedReferences
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        self.version = "0.8.6.5"
        super(MainWindow, self).__init__()

        self.initUI()
        self.minNorm = np.finfo(np.float32).min
        self.maxNorm = np.finfo(np.float32).max

    def initUI(self):

        self.centralWidget = QtWidgets.QWidget(self)

        self.setWindowTitle("CO5BOLDViewer {}".format(self.version))
        self.setGeometry(100, 100, 1000, 700)

        QtWidgets.QToolTip.setFont(QtGui.QFont('SansSerif', 10))

        self.initializeParams()
        self.setMenu()
        self.setGridLayout()
        self.statusBar().showMessage("ready")
        self.show()

    def initializeParams(self):

        # --- Read log-file if existing ---

        logfile = os.path.join(os.curdir, 'init.log')
        if os.path.exists(logfile):
            with open(logfile, 'r') as log:
                for line in log:
                    if 'stdDirMod' in line:
                        self.stdDirMod = line.split()[-1]
                    elif 'stdDirOpa' in line:
                        self.stdDirOpa = line.split()[-1]
                    elif 'stdDirEOS' in line:
                        self.stdDirEos = line.split()[-1]
        else:
            self.stdDirMod = None
            self.stdDirPar = None
            self.stdDirOpa = None
            self.stdDirEos = None

        # --- Initial time-index ---

        self.timind = 0

        # --- Timeline place-holder ---

        self.time = np.zeros((2, 3))

        # --- position in cube ---

        self.x1ind = 0
        self.x2ind = 0
        self.x3ind = 0

        # --- Axes of plot ---

        self.xc1 = np.linspace(0, 99, num=100)
        self.xc2 = np.linspace(0, 99, num=100)
        self.xc3 = np.linspace(0, 99, num=100)

        # --- Arbitrary parameters ---

        self.boxind = -1

        self.direction = 0
        self.dim = 0
        self.sameNorm = False

        self.data = np.outer(self.xc1, self.xc2)

        # --- math functions for post-processing data ---

        self.postmath = {'----': "data", '| |': "abs(data)", 'log': "log(data)", 'log(| |)': "log(abs(data))",
                         'log10': "log10(data)", 'log10(| |)': "log10(abs(data))", 'exp': "exp(data)"}

        # --- Available Colormaps (handle invert versions with checkbox) ---

        self.cmaps = [c for c in plt.colormaps() if not c.endswith('_r') ]
        self.cmaps.sort()

        # --- Message Box ---

        self.msgBox = QtWidgets.QMessageBox()

        # --- eos- and opta-file-control variables ---

        self.par = False
        self.eos = False
        self.opa = False

        self.eosname = False
        self.opaname = False

        # --- plot-control ---

        self.plot = False
        self.tauheight = None  # 2-tuple. First element is domain of 1D-plot, second element is height of tau=1-surface

    def setMenu(self):
        # --------------------------------------------------------------------
        # ------------------ "File" drop-down menu elements ------------------
        # --------------------------------------------------------------------

        # --- "Load Model" button config

        openModelAction = QtWidgets.QAction(QtGui.QIcon("open.png"), "Load &Model File", self)
        openModelAction.setShortcut("Ctrl+M")
        openModelAction.setStatusTip("Open a Model File (.mean, .full, .sta and .end).")
        openModelAction.setToolTip("Open a model-file. (.mean, .full and .end)")
        openModelAction.triggered.connect(self.showLoadModelDialog)

        # --- "Load parameter-File" button config

        self.openParAction = QtWidgets.QAction(QtGui.QIcon("open.png"), "Load &parameter File", self)
        self.openParAction.setShortcut("Ctrl+P")
        self.openParAction.setStatusTip("Open a parameter (.par)")
        self.openParAction.setToolTip("Open an eos-file.")
        self.openParAction.triggered.connect(self.showLoadParDialog)
        self.openParAction.setDisabled(True)

        # --- "Load EOS-File" button config

        self.openEosAction = QtWidgets.QAction(QtGui.QIcon("open.png"), "Load &EOS File", self)
        self.openEosAction.setShortcut("Ctrl+E")
        self.openEosAction.setStatusTip("Open an equation of state file (.eos)")
        self.openEosAction.setToolTip("Open an eos-file.")
        self.openEosAction.triggered.connect(self.showLoadEosDialog)
        self.openEosAction.setDisabled(True)

        # --- "Load opacity file" button config

        self.openOpaAction = QtWidgets.QAction(QtGui.QIcon("open.png"), "Load &opacity File", self)
        self.openOpaAction.setShortcut("Ctrl+O")
        self.openOpaAction.setStatusTip("Open an opacity file (.opta)")
        self.openOpaAction.setToolTip("Open an opacity file.")
        self.openOpaAction.triggered.connect(self.showLoadOpaDialog)
        self.openOpaAction.setDisabled(True)

        # --- "Exit" button config

        exitAction = QtWidgets.QAction(QtGui.QIcon("exit.png"), "&Exit", self)
        exitAction.setShortcut("Ctrl+Q")
        exitAction.setStatusTip("Exit application.")
        exitAction.setToolTip("Exit application.")
        exitAction.triggered.connect(self.close)

        # --------------------------------------------------------------------
        # ----------------- "Window" drop-down menu elements -----------------
        # --------------------------------------------------------------------

        stratificationAction = QtWidgets.QAction("Stratification", self)
        stratificationAction.setStatusTip("Plot quantities over optical or barometric height.")
        stratificationAction.setToolTip("Plot quantities over optical or barometric height.")
        stratificationAction.triggered.connect(self.showStratiWindow)

        # --------------------------------------------------------------------
        # ----------------- "Output" drop-down menu elements -----------------
        # --------------------------------------------------------------------

        saveImageAction = QtWidgets.QAction("Save &Image", self)
        saveImageAction.setShortcut("Ctrl+I")
        saveImageAction.setStatusTip("Save current plot, or sequences to image files.")
        saveImageAction.setToolTip("Save current plot, or sequences to image files")
        saveImageAction.triggered.connect(self.showImageSaveDialog)

        saveSliceHD5Action = QtWidgets.QAction("Save Slice", self)
        saveSliceHD5Action.setShortcut("Ctrl+H")
        saveSliceHD5Action.setStatusTip("Save current slice as HDF5 or FITS file.")
        saveSliceHD5Action.setToolTip("Save current slice as HDF5 or FITS file.")
        saveSliceHD5Action.triggered.connect(self.showSaveSliceDialog)

        # --------------------------------------------------------------------
        # ------------------------ Initialize menubar ------------------------
        # --------------------------------------------------------------------

        menubar = QtWidgets.QMenuBar(self)
        menubar.setNativeMenuBar(False)

        # --- "File" drop-down menu elements ---

        fileMenu = QtWidgets.QMenu("&File", self)
        fileMenu.addAction(openModelAction)
        fileMenu.addSeparator()
        fileMenu.addAction(self.openParAction)
        fileMenu.addAction(self.openEosAction)
        fileMenu.addAction(self.openOpaAction)
        fileMenu.addSeparator()
        fileMenu.addAction(exitAction)

        # --- "Window" drop-down menu elements ---

        self.windowMenu = QtWidgets.QMenu("&WIndow", self)
        self.windowMenu.addAction(stratificationAction)
        self.windowMenu.setDisabled(True)

        # --- "Output" drop-down menu elements ---

        self.outputMenu = QtWidgets.QMenu("&Output", self)
        self.outputMenu.addAction(saveImageAction)
        self.outputMenu.addAction(saveSliceHD5Action)
        self.outputMenu.setDisabled(True)

        menubar.addMenu(fileMenu)
        menubar.addMenu(self.windowMenu)
        menubar.addMenu(self.outputMenu)

        self.setMenuBar(menubar)

    def showStratiWindow(self):
        print("drin")

    def showImageSaveDialog(self):
        wind.showImageSaveDialog(self.modelfile, self.data, self.timeSlider.value(), self.quantityCombo.currentText(),
                                 self.time[:, 0], self.x1Slider.value(), self.xc1, self.x2Slider.value(), self.xc2,
                                 self.x3Slider.value(), self.xc3, self.cmCombo.currentIndex(),
                                 self.quantityCombo.currentIndex())

    def showLoadModelDialog(self):
        if self.stdDirMod is None:
            self.stdDirMod = os.path.curdir

        self.fname, fil = QtWidgets.QFileDialog.getOpenFileNames(self, "Open Model File", self.stdDirMod,
                                                                 "Model files (*.full *.end *.sta);;Mean files(*.mean)")
        Nfiles = len(self.fname)
        if Nfiles == 0:
            return

        self.stdDirMod = os.sep.join(self.fname[0].split(os.sep)[:-1])

        self.statusBar().showMessage("Read Modelfile...")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

        self.modelfile = []

        pd = QtWidgets.QProgressDialog("Load Model-files...", "Cancel", 0, Nfiles, self)

        for i in range(Nfiles):
            self.modelfile.append(uio.File(self.fname[i]))
            pd.setValue(i)

            if pd.wasCanceled():
                return

        pd.setValue(len(self.fname))

        if fil == "Mean files(*.mean)":
            self.meanfile = True
            # --- content from .mean file ---
            # --- Components depict box number from filestructure (see manual of CO5BOLD)

            self.quantityList = [OrderedDict([("Bolometric intensity", "intb3_r"),
                                              ("Intensity (bin 1)", "int01b3_r"),
                                              ("Intensity (bin 2)", "int02b3_r"),
                                              ("Intensity (bin 3)", "int03b3_r"),
                                              ("Intensity (bin 4)", "int04b3_r"),
                                              ("Intensity (bin 5)", "int05b3_r")]),
                                 OrderedDict([("Avg. density (x1)", "rho_xmean"),
                                              ("Squared avg. density (x1)", "rho_xmean2")])]
        elif fil == "Model files (*.full *.end *.sta)":
            self.meanfile = False
            # --- content from .full or .end file (has one box per dataset) ---
            # --- First list component: Data from file
            # --- Second list component: Data from post computed arrays
            # --- Third list component: Post computed MHD data, if present

            self.quantityList = [OrderedDict([("Density", "rho"), ("Internal energy", "ei"),
                                              ("Velocity x-component", "v1"), ("Velocity y-component", "v2"),
                                              ("Velocity z-component", "v3"), ("Velocity, absolute", "vabs"),
                                              ("Velocity, horizontal", "vhor"), ("Kinetic energy", "kinEn"),
                                              ("Momentum", "momentum"), ("Vert. mass flux (Rho*V3)", "massfl")])]

            # enable loading of parameter-, opacity- and eos-files
            self.openOpaAction.setDisabled(False)
            self.openEosAction.setDisabled(False)
            self.openParAction.setDisabled(False)

            path = os.path.split(self.fname[0])[0]

            # If another file is loaded, set the indicator to an "uncertain" state, i.e. it is not clear, if the
            # specific file corresponds to the recently loaded model

            if self.eos:
                self.eosFileLabel.setStyleSheet('color: orange')
                self.eosFileLabel.setToolTip("You loaded a new model, while using an eos-file loaded beforehand.\n"
                                             "Are you sure that the eos-file is still valid?\n"
                                             "If yes, then click on label. Otherwise load a new eos-file.")

            if self.opa:
                self.opaFileLabel.setStyleSheet('color: orange')
                self.opaFileLabel.setToolTip("You loaded a new model, while using an opa-file loaded beforehand.\n"
                                             "Are you sure that the opa-file is still valid?\n"
                                             "If yes, then click on label. Otherwise load a new opa-file.")

            if "rhd.par" in os.listdir(path):
                self.par = True
                self.parFile = ParFile(os.path.join(path, "rhd.par"))

                opaname = os.path.join(self.parFile['opapath'].data, self.parFile['opafile'].data)
                eosname = os.path.join(self.parFile['eospath'].data, self.parFile['eosfile'].data)

                self.showLoadEosDialog(eosname=eosname)
                self.showLoadOpaDialog(opaname=opaname)
                self.parFileLabel.setStyleSheet('color: green')
                self.parFileLabel.setToolTip("Parameter-file is available.")
            else:
                if self.par:
                    self.stdDirPar = self.stdDirMod
                    self.parFileLabel.setStyleSheet('color: orange')
                    self.parFileLabel.setToolTip("You loaded a new model, while using a par-file loaded beforehand.\n"
                                                 "Are you sure that the par-file is still valid?\n"
                                                 "If yes, then click on label. Otherwise load a new par-file.")
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
                                                      ("Magnetic energy", "bener"), ("Magnetic potential Phi", "phi"),
                                                      ("Alfven speed", "ca"), ("Electric current density jx", "jx"),
                                                      ("Electric current density jy", "jy"),
                                                      ("Electric current density jz", "jz"),
                                                      ("Electric current density |j|", "jabs")]))

        else:
            self.msgBox.setText("Data format unknown.")
            self.msgBox.exec_()

            for mod in self.modelfile:
                mod.close()

        # --- Fourth list component: EOS (and opacity) table
        # interpolated data, if already loaded

        if self.eos:
            self.quantityList.append(OrderedDict([("Temperature", "temp"), ("Entropy", "entr"), ("Pressure", "press"),
                                                  ("Adiabatic coefficient G1", "gamma1"), ("Mach Number", "mach"),
                                                  ("Adiabatic coefficient G3", "gamma3"), ("Sound velocity", "c_s"),
                                                  ("Mean molecular weight", "mu"), ("Plasma beta", "beta"),
                                                  ("c_s / c_A", "csca")]))

        if self.opa:
            self.quantityList[-1]["Opacity"] = "opa"
            self.quantityList[-1]["Optical depth"] = "optdep"

        if not self.modelfile[0].closed:
            self.quantityCombo.clear()
            # self.outputMenu.setDisabled(False)
            # self.windowMenu.setDisabled(False)

            for type in self.quantityList:
                self.quantityCombo.addItems(sorted(type.keys()))

            self.quantityCombo.setDisabled(False)

            self.initialLoad()

        QtWidgets.QApplication.restoreOverrideCursor()
        self.statusBar().showMessage("Loaded {f} files".format(f=str(Nfiles)))

    def showSaveSliceDialog(self):
        if not self.stdDir:
            self.stdDir = os.path.curdir

        fname, fil = QtWidgets.QFileDialog.getSaveFileName(self, "Save current slice (HD5)", self.stdDir,
                                                           "HDF5 file (*.h5);;FITS file (*.fits)")
        if len(fname) == 0:
            return

        if fname:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

            if fil == "HDF5 file (*.h5)":
                self.statusBar().showMessage("Save HDF5-file...")
                sc.saveHD5(fname, self.modelfile[self.modelind], self.quantityCombo.currentText(),
                           self.data, self.time[self.timind, 0], (self.x1ind,
                           self.x2ind, self.x3ind), self.planeCombo.currentText())
            elif fil == "FITS file (*.fits)":
                self.statusBar().showMessage("Save FITS-file...")
                sc.saveFits(fname, self.modelfile[self.modelind], self.quantityCombo.currentText(),
                            self.data, self.time[self.timind, 0], (self.x1ind,
                            self.x2ind, self.x3ind), self.planeCombo.currentText())
            QtWidgets.QApplication.restoreOverrideCursor()

            self.statusBar().showMessage("File {f} saved".format(f=fname))

    def showLoadParDialog(self):

        if self.stdDirPar is None:
            self.stdDirPar = os.path.curdir

        parname = QtWidgets.QFileDialog.getOpenFileName(self, "Open Parameter File", self.stdDirPar,
                                                             "parameter files (*.par)")[0]

        if len(parname) == 0:
            return      # return if QFileDialog canceled

        self.stdDirPar = os.sep.join(parname.split(os.sep)[:-1])

        self.statusBar().showMessage("Read parameter file...")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

        self.par = True
        self.parFile = ParFile(parname)

        if self.parFile['opapath'].data[-1] == os.path:
            opaname = os.path.join(self.parFile['opapath'].data[:-1], self.parFile['opafile'].data)
        else:
            opaname = os.path.join(self.parFile['opapath'].data, self.parFile['opafile'].data)

        if self.parFile['eospath'].data[-1] == os.path:
            eosname = os.path.join(self.parFile['eospath'].data[:-1], self.parFile['eosfile'].data)
        else:
            eosname = os.path.join(self.parFile['eospath'].data, self.parFile['eosfile'].data)

        self.showLoadEosDialog(eosname=eosname)
        self.showLoadOpaDialog(opaname=opaname)
        self.parFileLabel.setStyleSheet('color: green')
        self.parFileLabel.setToolTip("Parameter-file is available.")

    def showLoadEosDialog(self, eosname=False):
        if self.stdDirEos is None:
            self.stdDirEos = os.path.curdir

        if not eosname:
            self.eosname = QtWidgets.QFileDialog.getOpenFileName(self, "Open EOS File", self.stdDirEos,
                                                                 "EOS files (*.eos)")[0]
        else:
            if os.path.isfile(eosname):
                self.eosname = eosname
            else:
                return

        if len(self.eosname) == 0:
            return      # return if QFileDialog canceled

        self.stdDirEos = os.sep.join(self.eosname.split(os.sep)[:-1])

        if self.eosname:
            self.statusBar().showMessage("Read EOS file...")
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

            self.Eos = EosInter(self.eosname)

            # Check if self.quantityList exists (model file already loaded?)

            if not self.eos and hasattr(self, 'quantityList'):
                self.quantityList.append(OrderedDict([("Temperature", "temp"), ("Entropy", "entr"),
                                                      ("Pressure", "press"), ("Adiabatic coefficient G1", "gamma1"),
                                                      ("Adiabatic coefficient G3", "gamma3"), ("Sound velocity", "c_s"),
                                                      ("Mach Number", "mach"), ("Mean molecular weight", "mu"),
                                                      ("Plasma beta", "beta"), ("c_s / c_A", "csca")]))

            if self.opa:
                self.quantityList[-1]["Opacity"] = "opa"
                self.quantityList[-1]["Optical depth"] = "optdep"

                self.tauUnityCheck.setDisabled(False)
            self.quantityCombo.addItems(self.quantityList[-1].keys())

            self.eos = True

            QtWidgets.QApplication.restoreOverrideCursor()

            self.eosFileLabel.setStyleSheet('color: green')
            self.eosFileLabel.setToolTip("EOS-file is available.")

            self.statusBar().showMessage("Done")

    def showLoadOpaDialog(self, opaname=False):

        if self.stdDirOpa is None:
            self.stdDirOpa = os.path.curdir

        if not opaname:
            self.opaname = QtWidgets.QFileDialog.getOpenFileName(self, "Open Opacity File", self.stdDirOpa,
                                                                 "opacity files (*.opta)")[0]
        else:
            if os.path.isfile(opaname):
                self.opaname = opaname
            else:
                return

        if len(self.opaname) == 0:
            return      # return if QFileDialog canceled

        self.stdDirOpa = os.sep.join(self.opaname.split(os.sep)[:-1])

        if self.opaname:
            self.statusBar().showMessage("Read opacity file...")
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

            self.Opa = Opac(self.opaname)
            if self.eos:
                self.quantityList[-1]["Opacity"] = "opa"
                self.quantityList[-1]["Optical depth"] = "optdep"

                self.quantityCombo.addItem("Opacity")
                self.quantityCombo.addItem("Optical depth")

                self.tauUnityCheck.setDisabled(False)
            self.opa = True

            QtWidgets.QApplication.restoreOverrideCursor()

            self.opaFileLabel.setStyleSheet('color: green')
            self.opaFileLabel.setToolTip("Opacity-file is available.")

            self.statusBar().showMessage("Done")

    def setGridLayout(self):
        # --- Main Layout with splitter ---
        # --- (enables automatic resizing of widgets when window is resized)

        maingrid = QtWidgets.QHBoxLayout(self.centralWidget)

        # --- Splitter for dynamic seperation of control elements section and plot-box

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        maingrid.addWidget(splitter)

        # --- Layout with control elements ---

        leftgrid = QtWidgets.QVBoxLayout()

        # --- Widget consisting of layout for control elements ---
        # --- (enables adding of control-elements-layout to splitter)

        leftgridwid = QtWidgets.QWidget(self.centralWidget)
        leftgridwid.setLayout(leftgrid)

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

        currentTimeTitle = QtWidgets.QLabel("t/[s]:")

        self.actualTimeLabel = QtWidgets.QLabel(str(self.timeSlider.value()))
        self.actualTimeLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.actualTimeLabel.setMaximumWidth(55)
        self.actualTimeLabel.setMinimumWidth(55)
        self.actualTimeLabel.setObjectName("actual-time-Label")

        currentFileTitle = QtWidgets.QLabel("File:")

        self.currentFileLabel = QtWidgets.QLabel("")
        self.currentFileLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.currentFileLabel.setMaximumWidth(70)
        self.currentFileLabel.setMinimumWidth(70)
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

        x3SliderTitle = QtWidgets.QLabel("z-position:")
        currentX3Title = QtWidgets.QLabel("iz:")
        actualX3Title = QtWidgets.QLabel("z [km]:")

        self.currentX3Edit = QtWidgets.QLineEdit(str(self.x3Slider.value()))
        self.currentX3Edit.setMaximumWidth(40)
        self.currentX3Edit.setMinimumWidth(40)
        self.currentX3Edit.textChanged.connect(self.currentEditChange)
        self.currentX3Edit.setObjectName("current-z-Edit")

        self.actualX3Label = QtWidgets.QLabel(str(0))
        self.actualX3Label.setAlignment(QtCore.Qt.AlignCenter)
        self.actualX3Label.setMaximumWidth(55)
        self.actualX3Label.setMinimumWidth(55)

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

        posLayout.addWidget(x3SliderTitle, 3, 0)
        posLayout.addWidget(self.x3Slider, 3, 1)
        posLayout.addWidget(currentX3Title, 3, 2)
        posLayout.addWidget(self.currentX3Edit, 3, 3)
        posLayout.addWidget(actualX3Title, 3, 4)
        posLayout.addWidget(self.actualX3Label, 3, 5)

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
        self.normMinEdit = QtWidgets.QLineEdit("{dat:13.4g}".format(dat=self.data.min()))
        self.normMinEdit.setDisabled(True)
        self.normMinEdit.textChanged.connect(self.normChange)
        self.normMinEdit.setObjectName("norm-min-Edit")

        normMaxTitle = QtWidgets.QLabel("Max:")
        self.normMaxEdit = QtWidgets.QLineEdit("{dat:13.4g}".format(dat=self.data.max()))
        self.normMaxEdit.setDisabled(True)
        self.normMaxEdit.textChanged.connect(self.normChange)
        self.normMaxEdit.setObjectName("norm-max-Edit")

        normMeanTitle = QtWidgets.QLabel("Mean:")
        self.normMeanLabel = QtWidgets.QLabel("{dat:13.4g}".format(dat=self.data.mean()))
        self.normMeanLabel.setDisabled(True)

        unitTitle = QtWidgets.QLabel("Unit:")
        self.unitLabel = QtWidgets.QLabel("")

        # --- ComboBox with math-selection ---

        self.mathCombo = QtWidgets.QComboBox(self.centralWidget)
        self.mathCombo.clear()
        self.mathCombo.setDisabled(True)
        self.mathCombo.activated.connect(self.mathComboChange)
        self.mathCombo.addItems(self.postmath.keys())
        self.mathCombo.setCurrentText("----")
        self.mathCombo.setObjectName("math-Combo")

        # --- Radiobuttons for 2D-3D-selection ---

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

        # --- Setup of data-presentation-layout ---

        dataParamsLayout.addWidget(quantityLabel, 0, 0)
        dataParamsLayout.addWidget(self.quantityCombo, 0, 1, 1, 3)
        dataParamsLayout.addWidget(self.normCheck, 0, 4, 1, 2)
        dataParamsLayout.addWidget(self.mathCombo, 0, 6, 1, 1)

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

        dataParamsLayout.addWidget(twoDTitle, 4, 0)
        dataParamsLayout.addWidget(self.twoDRadio, 4, 1)

        dataParamsLayout.addWidget(threeDTitle, 5, 0)
        dataParamsLayout.addWidget(self.threeDRadio, 5, 1)

        dataParamsLayout.addWidget(tauUnityTitle, 4, 2)
        dataParamsLayout.addWidget(self.tauUnityCheck, 4, 3)

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
        self.vpScaleEdit.setDisabled(True)
        self.vpScaleEdit.textChanged.connect(self.generalPlotRoutine)

        vpXIncLabel = QtWidgets.QLabel("x-increment:")
        self.vpXIncEdit = QtWidgets.QLineEdit("{dat:8d}".format(dat=4))
        self.vpXIncEdit.setDisabled(True)
        self.vpXIncEdit.textChanged.connect(self.generalPlotRoutine)

        vpYIncLabel = QtWidgets.QLabel("y-increment:")
        self.vpYIncEdit = QtWidgets.QLineEdit("{dat:8d}".format(dat=4))
        self.vpYIncEdit.setDisabled(True)
        self.vpYIncEdit.textChanged.connect(self.generalPlotRoutine)

        vpAlphaLabel = QtWidgets.QLabel("Vector-opacity:")
        self.vpAlphaEdit = QtWidgets.QLineEdit("{dat:8d}".format(dat=1))
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

        # ---------------------------------------------------------------------
        # -------------- Groupbox with file-state indicators ------------------
        # ---------------------------------------------------------------------

        fileStateGroup = QtWidgets.QGroupBox("File available", self.centralWidget)
        fileStateLayout = QtWidgets.QHBoxLayout(fileStateGroup)
        fileStateGroup.setLayout(fileStateLayout)

        self.parFileLabel = QtWidgets.QLabel("parameter-file")
        self.parFileLabel.setStyleSheet('color: red')
        self.parFileLabel.setToolTip("Parameter-file is not available.")
        self.parFileLabel.setObjectName("parfilelabel")
        self.parFileLabel.mousePressEvent = self.labelParClick

        self.eosFileLabel = QtWidgets.QLabel("eos-file")
        self.eosFileLabel.setStyleSheet('color: red')
        self.eosFileLabel.setToolTip("EOS-file is not available.")
        self.eosFileLabel.setObjectName("eosfilelabel")
        self.eosFileLabel.mousePressEvent = self.labelEosClick

        self.opaFileLabel = QtWidgets.QLabel("opa-file")
        self.opaFileLabel.setStyleSheet('color: red')
        self.opaFileLabel.setToolTip("Opacity-file is not available.")
        self.opaFileLabel.setObjectName("opafilelabel")
        self.opaFileLabel.mousePressEvent = self.labelOpaClick

        fileStateLayout.addWidget(self.parFileLabel)
        fileStateLayout.addWidget(self.eosFileLabel)
        fileStateLayout.addWidget(self.opaFileLabel)

        # ---------------------------------------------------------------------
        # ---------------------------- Plot window ----------------------------
        # ---------------------------------------------------------------------

        self.plotBox = sc.PlotWidget(self.centralWidget)
        self.plotBox.mpl_connect("motion_notify_event", self.dataPlotMotion)
        self.plotBox.mpl_connect("button_press_event", self.dataPlotPress)

        # self.threeDPlotBox = sc.PlotWidget3D(self.centralWidget)

        # --- Add plot-widget and widget consisting of left layout to splitter

        splitter.addWidget(leftgridwid)
        splitter.addWidget(self.plotBox)
        # splitter.addWidget(self.threeDPlotBox)
        # self.threeDPlotBox.hide()

        # --- Fill up left layout with groups ---

        leftgrid.addWidget(timeGroup)
        leftgrid.addWidget(posGroup)
        leftgrid.addWidget(dataParamsGroup)
        leftgrid.addWidget(vectorPlotGroup)
        leftgrid.addWidget(fileStateGroup)

        self.centralWidget.setLayout(maingrid)
        self.setCentralWidget(self.centralWidget)

    def initialLoad(self):
        start = time.time()
        # --- Initiate post-computed arrays ---

        self.xc1 = self.modelfile[0].dataset[0].box[0]["xc1"].data.squeeze()*1.e-5
        self.xc2 = self.modelfile[0].dataset[0].box[0]["xc2"].data.squeeze()*1.e-5
        self.xc3 = self.modelfile[0].dataset[0].box[0]["xc3"].data.squeeze()*1.e-5

        self.xb1 = self.modelfile[0].dataset[0].box[0]["xb1"].data.squeeze()*1.e-5
        self.xb2 = self.modelfile[0].dataset[0].box[0]["xb2"].data.squeeze()*1.e-5
        self.xb3 = self.modelfile[0].dataset[0].box[0]["xb3"].data.squeeze()*1.e-5

        self.dx = np.diff(self.xb1).mean()
        self.dy = np.diff(self.xb2).mean()
        self.dz = np.diff(self.xb3).mean()

        self.constGrid = np.diff(self.xb3).std() < 0.01

        if len(self.modelfile):
            self.time = []
            for i in range(len(self.modelfile)):
                for j in range(len(self.modelfile[i].dataset)):
                    self.time.append([self.modelfile[i].dataset[j]["modeltime"].data, i, j])
        self.time = np.array(self.time)
        self.timlen = len(self.time[:,0])

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

        # --- determine slider boundaries ---

        if self.timlen > 1:
            self.timeSlider.setDisabled(False)
            self.timeSlider.setMaximum(self.timlen-1)

            self.prevTimeBtn.setDisabled(False)
            self.nextTimeBtn.setDisabled(False)

        self.x1Slider.setMaximum(len(self.xc1)-1)
        self.x2Slider.setMaximum(len(self.xc2)-1)
        self.x3Slider.setMaximum(len(self.xc3)-1)

        self.normMinEdit.setDisabled(False)
        self.normMaxEdit.setDisabled(False)

        if self.meanfile:
            self.vpAlphaEdit.setDisabled(True)
            self.vpCheck.setDisabled(True)
            self.vpMagRadio.setDisabled(True)
            self.vpScaleEdit.setDisabled(True)
            self.vpVelRadio.setDisabled(True)
            self.vpXIncEdit.setDisabled(True)
            self.vpYIncEdit.setDisabled(True)
        else:
            self.vpAlphaEdit.setDisabled(False)
            self.vpCheck.setDisabled(False)
            self.vpMagRadio.setDisabled(False)
            self.vpScaleEdit.setDisabled(False)
            self.vpVelRadio.setDisabled(False)
            self.vpXIncEdit.setDisabled(False)
            self.vpYIncEdit.setDisabled(False)

        self.mathCombo.setDisabled(False)
        self.crossCheck.setDisabled(False)

        # --------------------------------------
        # ---- update parameters of widgets ----
        # --------------------------------------

        self.timind = self.timeSlider.value()
        self.currentTimeEdit.setText(str(self.timind).rjust(4))

        self.modelind = 0
        self.dsind = 0

        self.plot = False
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

        self.dim = self.modelfile[self.modelind].dataset[self.dsind].box[self.boxind][self.typeind].data.ndim

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

            # self.threeDRadio.setDisabled(False)
            self.twoDRadio.setDisabled(False)

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

            self.threeDRadio.setDisabled(True)
            self.twoDRadio.setDisabled(True)

            self.tauUnityCheck.setDisabled(False)

            self.direction = self.modelfile[self.modelind].dataset[self.dsind].box[self.boxind][self.typeind].shape.\
                index(1)
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

            self.threeDRadio.setDisabled(True)
            self.twoDRadio.setDisabled(True)

            self.tauUnityCheck.setDisabled(True)

            self.direction = bisect.bisect(self.modelfile[self.modelind].dataset[self.dsind].box[self.boxind][self.
                                           typeind].shape, 2)
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
        self.data = self.setPlotData(self.modelind, self.dsind)

        self.unitLabel.setText(self.unit)
        if self.normCheck.checkState() == QtCore.Qt.Checked:
            self.getTotalMinMax()

        self.cmInvert.setDisabled(False)

        if self.cmInvert.checkState() == QtCore.Qt.Checked:
            self.invertCM(True)
        else:
            self.invertCM(False)

        self.planeCheck()
        print("Time needed for initial load:", time.time()-start)

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
                self.generalPlotRoutine()
            else:
                self.msgBox.setText("EOS or opacity-file not loaded!\nEOS: {0}\topacity: {1}".format(self.eos, self.opa))
        else:
            self.tauheight = None

    def getTotalMinMax(self):
        if self.dim == 3:
            self.globBound = []

            xmin, ymin, zmin, xmax, ymax, zmax = ([] for _ in range(6))

            for i, mod in enumerate(self.modelfile):
                for j, dat in enumerate(mod.dataset):
                    data = self.setPlotData(i, j)
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
            for i, mod in enumerate(self.modelfile):
                for j, dat in enumerate(mod.dataset):
                    data = self.setPlotData(i, j)
                    min.append(data.min())
                    max.append(data.max())
            self.globBound = [min, max]

    def labelParClick(self, event):
        if self.par:
            self.parFileLabel.setStyleSheet('color: green')
            self.parFileLabel.setToolTip("Parameter-file is available.")

    def labelEosClick(self, event):
        if self.eos:
            self.eosFileLabel.setStyleSheet('color: green')
            self.eosFileLabel.setToolTip("EOS-file is available.")

    def labelOpaClick(self, event):
        if self.opa:
            self.opaFileLabel.setStyleSheet('color: green')
            self.opaFileLabel.setToolTip("Opa-file is available.")

    def normCheckChange(self, state):
        if state == QtCore.Qt.Checked:
            self.getTotalMinMax()

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

    def setPlotData(self, mod, dat):
        self.statusBar().showMessage("Initialize arrays...")
        start = time.time()
        clight = 2.998e10
        const = 4.0 * np.pi

        ver = np.version.version
        # self.constGrid = False

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

        if not self.meanfile:
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
                x3 = self.modelfile[0].dataset[0].box[0]["xb3"].data.squeeze()*1.e-5
                bb3 = self.modelfile[mod].dataset[dat].box[0]["bb3"].data
                dz = np.diff(x3)

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
                data = self.modelfile[mod].dataset[dat].box[self.boxind][self.typeind].data
                self.unit = self.modelfile[mod].dataset[dat].box[self.boxind][self.typeind].\
                    params["u"]
        else:
            data = self.modelfile[mod].dataset[dat].box[self.boxind][self.typeind].data
            self.unit = self.modelfile[mod].dataset[dat].box[self.boxind][self.typeind].\
                params["u"]
        QtWidgets.QApplication.restoreOverrideCursor()
        text = "time needed for evaluation: {0:5.3g} s".format(time.time()-start)
        self.statusBar().showMessage(text)
        return ne.evaluate(self.postmath[self.mathCombo.currentText()], local_dict={'data': data})


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

    def mathComboChange(self):
        self.data = self.setPlotData(self.modelind, self.dsind)
        self.planeCheck()

    def SliderChange(self):
        sender = self.sender()

        if sender.objectName() == "time-Slider":
            self.timind = self.timeSlider.value()
            self.actualTimeLabel.setText("{dat:10.1f}".format(dat=self.time[self.timind, 0]))
            self.currentTimeEdit.setText(str(self.timind))

            self.modelind = int(self.time[self.timind, 1])
            self.dsind = int(self.time[self.timind, 2])
            fname = self.fname[self.modelind].split("/")[-1]
            self.currentFileLabel.setText(fname)

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
            self.actualX3Label.setText("{:10.1f}".format(self.xc3[self.x3ind]).rjust(0))

        self.planeCheck()

    def dataPlotMotion(self, event):
        try:
            if self.dim == 0:
                self.statusBar().showMessage("x: {xdat:13.6g}\ty: {ydat:13.6g}".format(xdat=event.xdata,
                                                                                       ydat=event.ydata))
            elif self.dim == 1:
                self.statusBar().showMessage("x: {xdat:13.6g} km\ty: {ydat:13.6g} {unit}".format(xdat=event.xdata,
                                                                                                  ydat=event.ydata,
                                                                                                  unit=self.unit))
                self.plotBox.setToolTip("x: {xdat:13.6g} km\ny: {ydat:13.6g} {unit}".format(xdat=event.xdata,
                                                                                            ydat=event.ydata,
                                                                                            unit=self.unit))
            elif self.dim == 2:
                idx = (np.abs(self.xc1-event.xdata)).argmin()
                idy = (np.abs(self.xc2-event.ydata)).argmin()

                self.statusBar().showMessage("x: {xdat:13.6g} km   y: {ydat:13.6g} km    value: {dat:13.6g} {unit}".
                                             format(xdat=event.xdata, ydat=event.ydata, dat = self.data[0, idy, idx],
                                                    unit=self.unit))
                self.plotBox.setToolTip("x: {xdat:13.6g} km\ny: {ydat:13.6g} km\nvalue: {dat:13.6g} {unit}".
                                        format(xdat=event.xdata, ydat=event.ydata, dat = self.data[0, idy, idx],
                                               unit=self.unit))

            elif self.dim == 3:
                if self.planeCombo.currentText() == "xy":
                    idx = (np.abs(self.xc1 - event.xdata)).argmin()
                    idy = (np.abs(self.xc2 - event.ydata)).argmin()

                    self.statusBar().showMessage("x: {xdat:13.6g} km\ty: {ydat:13.6g} km\tvalue: {dat:13.6g} {unit}".
                                                 format(xdat=event.xdata, ydat=event.ydata,
                                                        dat=self.data[self.x3ind, idy, idx], unit=self.unit))
                    self.plotBox.setToolTip("x: {xdat:13.6g} km\ny: {ydat:13.6g} km\nvalue: {dat:13.6g} {unit}".format(
                        xdat=event.xdata, ydat=event.ydata, dat=self.data[self.x3ind, idy, idx], unit=self.unit))

                elif self.planeCombo.currentText() == "xz":
                    idx = (np.abs(self.xc1 - event.xdata)).argmin()
                    idz = (np.abs(self.xc3 - event.ydata)).argmin()

                    self.statusBar().showMessage("x: {xdat:13.6g} km\tz: {zdat:13.6g} km\tvalue: {dat:13.6g} {unit}".
                                                 format(xdat=event.xdata, zdat=event.ydata,
                                                        dat=self.data[idz, self.x2ind, idx], unit=self.unit))
                    self.plotBox.setToolTip("x: {xdat:13.6g} km\nz: {zdat:13.6g} km\nvalue: {dat:13.6g} {unit}".
                                            format(xdat=event.xdata, zdat=event.ydata,
                                                   dat=self.data[idz, self.x2ind, idx], unit=self.unit))

                elif self.planeCombo.currentText() == "yz":
                    idy = (np.abs(self.xc2 - event.xdata)).argmin()
                    idz = (np.abs(self.xc3 - event.ydata)).argmin()

                    self.statusBar().showMessage("y: {ydat:13.6g} km\tz: {zdat:13.6g} km\tvalue: {dat:13.6g} {unit}".
                                                 format(ydat=event.xdata, zdat=event.ydata,
                                                        dat=self.data[idz, idy, self.x1ind], unit=self.unit))
                    self.plotBox.setToolTip("y: {ydat:13.6g} km\nz: {zdat:13.6g} km\nvalue: {dat:13.6g} {unit}".
                                            format(ydat=event.xdata, zdat=event.ydata,
                                                   dat=self.data[idz, idy, self.x1ind], unit=self.unit))

                sc.PlotWidget.linePlot(event.xdata, event.ydata)
        except Exception:
            pass

    def dataPlotPress(self, event):
        if event.xdata is not None and event.ydata is not None:
            self.pos = np.array([event.xdata, event.ydata])
            if self.dim == 3:
                if self.planeCombo.currentText() == "xy":
                    idx = (np.abs(self.xc1 - event.xdata)).argmin()
                    idy = (np.abs(self.xc2 - event.ydata)).argmin()

                    self.x1Slider.setValue(idx)
                    self.x2Slider.setValue(idy)

                elif self.planeCombo.currentText() == "xz":
                    idx = (np.abs(self.xc1 - event.xdata)).argmin()
                    idz = (np.abs(self.xc3 - event.ydata)).argmin()

                    self.x1Slider.setValue(idx)
                    self.x3Slider.setValue(idz)

                elif self.planeCombo.currentText() == "yz":
                    idy = (np.abs(self.xc2 - event.xdata)).argmin()
                    idz = (np.abs(self.xc3 - event.ydata)).argmin()

                    self.x2Slider.setValue(idy)
                    self.x3Slider.setValue(idz)
        else:
            pass
        if self.crossCheck.isChecked():
            self.generalPlotRoutine()

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
                if int(self.currentX3Edit.text()) > len(self.xc3):
                    self.currentX3Edit.setText(str(len(self.xc3)-1))
                elif int(self.currentX2Edit.text()) < 0:
                    self.currentX3Edit.setText(str(0))

                self.x3ind = int(self.currentX3Edit.text())
                self.x3Slider.setValue(self.x3ind)

            self.statusBar().showMessage("")
        except:
            self.statusBar().showMessage("Invalid input in currentEditChange!")

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

    # -----------------------------------------------
    # --- Change of 2D- to 3D-plot and vice versa ---
    # -----------------------------------------------

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

    def quantityChange(self):

        self.boxind = -1

        for i in range(len(self.quantityList)):
            if self.quantityCombo.currentText() in self.quantityList[i].keys():
                self.boxind = i
                break

        # --- First case: .mean file (all data from file)
        # --- Second case: .full or .end file and in first component of
        # ---               quantityList (data from file)
        # --- Else: Data from post computed arrays (not yet implemented)

        self.typeind = self.quantityList[self.boxind][self.quantityCombo.currentText()]

        # ---------------------------------------------------------------------
        # --- get new globally minimal and maximal values for normalization ---
        # ---------------------------------------------------------------------

        if self.normCheck.checkState() == QtCore.Qt.Checked:
            self.getTotalMinMax()

        # --------------------------------------
        # --- update parameters from widgets ---

        self.data = self.setPlotData(self.modelind, self.dsind)

        self.dim = 3 - self.data.shape.count(1)

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

            self.direction = self.modelfile[self.modelind].dataset[self.dsind].box[self.boxind][self.typeind].\
                shape.index(1)
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

            self.direction = bisect.bisect(self.modelfile[self.modelind].dataset[self.dsind].box[self.boxind]
                                           [self.typeind].shape, 2)

        self.unitLabel.setText(self.unit)

        self.planeCheck()

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
                x1 = self.modelfile[0].dataset[0].box[0]["xb1"].data.squeeze() * 1.e-5
                x2 = self.modelfile[0].dataset[0].box[0]["xb2"].data.squeeze() * 1.e-5
                x3 = self.modelfile[0].dataset[0].box[0]["xb3"].data.squeeze() * 1.e-5

                bb1 = self.modelfile[self.modelind].dataset[self.dsind].box[0]["bb1"].data*const
                bb2 = self.modelfile[self.modelind].dataset[self.dsind].box[0]["bb2"].data*const
                bb3 = self.modelfile[self.modelind].dataset[self.dsind].box[0]["bb3"].data*const

                self.u = ip.interp1d(x1, bb1, copy=False, assume_sorted=True)(self.xc1)
                self.v = ip.interp1d(x2, bb2, axis=1, copy=False, assume_sorted=True)(self.xc2)
                self.w = ip.interp1d(x3, bb3, axis=0, copy=False, assume_sorted=True)(self.xc3)

            self.generalPlotRoutine()
        else:
            self.vpMagRadio.setDisabled(True)
            self.vpVelRadio.setDisabled(True)

            self.generalPlotRoutine()

    def generalPlotRoutine(self):
        if self.crossCheck.isChecked():
            pos = self.pos
        else:
            pos = None
        if self.dim == 3:
            if self.threeDRadio.isChecked():
                pass
#                self.threeDPlotBox.Plot(self.data)
#                if self.vpCheck.isChecked():
#                    self.threeDPlotBox.visualization.update_vectors(self.u,
#                        self.v, self.w, float(self.vpXIncEdit.text()))
            elif self.twoDRadio.isChecked():
                if self.planeCombo.currentText() == "xy":
                    window = np.array([[self.x1min, self.x1max], [self.x2min, self.x2max]])
                    self.plotBox.plotFig(self.data[self.x3ind], window, vmin=self.minNorm, vmax=self.maxNorm,
                                         cmap=self.cmCombo.currentCmap, pos=pos)
                    if self.vpCheck.isChecked():
                        self.plotBox.vectorPlot(self.xc1, self.xc2, self.u[self.x3ind], self.v[self.x3ind],
                                                xinc=int(self.vpXIncEdit.text()), yinc=int(self.vpYIncEdit.text()),
                                                scale=float(self.vpScaleEdit.text()),
                                                alpha=float(self.vpAlphaEdit.text()))
                elif self.planeCombo.currentText() == "xz":
                    window = np.array([[self.x1min, self.x1max], [self.x3min, self.x3max]])

                    if self.tauUnityCheck.isChecked():
                        self.plotBox.plotFig(self.data[:, self.x2ind, :], window, vmin=self.minNorm, vmax=self.maxNorm,
                                             cmap=self.cmCombo.currentCmap, pos=pos,
                                             tauUnity=(self.xc1, self.tauheight[self.x2ind]))
                    else:
                        self.plotBox.plotFig(self.data[:, self.x2ind, :], window, vmin=self.minNorm, vmax=self.maxNorm,
                                             cmap=self.cmCombo.currentCmap, pos=pos)

                    if self.vpCheck.isChecked():
                        self.plotBox.vectorPlot(self.xc1, self.xc3, self.u[:, self.x2ind], self.w[:, self.x2ind],
                                                xinc=int(self.vpXIncEdit.text()), yinc=int(self.vpYIncEdit.text()),
                                                scale=float(self.vpScaleEdit.text()),
                                                alpha=float(self.vpAlphaEdit.text()))
                elif self.planeCombo.currentText() == "yz":
                    window = np.array([[self.x2min, self.x2max], [self.x3min, self.x3max]])
                    if self.tauUnityCheck.isChecked():
                        self.plotBox.plotFig(self.data[:, :, self.x1ind], window, vmin=self.minNorm, vmax=self.maxNorm,
                                             cmap=self.cmCombo.currentCmap, pos=pos,
                                             tauUnity=(self.xc2, self.tauheight[:, self.x1ind]))
                    else:
                        self.plotBox.plotFig(self.data[:, :, self.x1ind], window, vmin=self.minNorm, vmax=self.maxNorm,
                                             cmap=self.cmCombo.currentCmap, pos=pos)
                    if self.vpCheck.isChecked():
                        self.plotBox.vectorPlot(self.xc2, self.xc3, self.v[:, :, self.x1ind], self.w[:, :, self.x1ind],
                                                xinc=int(self.vpXIncEdit.text()), yinc=int(self.vpYIncEdit.text()),
                                                scale=float(self.vpScaleEdit.text()),
                                                alpha=float(self.vpAlphaEdit.text()))
                else:
                    self.msgBox.setText("Plane could not be identified.")
                    self.msgBox.exec_()
            else:
                self.msgBox.setText("Dimension of plot could not be identified.")
                self.msgBox.exec_()
        elif self.dim == 2:
            if self.direction == 0:
                window = np.array([[self.x1min, self.x1max], [self.x2min, self.x2max]])
                self.plotBox.plotFig(self.data[0], window, vmin=self.minNorm, vmax=self.maxNorm,
                                     cmap=self.cmCombo.currentCmap, pos=pos)
            elif self.direction == 1:
                window = np.array([[self.x1min, self.x1max], [self.x3min, self.x3max]])
                self.plotBox.plotFig(self.data[:, 0], window, vmin=self.minNorm, vmax=self.maxNorm,
                                     cmap=self.cmCombo.currentCmap, pos=pos)
            elif self.direction == 2:
                window = np.array([[self.x2min, self.x2max], [self.x3min, self.x3max]])
                self.plotBox.plotFig(self.data[:, :, 0], window, vmin=self.minNorm, vmax=self.maxNorm,
                                     cmap=self.cmCombo.currentCmap, pos=pos)
            else:
                self.msgBox.setText("Direction could not be identified.")
                self.msgBox.exec_()

        elif self.dim == 1:
            if self.direction == 0:
                window = np.array([self.x3min, self.x3max])
                self.plotBox.plotFig(self.data[:, 0, 0], window)
            elif self.direction == 1:
                window = np.array([self.x2min, self.x2max])
                self.plotBox.plotFig(self.data[0, :, 0], window)
            elif self.direction == 2:
                window = np.array([self.x1min, self.x1max])
                self.plotBox.plotFig(self.data[0, 0], window)
            else:
                self.msgBox.setText("Direction could not be identified.")
                self.msgBox.exec_()
        else:
            self.msgBox.setText("Dimension not legal.")
            self.msgBox.exec_()

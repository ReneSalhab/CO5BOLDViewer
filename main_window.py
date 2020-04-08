# -*- coding: utf-8 -*-
"""
Originally created on Tue Nov 05 10:12:33 2013

@author: René Georg Salhab

"""

import os
import numpy as np
from collections import OrderedDict
from PyQt5 import QtCore, QtGui, QtWidgets

import uio
from opta import Opac
import subclasses as sc
import windows as wind
from par import ParFile
from eosinter import EosInter
from nicole import Model, Profile
from MayaViPlotWidget import MayaViWidget as ThreeDPlotter


class MainWindow(wind.BasicWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("CO5BOLDViewer {}".format(self.version))
        self.setGeometry(100, 100, 1000, 700)

        QtWidgets.QToolTip.setFont(QtGui.QFont('SansSerif', 10))

        self.initParams()
        self.setMenu()
        self.addWidgets()

        self.show()

    def initParams(self):

        # --- Read log-file if existing ---

        logfile = os.path.join(os.curdir, 'init.log')

        self.stdDirMod = None
        self.stdDirPar = None
        self.stdDirOpa = None
        self.stdDirEos = None

        if os.path.exists(logfile):
            with open(logfile, 'r') as log:
                for line in log:
                    if 'stdDirMod' in line:
                        self.stdDirMod = line.split()[-1]
                    elif 'stdDirOpa' in line:
                        self.stdDirOpa = line.split()[-1]
                    elif 'stdDirEOS' in line:
                        self.stdDirEos = line.split()[-1]

        # --- Axes of plot ---

        self.xc1 = np.linspace(0, 99, num=100)
        self.xc2 = np.linspace(0, 99, num=100)
        self.xc3 = np.linspace(0, 99, num=100)

        # --- Data-array for plotting ---

        self.data = np.outer(self.xc1, self.xc2)

        # --- eos- and opta-file-names ---

        self.eosname = False
        self.opaname = False

        self.senders = []

        # other parameters initialized in BasicWindow class

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

        multiPlotAction = QtWidgets.QAction("Multi-Plot Window", self)
        multiPlotAction.setStatusTip("Open window with multi-plot ability.")
        multiPlotAction.setToolTip("Open window with multi-plot ability.")
        multiPlotAction.triggered.connect(self.showMultiPlot)

        # --------------------------------------------------------------------
        # ----------------- "Output" drop-down menu elements -----------------
        # --------------------------------------------------------------------

        saveImageAction = QtWidgets.QAction("Save &Image", self)
        saveImageAction.setShortcut("Ctrl+I")
        saveImageAction.setStatusTip("Save current plot, or sequences to image files.")
        saveImageAction.setToolTip("Save current plot, or sequences to image files")
        saveImageAction.triggered.connect(self.showSaveDialog)

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

        self.windowMenu = QtWidgets.QMenu("&Window", self)
        self.windowMenu.addAction(multiPlotAction)
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

    def addWidgets(self):
        # BasicWindow consists of all elements, but plot-element. Therefore, layout is already set. Only plot-area has
        # to be defined

        # ---------------------------------------------------------------------
        # ---------------------------- Plot window ----------------------------
        # ---------------------------------------------------------------------

        self.plotBox = sc.PlotWidget(self.centralWidget)
        self.plotBox.mpl_connect("motion_notify_event", self.dataPlotMotion)
        self.plotBox.mpl_connect("button_press_event", self.dataPlotPress)

        # self.vtkPlot = sc.VTKPlotWidget(self.centralWidget)
        # self.vtkPlot.hide()

        # ---------------------------------------------------------------------
        # -------------- Groupbox with file-state indicators ------------------
        # ---------------------------------------------------------------------

        fileStateGroup = QtWidgets.QGroupBox("File availability", self.centralWidget)
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

        self.threeDPlotBox = ThreeDPlotter()

        # --- Add plot-widget to inhereted splitter

        self.splitter.addWidget(self.plotBox)

        # --- Add aditional groups to control panel ---

        self.controlgrid.addWidget(fileStateGroup)

    def showSaveDialog(self):
        pass
        # wind.showImageSaveDialog(self.modelfile, self.data, self.timeSlider.value(), self.quantityCombo.currentText(),
        #                          self.time[:, 0], self.x1Slider.value(), self.xc1, self.x2Slider.value(), self.xc2,
        #                          self.x3Slider.value(), self.xc3, self.cmCombo.currentIndex(),
        #                          self.quantityCombo.currentIndex())

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
                sc.saveHD5(fname, self.modelfile[self.modelind], self.quantityCombo.currentText(), self.data,
                           self.time[self.timind, 0], (self.x1ind, self.x2ind, self.x3ind), self.planeCombo.currentText())
            elif fil == "FITS file (*.fits)":
                self.statusBar().showMessage("Save FITS-file...")
                sc.saveFits(fname, self.modelfile[self.modelind], self.quantityCombo.currentText(), self.data,
                            self.time[self.timind, 0], (self.x1ind, self.x2ind, self.x3ind), self.planeCombo.currentText())
            QtWidgets.QApplication.restoreOverrideCursor()

            self.statusBar().showMessage("File {f} saved".format(f=fname))

    # --------------------
    # --- Load dialogs ---
    # --------------------

    def showLoadModelDialog(self):
        if self.stdDirMod is None:
            self.stdDirMod = os.path.curdir

        # get list of model-file-names
        fname, fil = QtWidgets.QFileDialog.getOpenFileNames(self, "Open Model File", self.stdDirMod,
                                                                 "Model files (*.full *.end *.sta);;Mean files(*.mean);;"
                                                                 "NICOLE profiles (*.prof);;NICOLE model files (*.bin)")
        Nfiles = len(fname)
        if Nfiles == 0:
            return
        self.fname = fname

        # set standard directory for Model Load-Dialog to current directory
        self.stdDirMod = "/".join(self.fname[0].split("/")[:-1])

        self.statusBar().showMessage("Read Modelfile(s)...")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

        # if modelfile is already existent, close all files
        if isinstance(self.modelfile, list):
            if len(self.modelfile) > 0:
                for mod in self.modelfile:
                    mod.close()

        self.modelfile = []

        if fil == "Mean files(*.mean)":
            self.fileType = "mean"

            self.showProgressBar(uio.File)
            if len(self.modelfile) == 0:
                return

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
        elif fil == "Model files (*.full *.end *.sta)":
            self.fileType = "cobold"

            self.showProgressBar(uio.File)
            if len(self.modelfile) == 0:
                return

            # --- content from .full or .end file (has one box per dataset) ---
            # --- First list component: Data from file
            # --- Second list component: Data from post computed arrays
            # --- Third list component: Post computed MHD data, if present

            self.quantityList = [OrderedDict([("Density", "rho"), ("Internal energy", "ei"),
                                              ("Velocity (x-component)", "v1"), ("Velocity (y-component)", "v2"),
                                              ("Velocity (z-component)", "v3"), ("Velocity, absolute", "vabs"),
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
                try:
                    parpath = os.path.normpath(os.path.join(path, "rhd.par"))
                    self.parFile = ParFile(parpath)
                    self.par = True

                    opaname = os.path.join(self.parFile['opapath'].data, self.parFile['opafile'].data)
                    eosname = os.path.join(self.parFile['eospath'].data, self.parFile['eosfile'].data)

                    self.showLoadEosDialog(eosname=eosname)
                    self.showLoadOpaDialog(opaname=opaname)
                    self.parFileLabel.setStyleSheet('color: green')
                    self.parFileLabel.setToolTip("Parameter-file is available.")
                except OSError:
                    self.par = False
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
                                                      ("Magnetic energy", "bener"),# ("Magnetic potential Phi", "phi"),
                                                      ("Alfven speed", "ca"), ("Electric current density jx", "jx"),
                                                      ("Electric current density jy", "jy"),
                                                      ("Electric current density jz", "jz"),
                                                      ("Electric current density |j|", "jabs")]))
        elif fil == "NICOLE profiles (*.prof)":
            self.fileType = "profile"
            self.eos = False
            self.opa = False

            self.showProgressBar(Profile)
            if len(self.modelfile) == 0:
                return

            self.quantityList = [OrderedDict([("Stokes I", "I"), ("Stokes Q", "Q"), ("Stokes U", "U"),
                                              ("Stokes V", "V")])]
        elif fil == "NICOLE model files (*.bin)":
            self.fileType = "nicole"
            self.eos = False
            self.opa = False

            self.showProgressBar(Model)
            if len(self.modelfile) == 0:
                return

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
            self.msgBox.setText("Data format unknown.")
            self.msgBox.exec_()

            for mod in self.modelfile:
                mod.close()

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
            # self.outputMenu.setDisabled(False)
            # self.windowMenu.setDisabled(False)

            for type in self.quantityList:
                self.quantityCombo.addItems(type.keys())

            self.initialLoad()
            self.windowMenu.setDisabled(False)

        QtWidgets.QApplication.restoreOverrideCursor()
        self.statusBar().showMessage("Loaded {f} files".format(f=str(Nfiles)))

    def showLoadParDialog(self):

        if self.stdDirPar is None:
            self.stdDirPar = os.path.curdir

        parname = QtWidgets.QFileDialog.getOpenFileName(self, "Open Parameter File", self.stdDirPar,
                                                        "parameter files (*.par)")[0]

        if len(parname) == 0:
            return      # return if QFileDialog canceled

        self.stdDirPar = "/".join(parname.split("/")[:-1])

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

    def showLoadEosDialog(self, eosname=''):
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

        # return if QFileDialog canceled
        if len(self.eosname) == 0:
            return

        self.stdDirEos = "/".join(self.eosname.split("/")[:-1])

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
                self.x3Combo.setDisabled(False)
            self.quantityCombo.addItems(self.quantityList[-1].keys())

            self.eos = True

            QtWidgets.QApplication.restoreOverrideCursor()

            self.eosFileLabel.setStyleSheet('color: green')
            self.eosFileLabel.setToolTip("EOS-file is available. File: {}".format(self.eosname))

            self.statusBar().showMessage("Done")

    def showLoadOpaDialog(self, opaname=''):

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

        self.stdDirOpa = "/".join(self.opaname.split("/")[:-1])

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
                self.x3Combo.setDisabled(False)
            self.opa = True

            QtWidgets.QApplication.restoreOverrideCursor()

            self.opaFileLabel.setStyleSheet('color: green')
            self.opaFileLabel.setToolTip("Opacity-file is available. File: {}".format(self.opaname))

            self.statusBar().showMessage("Done")

    def showProgressBar(self, func):
        Nfiles = len(self.fname)

        pd = QtWidgets.QProgressDialog("Load files...", "Cancel", 0, Nfiles, self)
        # pd.setWindowTitle("Loading files...")
        pd.show()

        for i in range(Nfiles):
            name = self.fname[i].split("/")[-1]
            pd.setLabelText("Load {0}".format(name))
            try:
                self.modelfile.append(func(self.fname[i]))
            except OSError:
                print("{0} could not be loaded.".format(name))
            pd.setValue(i + 1)
            QtGui.QGuiApplication.processEvents()

            if pd.wasCanceled():
                self.modelfile = []
                return

    # -------------------------------
    # --- Open additional windows ---
    # -------------------------------

    def showMultiPlot(self):
        if self.eos and self.opa:
            self.multiPlot = wind.MultiPlotWind(self.fname, self.modelfile, self.fileType, eos=self.Eos, opa=self.Opa)
        elif self.eos:
            self.multiPlot = wind.MultiPlotWind(self.fname, self.modelfile, self.fileType, eos=self.Eos)
        elif self.opa:
            self.multiPlot = wind.MultiPlotWind(self.fname, self.modelfile, self.fileType, opa=self.Opa)
        else:
            self.multiPlot = wind.MultiPlotWind(self.fname, self.modelfile, self.fileType)

    def showDataPicker(self):
        pass

    # -------------
    # --- Slots ---
    # -------------

    def labelParClick(self, event):
        if self.par:
            self.parFileLabel.setStyleSheet('color: green')
            self.parFileLabel.setToolTip("Parameter-file is available.")

    def labelEosClick(self, event):
        if self.eos:
            self.eosFileLabel.setStyleSheet('color: green')
            self.eosFileLabel.setToolTip("EOS-file is available. File: {}".format(self.eosname))

    def labelOpaClick(self, event):
        if self.opa:
            self.opaFileLabel.setStyleSheet('color: green')
            self.opaFileLabel.setToolTip("Opa-file is available. File: {}".format(self.opaname))

    def dataPlotMotion(self, event):
        if self.funcCombo.currentText() in ["log10", "log10(| |)"]:
            unit = "log10(" + self.unit + ")"
        else:
            unit = self.unit

        try:
            if self.DataDim == 0:
                self.statusBar().showMessage("x: {xdat:13.6g}\ty: {ydat:13.6g}".format(xdat=event.xdata,
                                                                                       ydat=event.ydata))
            elif self.DataDim == 1:
                self.statusBar().showMessage("x: {xdat:13.6g} km\ty: {ydat:13.6g} {unit}".format(xdat=event.xdata,
                                                                                                  ydat=event.ydata,
                                                                                                  unit=unit))
                self.plotBox.setToolTip("x: {xdat:13.6g} km\ny: {ydat:13.6g} {unit}".format(xdat=event.xdata,
                                                                                            ydat=event.ydata,
                                                                                            unit=unit))
            elif self.DataDim == 2:
                idx = (np.abs(self.xc1-event.xdata)).argmin()
                idy = (np.abs(self.xc2-event.ydata)).argmin()

                self.statusBar().showMessage("x: {xdat:13.6g} km   y: {ydat:13.6g} km    value: {dat:13.6g} {unit}".
                                             format(xdat=event.xdata, ydat=event.ydata, dat = self.data[0, idy, idx],
                                                    unit=unit))
                self.plotBox.setToolTip("x: {xdat:13.6g} km\ny: {ydat:13.6g} km\nvalue: {dat:13.6g} {unit}".
                                        format(xdat=event.xdata, ydat=event.ydata, dat = self.data[0, idy, idx],
                                               unit=unit))

            elif self.DataDim == 3:
                if self.planeCombo.currentText() == "xy":
                    idx = (np.abs(self.xc1 - event.xdata)).argmin()
                    idy = (np.abs(self.xc2 - event.ydata)).argmin()

                    self.statusBar().showMessage("x: {xdat:13.6g} km\ty: {ydat:13.6g} km\tvalue: {dat:13.6g} {unit}".
                                                 format(xdat=event.xdata, ydat=event.ydata,
                                                        dat=self.data[self.x3ind, idy, idx], unit=unit))
                    self.plotBox.setToolTip("x: {xdat:13.6g} km\ny: {ydat:13.6g} km\nvalue: {dat:13.6g} {unit}".format(
                        xdat=event.xdata, ydat=event.ydata, dat=self.data[self.x3ind, idy, idx], unit=unit))

                elif self.planeCombo.currentText() == "xz":
                    idx = (np.abs(self.xc1 - event.xdata)).argmin()
                    if self.x3Combo.currentIndex() == 0:
                        idz = (np.abs(self.xc3 - event.ydata)).argmin()
                    else:
                        idz = (np.abs(np.log10(self.tauRange) - event.ydata)).argmin()
                    self.statusBar().showMessage("x: {xdat:13.6g} km\tz: {zdat:13.6g} km\tvalue: {dat:13.6g} {unit}".
                                                 format(xdat=event.xdata, zdat=event.ydata,
                                                        dat=self.data[idz, self.x2ind, idx], unit=unit))
                    self.plotBox.setToolTip("x: {xdat:13.6g} km\nz: {zdat:13.6g} km\nvalue: {dat:13.6g} {unit}".
                                            format(xdat=event.xdata, zdat=event.ydata,
                                                   dat=self.data[idz, self.x2ind, idx], unit=unit))

                elif self.planeCombo.currentText() == "yz":
                    idy = (np.abs(self.xc2 - event.xdata)).argmin()
                    if self.x3Combo.currentIndex() == 0:
                        idz = (np.abs(self.xc3 - event.ydata)).argmin()
                    else:
                        idz = (np.abs(np.log10(self.tauRange) - event.ydata)).argmin()

                    self.statusBar().showMessage("y: {ydat:13.6g} km\tz: {zdat:13.6g} km\tvalue: {dat:13.6g} {unit}".
                                                 format(ydat=event.xdata, zdat=event.ydata,
                                                        dat=self.data[idz, idy, self.x1ind], unit=unit))
                    self.plotBox.setToolTip("y: {ydat:13.6g} km\nz: {zdat:13.6g} km\nvalue: {dat:13.6g} {unit}".
                                            format(ydat=event.xdata, zdat=event.ydata,
                                                   dat=self.data[idz, idy, self.x1ind], unit=unit))

                sc.PlotWidget.linePlot(event.xdata, event.ydata)
        except Exception:
            pass

    def dataPlotPress(self, event):
        if event.xdata is not None and event.ydata is not None:
            if self.DataDim == 3:
                if self.planeCombo.currentText() == "xy":
                    idx = (np.abs(self.xc1 - event.xdata)).argmin()
                    idy = (np.abs(self.xc2 - event.ydata)).argmin()

                    self.x1Slider.setValue(idx)
                    self.x2Slider.setValue(idy)

                elif self.planeCombo.currentText() == "xz":
                    idx = (np.abs(self.xc1 - event.xdata)).argmin()
                    if self.x3Combo.currentIndex() == 0:
                        idz = (np.abs(self.xc3 - event.ydata)).argmin()
                    else:
                        idz = (np.abs(np.log10(self.tauRange) - event.ydata)).argmin()

                    self.x1Slider.setValue(idx)
                    self.x3Slider.setValue(idz)

                elif self.planeCombo.currentText() == "yz":
                    idy = (np.abs(self.xc2 - event.xdata)).argmin()
                    if self.x3Combo.currentIndex() == 0:
                        idz = (np.abs(self.xc3 - event.ydata)).argmin()
                    else:
                        idz = (np.abs(np.log10(self.tauRange) - event.ydata)).argmin()

                    self.x2Slider.setValue(idy)
                    self.x3Slider.setValue(idz)
        else:
            pass
        if self.crossCheck.isChecked():
            self.plotRoutine()

    def getPlotData(self):
        if self.plotDim == 3:
            return self.data, np.array([[self.x1min, self.x1max], [self.x2min, self.x2max]])
        elif self.plotDim == 2:
            if self.DataDim == 3:
                if self.planeCombo.currentText() == "xy":
                    limits = np.array([[self.x1min, self.x1max], [self.x2min, self.x2max]])
                    return self.data[self.x3ind], limits

                elif self.planeCombo.currentText() == "xz":
                    if self.x3Combo.currentIndex() == 0:
                        limits = np.array([[self.x1min, self.x1max], [self.x3min, self.x3max]])
                    else:
                        limits = np.array([[self.x1min, self.x1max], [float(self.maxTauEdit.text()),
                                                                      float(self.minTauEdit.text())]])
                    return self.data[:, self.x2ind], limits

                elif self.planeCombo.currentText() == "yz":
                    if self.x3Combo.currentIndex() == 0:
                        limits = np.array([[self.x2min, self.x2max], [self.x3min, self.x3max]])
                    else:
                        limits = np.array([[self.x2min, self.x2max], [float(self.maxTauEdit.text()),
                                                                      float(self.minTauEdit.text())]])
                    return self.data[:, :, self.x1ind], limits

                else:
                    self.msgBox.setText("Plane could not be identified.")
                    self.msgBox.exec_()
                    return None, None

            elif self.DataDim == 2:
                if self.direction == 0:
                    limits = np.array([[self.x1min, self.x1max], [self.x2min, self.x2max]])
                elif self.direction == 1:
                    limits = np.array([[self.x1min, self.x1max], [self.x3min, self.x3max]])
                elif self.direction == 2:
                    limits = np.array([[self.x2min, self.x2max], [self.x3min, self.x3max]])
                else:
                    self.msgBox.setText("Direction could not be identified.")
                    self.msgBox.exec_()
                return self.data, limits

        elif self.plotDim == 1:
            if self.DataDim == 3:
                if self.planeCombo.currentText() == "xy":
                    limits = np.array([self.x3min, self.x3max])
                    axis = (1, 2)
                    plotSlice = (slice(None, None), slice(self.x2ind, self.x2ind + 1), slice(self.x1ind, self.x1ind + 1))
                elif self.planeCombo.currentText() == "xz":
                    limits = np.array([self.x2min, self.x2max])
                    axis = (0, 2)
                    plotSlice = (slice(self.x3ind, self.x3ind + 1), slice(None, None), slice(self.x1ind, self.x1ind + 1))
                elif self.planeCombo.currentText() == "yz":
                    limits = np.array([self.x1min, self.x1max])
                    axis = (0, 1)
                    plotSlice = (slice(self.x3ind, self.x3ind + 1), slice(self.x2ind, self.x2ind + 1), slice(None, None))
                if self.oneDDataCombo.currentText() == "average":
                    return self.data.mean(axis=axis), limits
                else:
                    return self.data[plotSlice].squeeze(), limits
            else:
                self.msgBox.setText("Dimension of plot could not be identified.")
                self.msgBox.exec_()
                return None
        else:
            self.msgBox.setText("Dimension not legal.")
            self.msgBox.exec_()
            return None

    def plotRoutine(self):
        plotCond = not self.vpCheck.isChecked() and "tauUnityCheck" not in self.senders and not\
            self.dataRangeCheck.isChecked() and not self.crossCheck.isChecked()

        if plotCond:
            if "norm-max-Edit" not in self.senders and len(self.senders) > 1:
                self.senders = []
                return

        self.senders.append(self.sender().objectName())

        data, limits = self.getPlotData()

        if not self.fixPlotWindowCheck.isChecked():
            if self.plotDim == 1:
                self.plotBox.ax.set_xlim(limits)
            elif self.plotDim == 2:
                self.plotBox.ax.set_xlim(limits[0])
                self.plotBox.ax.set_ylim(limits[1])

        if self.crossCheck.isChecked():
            pos = self.pos
        else:
            pos = None

        # --- Update plot, or don´t plot at all ---
        # -----------------------------------------
        if plotCond and self.senders[-1] not in ["cross-Check", "vp-Check"] and not self.tauUnityCheck.isChecked():
            if np.all(limits == self.oldLimits) and self.plotDim == 2:
                self.plotBox.updatePlot(data, self.minNorm, self.maxNorm)
                self.oldData = data
                self.oldLimits = limits
                self.senders = []
                return

            if data is None or np.all(data == self.oldData):
                self.senders = []
                return

        # -----------------------------------------

        if self.fixPlotWindowCheck.isChecked():
            if self.DataDim == 1:
                window = np.array(self.plotBox.ax.get_xlim())
            else:
                window = np.array([self.plotBox.ax.get_xlim(), self.plotBox.ax.get_ylim()])
        else:
            window = None

        if self.plotDim == 3:
            self.threeDPlotBox.plot(np.swapaxes(self.data, 0, 2))
            # if self.vpCheck.isChecked():
            #     self.threeDPlotBox.visualization.update_vectors(self.u, self.v, self.w, float(self.vpXIncEdit.text()))
        elif self.plotDim == 2:
            if self.DataDim == 3:
                if self.planeCombo.currentText() == "xy":
                    self.plotBox.plotFig(data, limits=limits, vmin=self.minNorm, vmax=self.maxNorm,
                                         cmap=self.cmCombo.currentCmap, pos=pos, window=window)
                    if self.vpCheck.isChecked():
                        try:
                            self.plotBox.vectorPlot(self.xc1, self.xc2, self.u[self.x3ind], self.v[self.x3ind],
                                                    xinc=int(self.vpXIncEdit.text()), yinc=int(self.vpYIncEdit.text()),
                                                    scale=float(self.vpScaleEdit.text()),
                                                    alpha=float(self.vpAlphaEdit.text()))
                        except ValueError:
                            pass
                elif self.planeCombo.currentText() == "xz":
                    if self.tauUnityCheck.isChecked():
                        self.plotBox.plotFig(data, limits=limits, vmin=self.minNorm, vmax=self.maxNorm,
                                             cmap=self.cmCombo.currentCmap, pos=pos,
                                             tauUnity=(self.xc1, self.tauheight[self.x2ind]), window=window)
                    else:
                        self.plotBox.plotFig(data, limits=limits, vmin=self.minNorm, vmax=self.maxNorm,
                                             cmap=self.cmCombo.currentCmap, pos=pos, window=window)

                    if self.vpCheck.isChecked():
                        try:
                            self.plotBox.vectorPlot(self.xc1, self.xc3, self.u[:, self.x2ind], self.w[:, self.x2ind],
                                                    xinc=int(self.vpXIncEdit.text()), yinc=int(self.vpYIncEdit.text()),
                                                    scale=float(self.vpScaleEdit.text()),
                                                    alpha=float(self.vpAlphaEdit.text()))
                        except ValueError:
                            pass
                elif self.planeCombo.currentText() == "yz":
                    if self.tauUnityCheck.isChecked():
                        self.plotBox.plotFig(data, limits=limits, vmin=self.minNorm, vmax=self.maxNorm,
                                             cmap=self.cmCombo.currentCmap, pos=pos,
                                             tauUnity=(self.xc2, self.tauheight[:, self.x1ind]), window=window)
                    else:
                        self.plotBox.plotFig(data, limits=limits, vmin=self.minNorm, vmax=self.maxNorm,
                                             cmap=self.cmCombo.currentCmap, pos=pos, window=window)
                    if self.vpCheck.isChecked():
                        try:
                            self.plotBox.vectorPlot(self.xc2, self.xc3, self.v[:, :, self.x1ind], self.w[:, :, self.x1ind],
                                                    xinc=int(self.vpXIncEdit.text()), yinc=int(self.vpYIncEdit.text()),
                                                    scale=float(self.vpScaleEdit.text()),
                                                    alpha=float(self.vpAlphaEdit.text()))
                        except ValueError:
                            pass
                else:
                    self.msgBox.setText("Plane could not be identified.")
                    self.msgBox.exec_()
            elif self.DataDim == 2:

                self.plotBox.plotFig(data, limits=limits, vmin=self.minNorm, vmax=self.maxNorm,
                                     cmap=self.cmCombo.currentCmap, pos=pos, window=window)
            elif self.DataDim == 1:
                self.plotBox.plotFig(data, limits=limits, window=window)
            else:
                self.msgBox.setText("Dimension of plot could not be identified.")
                self.msgBox.exec_()

        elif self.plotDim == 1:
            if self.DataDim == 3:
                if self.oneDDataCombo.currentText() == "average":
                    self.plotBox.plotFig(data, limits=limits, window=window)
                else:
                    self.plotBox.plotFig(data, limits=limits, window=window)
            else:
                self.msgBox.setText("Dimension of plot could not be identified.")
                self.msgBox.exec_()
        else:
            self.msgBox.setText("Dimension not legal.")
            self.msgBox.exec_()

        self.oldData = data
        self.oldLimits = limits
        self.senders = []

# -*- coding: utf-8 -*-
"""
Originally created on Tue Nov 05 10:12:33 2013

@author: René Georg Salhab

Modifications:
    Aug 08 2017 : René Georg Salhab.
                   Split MainWindow (including all attributes) into BasicWindow (basic class) and
                   MainWindow.
"""

import os
from collections import OrderedDict

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

import windows as wind
from cobopy import EosInter, Opac, ParFile, Uio
from nicole import Model, Profile
from streaming.init_file import InitFileHandler
from windows import BasicWindow
from windows.multi_plot_window import MultiPlotWindow
from windows.widgets import PlotWidget


class MainWindow(BasicWindow):
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

        self.init_file_name = './resources/init.json'
        self.init_file_loader = InitFileHandler(self.init_file_name)
        self.init_data = self.init_file_loader.load_parameters()

        self.std_dir_mod = self.init_data['stdDirMod']
        self.std_dir_par = None
        self.std_dir_opa = self.init_data['stdDirOpa']
        self.std_dir_eos = self.init_data['stdDirEOS']

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

        open_model_action = QtWidgets.QAction(QtGui.QIcon("open.png"), "Load &Model File", self)
        open_model_action.setShortcut("Ctrl+M")
        open_model_action.setStatusTip("Open a Model File (.mean, .full, .sta and .end).")
        open_model_action.setToolTip("Open a model-file. (.mean, .full and .end)")
        open_model_action.triggered.connect(self.show_load_model_dialog)

        # --- Sub-menu for recently loaded models

        self.recent_model_menu = QtWidgets.QMenu("Recent Models")
        self.recent_model_menu.setStatusTip("Recently loaded models.")
        self.recent_model_menu.setToolTip("Recently loaded models.")

        new_action = None
        for recmod in self.init_data['recentModels']:
            if isinstance(recmod, str):
                new_action = QtWidgets.QAction(recmod, self)
                new_action.setObjectName(recmod)
            elif isinstance(recmod, list):
                new_action = QtWidgets.QAction(recmod[-1], self)
                new_action.setObjectName(recmod[-1])

            self.recent_model_menu.addAction(new_action)
            new_action.triggered.connect(self.load_recent_model)

        # --- "Load parameter-File" button config

        self.open_par_action = QtWidgets.QAction(QtGui.QIcon("open.png"), "Load &parameter File", self)
        self.open_par_action.setShortcut("Ctrl+P")
        self.open_par_action.setStatusTip("Open a parameter (.par)")
        self.open_par_action.setToolTip("Open an eos-file.")
        self.open_par_action.triggered.connect(self.show_load_par_dialog)
        self.open_par_action.setDisabled(True)

        # --- "Load EOS-File" button config

        self.open_eos_action = QtWidgets.QAction(QtGui.QIcon("open.png"), "Load &EOS File", self)
        self.open_eos_action.setShortcut("Ctrl+E")
        self.open_eos_action.setStatusTip("Open an equation of state file (.eos)")
        self.open_eos_action.setToolTip("Open an eos-file.")
        self.open_eos_action.triggered.connect(self.show_load_eos_dialog)
        self.open_eos_action.setDisabled(True)

        # --- "Load opacity file" button config

        self.openOpaAction = QtWidgets.QAction(QtGui.QIcon("open.png"), "Load &opacity File", self)
        self.openOpaAction.setShortcut("Ctrl+O")
        self.openOpaAction.setStatusTip("Open an opacity file (.opta)")
        self.openOpaAction.setToolTip("Open an opacity file.")
        self.openOpaAction.triggered.connect(self.show_load_opa_dialog)
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

        multi_plot_action = QtWidgets.QAction("Multi-Plot Window", self)
        multi_plot_action.setStatusTip("Opens window with multi-plot ability.")
        multi_plot_action.setToolTip("Opens window with multi-plot ability.")
        multi_plot_action.triggered.connect(self.show_multi_plot)

        file_descriptor_action = QtWidgets.QAction("File Description", self)
        file_descriptor_action.setStatusTip("Opens window with full information about file.")
        file_descriptor_action.setToolTip("Opens window with full information about file.")
        file_descriptor_action.triggered.connect(self.show_file_descriptor)
        file_descriptor_action.setDisabled(True)

        # --------------------------------------------------------------------
        # ----------------- "Output" drop-down menu elements -----------------
        # --------------------------------------------------------------------

        # --------------------------------------------------------------------
        # ------------------------ Initialize menubar ------------------------
        # --------------------------------------------------------------------

        menubar = QtWidgets.QMenuBar(self)
        menubar.setNativeMenuBar(False)

        # --- "File" drop-down menu elements ---

        file_menu = QtWidgets.QMenu("&File", self)
        file_menu.addAction(open_model_action)
        file_menu.addMenu(self.recent_model_menu)
        file_menu.addSeparator()
        file_menu.addAction(self.open_par_action)
        file_menu.addAction(self.open_eos_action)
        file_menu.addAction(self.openOpaAction)
        file_menu.addSeparator()
        file_menu.addAction(exitAction)

        # --- "Window" drop-down menu elements ---

        self.window_menu = QtWidgets.QMenu("&Window", self)
        self.window_menu.addAction(multi_plot_action)
        self.window_menu.addAction(file_descriptor_action)
        self.window_menu.setDisabled(True)

        # --- "Output" drop-down menu elements ---

        self.output_menu = QtWidgets.QMenu("&Output", self)
        self.output_menu.setDisabled(True)

        menubar.addMenu(file_menu)
        menubar.addMenu(self.window_menu)
        menubar.addMenu(self.output_menu)

        self.setMenuBar(menubar)

    def addWidgets(self):
        # BasicWindow consists of all elements, but plot-element. Therefore, layout is already set. Only plot-area has
        # to be defined

        # ---------------------------------------------------------------------
        # ---------------------------- Plot window ----------------------------
        # ---------------------------------------------------------------------

        self.plotBox = PlotWidget(self.centralWidget)
        self.plotBox.mpl_connect("motion_notify_event", self.data_plot_motion)
        self.plotBox.mpl_connect("button_press_event", self.data_plot_press)

        # self.vtkPlot = sc.VTKPlotWidget(self.centralWidget)
        # self.vtkPlot.hide()

        # ---------------------------------------------------------------------
        # -------------- Groupbox with file-state indicators ------------------
        # ---------------------------------------------------------------------

        file_state_group = QtWidgets.QGroupBox("File availability", self.centralWidget)
        file_state_layout = QtWidgets.QHBoxLayout(file_state_group)
        file_state_group.setLayout(file_state_layout)

        self.par_file_label = QtWidgets.QLabel("parameter-file")
        self.par_file_label.setStyleSheet('color: red')
        self.par_file_label.setToolTip("Parameter-file is not available.")
        self.par_file_label.setObjectName("parfilelabel")
        self.par_file_label.mousePressEvent = self.label_par_click

        self.eos_file_label = QtWidgets.QLabel("eos-file")
        self.eos_file_label.setStyleSheet('color: red')
        self.eos_file_label.setToolTip("EOS-file is not available.")
        self.eos_file_label.setObjectName("eosfilelabel")
        self.eos_file_label.mousePressEvent = self.label_eos_click

        self.opa_file_label = QtWidgets.QLabel("opa-file")
        self.opa_file_label.setStyleSheet('color: red')
        self.opa_file_label.setToolTip("Opacity-file is not available.")
        self.opa_file_label.setObjectName("opafilelabel")
        self.opa_file_label.mousePressEvent = self.label_opa_click

        file_state_layout.addWidget(self.par_file_label)
        file_state_layout.addWidget(self.eos_file_label)
        file_state_layout.addWidget(self.opa_file_label)

        # self.threeDPlotBox = sc.PlotWidget3D(self.centralWidget)

        # --- Add plot-widget to inherited splitter

        self.splitter.addWidget(self.plotBox)

        # splitter.addWidget(self.threeDPlotBox)
        # self.threeDPlotBox.hide()

        # --- Add aditional groups to control panel ---

        self.controlgrid.addWidget(file_state_group)

    # --------------------
    # --- Load dialogs ---
    # --------------------

    def load_recent_model(self, sender):
        pass

    def show_load_model_dialog(self):
        if self.std_dir_mod is None:
            self.std_dir_mod = os.path.curdir

        # get list of model-file-names
        fname, fileExtension = QtWidgets.QFileDialog.getOpenFileNames(self, "Open Model File", self.std_dir_mod,
                                                                      "Model files (*.full *.end *.sta);;"
                                                                      "Mean files(*.mean);;"
                                                                      "NICOLE profiles (*.prof);;"
                                                                      "NICOLE model files (*.bin)")
        n_files = len(fname)

        # if dialog is canceled do nothing
        if n_files == 0:
            return
        self.fname = fname

        if 'recentModels' in self.init_data and fname not in self.init_data['recentModels'].values():
            self.init_file_loader.add_recent_file(self.fname)
        else:
            self.init_data['recentModels'] = [1, fname]
            self.init_file_loader.set_parameter('recentModels', self.init_data)


        # set standard directory for Model Load-Dialog to current directory
        self.std_dir_mod = "/".join(self.fname[0].split("/")[:-1])

        self.statusBar().showMessage("Read Model-file(s)...")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

        # if a former modelfile is already existent, close all files
        if isinstance(self.modelfile, list) and len(self.modelfile) > 0:
            for mod in self.modelfile:
                mod.close()

        self.modelfile = []

        if fileExtension == "Mean files(*.mean)":
            self.file_type = "mean"

            self.load_model_files(Uio)
            if len(self.modelfile) == 0:
                return

            # --- content from .mean file ---
            # --- Components depict box number according to file-structure (see manual of CO5BOLD)

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
        elif fileExtension == "Model files (*.full *.end *.sta)":
            self.file_type = "cobold"

            self.load_model_files(Uio)
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
            self.open_eos_action.setDisabled(False)
            self.open_par_action.setDisabled(False)

            path = os.path.split(self.fname[0])[0]

            # If another file is loaded, set the indicator to an "uncertain" state, i.e. it is not clear, if the
            # specific file corresponds to the recently loaded model

            if self.eos:
                self.eos_file_label.setStyleSheet('color: orange')
                self.eos_file_label.setToolTip("You loaded a new model, while using an eos-file loaded beforehand.\n"
                                               "Are you sure that the eos-file is still valid?\n"
                                               "If yes, then click on label. Otherwise load a new eos-file.")

            if self.opa:
                self.opa_file_label.setStyleSheet('color: orange')
                self.opa_file_label.setToolTip("You loaded a new model, while using an opa-file loaded beforehand.\n"
                                               "Are you sure that the opa-file is still valid?\n"
                                               "If yes, then click on label. Otherwise load a new opa-file.")

            if "rhd.par" in os.listdir(path):
                try:
                    parpath = os.path.normpath(os.path.join(path, "rhd.par"))
                    self.parFile = ParFile(parpath)
                    self.par = True

                    opaname = os.path.join(self.parFile['opapath'].data, self.parFile['opafile'].data)
                    eosname = os.path.join(self.parFile['eospath'].data, self.parFile['eosfile'].data)

                    self.show_load_eos_dialog(eosname=eosname)
                    self.show_load_opa_dialog(opaname=opaname)
                    self.par_file_label.setStyleSheet('color: green')
                    self.par_file_label.setToolTip("Parameter-file is available.")
                except OSError:
                    self.par = False
            else:
                if self.par:
                    self.std_dir_par = self.std_dir_mod
                    self.par_file_label.setStyleSheet('color: orange')
                    self.par_file_label.setToolTip("You loaded a new model, while using a par-file loaded beforehand.\n"
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
                                                      ("Magnetic energy", "bener"),
                                                      ("Alfven speed", "ca"), ("Electric current density jx", "jx"),
                                                      ("Electric current density jy", "jy"),
                                                      ("Electric current density jz", "jz"),
                                                      ("Electric current density |j|", "jabs")]))
        elif fileExtension == "NICOLE profiles (*.prof)":
            self.file_type = "profile"
            self.eos = False
            self.opa = False

            self.load_model_files(Profile)
            if len(self.modelfile) == 0:
                return

            self.quantityList = [OrderedDict([("Stokes I", "I"), ("Stokes Q", "Q"), ("Stokes U", "U"),
                                              ("Stokes V", "V")])]
        elif fileExtension == "NICOLE model files (*.bin)":
            self.file_type = "nicole"
            self.eos = False
            self.opa = False

            self.load_model_files(Model)
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

        if self.eos and (self.file_type == "mean" or self.file_type == "cobold"):
            self.quantityList.append(OrderedDict([("Temperature", "temp"), ("Entropy", "entr"), ("Pressure", "press"),
                                                  ("Adiabatic coefficient G1", "gamma1"), ("Mach Number", "mach"),
                                                  ("Adiabatic coefficient G3", "gamma3"), ("Sound velocity", "c_s"),
                                                  ("Mean molecular weight", "mu"), ("Plasma beta", "beta"),
                                                  ("c_s / c_A", "csca")]))

        if self.opa and (self.file_type == "mean" or self.file_type == "cobold"):
            self.quantityList[-1]["Opacity"] = "opa"
            self.quantityList[-1]["Optical depth"] = "optdep"

        if not self.modelfile[0].closed:
            self.quantityCombo.clear()
            # self.outputMenu.setDisabled(False)
            # self.windowMenu.setDisabled(False)

            for type in self.quantityList:
                self.quantityCombo.addItems(type.keys())

            self.initialLoad()
            self.window_menu.setDisabled(False)

        QtWidgets.QApplication.restoreOverrideCursor()
        self.statusBar().showMessage("Loaded {f} files".format(f=str(n_files)))

    def show_load_par_dialog(self):

        if self.std_dir_par is None:
            self.std_dir_par = os.path.curdir

        parname = QtWidgets.QFileDialog.getOpenFileName(self, "Open Parameter File", self.std_dir_par,
                                                        "parameter files (*.par)")[0]

        if len(parname) == 0:
            return      # return if QFileDialog canceled

        self.std_dir_par = "/".join(parname.split("/")[:-1])

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

        self.show_load_eos_dialog(eosname=eosname)
        self.show_load_opa_dialog(opaname=opaname)
        self.par_file_label.setStyleSheet('color: green')
        self.par_file_label.setToolTip("Parameter-file is available.")

    def show_load_eos_dialog(self, eosname=False):
        if self.std_dir_eos is None:
            self.std_dir_eos = os.path.curdir

        if not eosname:
            self.eosname = QtWidgets.QFileDialog.getOpenFileName(self, "Open EOS File", self.std_dir_eos,
                                                                 "EOS files (*.eos)")[0]
        else:
            if os.path.isfile(eosname):
                self.eosname = eosname
            else:
                return

        # return if QFileDialog canceled
        if len(self.eosname) == 0:
            return

        self.std_dir_eos = "/".join(self.eosname.split("/")[:-1])

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

            self.eos_file_label.setStyleSheet('color: green')
            self.eos_file_label.setToolTip("EOS-file is available. File: {}".format(self.eosname))

            self.statusBar().showMessage("Done")

    def show_load_opa_dialog(self, opaname=False):

        if self.std_dir_opa is None:
            self.std_dir_opa = os.path.curdir

        if not opaname:
            self.opaname = QtWidgets.QFileDialog.getOpenFileName(self, "Open Opacity File", self.std_dir_opa,
                                                                 "opacity files (*.opta)")[0]
        else:
            if os.path.isfile(opaname):
                self.opaname = opaname
            else:
                return

        if len(self.opaname) == 0:
            return      # return if QFileDialog canceled

        self.std_dir_opa = "/".join(self.opaname.split("/")[:-1])

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

            self.opa_file_label.setStyleSheet('color: green')
            self.opa_file_label.setToolTip("Opacity-file is available. File: {}".format(self.opaname))

            self.statusBar().showMessage("Done")

    def load_model_files(self, func):
        n_files = len(self.fname)

        pd = QtWidgets.QProgressDialog("Load files...", "Cancel", 0, n_files, self)
        # pd.setWindowTitle("Loading files...")
        pd.show()

        for i in range(n_files):
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

    def show_multi_plot(self):
        if self.eos and self.opa:
            self.multiPlot = MultiPlotWindow(self.fname, self.modelfile, self.file_type, eos=self.Eos, opa=self.Opa)
        elif self.eos:
            self.multiPlot = MultiPlotWindow(self.fname, self.modelfile, self.file_type, eos=self.Eos)
        elif self.opa:
            self.multiPlot = MultiPlotWindow(self.fname, self.modelfile, self.file_type, opa=self.Opa)
        else:
            self.multiPlot = MultiPlotWindow(self.fname, self.modelfile, self.file_type)

    def show_file_descriptor(self):
        self.file_descriptor = wind.FileDescriptor(self.modelfile, self.fname, self.modelind, self.dsind) # TODO: right arguments


    def showDataPicker(self):
        pass

    # -------------
    # --- Slots ---
    # -------------

    def label_par_click(self):
        if self.par:
            self.par_file_label.setStyleSheet('color: green')
            self.par_file_label.setToolTip("Parameter-file is available.")

    def label_eos_click(self):
        if self.eos:
            self.eos_file_label.setStyleSheet('color: green')
            self.eos_file_label.setToolTip("EOS-file is available. File: {}".format(self.eosname))

    def label_opa_click(self):
        if self.opa:
            self.opa_file_label.setStyleSheet('color: green')
            self.opa_file_label.setToolTip("Opa-file is available. File: {}".format(self.opaname))

    def data_plot_motion(self, event):
        if self.func_combo.currentText() in ["log10", "log10(| |)"]:
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

                PlotWidget.linePlot(event.xdata, event.ydata)
        except Exception:
            pass

    def plot_dimension_change(self):
        sender = self.sender()

        if sender.objectName() == "2DRadio":
            self.planeCheck()
            # self.vtkPlot.hide()
            # self.plotBox.show()

        elif sender.objectName() == "3DRadio":
            self.threeDPlotBox.Plot(self.data)
            self.plotBox.hide()
            # self.vtkPlot.show()

    def data_plot_press(self, event):
        if event.xdata is not None and event.ydata is not None:
            # self.pos = np.array([event.xdata, event.ydata])
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

    def get_plot_data(self):
        if self.plotDim == 3:
            return self.data
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
                    plot_slice = (slice(None, None), slice(self.x2ind, self.x2ind + 1), slice(self.x1ind, self.x1ind + 1))
                elif self.planeCombo.currentText() == "xz":
                    limits = np.array([self.x2min, self.x2max])
                    axis = (0, 2)
                    plot_slice = (slice(self.x3ind, self.x3ind + 1), slice(None, None), slice(self.x1ind, self.x1ind + 1))
                elif self.planeCombo.currentText() == "yz":
                    limits = np.array([self.x1min, self.x1max])
                    axis = (0, 1)
                    plot_slice = (slice(self.x3ind, self.x3ind + 1), slice(self.x2ind, self.x2ind + 1), slice(None, None))
                if self.oneDDataCombo.currentText() == "average":
                    return self.data.mean(axis=axis), limits
                else:
                    return self.data[plot_slice].squeeze(), limits
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

        if self.sender() is not None:
            self.senders.append(self.sender().objectName())

        data, limits = self.get_plot_data()

        if not self.fixPlotWindowCheck.isChecked():
            if self.plotDim == 1:
                self.plotBox.ax.set_xlim(limits)
            else:
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
                self.old_data = data
                self.oldLimits = limits
                self.senders = []
                return

            if data is None or np.all(data == self.old_data):
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
            pass
#            self.threeDPlotBox.Plot(self.data)
#            if self.vpCheck.isChecked():
#            self.threeDPlotBox.visualization.update_vectors(self.u, self.v, self.w, float(self.vpXIncEdit.text()))
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

        self.old_data = data
        self.oldLimits = limits
        self.senders = []

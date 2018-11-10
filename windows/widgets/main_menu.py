# -*- coding: utf-8 -*-
"""
Created on 18 Mai 16:51 2018

@author: Rene Georg Salhab
"""

#set  = "Rene Georg Salhab"

from PyQt5 import QtWidgets, QtGui

class MainMenu(QtWidgets.QMenuBar):

    def __init__(self, parent):
        super(MainMenu, self).__init__(parent)

        # --------------------------------------------------------------------
        # ------------------ "File" drop-down menu elements ------------------
        # --------------------------------------------------------------------

        # --- "Load Model" button config


        self.openModelAction = QtWidgets.QAction(QtGui.QIcon("open.png"), "Load &Model File", self)
        self.openModelAction.setShortcut("Ctrl+M")
        self.openModelAction.setStatusTip("Open a Model File (.mean, .full, .sta and .end).")
        self.openModelAction.setToolTip("Open a model-file. (.mean, .full and .end)")

        # --- Sub-menu for recently loaded models

        self.recent_model_menu = QtWidgets.QMenu("Recent Models")
        self.recent_model_menu.setStatusTip("Recently loaded models.")
        self.recent_model_menu.setToolTip("Recently loaded models.")
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
        multiPlotAction.setStatusTip("Opens window with multi-plot ability.")
        multiPlotAction.setToolTip("Opens window with multi-plot ability.")
        multiPlotAction.triggered.connect(self.showMultiPlot)

        fileDescriptorAction = QtWidgets.QAction("File Description", self)
        fileDescriptorAction.setStatusTip("Opens window with full information about file.")
        fileDescriptorAction.setToolTip("Opens window with full information about file.")
        fileDescriptorAction.triggered.connect(self.showFileDescriptor)
        fileDescriptorAction.setDisabled(True)

        # --------------------------------------------------------------------
        # ----------------- "Output" drop-down menu elements -----------------
        # --------------------------------------------------------------------

        # --------------------------------------------------------------------
        # ------------------------ Initialize menubar ------------------------
        # --------------------------------------------------------------------

        menubar = QtWidgets.QMenuBar(self)
        menubar.setNativeMenuBar(False)

        # --- "File" drop-down menu elements ---

        fileMenu = QtWidgets.QMenu("&File", self)
        fileMenu.addAction(openModelAction)
        fileMenu.addMenu(self.recent_model_menu)
        fileMenu.addSeparator()
        fileMenu.addAction(self.openParAction)
        fileMenu.addAction(self.openEosAction)
        fileMenu.addAction(self.openOpaAction)
        fileMenu.addSeparator()
        fileMenu.addAction(exitAction)

        # --- "Window" drop-down menu elements ---

        self.windowMenu = QtWidgets.QMenu("&Window", self)
        self.windowMenu.addAction(multiPlotAction)
        self.windowMenu.addAction(fileDescriptorAction)
        self.windowMenu.setDisabled(True)

        # --- "Output" drop-down menu elements ---

        self.outputMenu = QtWidgets.QMenu("&Output", self)
        self.outputMenu.setDisabled(True)

        menubar.addMenu(fileMenu)
        menubar.addMenu(self.windowMenu)
        menubar.addMenu(self.outputMenu)

        self.setMenuBar(menubar)

    def connectopenModelAction(self, action):
        self.openModelAction.triggered.connect(self.showLoadModelDialog)
# -*- coding: utf-8 -*-
"""
Created on 03 Mai 07:50 2018

@author: 
"""

#set  = "Rene Georg Salhab"

from collections import OrderedDict

import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import pyqtSlot

import windows as wind
from windows import mdis


class MultiPlotWindow(wind.BasicWindow):

    def __init__(self, fname, modelfile, fileType, eos=None, opa=None):
        super(MultiPlotWindow, self).__init__()

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
        self.plotWinds = {"z-position:": {}, u"\u03C4-position:": {}}
        self.plotInds = {"z-position:": [], u"\u03C4-position:": []}
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
        axind = self.x3Combo.currentText()
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

            sub = mdis.MdiSubWindow(self.plotArea)
            sub.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
            self.plotWinds[axind][ind] = mdis.MDIPlotWidget(self.data, parent=sub, dimension=self.plotDim, axis=axis)
            self.plotInds[axind].append(self.plotWindsN)
            sub.setWidget(self.plotWinds[axind][ind])
            sub.setObjectName(ind)
            sub.closed.connect(self.closedSubWindow)

            if self.x3Combo.currentIndex() == 0:
                title = self.quantityCombo.currentText() + " z"
            else:
                title = self.quantityCombo.currentText() + " tau"
            sub.setWindowTitle(title)
            self.plotArea.addSubWindow(sub)
            sub.show()
            self.plotWindsN += 1
        self.plotRoutine()

    @pyqtSlot()
    def x3ComboChange(self):
        super(MultiPlotWindow, self).x3ComboChange()
        axind = self.x3Combo.currentIndex()

        for i in range(self.x3Combo.count()):
            if i == axind:
                for j in self.plotInds[i]:
                    self.plotArea.subWindowList()[i].show()
            else:
                for j in self.plotInds[i]:
                    self.plotArea.subWindowList()[i].hide()

    @pyqtSlot(str)
    def closedSubWindow(self, name):
        axind = self.x3Combo.currentText()
        self.plotWinds[axind].pop(name, None)

    @pyqtSlot()
    def plotRoutine(self):
        axind = self.x3Combo.currentText()
        sender = self.sender()
        if sender.objectName() == "quantity-Combo":
            pass
        for plot in self.plotWinds[axind]:
            if self.crossCheck.isChecked():
                pos = self.pos
            else:
                pos = None
            if self.fixPlotWindowCheck.isChecked():
                if self.plotWinds[axind][plot].dim == 1:
                    window = np.array(self.plotWinds[axind][plot].ax.get_xlim())
                else:
                    window = np.array([self.plotWinds[axind][plot].ax.get_xlim(),
                                       self.plotWinds[axind][plot].ax.get_ylim()])
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
                self.plotWinds[axind][plot].Plot(ind=ind, limits=limits, window=window, cmap=self.cmCombo.currentCmap,
                                                 pos=pos, tauUnity=(self.xc1, self.tauheight[self.x2ind]))
            else:
                self.plotWinds[axind][plot].Plot(ind=ind, limits=limits, window=window, cmap=self.cmCombo.currentCmap,
                                                 pos=pos)

            if self.vpCheck.isChecked():
                try:
                    self.plotWinds[axind][plot].vectorPlot(self.xc1, self.xc3, self.u[:, self.x2ind],
                                                           self.w[:, self.x2ind], xinc=int(self.vpXIncEdit.text()),
                                                           yinc=int(self.vpYIncEdit.text()),
                                                           scale=float(self.vpScaleEdit.text()),
                                                           alpha=float(self.vpAlphaEdit.text()))
                except ValueError:
                    pass
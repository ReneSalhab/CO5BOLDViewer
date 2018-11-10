# -*- coding: utf-8 -*-
"""
Created on Apr 29 19:06 2017

:author: René Georg Salhab
"""

import numpy as np
from PyQt5 import QtWidgets

from windows.basic_window import BasicWindow


class ModelSaveDialog(BasicWindow):
    def __init__(self, modelfile):

        # --- convert input parameters to global parameters

        self.modelfile = modelfile

    def saveEvent(self):
        pass


class FileDescriptor(QtWidgets.QMainWindow):
    def __init__(self, files, fname, curmodind, curdsind):
        super(FileDescriptor, self).__init__()
        self.centralWidget = QtWidgets.QWidget(self)

        self.files = files
        self.fname = fname

        self.curmodind = curmodind
        self.curdsind = curdsind

        self.initUI()

        self.show()

    def initUI(self):

        # --- main layout just consists of the file selector and the files's description

        mainGrid = QtWidgets.QVBoxLayout(self.centralWidget)

        # --- GroupBox of file with info about file ---

        fileGroup = QtWidgets.QGroupBox("File", self.centralWidget)
        fileLayout = QtWidgets.QVBoxLayout()
        fileGroup.setLayout(fileLayout)

        # --- Content of file-GroupBox ---

        self.fileCommbo = QtWidgets.QComboBox(self.centralWidget)
        for i in range(len(self.fname)):
            self.fileCommbo.addItem(self.fname[i].split("/")[-1])

        self.fileCommbo.setCurrentText(self.fname[self.curmodind].split("/")[-1])

        fHeaderGroup = QtWidgets.QGroupBox(self.centralWidget)
        fHeaderLayout = QtWidgets.QGridLayout(fHeaderGroup)
        fHeaderGroup.setLayout(fHeaderLayout)

        i = 0
        for k, v in self.files[self.curmodind].items():
            fHeaderLayout.addWidget(QtWidgets.QLabel(k), i, 0)
            headvals = QtWidgets.QLabel(fHeaderGroup)
            fHeaderLayout.addWidget(headvals, i, 1)

            for val in v.data:
                headvals.setText(headvals.text() + "\n" + str(val))
                
            i += 1
        
        dsSelectCombo = QtWidgets.QComboBox(fileGroup)
        dsSelectCombo.addItems(np.arange(len(self.files[self.curmodind].dataset)))
        dsSelectCombo.setCurrentIndex(self.curdsind)

        # --- Dataset GroupBox with info about dataset ---

        dsGroup = QtWidgets.QGroupBox("Dataset", self.centralWidget)
        dsLayout = QtWidgets.QGridLayout(dsGroup)
        dsGroup.setLayout(dsLayout)

        # --- Content of Dataset GroupBox ---

        dsHeaderGroup = QtWidgets.QGroupBox(self.centralWidget)
        dsHeaderLayout = QtWidgets.QGridLayout(dsHeaderGroup)



        # --- add items to fileLayout

        fileLayout.addItem(self.fileCommbo)
        fileLayout.addItem(fHeaderGroup)
        fileLayout.addItem(dsSelectCombo)
        fileLayout.addItem(dsGroup)


        self.centralWidget.setLayout(maingrid)


class PolePickerWind(QtWidgets.QMainWindow):
    def __init__(self):
        super(PolePickerWind, self).__init__()

        self.setWindowTitle("Data Picker")
        self.centralWidget = QtWidgets.QWidget(self)

        self.MainLayout = QtWidgets.QBoxLayout(parent=self.centralWidget)

        self.show()


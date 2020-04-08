 # -*- coding: utf-8 -*-
"""
Created on Tue Nov 05 08:49:34 2013

@author: Rene Georg Salhab
"""

import sys

from PyQt5 import QtWidgets, QtGui, Qt, QtCore

def main():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    splash_pix = QtGui.QPixmap('loading_screen.png')
    splash = QtWidgets.QSplashScreen(splash_pix)
    splash.setMask(splash_pix.mask())
    font = QtGui.QFont(splash.font())
    font.setPointSize(font.pointSize() + 5)
    splash.setFont(font)
    splash.show()
    app.processEvents()

    col = QtGui.QColor(200, 0, 0)

    splash.showMessage("Load module: bisect", Qt.Qt.AlignCenter | Qt.Qt.AlignBottom, col)
    import bisect
    splash.showMessage("Load backend_qt5agg from matplotlib", Qt.Qt.AlignCenter | Qt.Qt.AlignBottom, col)
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    splash.showMessage("Load colorbar from matplotlib", Qt.Qt.AlignCenter | Qt.Qt.AlignBottom, col)
    import matplotlib.colorbar as clbar
    splash.showMessage("Load colors from matplotlib", Qt.Qt.AlignCenter | Qt.Qt.AlignBottom, col)
    import matplotlib.colors as cl
    splash.showMessage("Load interpolate from scipy", Qt.Qt.AlignCenter | Qt.Qt.AlignBottom, col)
    from scipy import interpolate as ip
    splash.showMessage("Load module: math", Qt.Qt.AlignCenter | Qt.Qt.AlignBottom, col)
    import math
    splash.showMessage("Load module: numpy", Qt.Qt.AlignCenter | Qt.Qt.AlignBottom, col)
    import numpy as np
    splash.showMessage("Load module: numexpr", Qt.Qt.AlignCenter | Qt.Qt.AlignBottom, col)
    import numexpr as ne
    ne.use_vml = False
    splash.showMessage("Load OrderedGrid from collections", Qt.Qt.AlignCenter | Qt.Qt.AlignBottom, col)
    from collections import OrderedDict
    splash.showMessage("Load module: os", Qt.Qt.AlignCenter | Qt.Qt.AlignBottom, col)
    import os
    splash.showMessage("Load pyplot from matplotlib", Qt.Qt.AlignCenter | Qt.Qt.AlignBottom, col)
    import matplotlib.pyplot as plt
    splash.showMessage("Load module: time", Qt.Qt.AlignCenter | Qt.Qt.AlignBottom, col)
    import time

    splash.showMessage("Load Eosinter from eosinter", Qt.Qt.AlignCenter | Qt.Qt.AlignBottom, col)
    from eosinter import EosInter
    splash.showMessage("Load uio", Qt.Qt.AlignCenter | Qt.Qt.AlignBottom, col)
    import uio
    splash.showMessage("Load Opac from opta", Qt.Qt.AlignCenter | Qt.Qt.AlignBottom, col)
    from opta import Opac
    splash.showMessage("Load Model and ProfileReader from nicole", Qt.Qt.AlignCenter | Qt.Qt.AlignBottom, col)
    from nicole import Model, Profile
    splash.showMessage("Load ParFile from par", Qt.Qt.AlignCenter | Qt.Qt.AlignBottom, col)
    from par import ParFile
    splash.showMessage("Load subclasses", Qt.Qt.AlignCenter | Qt.Qt.AlignBottom, col)
    import subclasses as sc
    splash.showMessage("Load windows", Qt.Qt.AlignCenter | Qt.Qt.AlignBottom, col)
    import windows as wind

    splash.showMessage("Load main window", Qt.Qt.AlignCenter | Qt.Qt.AlignBottom, col)
    from main_window import MainWindow

    MainWindow = MainWindow()
    splash.finish(MainWindow)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

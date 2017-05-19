 # -*- coding: utf-8 -*-
"""
Created on Tue Nov 05 08:49:34 2013

@author: Rene Georg Salhab
"""

import sys

from PyQt5 import QtWidgets, QtGui, Qt

def main():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    splash_pix = QtGui.QPixmap('loading_screen.png')
    splash = QtWidgets.QSplashScreen(splash_pix)
    splash.setMask(splash_pix.mask())
    splash.show()
    app.processEvents()
    splash.showMessage("Load module: os", Qt.Qt.AlignCenter | Qt.Qt.AlignBottom)
    import os
    splash.showMessage("Load module: time", Qt.Qt.AlignCenter | Qt.Qt.AlignBottom)
    import time
    splash.showMessage("Load module: math", Qt.Qt.AlignCenter | Qt.Qt.AlignBottom)
    import math
    splash.showMessage("Load module: bisect", Qt.Qt.AlignCenter | Qt.Qt.AlignBottom)
    import bisect
    splash.showMessage("Load module: numpy", Qt.Qt.AlignCenter | Qt.Qt.AlignBottom)
    import numpy as np
    splash.showMessage("Load module: numexpr", Qt.Qt.AlignCenter | Qt.Qt.AlignBottom)
    import numexpr as ne
    splash.showMessage("Load OrderedGrid from collections", Qt.Qt.AlignCenter | Qt.Qt.AlignBottom)
    from collections import OrderedDict
    splash.showMessage("Load interpolate from scipy", Qt.Qt.AlignCenter | Qt.Qt.AlignBottom)
    from scipy import interpolate as ip
    splash.showMessage("Load QtCore from PyQt5", Qt.Qt.AlignCenter | Qt.Qt.AlignBottom)
    from PyQt5 import QtCore

    splash.showMessage("Load pyplot from matplotlib", Qt.Qt.AlignCenter | Qt.Qt.AlignBottom)
    import matplotlib.pyplot as plt
    splash.showMessage("Load colors from matplotlib", Qt.Qt.AlignCenter | Qt.Qt.AlignBottom)
    import matplotlib.colors as cl
    splash.showMessage("Load colorbar from matplotlib", Qt.Qt.AlignCenter | Qt.Qt.AlignBottom)
    import matplotlib.colorbar as clbar
    splash.showMessage("Load backend_qt5agg from matplotlib", Qt.Qt.AlignCenter | Qt.Qt.AlignBottom)
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

    splash.showMessage("Load uio", Qt.Qt.AlignCenter | Qt.Qt.AlignBottom)
    import uio
    splash.showMessage("Load Opac from opta", Qt.Qt.AlignCenter | Qt.Qt.AlignBottom)
    from opta import Opac
    splash.showMessage("Load subclasses", Qt.Qt.AlignCenter | Qt.Qt.AlignBottom)
    import subclasses as sc
    splash.showMessage("Load windows", Qt.Qt.AlignCenter | Qt.Qt.AlignBottom)
    import windows as wind
    splash.showMessage("Load ParFile from par", Qt.Qt.AlignCenter | Qt.Qt.AlignBottom)
    from par import ParFile
    splash.showMessage("Load Eosinter from eosinter", Qt.Qt.AlignCenter | Qt.Qt.AlignBottom)
    # from eosinter import EosInter

    splash.showMessage("Load rootelements", Qt.Qt.AlignCenter | Qt.Qt.AlignBottom)
    import rootelements as cre

    MainWindow = cre.MainWindow()
    splash.finish(MainWindow)
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
 # -*- coding: utf-8 -*-
"""
Created on Tue Nov 05 08:49:34 2013

@author: Rene Georg
"""

import sys

from PyQt5 import QtWidgets

import rootelements as cre

def main():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    MainWindow = cre.MainWindow()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
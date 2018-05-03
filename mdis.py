"""
Created on Aug 29 14:19 2017

:author: René Georg Salhab
"""

from PyQt5 import QtCore,  QtWidgets
import numpy as np

import matplotlib
matplotlib.use('Qt5Agg', force=True)
matplotlib.rcParams['backend'] = 'Qt5Agg'
matplotlib.rcParams['backend.qt5'] = 'PyQt5'
import matplotlib.pyplot as plt
import matplotlib.colors as cl
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavTool
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class MDIPlotWidget(FigureCanvas):
    def __init__(self, data, parent=None, dimension=2, axis=None):
        self.fig = plt.figure(figsize=(8, 6), dpi=100)
        super(MDIPlotWidget, self).__init__(self.fig)

        FigureCanvas.__init__(self, self.fig)

        self.dim = dimension
        self.data = data
        self.axis = axis

        self.setParent(parent)

        self.ax = self.fig.add_subplot(111)

        self.toolbar = NavTool(self, self)

        if self.dim == 1:
            self.Plot = self.plot1D
        elif self.dim == 2:
            self.Plot = self.plot2D
        else:
            raise ValueError("Invalid dimension.")

    # Plot-methods. Have to have same parameters, as usage is equal to each other.

    def plot1D(self, ind=0, limits=None, window=None, cmap="inferno", pos=None, tauUnity=None):
        self.ax.cla()
        axes = np.arange(3)
        print(axes, self.axis)
        axes = tuple(axes[axes != self.axis])

        if limits is None:
            self.plot = self.ax.plot(self.data.mean(axis=axes))
            self.ax.set_aspect('auto')
        else:
            self.plot = self.ax.plot(self.data.mean(axis=axes))
            self.ax.set_aspect('auto')
        self._draw_plot(window)

    def plot2D(self, ind=0, limits=None, window=None, cmap="inferno", pos=None, tauUnity=None):
        """
        Description
        -----------
            Plots data.
        :param data: ndarray, 2D, or 1D. Data that is plotted
        :param limits: ndarray, shape is (2, 2). Extent of the plotRoutine.
        :param vmin: float, minimum value for normalizing the plotRoutine
        :param vmax: float, maximum value for normalizing the plotRoutine
        :param cmap:

        :param pos:
        :return:
        """

        if cmap == "inferno" and "inferno" not in cl.cmap:
            cmap = "jet"
        
        valer = ValueError("Either limits is not None, or an ndarray, or has wrong shape.")

        self.ax.cla()

        if self.axis is not None:
            tup = [None for _ in range(3)]
            tup[self.axis] = ind
            tup = tuple(tup)
            data = self.data[tup].squeeze()

        vmin = data.min()
        vmax = data.max()

        if limits is None:
            self.image = self.ax.imshow(data, interpolation="bilinear", origin="bottom", cmap=cmap,
                                        norm=cl.Normalize(vmin=vmin, vmax=vmax))

            if pos is not None:
                self.image.axes.axvline(x=pos[0], ymin=data.min(axis=1), ymax=data.max(axis=1), color="black")
                self.ax.axvline(x=pos[0], ymin=data.min(axis=1), ymax=data.max(axis=1), linestyles='dashed',
                                color="white")
                self.ax.hlines(y=pos[1], xmin=data.min(axis=0), xmax=data.max(axis=0), color="black")
                self.ax.hlines(y=pos[1], xmin=data.min(axis=0), xmax=data.max(axis=0), linestyles='dashed',
                               color="white")
        elif np.shape(limits) == (2, 2):
            dx = np.abs(limits[0, 1] - limits[0, 0])
            dy = np.abs(limits[1, 1] - limits[1, 0])
            rat = dy / dx
            if rat >= 10:
                aspect = rat
            elif rat <= 0.1:
                aspect = 1 / rat
            else:
                aspect = 1
            self.image = self.ax.imshow(data, interpolation="bilinear", origin="bottom", cmap=cmap,
                                        extent=(limits[0, 0], limits[0, 1], limits[1, 0], limits[1, 1]),
                                        norm=cl.Normalize(vmin=vmin, vmax=vmax), aspect=aspect)
            if pos is not None:
                self.ax.vlines(x=pos[0], ymin=limits[1, 0], ymax=limits[1, 1], color="black")
                self.ax.vlines(x=pos[0], ymin=limits[1, 0], ymax=limits[1, 1], color="white", linestyles='dashed')
                self.ax.hlines(y=pos[1], xmin=limits[0, 0], xmax=limits[0, 1], color="black")
                self.ax.hlines(y=pos[1], xmin=limits[0, 0], xmax=limits[0, 1], color="white", linestyles='dashed')
        else:
            raise valer
        if isinstance(tauUnity, tuple):
            if isinstance(tauUnity[1], np.ndarray):
                if tauUnity[1].ndim == 1:
                    self.ax.plot(tauUnity[0], tauUnity[1])
        self._draw_plot(window)
    
    def vectorPlot(self, x, y, u, v, xinc=1, yinc=1, scale=1.0, alpha=1.0):
        x, y = np.meshgrid(x, y)
        self.vector = self.ax.quiver(x[::xinc, ::yinc], y[::xinc, ::yinc], u[::xinc, ::yinc], v[::xinc, ::yinc],
                                     scale=1.0/scale, alpha=alpha, edgecolor='k', facecolor='white', linewidth=0.5)

        self.fig.tight_layout()
        self.draw()

    def _draw_plot(self, window):
        if window is not None:
            self.ax.set_xlim(window[0])
            self.ax.set_ylim(window[1])

        self.fig.tight_layout()
        self.draw()

class MdiSubWindow(QtWidgets.QMdiSubWindow):
    closed = QtCore.pyqtSignal(str)

    def closeEvent(self, event):
        self.closed.emit(self.objectName())
        event.accept()

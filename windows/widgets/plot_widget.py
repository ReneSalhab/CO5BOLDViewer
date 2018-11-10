# -*- coding: utf-8 -*-
"""
Created on 08 Mai 20:48 2018

@author: Rene Georg Salhab
"""

import numpy as np
from matplotlib import pyplot as plt, colors as cl
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavTool
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class PlotWidget(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = plt.figure(figsize=(8, 6), dpi=100)
        super(PlotWidget, self).__init__(self.fig)

        FigureCanvas.__init__(self, self.fig)

        self.setParent(parent)

        self.ax = self.fig.add_subplot(111)
#        self.ax = self.fig.add_axes(aspect="equal",extent=[0, 10, 0, 10])

        self.toolbar = NavTool(self, self)

        x1 = np.linspace(0, np.pi, 100)
        x2 = np.linspace(0, np.pi, 100)
        data = np.outer(x1, x2)

        self.image = self.ax.imshow(data, interpolation="bilinear", origin="bottom")
        self.fig.tight_layout()

    def plotFig(self, data, limits=None, window=None, vmin=None, vmax=None, cmap="inferno", pos=None, tauUnity=None):
        """
        Description
        -----------
            Plots data.
        :param data: ndarray, 2D, or 1D. Data that is plotted
        :param limits: ndarray, shape is (2, 2). Extent of the plotRoutine.
        :param vmin: float, minimum saturation-value
        :param vmax: float, maximum saturation-value
        :param cmap:

        :param pos:
        :return:
        """
        valer = ValueError("Either limits is not None, or an ndarray, or has wrong shape.")

        if vmin is None:
            vmin = data.min()
        if vmax is None:
            vmax = data.max()

        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self.ax.cla()

        if data.ndim == 2:
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
                rat = dy/dx
                if rat >= 10:
                    aspect = rat
                elif  rat <= 0.1:
                    aspect = 1/rat
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
        elif data.ndim == 1:
            if limits is None:
                self.plot = self.ax.plot(data)
                self.ax.set_aspect('auto')
            elif np.shape(limits) == (2,):
                x = np.linspace(limits[0], limits[1], len(data), endpoint=True)
                self.plot = self.ax.plot(x, data)
                self.ax.set_aspect('auto')
            else:
                raise valer
        else:
            raise ValueError("data has wrong dimension!")

        if window is not None:
            self.ax.set_xlim(window[0])
            self.ax.set_ylim(window[1])

        self.fig.tight_layout()
        self.draw()

    def vectorPlot(self, x, y, u, v, xinc=1, yinc=1, scale=1.0, alpha=1.0):
        x, y = np.meshgrid(x, y)
        self.vector = self.ax.quiver(x[::xinc, ::yinc], y[::xinc, ::yinc], u[::xinc, ::yinc], v[::xinc, ::yinc],
                                     scale=1.0/scale, alpha=alpha, edgecolor='k', facecolor='white', linewidth=0.5)

        self.fig.tight_layout()
        self.draw()

    def colorChange(self, cmap="jet"):
        self.image.set_cmap(cmap)
        self.fig.tight_layout()
        self.draw()

    def updatePlot(self, data, vmin, vmax):
        self.image.set_array(data)
        self.image.set_clim(vmin, vmax)
        self.fig.tight_layout()
        self.draw()
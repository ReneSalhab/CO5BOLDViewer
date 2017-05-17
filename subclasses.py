# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 09:18:20 2014

@author: René Georg Salhab
"""

import numpy as np
from scipy import interpolate

from PyQt5 import QtWidgets

import matplotlib.pyplot as plt
import matplotlib.colors as cl
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from astropy.io import fits
import h5py
        
# ---------------

# ----------------------------------
# --- The Matplotlib-Plot-Widget ---
# ----------------------------------

class PlotWidget(FigureCanvas):
    def __init__(self, parent=None):

        self.msgBox = QtWidgets.QMessageBox()

        self.fig = plt.Figure(figsize=(8, 6), dpi=100)

        FigureCanvas.__init__(self, self.fig)

        self.setParent(parent)
        
        self.ax = self.fig.add_subplot(111)
#        self.ax = self.fig.add_axes(aspect="equal",extent=[0, 10, 0, 10])
        
        self.fig.tight_layout()

        self.toolbar = NavigationToolbar(self, self)
        
        x1 = np.linspace(0, np.pi, 100)
        x2 = np.linspace(0, np.pi, 100)
        data = np.outer(x1, x2)

        self.image = self.ax.imshow(data, interpolation="bilinear", origin="bottom")

    def plotFig(self, data, window=None, vmin=None, vmax=None, cmap="inferno", pos=None, tauUnity=None):
        """
        Description
        -----------
            Plots data.
        :param data: ndarray, 2D, or 1D. Data that is plotted
        :param window: ndarray, shape is (2, 2). Extent of the plot.
        :param vmin: float, minimum value for normalizing the plot of
        :param vmax: float, minimum value for normalizing the plot of
        :param cmap: 
        
        :param pos: 
        :return: 
        """
        valer = ValueError("Either window is not None, or an ndarray, or has wrong shape.")

        if vmin is None:
            vmin = data.min()
        if vmax is None:
            vmax = data.max()

        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self.ax.cla()

        if data.ndim == 2:
            if window is None:
                self.image = self.ax.imshow(data, interpolation="bilinear", origin="bottom", cmap=cmap,
                                            norm=cl.Normalize(vmin=vmin, vmax=vmax))

                if pos is not None:
                    self.image.axes.axvline(x=pos[0], ymin=data.min(axis=1), ymax=data.max(axis=1), color="black")
                    self.ax.axvline(x=pos[0], ymin=data.min(axis=1), ymax=data.max(axis=1), linestyles='dashed',
                                    color="white")
                    self.ax.hlines(y=pos[1], xmin=data.min(axis=0), xmax=data.max(axis=0), color="black")
                    self.ax.hlines(y=pos[1], xmin=data.min(axis=0), xmax=data.max(axis=0), linestyles='dashed',
                                   color="white")
            elif np.shape(window) == (2, 2):
                self.image = self.ax.imshow(data, interpolation="bilinear", origin="bottom", cmap=cmap,
                                            extent=(window[0, 0], window[0, 1], window[1, 0], window[1, 1]),
                                            norm=cl.Normalize(vmin=vmin, vmax=vmax))
                if pos is not None:
                    self.ax.vlines(x=pos[0], ymin=window[1, 0], ymax=window[1, 1], color="black")
                    self.ax.vlines(x=pos[0], ymin=window[1, 0], ymax=window[1, 1], color="white", linestyles='dashed')
                    self.ax.hlines(y=pos[1], xmin=window[0, 0], xmax=window[0, 1], color="black")
                    self.ax.hlines(y=pos[1], xmin=window[0, 0], xmax=window[0, 1], color="white", linestyles='dashed')
            else:
                raise valer
            if isinstance(tauUnity, tuple):
                if isinstance(tauUnity[1], np.ndarray):
                    if tauUnity[1].ndim == 1:
                        self.ax.plot(tauUnity[0], tauUnity[1])
        elif data.dim == 1:
            if window is None:
                self.plot = self.ax.plot(data)
            elif np.shape(window) == (2,):
                x = np.linspace(window[0], window[1], len(data), endpoint=True)
                self.plot = self.ax.plot(x, data)
            else:
                raise valer
        else:
            raise ValueError("data has wrong dimension!")

        self.draw()
    
    def vectorPlot(self, x, y, u, v, xinc=1, yinc=1, scale=1.0, alpha=1.0):
        x, y = np.meshgrid(x, y)
        self.vector = self.ax.quiver(x[::xinc, ::yinc], y[::xinc, ::yinc], u[::xinc, ::yinc], v[::xinc, ::yinc],
                                     scale=1.0/scale, alpha=alpha, edgecolor='k', facecolor='white', linewidth=0.5)

        self.draw()

    def colorChange(self, cmap="jet"):
        self.image.set_cmap(cmap)
        self.draw()

    def updatePlot(self,data,vmin,vmax):
        self.image.set_clim(vmin, vmax)
        self.image.set_array(data)
        self.draw()

# ----------------------------------------------------------------------------
# --------------------------------- Functions --------------------------------       
# ----------------------------------------------------------------------------

def saveFits(filename, modelfile, datatype, data, time, pos, plane):
    
    if plane == "xy":
        dataHDU = fits.PrimaryHDU(data[pos[2],:,:])
        dataHDU.header["z-pos"] = pos[2]
        
        x1 = modelfile.dataset[0].box[0]["xc1"].data.flatten()
        x2 = modelfile.dataset[0].box[0]["xc2"].data.flatten()
        
        col1 = fits.Column(name='x-axis', format='E', array=x1)
        col2 = fits.Column(name='y-axis', format='E', array=x2)
    elif plane == "xz":
        dataHDU = fits.PrimaryHDU(data[:,pos[1],:])
        dataHDU.header["y-pos"] = pos[1]
        
        x1 = modelfile.dataset[0].box[0]["xc1"].data.flatten()
        x2 = modelfile.dataset[0].box[0]["xc3"].data.flatten()
        
        col1 = fits.Column(name='x-axis', format='E', array=x1)
        col2 = fits.Column(name='z-axis', format='E', array=x2)
    elif plane == "yz":
        dataHDU = fits.PrimaryHDU(data[:,:,pos[0]])
        dataHDU.header["x-pos"] = pos[0]
        
        x1 = modelfile.dataset[0].box[0]["xc2"].data.flatten()
        x2 = modelfile.dataset[0].box[0]["xc3"].data.flatten()
        
        col1 = fits.Column(name='y-axis', format='E', array=x1)
        col2 = fits.Column(name='z-axis', format='E', array=x2)
   
    dataHDU.header["plane"] = plane
    dataHDU.header["time"] = time
    dataHDU.header["datatype"] = datatype
    
    cols = fits.ColDefs([col1,col2])
    
    tbhdu = fits.new_table(cols)
    
    HDUlist = fits.HDUList([dataHDU,tbhdu])
    
    HDUlist.writeto(filename)
    
def saveHD5(filename, modelfile, datatype, data, time, pos, plane):
    
    HD5file = h5py.File(filename, "w")
    
    datagroup = HD5file.create_group(datatype)
    
    datagroup["x"] = modelfile.dataset[0].box[0]["xc1"].data.flatten()
    datagroup["y"] = modelfile.dataset[0].box[0]["xc2"].data.flatten()
    datagroup["z"] = modelfile.dataset[0].box[0]["xc3"].data.flatten()
    
    datagroup["time"] = time
    datagroup["position"] = (datagroup["x"][pos[0]],datagroup["y"][pos[1]],
                             datagroup["z"][pos[2]])
    
    datagroup["plane"] = plane
    
    if plane == "xy":
        datagroup["data"] = data[pos[2], :, :]
        
        datagroup["data"].dims[0].label = "x"
        datagroup["data"].dims[1].label = "y"
        
        datagroup['data'].dims.create_scale(datagroup['x'])
        datagroup['data'].dims.create_scale(datagroup['y'])
        
        datagroup['data'].dims[0].attach_scale(datagroup['x'])
        datagroup['data'].dims[1].attach_scale(datagroup['y'])
        
    elif plane == "xz":
        datagroup["data"] = data[:, pos[1], :]
        
        datagroup["data"].dims[0].label = "x"
        datagroup["data"].dims[1].label = "z"
        
        datagroup['data'].dims.create_scale(datagroup['x'])
        datagroup['data'].dims.create_scale(datagroup['z'])
        
        datagroup['data'].dims[0].attach_scale(datagroup['x'])
        datagroup['data'].dims[1].attach_scale(datagroup['z'])
        
    elif plane == "yz":
        datagroup["data"] = data[:, :, pos[0]]
        
        datagroup["data"].dims[0].label = "y"
        datagroup["data"].dims[1].label = "z"
        
        datagroup['data'].dims.create_scale(datagroup['y'])
        datagroup['data'].dims.create_scale(datagroup['z'])
        
        datagroup['data'].dims[0].attach_scale(datagroup['y'])
        datagroup['data'].dims[1].attach_scale(datagroup['z'])
        
    HD5file.close()


def Deriv(qc, vc, vb, axis=-1):
    """
    qc: 3D array containing the variable to be differentiated, 
        values cell-centered\n
    vc: differentiate with respect to this axis, values cell-centered\n
    vb: differentiate with respect to this axis, values at cell boundaries\n
    axis: Axis along derivative shall be computed
    """
    vc = vc.squeeze()
    vb = vb.squeeze()

    vb = np.clip(vb, vc.min(), vc.max())
    dv = np.diff(vb)

    ind = [np.newaxis for _ in range(qc.ndim)]
    ind[axis] = ...
    ind = tuple(ind)
    qb = interpolate.interp1d(vc, qc, axis=axis, copy=False, assume_sorted=True)(vb)
    return np.diff(qb, axis=axis)/dv[ind]

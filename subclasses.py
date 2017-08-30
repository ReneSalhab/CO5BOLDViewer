# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 09:18:20 2014

@author: René Georg Salhab
"""

import numpy as np
from scipy import interpolate

import matplotlib.pyplot as plt
import matplotlib.colors as cl
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavTool
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from astropy.io import fits
import h5py
        
# ---------------

# ----------------------------------
# --- The Matplotlib-Plot-Widget ---
# ----------------------------------

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
        :param limits: ndarray, shape is (2, 2). Extent of the plot.
        :param vmin: float, minimum value for normalizing the plot
        :param vmax: float, maximum value for normalizing the plot
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

    def updatePlot(self,data,vmin,vmax):
        self.image.set_clim(vmin, vmax)
        self.image.set_array(data)
        self.fig.tight_layout()
        self.draw()


class PlotGridWidget(FigureCanvas):
    def __init__(self, parent=None):

        self.fig, self.ax = plt.subplots(2, 2, True, True, figsize=(8, 6), dpi=100)

        FigureCanvas.__init__(self, self.fig)

        self.setParent(parent)
        self.toolbar = NavTool(self, self)

        x1 = np.linspace(0, np.pi, 100)
        x2 = np.linspace(0, np.pi, 100)
        data = np.outer(x1, x2)

        self.image = self.ax.imshow(data, interpolation="bilinear", origin="bottom")
        self.fig.tight_layout()
        self.draw()

    def plotFig(self, data, limits=None, vmin=None, vmax=None, cmap="inferno", pos=None, tauUnity=None):
        """
        Description
        -----------
            Plots data.
        :param data: ndarray, 2D, or 1D. Data that is plotted
        :param limits: ndarray, shape is (2, 2). Extent of the plot.
        :param vmin: float, minimum value for normalizing the plot of
        :param vmax: float, minimum value for normalizing the plot of
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
                self.image = self.ax.imshow(data, interpolation="bilinear", origin="bottom", cmap=cmap,
                                            extent=(limits[0, 0], limits[0, 1], limits[1, 0], limits[1, 1]),
                                            norm=cl.Normalize(vmin=vmin, vmax=vmax))
                if pos is not None:
                    self.ax.vlines(x=pos[0], ymin=limits[1, 0], ymax=limits[1, 1], color="black")
                    self.ax.vlines(x=pos[0], ymin=limits[1, 0], ymax=limits[1, 1], color="white",
                                   linestyles='dashed')
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
            raise ValueError("Data has wrong dimension!")

        self.fig.tight_layout()
        self.draw()

# ----------------------------------------------------------------------------
# --------------------------------- Functions --------------------------------       
# ----------------------------------------------------------------------------

def saveFits(filename, modelfile, datatype, data, time, pos, plane):
    
    if plane == "xy":
        dataHDU = fits.PrimaryHDU(data[pos[2],:,:])
        dataHDU.header["z-pos"] = pos[2]
        
        x1 = modelfile.dataset[0].box[0]["xc1"].data.ravel()
        x2 = modelfile.dataset[0].box[0]["xc2"].data.ravel()
        
        col1 = fits.Column(name='x-axis', format='E', array=x1)
        col2 = fits.Column(name='y-axis', format='E', array=x2)
    elif plane == "xz":
        dataHDU = fits.PrimaryHDU(data[:,pos[1],:])
        dataHDU.header["y-pos"] = pos[1]
        
        x1 = modelfile.dataset[0].box[0]["xc1"].data.ravel()
        x2 = modelfile.dataset[0].box[0]["xc3"].data.ravel()
        
        col1 = fits.Column(name='x-axis', format='E', array=x1)
        col2 = fits.Column(name='z-axis', format='E', array=x2)
    elif plane == "yz":
        dataHDU = fits.PrimaryHDU(data[:,:,pos[0]])
        dataHDU.header["x-pos"] = pos[0]
        
        x1 = modelfile.dataset[0].box[0]["xc2"].data.ravel()
        x2 = modelfile.dataset[0].box[0]["xc3"].data.ravel()
        
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
    
    datagroup["x"] = modelfile.dataset[0].box[0]["xc1"].data.ravel()
    datagroup["y"] = modelfile.dataset[0].box[0]["xc2"].data.ravel()
    datagroup["z"] = modelfile.dataset[0].box[0]["xc3"].data.ravel()
    
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

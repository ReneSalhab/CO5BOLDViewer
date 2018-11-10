# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 09:18:20 2014

@author: René Georg Salhab
"""

import numpy as np
from scipy import interpolate

# class VTKPlotWidget(QVTKRenderWindowInteractor):
#
#     def __init__(self, parent):
#         super(VTKPlotWidget, self).__init__()
#         self.setParent(parent)
#
#         self.ren = vtk.vtkRenderer()
#         self.GetRenderWindow().AddRenderer(self.ren)
#         self.iren = self.GetRenderWindow().GetInteractor()
#
#         self.show()
#         self.iren.Initialize()

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

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 19:27:59 2015

@author: René Georg Salhab
"""

from __future__ import division, print_function

import numexpr as ne
import numpy as np

try:
    import eosinterx as eosx
    eosx_available = True
except:
    eosx_available = False

import uio

print("eosx available (eosinter):", eosx_available)

class EosInter:
    def __init__(self, fname):
        """
            Description
            -----------
                Opens an .eos-file given by :param fname.
            Input
            -----
                :param fname: string, path and name of file with eos-related tables.
        """

        self.eosf = uio.File(fname)

        self.cent = self.eosf.block[0]['c1'].data.T
        self.cpress = self.eosf.block[0]['c2'].data.T
        self.ctemp = self.eosf.block[0]['c3'].data.T

        if self.cent.dtype.byteorder != np.dtype('f').byteorder:
            self.cent = self.cent.newbyteorder()
            self.cent = self.cent.byteswap()
            self.cpress = self.cpress.newbyteorder()
            self.cpress = self.cpress.byteswap()
            self.ctemp = self.ctemp.newbyteorder()
            self.ctemp = self.ctemp.byteswap()

        self.lnx11d = np.log(self.eosf.block[0]['x1'].data + self.eosf.block[0]['x1shift'].data).squeeze()
        self.lnx21d = np.log(self.eosf.block[0]['x2'].data + self.eosf.block[0]['x2shift'].data).squeeze()
        self.x2shift = self.eosf.block[0]['x2shift'].data

        self.n1 = self.lnx11d.size - 1
        self.n2 = self.lnx21d.size - 1

        self.x1fac = self.n1 / (self.lnx11d.max() - self.lnx11d.min())
        self.x2fac = self.n2 / (self.lnx21d.max() - self.lnx21d.min())

    def _prep(self, rho, ei):
        nx1 = rho.size - 1
        nx2 = rho.size - 2

        x2 = self.x2shift

        lnx1 = ne.evaluate("log(rho)")
        lnx2 = ne.evaluate("log(ei+x2)")

        x1fac, x2fac = self.x1fac, self.x2fac
        x1off, x2off = self.lnx11d[0], self.lnx21d[0]

        i1 = ne.evaluate("(lnx1 - x1off) * x1fac").astype(np.int32).clip(0, min(nx1, self.n1))
        i2 = ne.evaluate("(lnx2 - x2off) * x2fac").astype(np.int32).clip(0, min(nx2, self.n2))

        lnx11d = self.lnx11d[i1]
        lnx21d = self.lnx21d[i2]

        x1ta = ne.evaluate('lnx1 - lnx11d')
        x2ta = ne.evaluate('lnx2 - lnx21d')

        return x1ta, x2ta, i1, i2

    def unit(self, quantity="Pressure"):
        """
            Description
            -----------
                returns the unit of :param quantity
            Input
            -----
                :param quantity: string, the demanded quantity.
                    Possibilities: Entropy: "Entropy", "entropy", "E", "e"
                                   Pressure: "Pressure", "pressure", "P", "p"
                                   Temperature: "Temperature", "temperature", "T", "t"
            Output
            ------
                :return: string, unit of :param quantity:.
        """
        if quantity in ["Entropy", "entropy", "E", "e"]:
            unit = self.eosf.block[0]['c1'].params['u']
        elif quantity in ["Pressure", "pressure", "P", "p"]:
            unit = self.eosf.block[0]['c2'].params['u']
        elif quantity in ["Temperature", "temperature", "T", "t"]:
            unit = self.eosf.block[0]['c3'].params['u']
        else:
            raise ValueError('{0} is not a valid quantity'.format(quantity))
        return unit


    def STP(self, rho, ei, quantity="Pressure"):
        """
            Description
            -----------
                Computes the entropy, gas-pressure, or temperature.
            Input
            -----
                :param rho: ndarray, shape of simulation box, density-field
                :param ei: ndarray, shape of simulation box, internal energy-field
                :param quantity: string, the demanded quantity.
                    Possibilities: Entropy: "Entropy", "entropy", "E", "e"
                                   Pressure: "Pressure", "pressure", "P", "p"
                                   Temperature: "Temperature", "temperature", "T", "t"
            Output
            ------
                :return: ndarray, shape of simulation box, values of :param quantity:.
        """
        if not eosx_available:
            raise IOError("Compilation of eosinterx.pyx necessary.")
        x1ta, x2ta, i1, i2 = self._prep(rho, ei)

        funcs = {3: eosx.STP3D, 4: eosx.STP4D}
        try:
            func = funcs[rho.ndim]
        except KeyError:
            raise ValueError("Wrong dimension. Only 3D- and 4D-arrays supported.")

        if quantity in ["Entropy", "entropy", "E", "e", "S", "s"]:
            C = self.cent
        elif quantity in ["Pressure", "pressure", "P", "p"]:
            C = self.cpress
        elif quantity in ["Temperature", "temperature", "T", "t"]:
            C = self.ctemp
        else:
            raise ValueError("{0} as quantity is not supported.".format(quantity))

        if quantity in ["Entropy", "entropy", "E", "e"]:
            return func(C, i1, i2, x1ta, x2ta)
        else:
            return ne.evaluate("exp(val)", local_dict={'val': func(C, i1, i2, x1ta, x2ta)})

    def PandT(self, rho, ei):
        """
            Description
            -----------
                Computes gas-pressure and temperature.
            Input
            -----
                :param rho: ndarray, shape of simulation box, density-field
                :param ei: ndarray, shape of simulation box, internal energy-field
            Output
            ------
                :return: ndarray, shape of simulation box, pressure,
                         ndarray, shape of simulation box, temperature.
        """
        if not eosx_available:
            raise IOError("Compilation of eosinterx.pyx necessary.")
        x1ta, x2ta, i1, i2 = self._prep(rho, ei)

        if rho.ndim == 3:
            P, T = eosx.PandT3D(self.cpress, self.ctemp, i1, i2, x1ta, x2ta)
        elif rho.ndim == 4:
            P, T = eosx.PandT4D(self.cpress, self.ctemp, i1, i2, x1ta, x2ta)
        else:
            raise ValueError("Wrong dimension. Only 3D- and 4D-arrays supported.")
        return ne.evaluate("exp(P)"), ne.evaluate("exp(T)")

    def Pall(self, rho, ei):
        """
            Description
            -----------
                Computes gas-pressure and its derivations dPdrho and dPdei.
            Input
            -----
                :param rho: ndarray, shape of simulation box, density-field
                :param ei: ndarray, shape of simulation box, internal energy-field
            Output
            ------
                :return: ndarray, shape of simulation box, pressure,
                         ndarray, shape of simulation box, dPdrho.
                         ndarray, shape of simulation box, dPdei.
        """
        if not eosx_available:
            raise IOError("Compilation of eosinterx.pyx necessary.")
        x1ta, x2ta, i1, i2 = self._prep(rho, ei)

        if rho.ndim == 3:
            return eosx.Pall3D(rho, ei, self.cpress, i1, i2, x1ta, x2ta, self.x2shift)
        elif rho.ndim == 4:
            return eosx.Pall4D(rho, ei, self.cpress, i1, i2, x1ta, x2ta, self.x2shift)
        else:
            raise ValueError("Wrong dimension. Only 3D- and 4D-arrays supported.")

    def Tall(self, rho, ei):
        """
            Description
            -----------
                Computes temperature and its derivations dTdei.
            Input
            -----
                :param rho: ndarray, shape of simulation box, density-field
                :param ei: ndarray, shape of simulation box, internal energy-field
            Output
            ------
                :return: ndarray, shape of simulation box, pressure,
                         ndarray, shape of simulation box, dTdei.
        """
        if not eosx_available:
            raise IOError("Compilation of eosinterx.pyx necessary.")
        x1ta, x2ta, i1, i2 = self._prep(rho, ei)

        if rho.ndim == 3:
            return eosx.Tall3D(ei, self.ctemp, i1, i2, x1ta, x2ta, self.x2shift)
        elif rho.ndim == 4:
            return eosx.Tall4D(ei, self.ctemp, i1, i2, x1ta, x2ta, self.x2shift)
        else:
            raise ValueError("Wrong dimension. Only 3D- and 4D-arrays supported.")

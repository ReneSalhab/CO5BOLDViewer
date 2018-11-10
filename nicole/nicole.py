# -*- coding: utf-8 -*-
"""
Created on Jul 28 13:55 2017

:author: René Georg Salhab
"""

import numpy as np


class Profile:
    def __init__(self, fname):
        self._pos = {'I': 0, 'Q': 1, 'U': 2, 'V': 3}
        self.fname = fname


        self.dat = np.memmap(fname, dtype='<f8', mode='r')
        self.cl = False

        self.header = self.dat[:2].view('<16a')
        self.nx, self.ny, self.nlam, _ = self.dat[2:4].view('<i4')

        self.Nrec = self.nx * self.ny
        self.shape = (self.nx, self.ny, self.nlam)
        self._sizerec = 4 * self.nlam  # 8 bytes for each I, Q, U, V
        self.dat = self.dat[self._sizerec:]

    def __getitem__(self, key):
        for e in self._pos:
            if e == key:
                return self.data(e)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def closed(self):
        return self.cl

    def unit(self, key):
        if key in self._pos.keys():
            return "erg cm^-3 s^-1 sr-1"
        else:
            raise KeyError("Key unknown")

    def keys(self):
        return self._pos.keys()

    def close(self):
        del self.dat
        self.cl = True

    def data(self, key):
        return self.dat[self._pos[key]:: 4].reshape(self.shape)


class Model:
    def __init__(self, fname):
        self._pos = {'z': [0, "km"], 'tau': [1, ""], 'T': [2, "K"], 'P': [3, "dyn cm^-2"], 'rho': [4, "g cm^-3"],
                     'el_p': [5, "dyn cm^-2"], 'v_los': [6, "cm s^-1"], 'v_mic': [7, "cm s^-1"], 'b_long': [8, "G"],
                     'b_x': [9, "G"], 'b_y': [10, "G"], 'b_local_x': [11, "G"], 'b_local_y': [12, "G"],
                     'b_local_z': [13, "G"], 'v_local_x': [14, "cm s^-1"], 'v_local_y': [15, "cm s^-1"],
                     'v_local_z': [16, "cm s^-1"]}

        self.dat = np.memmap(fname, dtype='<f8', mode='r')
        self.cl = False

        self.header = self.dat[:2].view('<16a')
        self.nx, self.ny, self.nz, _ = self.dat[2:4].view('<i4')

        self.Nrec = self.nx * self.ny
        self.shape = (self.nx, self.ny, self.nz)
        self._sizerec = 22 * self.nz + 3 + 8 + 92  # Floats (multiply by 8 to convert to bytes)

        self.dat = self.dat[self._sizerec:].reshape(self.nx, self.ny, self._sizerec)

    def __getitem__(self, key):
        for e in self._pos:
            if e == key:
                return self.data(e)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def closed(self):
        return self.cl

    def unit(self, key):
        for e in self._pos:
            if e == key:
                return self._pos[key][1]

    def keys(self):
        return self._pos.keys()

    def close(self):
        for d in self.dat:
            del d
        del self.dat
        self.cl = True

    def data(self, key):
        l, u = self._pos[key][0]*self.nz, (self._pos[key][0] + 1)*self.nz
        return self.dat[:, :, l: u]

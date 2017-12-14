# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 14:21:59 2016

@author: René Georg Salhab
"""

from __future__ import division, print_function

import numpy as np
from cython.parallel import prange

cimport cython
cimport numpy as np
from libc.math cimport exp, log10, log

DTYPE = np.int32
DTYPEf = np.float32
DTYPEf64 = np.float64

ctypedef np.int32_t DTYPE_t
ctypedef np.float32_t DTYPEf_t
ctypedef np.float64_t DTYPEf64_t

cdef inline DTYPE_t int_max(DTYPE_t a, DTYPE_t b): return a if a >= b else b
cdef inline DTYPE_t int_min(DTYPE_t a, DTYPE_t b): return a if a <= b else b

cdef inline DTYPEf_t float_max(DTYPEf_t a, DTYPEf_t b): return a if a >= b else b
cdef inline DTYPEf_t float_min(DTYPEf_t a, DTYPEf_t b): return a if a <= b else b


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef STP3D(np.ndarray[DTYPEf_t, ndim=3] C, np.ndarray[DTYPE_t, ndim=3] i1, np.ndarray[DTYPE_t, ndim=3] i2,
            np.ndarray[DTYPEf_t, ndim=3] x1ta, np.ndarray[DTYPEf_t, ndim=3] x2ta):
    """
        Description:
            Computes the entropy, pressure, or temperature with the help of the eos-file.
        inputs:
            :param rho: numpy-ndarray, 3D, shape of simulation box, density-field
            :param ei: numpy-ndarray, 3D, shape of simulation box, internal energy-field
            :param C: ndarray, interpolation coefficients of eos-file
            :param i1: numpy-ndarray, 3D, indices for pressure-dimension of C
            :param i2: numpy-ndarray, 3D, indices for temperature-dimension of C
            :param x1ta: numpy-ndarray, 3D, interpolation-factors along pressure-dimension
            :param x2ta: numpy-ndarray, 3D, interpolation-factors along temperature-dimension
        output:
            :return: ndarray, shape of simulation box, values of :param quantity:. The function returns entropy, or
            log(P), or log(T)
    """
    cdef DTYPE_t i, j, k, ti1, ti2
    cdef DTYPE_t nx = i1.shape[0]
    cdef DTYPE_t ny = i1.shape[1]
    cdef DTYPE_t nz = i1.shape[2]
    cdef np.ndarray[DTYPEf_t, ndim=3] out = np.empty((nx, ny, nz), dtype=DTYPEf)

    # for i in range(nx):
    for i in prange(nx, nogil=True):
        for j in range(ny):
            for k in range(nz):
                ti1 = i1[i, j, k]
                ti2 = i2[i, j, k]
                out[i,j,k] = C[0,ti1,ti2] + x1ta[i,j,k] * (C[1,ti1,ti2] + x1ta[i,j,k] * (C[2,ti1,ti2] + x1ta[i,j,k] *
                    C[3,ti1,ti2])) + x2ta[i,j,k] * (C[4,ti1,ti2] + x1ta[i,j,k] * (C[5,ti1, ti2] + x1ta[i,j,k] *
                    (C[6,ti1,ti2] + x1ta[i,j,k] * C[7,ti1,ti2])) + x2ta[i,j,k] * (C[8,ti1,ti2] + x1ta[i,j,k] *
                    (C[9,ti1, ti2] + x1ta[i,j,k] * (C[10,ti1,ti2] + x1ta[i,j,k] * C[11,ti1, ti2])) + x2ta[i,j,k] *
                    (C[12,ti1,ti2] + x1ta[i,j,k] * (C[13,ti1,ti2] + x1ta[i,j,k] * (C[14,ti1, ti2] + x1ta[i,j,k] *
                    C[15,ti1,ti2])))))
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef STP4D(np.ndarray[DTYPEf_t, ndim=3] C, np.ndarray[DTYPE_t, ndim=4] i1, np.ndarray[DTYPE_t, ndim=4] i2,
            np.ndarray[DTYPEf_t, ndim=4] x1ta, np.ndarray[DTYPEf_t, ndim=4] x2ta):
    """
        Description:
            Computes the entropy, pressure, or temperature with the hekp of the eos-file.
        inputs:
            :param rho: ndarray, shape of simulation box, density-field
            :param ei: ndarray, shape of simulation box, internal energy-field
            :param quantity: string, the demanded quantity
        output:
            :return: ndarray, shape of simulation box, values of :param quantity:. The function returns entropy, or
            log(P), or log(T)
    """
    cdef DTYPE_t i, j, k, l, ti1, ti2
    cdef DTYPE_t nt = i1.shape[0]
    cdef DTYPE_t nx = i1.shape[1]
    cdef DTYPE_t ny = i1.shape[2]
    cdef DTYPE_t nz = i1.shape[3]
    cdef np.ndarray[DTYPEf_t, ndim = 4] out = np.empty((nt, nx, ny, nz), dtype=DTYPEf)

    for i in range(nt):
        # for j in range(nx):
        for j in prange(nx, nogil=True):
            for k in range(ny):
                for l in range(nz):
                    ti1 = i1[i, j, k, l]
                    ti2 = i2[i, j, k, l]
                    out[i,j,k,l] = C[0,ti1,ti2] + x1ta[i, j, k, l] * (C[1, ti1, ti2] + x1ta[i, j, k, l] *
                        (C[2, ti1, ti2] + x1ta[i, j, k, l] * C[3, ti1, ti2])) + x2ta[i, j, k, l] * (C[4, ti1, ti2] +
                        x1ta[i, j, k, l] * (C[5, ti1, ti2] + x1ta[i, j, k, l] * (C[6, ti1, ti2] + x1ta[i, j, k, l] *
                        C[7, ti1, i2[i, j, k ,l]])) + x2ta[i, j, k, l] * (C[8, ti1, ti2] + x1ta[i, j, k, l] *
                        (C[9, ti1, ti2] + x1ta[i, j, k, l] * (C[10, ti1, ti2] + x1ta[i, j, k, l] * C[11, ti1, ti2])) +
                        x2ta[i, j, k ,l] * (C[12, ti1, ti2] + x1ta[i, j, k, l] * (C[13, ti1, ti2] + x1ta[i, j, k, l] *
                        (C[14, ti1, ti2] + x1ta[i, j, k, l] * C[15, ti1, ti2])))))
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef PandT3D(np.ndarray[DTYPEf_t, ndim=3] CP, np.ndarray[DTYPEf_t, ndim=3] CT, np.ndarray[DTYPE_t, ndim=3] i1,
              np.ndarray[DTYPE_t, ndim=3] i2, np.ndarray[DTYPEf_t, ndim=3] x1ta, np.ndarray[DTYPEf_t, ndim=3] x2ta):
    cdef DTYPE_t i, j, k, ti1, ti2
    cdef DTYPE_t nx = i1.shape[0]
    cdef DTYPE_t ny = i1.shape[1]
    cdef DTYPE_t nz = i1.shape[2]
    cdef np.ndarray[DTYPEf_t, ndim = 3] P = np.empty((nx, ny, nz), dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim = 3] T = np.empty((nx, ny, nz), dtype=DTYPEf)

    # for i in range(nx):
    for i in prange(nx, nogil=True):
        for j in range(ny):
            for k in range(nz):
                ti1 = i1[i, j, k]
                ti2 = i2[i, j, k]
                P[i, j, k] = CP[0, ti1, ti2] + x1ta[i, j, k] * (CP[1, ti1, ti2] + x1ta[i, j, k] * (CP[2, ti1, ti2] +
                    x1ta[i, j, k] * CP[3, ti1, ti2])) + x2ta[i, j, k] * (CP[4, ti1, ti2] + x1ta[i, j, k] *
                    (CP[5, ti1, ti2] + x1ta[i, j, k] * (CP[6, ti1, ti2] + x1ta[i, j, k] * CP[7, ti1, ti2])) +
                    x2ta[i, j, k] * (CP[8, ti1, ti2] + x1ta[i, j, k] * (CP[9, ti1, ti2] + x1ta[i, j, k] *
                    (CP[10, ti1, ti2] + x1ta[i, j, k] * CP[11, ti1, ti2])) + x2ta[i, j, k] * (CP[12, ti1, ti2] +
                    x1ta[i, j, k] * (CP[13, ti1, ti2] + x1ta[i, j, k] * (CP[14, ti1, ti2] + x1ta[i, j, k] *
                    CP[15, ti1, ti2])))))
                T[i, j, k] = CT[0, ti1, ti2] + x1ta[i, j, k] * (CT[1, ti1, ti2] + x1ta[i, j, k] * (CT[2, ti1, ti2] +
                    x1ta[i, j, k] * CT[3, ti1, ti2])) + x2ta[i, j, k] * (CT[4, ti1, ti2] + x1ta[i, j, k] *
                    (CT[5, ti1, ti2] + x1ta[i, j, k] * (CT[6, ti1, ti2] + x1ta[i, j, k] * CT[7, ti1, ti2])) +
                    x2ta[i, j, k] * (CT[8, ti1, ti2] + x1ta[i, j, k] * (CT[9, ti1, ti2] + x1ta[i, j, k] *
                    (CT[10, ti1, ti2] + x1ta[i, j, k] * CT[11, ti1, ti2])) + x2ta[i, j, k] * (CT[12, ti1, ti2] +
                    x1ta[i, j, k] * (CT[13, ti1, ti2] + x1ta[i, j, k] * (CT[14, ti1, ti2] + x1ta[i, j, k] *
                    CT[15, ti1, ti2])))))
    return P, T


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef PandT4D(np.ndarray[DTYPEf_t, ndim=3] CP, np.ndarray[DTYPEf_t, ndim=3] CT, np.ndarray[DTYPE_t, ndim=4] i1,
              np.ndarray[DTYPE_t, ndim=4] i2, np.ndarray[DTYPEf_t, ndim=4] x1ta, np.ndarray[DTYPEf_t, ndim=4] x2ta):
    cdef DTYPE_t i, j, k, l, ti1, ti2
    cdef DTYPE_t nt = i1.shape[0]
    cdef DTYPE_t nx = i1.shape[1]
    cdef DTYPE_t ny = i1.shape[2]
    cdef DTYPE_t nz = i1.shape[3]
    cdef np.ndarray[DTYPEf_t, ndim = 4] P = np.empty((nt, nx, ny, nz), dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim = 4] T = np.empty((nt, nx, ny, nz), dtype=DTYPEf)

    for i in range(nt):
        # for j in range(nx):
        for j in prange(nx, nogil=True):
            for k in range(ny):
                for l in range(nz):
                    ti1 = i1[i, j, k, l]
                    ti2 = i2[i, j, k, l]
                    P[i, j, k, l] = CP[0, ti1, ti2] + x1ta[i, j, k, l] * (CP[1, ti1, ti2] + x1ta[i, j, k, l] *
                        (CP[2, ti1, ti2] + x1ta[i, j, k, l] * CP[3, ti1, ti2])) + x2ta[i, j, k, l] * (CP[4, ti1, ti2] +
                        x1ta[i, j, k, l] * (CP[5, ti1, ti2] + x1ta[i, j, k, l] * (CP[6, ti1, ti2] + x1ta[i, j, k, l] *
                        CP[7, ti1, i2[i, j, k ,l]])) + x2ta[i, j, k, l] * (CP[8, ti1, ti2] + x1ta[i, j, k, l] *
                        (CP[9, ti1, ti2] + x1ta[i, j, k, l] * (CP[10, ti1, ti2] + x1ta[i, j, k, l] * CP[11, ti1, ti2])) +
                        x2ta[i, j, k ,l] * (CP[12, ti1, ti2] + x1ta[i, j, k, l] * (CP[13, ti1, ti2] + x1ta[i, j, k, l] *
                        (CP[14, ti1, ti2] + x1ta[i, j, k, l] * CP[15, ti1, ti2])))))
                    T[i, j, k, l] = CT[0, ti1, ti2] + x1ta[i, j, k, l] * (CT[1, ti1, ti2] + x1ta[i, j, k, l] *
                        (CT[2, ti1, ti2] + x1ta[i, j, k, l] * CT[3, ti1, ti2])) + x2ta[i, j, k, l] * (CT[4, ti1, ti2] +
                        x1ta[i, j, k, l] * (CT[5, ti1, ti2] + x1ta[i, j, k, l] * (CT[6, ti1, ti2] + x1ta[i, j, k, l] *
                        CT[7, ti1, i2[i, j, k ,l]])) + x2ta[i, j, k, l] * (CT[8, ti1, ti2] + x1ta[i, j, k, l] *
                        (CT[9, ti1, ti2] + x1ta[i, j, k, l] * (CT[10, ti1, ti2] + x1ta[i, j, k, l] * CT[11, ti1, ti2])) +
                        x2ta[i, j, k ,l] * (CT[12, ti1, ti2] + x1ta[i, j, k, l] * (CT[13, ti1, ti2] + x1ta[i, j, k, l] *
                        (CT[14, ti1, ti2] + x1ta[i, j, k, l] * CT[15, ti1, ti2])))))
    return P, T


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef Pall3D(np.ndarray[DTYPEf_t, ndim=3] rho, np.ndarray[DTYPEf_t, ndim=3] ei, np.ndarray[DTYPEf_t, ndim=3] C,
             np.ndarray[DTYPE_t, ndim=3] i1, np.ndarray[DTYPE_t, ndim=3] i2, np.ndarray[DTYPEf_t, ndim=3] x1ta,
             np.ndarray[DTYPEf_t, ndim=3] x2ta, DTYPEf_t x2shift):
    cdef DTYPE_t i, j, k, ti1, ti2
    cdef DTYPE_t nx = rho.shape[0]
    cdef DTYPE_t ny = rho.shape[1]
    cdef DTYPE_t nz = rho.shape[2]
    cdef np.ndarray[DTYPEf_t, ndim = 3] P = np.empty((nx, ny, nz), dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim = 3] dPdrho = np.empty((nx, ny, nz), dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim = 3] dPde = np.empty((nx, ny, nz), dtype=DTYPEf)


    # for i in range(nx):
    for i in prange(nx, nogil=True):
        for j in range(ny):
            for k in range(nz):
                ti1 = i1[i, j, k]
                ti2 = i2[i, j, k]
                P[i, j, k] = exp(C[0, ti1, ti2] + x1ta[i, j, k] * (C[1, ti1, ti2] + x1ta[i, j, k] * (C[2, ti1, ti2] +
                    x1ta[i, j, k] * C[3, ti1, ti2])) + x2ta[i, j, k] * (C[4, ti1, ti2] + x1ta[i, j, k] * (C[5, ti1, ti2] +
                    x1ta[i, j, k] * (C[6, ti1, ti2] + x1ta[i, j, k] * C[7, ti1, ti2])) + x2ta[i, j, k] * (C[8, ti1, ti2] +
                    x1ta[i, j, k] * (C[9, ti1, ti2] + x1ta[i, j, k] * (C[10, ti1, ti2] + x1ta[i, j, k] * C[11, ti1, ti2])) +
                    x2ta[i, j, k] * (C[12, ti1, ti2] + x1ta[i, j, k] * (C[13, ti1, ti2] + x1ta[i, j, k] * (C[14, ti1, ti2] +
                    x1ta[i, j, k] * C[15, ti1, ti2]))))))

                dPdrho[i,j,k] = P[i,j,k] / rho[i,j,k] * (C[1,ti1,ti2] + x1ta[i,j,k] * (2 * C[2,ti1,ti2] + x1ta[i,j,k] *
                    3 * C[3,ti1,ti2]) + x2ta[i,j,k]*(C[5,ti1,ti2] + x1ta[i,j,k] * (2 * C[6,ti1,ti2] + x1ta[i,j,k] * 3 *
                    C[7,ti1,ti2]) + x2ta[i,j,k] * (C[9,ti1,ti2] + x1ta[i,j,k] * (2 * C[10,ti1,ti2] + x1ta[i,j,k] * 3 *
                    C[11,ti1,ti2]) + x2ta[i,j,k] * (C[13,ti1,ti2] + x1ta[i,j,k] * (2 * C[14,ti1,ti2] + x1ta[i,j,k] * 3 *
                    C[15,ti1,ti2])))))

                dPde[i,j,k] = P[i,j,k] / (ei[i,j,k] + x2shift) * (C[4,ti1,ti2] + x1ta[i,j,k] * (C[5,ti1,ti2] + x1ta[i,j,k] *
                    (C[6,ti1,ti2] + x1ta[i,j,k] * C[7,ti1,ti2])) + 2 * x2ta[i,j,k] * (C[8,ti1,ti2] + x1ta[i,j,k] *
                    (C[9,ti1,ti2] + x1ta[i,j,k] * (C[10,ti1,ti2] + x1ta[i,j,k] * C[11,ti1,ti2])) + 1.5 * x2ta[i,j,k] *
                    (C[12,ti1,ti2] + x1ta[i,j,k] * (C[13,ti1,ti2] + x1ta[i,j,k] * (C[14,ti1,ti2] + x1ta[i,j,k] *
                    C[15,ti1,ti2])))))
    return P, dPdrho, dPde


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef Pall4D(np.ndarray[DTYPEf_t, ndim=4] rho, np.ndarray[DTYPEf_t, ndim=4] ei, np.ndarray[DTYPEf_t, ndim=3] C,
             np.ndarray[DTYPE_t, ndim=4] i1, np.ndarray[DTYPE_t, ndim=4] i2, np.ndarray[DTYPEf_t, ndim=4] x1ta,
             np.ndarray[DTYPEf_t, ndim=4] x2ta, DTYPEf_t x2shift):
    cdef DTYPE_t i, j, k, l, ti1, ti2
    cdef DTYPE_t nt = rho.shape[0]
    cdef DTYPE_t nx = rho.shape[1]
    cdef DTYPE_t ny = rho.shape[2]
    cdef DTYPE_t nz = rho.shape[3]
    cdef np.ndarray[DTYPEf_t, ndim = 4] P = np.empty((nt, nx, ny, nz), dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim = 4] dPdrho = np.empty((nt, nx, ny, nz), dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim = 4] dPde = np.empty((nt, nx, ny, nz), dtype=DTYPEf)

    for i in range(nt):
        # for j in range(nx):
        for j in prange(nx, nogil=True):
            for k in range(ny):
                for l in range(nz):
                    ti1 = i1[i,j,k,l]
                    ti2 = i2[i,j,k,l]
                    P[i,j,k,l] = exp(C[0,ti1, ti2] + x1ta[i,j,k,l] * (C[1,ti1,ti2] + x1ta[i,j,k,l] * (C[2,ti1,ti2] +
                        x1ta[i,j,k,l] * C[3,ti1,ti2])) + x2ta[i,j,k,l] * (C[4,ti1,ti2] + x1ta[i,j,k,l] * (C[5,ti1,ti2] +
                        x1ta[i,j,k,l] * (C[6,ti1,ti2] + x1ta[i,j,k,l] * C[7,ti1,ti2])) + x2ta[i,j,k,l] * (C[8,ti1,ti2] +
                        x1ta[i,j,k,l] * (C[9,ti1,ti2] + x1ta[i,j,k,l] * (C[10,ti1,ti2] + x1ta[i,j,k,l] * C[11,ti1, ti2])) +
                        x2ta[i,j,k,l] * (C[12, ti1, ti2] + x1ta[i,j,k,l] * (C[13, ti1, ti2] + x1ta[i,j,k,l] * (C[14, ti1, ti2] +
                        x1ta[i,j,k,l] * C[15, ti1, ti2]))))))

                    dPdrho[i,j,k,l] = P[i,j,k,l] / rho[i,j,k,l] * (C[1,ti1,ti2] + x1ta[i,j,k,l] * (2 * C[2,ti1,ti2] +
                        x1ta[i,j,k,l] * 3 * C[3,ti1,ti2]) + x2ta[i,j,k,l]*(C[5,ti1,ti2] + x1ta[i,j,k,l] *
                        (2 * C[6,ti1,ti2]+ x1ta[i,j,k,l] * 3 * C[7,ti1,ti2]) + x2ta[i,j,k,l] * (C[9,ti1,ti2] +
                        x1ta[i,j,k,l] * (2 * C[10,ti1,ti2] + x1ta[i,j,k,l] * 3 * C[11,ti1,ti2]) + x2ta[i,j,k,l] *
                        (C[13,ti1,ti2] + x1ta[i,j,k,l] * (2 * C[14,ti1,ti2] + x1ta[i,j,k,l] * 3 * C[15,ti1,ti2])))))

                    dPde[i,j,k,l] = P[i,j,k,l] / (ei[i,j,k,l] + x2shift) * (C[4,ti1,ti2] + x1ta[i,j,k,l] * (C[5,ti1,ti2] +
                        x1ta[i,j,k,l] * (C[6,ti1,ti2] + x1ta[i,j,k,l] * C[7,ti1,ti2])) + 2 * x2ta[i,j,k,l] * (C[8,ti1,ti2] +
                        x1ta[i,j,k,l] * (C[9,ti1,ti2] + x1ta[i,j,k,l] * (C[10,ti1,ti2] + x1ta[i,j,k,l] * C[11,ti1,ti2])) +
                        1.5 * x2ta[i,j,k,l] * (C[12,ti1,ti2] + x1ta[i,j,k,l] * (C[13,ti1,ti2] + x1ta[i,j,k,l] *
                        (C[14,ti1,ti2] + x1ta[i,j,k,l] * C[15,ti1,ti2])))))
    return P, dPdrho, dPde


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef Tall3D(np.ndarray[DTYPEf_t, ndim=3] ei, np.ndarray[DTYPEf_t, ndim=3] C, np.ndarray[DTYPE_t, ndim=3] i1,
             np.ndarray[DTYPE_t, ndim=3] i2, np.ndarray[DTYPEf_t, ndim=3] x1ta, np.ndarray[DTYPEf_t, ndim=3] x2ta,
             DTYPEf_t x2shift):
    cdef DTYPE_t i, j, k, ti1, ti2
    cdef DTYPE_t nx = ei.shape[0]
    cdef DTYPE_t ny = ei.shape[1]
    cdef DTYPE_t nz = ei.shape[2]
    cdef np.ndarray[DTYPEf_t, ndim=3] T = np.empty((nx, ny, nz), dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=3] dTde = np.empty((nx, ny, nz), dtype=DTYPEf)

    # for i in range(nx):
    for i in prange(nx, nogil=True):
        for j in range(ny):
            for k in range(nz):
                ti1 = i1[i, j, k]
                ti2 = i2[i, j, k]
                T[i, j, k] = exp(C[0, ti1, ti2] + x1ta[i, j, k] * (C[1, ti1, ti2] + x1ta[i, j, k] * (C[2, ti1, ti2] +
                    x1ta[i, j, k] * C[3, ti1, ti2])) + x2ta[i, j, k] * (C[4, ti1, ti2] + x1ta[i, j, k] * (C[5, ti1, ti2] +
                    x1ta[i, j, k] * (C[6, ti1, ti2] + x1ta[i, j, k] * C[7, ti1, ti2])) + x2ta[i, j, k] * (C[8, ti1, ti2] +
                    x1ta[i, j, k] * (C[9, ti1, ti2] + x1ta[i, j, k] * (C[10, ti1, ti2] + x1ta[i, j, k] * C[11, ti1, ti2])) +
                    x2ta[i, j, k] * (C[12, ti1, ti2] + x1ta[i, j, k] * (C[13, ti1, ti2] + x1ta[i, j, k] * (C[14, ti1, ti2] +
                    x1ta[i, j, k] * C[15, ti1, ti2]))))))

                dTde[i,j,k] = T[i,j,k] / (ei[i,j,k] + x2shift) * (C[4,ti1,ti2] + x1ta[i,j,k] * (C[5,ti1,ti2] + x1ta[i,j,k] *
                    (C[6,ti1,ti2] + x1ta[i,j,k] * C[7,ti1,ti2])) + 2 * x2ta[i,j,k] * (C[8,ti1,ti2] + x1ta[i,j,k] *
                    (C[9,ti1,ti2] + x1ta[i,j,k] * (C[10,ti1,ti2] + x1ta[i,j,k] * C[11,ti1,ti2])) + 1.5 * x2ta[i,j,k] *
                    (C[12,ti1,ti2] + x1ta[i,j,k] * (C[13,ti1,ti2] + x1ta[i,j,k] * (C[14,ti1,ti2] + x1ta[i,j,k] *
                    C[15,ti1,ti2])))))
    return T, dTde


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef Tall4D(np.ndarray[DTYPEf_t, ndim=4] ei, np.ndarray[DTYPEf_t, ndim=3] C, np.ndarray[DTYPE_t, ndim=4] i1,
             np.ndarray[DTYPE_t, ndim=4] i2, np.ndarray[DTYPEf_t, ndim=4] x1ta, np.ndarray[DTYPEf_t, ndim=4] x2ta,
             DTYPEf_t x2shift):
    cdef DTYPE_t i, j, k, l, ti1, ti2
    cdef DTYPE_t nt = ei.shape[0]
    cdef DTYPE_t nx = ei.shape[1]
    cdef DTYPE_t ny = ei.shape[2]
    cdef DTYPE_t nz = ei.shape[3]
    cdef np.ndarray[DTYPEf_t, ndim=4] T = np.empty((nx, ny, nz), dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=4] dTde = np.empty((nx, ny, nz), dtype=DTYPEf)

    for i in range(nt):
        # for j in range(nx):
        for j in prange(nx, nogil=True):
            for k in range(ny):
                for l in range(nz):
                    ti1 = i1[i,j,k,l]
                    ti2 = i2[i,j,k,l]
                    T[i,j,k,l] = exp(C[0,ti1, ti2] + x1ta[i,j,k,l] * (C[1,ti1,ti2] + x1ta[i,j,k,l] * (C[2,ti1,ti2] +
                        x1ta[i,j,k,l] * C[3,ti1,ti2])) + x2ta[i,j,k,l] * (C[4,ti1,ti2] + x1ta[i,j,k,l] * (C[5,ti1,ti2] +
                        x1ta[i,j,k,l] * (C[6,ti1,ti2] + x1ta[i,j,k,l] * C[7,ti1,ti2])) + x2ta[i,j,k,l] * (C[8,ti1,ti2] +
                        x1ta[i,j,k,l] * (C[9,ti1,ti2] + x1ta[i,j,k,l] * (C[10,ti1,ti2] + x1ta[i,j,k,l] * C[11,ti1, ti2])) +
                        x2ta[i,j,k,l] * (C[12, ti1, ti2] + x1ta[i,j,k,l] * (C[13, ti1, ti2] + x1ta[i,j,k,l] *
                        (C[14, ti1, ti2] + x1ta[i,j,k,l] * C[15, ti1, ti2]))))))

                    dTde[i,j,k,l] = T[i,j,k,l] / (ei[i,j,k,l] + x2shift) * (C[4,ti1,ti2] + x1ta[i,j,k,l] * (C[5,ti1,ti2] +
                        x1ta[i,j,k,l] * (C[6,ti1,ti2] + x1ta[i,j,k,l] * C[7,ti1,ti2])) + 2 * x2ta[i,j,k,l] * (C[8,ti1,ti2] +
                        x1ta[i,j,k,l] * (C[9,ti1,ti2] + x1ta[i,j,k,l] * (C[10,ti1,ti2] + x1ta[i,j,k,l] * C[11,ti1,ti2])) +
                        1.5 * x2ta[i,j,k,l] * (C[12,ti1,ti2] + x1ta[i,j,k,l] * (C[13,ti1,ti2] + x1ta[i,j,k,l] *
                        (C[14,ti1,ti2] + x1ta[i,j,k,l] * C[15,ti1,ti2])))))
    return T, dTde


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef logPT2kappa3D(np.ndarray[DTYPEf_t, ndim=3] log10P, np.ndarray[DTYPEf_t, ndim=3] log10T,
                    np.ndarray[DTYPEf_t, ndim=1] tabP, np.ndarray[DTYPEf_t, ndim=1] tabT,
                    np.ndarray[DTYPEf_t, ndim=3] tabKap, np.ndarray[DTYPEf_t, ndim=1] tabTBN,
                    np.ndarray[DTYPEf_t, ndim=1] tabDTB, np.ndarray[DTYPE_t, ndim=1] idxTBN,
                    np.ndarray[DTYPEf_t, ndim=1] tabPBN, np.ndarray[DTYPEf_t, ndim=1] tabDPB,
                    np.ndarray[DTYPE_t, ndim=1] idxPBN, DTYPE_t iband):
    cdef size_t i, j, k, iTx, iTx0, iTx1, iTx2, iPx, iPx0, iPx1, iPx2, nTx, nPx
    cdef DTYPE_t t1, t2, t3, t4, p1, p2, p3, p4
    cdef DTYPE_t nx = log10T.shape[0]
    cdef DTYPE_t ny = log10T.shape[1]
    cdef DTYPE_t nz = log10T.shape[2]
    cdef DTYPE_t NT = tabT.size - 1
    cdef DTYPE_t NP = tabP.size - 1
    cdef DTYPEf_t Tx, Px, gT1, gT2, gT3, gT4, gP1, gP2, gP3, gP4, fP1, fP2, fP3, fP4, dT, dT1, dtabT1, dtabT01
    cdef np.ndarray[DTYPEf_t, ndim=3] xkaros = np.empty((nx, ny, nz), dtype=DTYPEf)

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                Tx = float_min(tabT[NT - 1], float_max(tabT[1], log10T[i, j, k]))
                Px = float_min(tabP[NP - 1], float_max(tabP[1], log10P[i, j, k]))

                if Tx < tabTBN[1]:
                    nTx = <int>((Tx - tabTBN[0]) / tabDTB[0]) + idxTBN[0]
                elif Tx < tabTBN[2]:
                    nTx = <int>((Tx - tabTBN[1]) / tabDTB[1]) + idxTBN[1]
                elif Tx < tabTBN[3]:
                    nTx = <int>((Tx - tabTBN[2]) / tabDTB[2]) + idxTBN[2]
                else:
                    nTx = NT
                iTx = int_min(int_max(1, nTx), NT)
                iTx0 = iTx - 1
                iTx1 = iTx + 1
                iTx2 = iTx + 2

                if Px < tabPBN[1]:
                    nPx = <int>((Px - tabPBN[0]) / tabDPB[0]) + idxPBN[0]
                elif Px < tabPBN[2]:
                    nPx = <int>((Px - tabPBN[1]) / tabDPB[1]) + idxPBN[1]
                elif Px < tabPBN[3]:
                    nPx = <int>((Px - tabPBN[2]) / tabDPB[2]) + idxPBN[2]
                else:
                    nPx = NP
                iPx = int_min(int_max(1, nPx), NP)
                iPx0 = iPx - 1
                iPx1 = iPx + 1
                iPx2 = iPx + 2

                dT = Tx - tabT[iTx]
                dT1 = Tx - tabT[iTx1]
                dtabT1 = tabT[iTx] - tabT[iTx1]
                dtabT01 = tabT[iTx0] - tabT[iTx1]

                gT1 = dT * dT1 * dT1 / ( (tabT[iTx0] - tabT[iTx] ) * dtabT01 * dtabT1)
                gT2 = dT1 * ((tabT[iTx2] - tabT[iTx0]) * dT * dT - dtabT1 * (tabT[iTx] - tabT[iTx2]) * (Tx - tabT[iTx0])) /\
                      ( (tabT[iTx0] - tabT[iTx] ) * dtabT1 * dtabT1 * (tabT[iTx]  - tabT[iTx2]))
                gT3 = dT * ( (tabT[iTx0] - tabT[iTx2]) * dT1 * dT1 - dtabT01 * dtabT1 * (Tx - tabT[iTx2])) /\
                      ( dtabT01 * dtabT1 * dtabT1 * (tabT[iTx1] - tabT[iTx2]))
                gT4 =  -dT * dT * dT1 / ( dtabT1 * (tabT[iTx]  - tabT[iTx2]) * (tabT[iTx1]- tabT[iTx2]))

                dT = Px - tabP[iPx]
                dT1 = Px - tabP[iPx1]
                dtabT1 = tabP[iPx] - tabP[iPx1]
                dtabT01 = tabP[iPx0] - tabP[iPx1]

                gP1 = dT * dT1 * dT1 / ((tabP[iPx0] - tabP[iPx]) * dtabT01 * dtabT1)
                gP2 = dT1 * ((tabP[iPx2] - tabP[iPx0]) * dT * dT - dtabT1 * (tabP[iPx] - tabP[iPx2]) * (Px - tabP[iPx0])) / (
                        (tabP[iPx0] - tabP[iPx]) * dtabT1 * dtabT1 *
                        (tabP[iPx] - tabP[iPx2]))
                gP3 = dT * ((tabP[iPx0] - tabP[iPx2]) * dT1 * dT1 - dtabT01 * dtabT1 * (Px - tabP[iPx2])) / (
                    dtabT01 * dtabT1 * dtabT1 * (tabP[iPx1] - tabP[iPx2]))
                gP4 = -dT * dT * dT1 / (dtabT1 * (tabP[iPx] - tabP[iPx2]) * (tabP[iPx1] - tabP[iPx2]))

                fP1 = tabKap[iTx0, iPx0, iband] * gT1 + tabKap[iTx, iPx0, iband] * gT2 + \
                      tabKap[iTx1, iPx0, iband] * gT3 + tabKap[iTx2, iPx0, iband] * gT4
                fP2 = tabKap[iTx0, iPx, iband] * gT1 + tabKap[iTx, iPx, iband] * gT2 + \
                      tabKap[iTx1, iPx, iband] * gT3 + tabKap[iTx2, iPx, iband] * gT4
                fP3 = tabKap[iTx0, iPx1, iband] * gT1 + tabKap[iTx, iPx1, iband] * gT2 + \
                      tabKap[iTx1, iPx1, iband] * gT3 + tabKap[iTx2, iPx1, iband] * gT4
                fP4 = tabKap[iTx0, iPx2, iband] * gT1 + tabKap[iTx, iPx2, iband] * gT2 + \
                      tabKap[iTx1, iPx2, iband] * gT3 + tabKap[iTx2, iPx2, iband] * gT4

                xkaros[i, j, k] = fP1 * gP1 + fP2 * gP2 + fP3 * gP3 + fP4 * gP4
    return xkaros


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef logPT2kappa4D(np.ndarray[DTYPEf_t, ndim=4] log10P, np.ndarray[DTYPEf_t, ndim=4] log10T,
                    np.ndarray[DTYPEf_t, ndim=1] tabP, np.ndarray[DTYPEf_t, ndim=1] tabT,
                    np.ndarray[DTYPEf_t, ndim=3] tabKap, np.ndarray[DTYPEf_t, ndim=1] tabTBN,
                    np.ndarray[DTYPEf_t, ndim=1] tabDTB, np.ndarray[DTYPE_t, ndim=1] idxTBN,
                    np.ndarray[DTYPEf_t, ndim=1] tabPBN, np.ndarray[DTYPEf_t, ndim=1] tabDPB,
                    np.ndarray[DTYPE_t, ndim=1] idxPBN, DTYPE_t iband):
    cdef DTYPE_t i, j, k, l, iTx, iTx0, iTx1, iTx2, iPx, iPx0, iPx1, iPx2, nTx, nPx
    cdef DTYPE_t nt = log10T.shape[0]
    cdef DTYPE_t nx = log10T.shape[1]
    cdef DTYPE_t ny = log10T.shape[2]
    cdef DTYPE_t nz = log10T.shape[2]
    cdef DTYPE_t NT = tabT.size - 1
    cdef DTYPE_t NP = tabP.size - 1
    cdef DTYPEf_t Tx, Px, gT1, gT2, gT3, gT4, gP1, gP2, gP3, gP4, fP1, fP2, fP3, fP4, dT, dT1, dtabT1, dtabT01
    cdef np.ndarray[DTYPEf_t, ndim=4] xkaros = np.empty((nt, nx, ny, nz), dtype=DTYPEf)

    for i in range(nt):
        for j in range(nx):
            for k in range(ny):
                for l in range(nz):
                    Tx = float_min(tabT[NT-1], float_max(tabT[1], log10T[i, j, k, l]))
                    Px = float_min(tabP[NP-1], float_max(tabP[1], log10P[i, j, k, l]))

                    if Tx < tabTBN[1]:
                        nTx = <int>((Tx - tabTBN[0]) / tabDTB[0]) + idxTBN[0]
                    elif Tx < tabTBN[2]:
                        nTx = <int>((Tx - tabTBN[1]) / tabDTB[1]) + idxTBN[1]
                    elif Tx < tabTBN[3]:
                        nTx = <int>((Tx - tabTBN[2]) / tabDTB[2]) + idxTBN[2]
                    else:
                        nTx = NT
                    iTx = int_min(int_max(1, nTx), NT)
                    iTx0 = iTx - 1
                    iTx1 = iTx + 1
                    iTx2 = iTx + 2

                    if Px < tabPBN[1]:
                        nPx = <int>((Px - tabPBN[0]) / tabDPB[0]) + idxPBN[0]
                    elif Px < tabPBN[2]:
                        nPx = <int>((Px - tabPBN[1]) / tabDPB[1]) + idxPBN[1]
                    elif Px < tabPBN[3]:
                        nPx = <int>((Px - tabPBN[2]) / tabDPB[2]) + idxPBN[2]
                    else:
                        nPx = NP
                    iPx = int_min(int_max(1, nPx), NP)
                    iPx0 = iPx - 1
                    iPx1 = iPx + 1
                    iPx2 = iPx + 2

                    dT = Tx - tabT[iTx]
                    dT1 = Tx - tabT[iTx1]
                    dtabT1 = tabT[iTx] - tabT[iTx1]
                    dtabT01 = tabT[iTx0] - tabT[iTx1]

                    gT1 = dT * dT1 * dT1 / ( (tabT[iTx0] - tabT[iTx] ) * dtabT01 * dtabT1)
                    gT2 = dT1 * ((tabT[iTx2] - tabT[iTx0]) * dT * dT - dtabT1 * (tabT[iTx] - tabT[iTx2]) * (Tx - tabT[iTx0])) /\
                          ( (tabT[iTx0] - tabT[iTx] ) * dtabT1 * dtabT1 * (tabT[iTx]  - tabT[iTx2]))
                    gT3 = dT * ( (tabT[iTx0] - tabT[iTx2]) * dT1 * dT1 - dtabT01 * dtabT1 * (Tx - tabT[iTx2])) /\
                          ( dtabT01 * dtabT1 * dtabT1 * (tabT[iTx1] - tabT[iTx2]))
                    gT4 =  -dT * dT * dT1 / ( dtabT1 * (tabT[iTx]  - tabT[iTx2]) * (tabT[iTx1]- tabT[iTx2]))

                    dT = Px - tabP[iPx]
                    dT1 = Px - tabP[iPx1]
                    dtabT1 = tabP[iPx] - tabP[iPx1]
                    dtabT01 = tabP[iPx0] - tabP[iPx1]

                    gP1 = dT * dT1 * dT1 / ((tabP[iPx0] - tabP[iPx]) * dtabT01 * dtabT1)
                    gP2 = dT1 * ((tabP[iPx2] - tabP[iPx0]) * dT * dT - dtabT1 * (tabP[iPx] - tabP[iPx2]) * (Px - tabP[iPx0])) / (
                            (tabP[iPx0] - tabP[iPx]) * dtabT1 * dtabT1 * (tabP[iPx] - tabP[iPx2]))
                    gP3 = dT * ((tabP[iPx0] - tabP[iPx2]) * dT1 * dT1 - dtabT01 * dtabT1 * (Px - tabP[iPx2])) / (
                        dtabT01 * dtabT1 * dtabT1 * (tabP[iPx1] - tabP[iPx2]))
                    gP4 = -dT * dT * dT1 / (dtabT1 * (tabP[iPx] - tabP[iPx2]) * (tabP[iPx1] - tabP[iPx2]))

                    fP1 = tabKap[iTx0, iPx0, iband] * gT1 + tabKap[iTx, iPx0, iband] * gT2 + \
                          tabKap[iTx1, iPx0, iband] * gT3 + tabKap[iTx2, iPx0, iband] * gT4
                    fP2 = tabKap[iTx0, iPx, iband] * gT1 + tabKap[iTx, iPx, iband] * gT2 + \
                          tabKap[iTx1, iPx, iband] * gT3 + tabKap[iTx2, iPx, iband] * gT4
                    fP3 = tabKap[iTx0, iPx1, iband] * gT1 + tabKap[iTx, iPx1, iband] * gT2 + \
                          tabKap[iTx1, iPx1, iband] * gT3 + tabKap[iTx2, iPx1, iband] * gT4
                    fP4 = tabKap[iTx0, iPx2, iband] * gT1 + tabKap[iTx, iPx2, iband] * gT2 + \
                          tabKap[iTx1, iPx2, iband] * gT3 + tabKap[iTx2, iPx2, iband] * gT4

                    xkaros[i, j, k, l] = fP1 * gP1 + fP2 * gP2 + fP3 * gP3 + fP4 * gP4
    return xkaros



@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tau3D(np.ndarray[DTYPEf_t, ndim=3] kaprho, np.ndarray[DTYPEf_t, ndim=1] dz, DTYPEf_t radHtautop):
    cdef DTYPE_t i, j, k, kt
    cdef DTYPE_t nx = kaprho.shape[0]
    cdef DTYPE_t ny = kaprho.shape[1]
    cdef DTYPE_t nz = kaprho.shape[2]
    cdef DTYPE_t k1 = nz - 1
    cdef DTYPE_t k2 = nz - 2
    cdef DTYPE_t k3 = nz - 3
    cdef DTYPEf_t s3, s4, s5
    cdef np.ndarray[DTYPEf_t, ndim=1] dkds = np.empty((nz), dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=1] dxds = np.empty((nz), dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=3] tau = np.empty((nx, ny, nz), dtype=DTYPEf)

    for i in range(nx):
        for j in range(ny):
            tau[i, j, k1] = kaprho[i, j, k1] * radHtautop

            for k in range(k1):
                dkds[k] = -(kaprho[i, j, k+1] - kaprho[i, j, k]) / dz[k]

            s3 = (dkds[k2] * dz[k3] + dkds[k3] * dz[k2]) / (2 * (dz[k2] + dz[k3]))
            s4 = float_min(float_min(s3, dkds[k2]), dkds[k3])
            s5 = float_max(float_max(s3, dkds[k2]), dkds[k3])
            dxds[k1] = 1.5 * dkds[k2] - (float_max(s4, 0.0) + float_min(s5, 0.0))

            for k in range(k2, 0, -1):
                kt = k - 1

                s3 = (dkds[k] * dz[kt] + dkds[kt] * dz[k]) / (2.0 * (dz[k] + dz[kt]))
                s4 = float_min(float_min(s3, dkds[k]), dkds[kt])
                s5 = float_max(float_max(s3, dkds[k]), dkds[kt])
                dxds[k] = 2.0 * (float_max(s4, 0.0) + float_min(s5, 0.0))

            dxds[0] = 1.5 * dkds[0] - 0.5 * dxds[1]

            for k in range(k2, -1, -1):
                kt = k + 1
                tau[i, j, k] = tau[i, j, kt] + dz[k] * (0.5 * (kaprho[i, j, kt] + kaprho[i, j, k]) +
                                                        dz[k] * (dxds[kt] - dxds[k]) / 12.0)
    return tau


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tau4D(np.ndarray[DTYPEf_t, ndim=4] kaprho, np.ndarray[DTYPEf_t, ndim=1] dz, DTYPEf_t radHtautop):
    cdef DTYPE_t i, j, k, l, lt
    cdef DTYPE_t nt = kaprho.shape[0]
    cdef DTYPE_t nx = kaprho.shape[1]
    cdef DTYPE_t ny = kaprho.shape[2]
    cdef DTYPE_t nz = kaprho.shape[3]
    cdef DTYPE_t l1 = nz - 1
    cdef DTYPE_t l2 = nz - 2
    cdef DTYPE_t l3 = nz - 3
    cdef DTYPEf_t s3, s4, s5
    cdef np.ndarray[DTYPEf_t, ndim=1] dkds = np.empty((nz), dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=1] dxds = np.empty((nz), dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=4] tau = np.empty((nt, nx, ny, nz), dtype=DTYPEf)

    for i in range(nt):
        for j in range(nx):
            for k in range(ny):
                tau[i, j, k, l1] = kaprho[i, j, k, l1] * radHtautop

                for l in range(l1):
                    dkds[l] = -(kaprho[i, j, k, l+1] - kaprho[i, j, k, l]) / dz[l]

                s3 = (dkds[l2] * dz[l3] + dkds[l3] * dz[l2]) / (2 * (dz[l2] + dz[l3]))
                s4 = float_min(float_min(s3, dkds[l2]), dkds[l3])
                s5 = float_max(float_max(s3, dkds[l2]), dkds[l3])
                dxds[l1] = 1.5 * dkds[l2] - (float_max(s4, 0.0 ) + float_min(s5, 0.0))

                for l in range(l2, 0, -1):
                    lt = l - 1

                    s3 = (dkds[l] * dz[lt] + dkds[lt] * dz[l]) / (2.0 * (dz[l] + dz[lt]))
                    s4 = float_min(float_min(s3, dkds[l]), dkds[lt])
                    s5 = float_max(float_max(s3, dkds[l]), dkds[lt])
                    dxds[l] = 2.0 * (float_max(s4, 0.0) + float_min(s5, 0.0))

                dxds[0] = 1.5 * dkds[0] - 0.5 * dxds[1]

                for l in range(l2, -1, -1):
                    lt = l + 1
                    tau[i, j, k, l] = tau[i, j, k, lt] + dz[l] * (0.5 * (kaprho[i, j, k, lt] + kaprho[i, j, k, l]) +
                                                                  dz[l] * (dxds[lt] - dxds[l]) / 12.0)
    return tau

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef height3D(np.ndarray[DTYPEf_t, ndim=3] tau, np.ndarray[DTYPEf_t, ndim=1] z, DTYPEf_t val):
    """
    Description:
        Computes the height-surface of tau=val. It assumes linear interpolation of tau along the last axis.
    input:
        tau: 3D ndarray (float64), Containing optical depth. Last dimension is height-dimension
        z: 1D ndarray (float64), Vertical axis along last dimension of tau.
        val: float64, Value of tau, at which geometrical height is to be computed.

    out: 2D ndarray (float32), height of tau=val-level
    """
    tau -= val
    cdef DTYPE_t i, j, k
    cdef DTYPE_t nx = tau.shape[0]
    cdef DTYPE_t ny = tau.shape[1]
    cdef DTYPE_t nz = tau.shape[2]
    cdef np.ndarray[DTYPEf_t, ndim=2] out = np.empty((nx, ny), dtype=DTYPEf)
    # for i in range(nx):
    for i in prange(nx, nogil=True):
        for j in range(ny):
            for k in range(nz):
                if tau[i, j, k] < 0:
                    out[i, j] = (z[k - 1] - z[k]) / (tau[i, j, k] - tau[i, j, k-1]) * tau[i, j, k] + z[k]
                    break

    return out

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef height4D(np.ndarray[DTYPEf_t, ndim=4] tau, np.ndarray[DTYPEf_t, ndim=1] z, DTYPEf_t val):
    """
    Description:
        Computes the height-surface of tau=val. It assumes linear interpolation of tau along the last axis.
    input:
        tau: 4D ndarray (float64), Containing optical depth. Last dimension is height-dimension
        z: 1D ndarray (float64), Vertical axis along last dimension of tau.
        val: float64, Value of tau, at which geometrical height is to be computed.

    out: 3D ndarray (float32), height of tau=val-level
    """
    tau -= val
    cdef DTYPE_t i, j, k, l
    cdef DTYPE_t nt = tau.shape[0]
    cdef DTYPE_t nx = tau.shape[1]
    cdef DTYPE_t ny = tau.shape[2]
    cdef DTYPE_t nz = tau.shape[3]
    cdef np.ndarray[DTYPEf_t, ndim=3] out = np.empty((nt, nx, ny), dtype=DTYPEf)
    for i in range(nt):
        # for j in range(nx):
        for j in prange(nx, nogil=True):
            for k in range(ny):
                for l in range(1, nz):
                    if tau[i, j, k , l] < 0:
                        out[i, j, k] = (z[l - 1] - z[l]) / (tau[i, j, k, l] - tau[i, j, k, l-1]) * tau[i, j, k, l] + z[l]
                        break

    return out

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef height3Dvec(np.ndarray[DTYPEf_t, ndim=3] tau, np.ndarray[DTYPEf_t, ndim=1] z,
                     np.ndarray[DTYPEf_t, ndim=1] val):
    """
    Description:
        Computes the geometrical height of tau=val. It assumes linear interpolation of tau along the last axis.
    input:
        tau: 3D ndarray (float64), Containing optical depth. Last dimension is height-dimension
        z: 1D ndarray (float64), Vertical axis along last dimension of tau.
        val: 1D ndarray (float64), Values of tau, at which geometrical height is to be computed.

    out: 3D ndarray (float32), heights of tau=val-level
    """
    cdef DTYPE_t i, j, k, l
    cdef DTYPE_t nx = tau.shape[0]
    cdef DTYPE_t ny = tau.shape[1]
    cdef DTYPE_t nz = tau.shape[2]
    cdef DTYPE_t nv = val.shape[0]
    cdef np.ndarray[DTYPEf_t, ndim=3] out = np.empty((nx, ny, nv), dtype=DTYPEf)
    # for i in range(nx):
    for i in prange(nx, nogil=True):
        for j in range(ny):
            for l in range(nv):
                for k in range(1, nz):
                    if tau[i, j, k] - val[l] < 0:
                        out[i, j, l] = (z[k - 1] - z[k]) / (tau[i, j, k] - tau[i, j, k-1]) * tau[i, j, k] + z[k]
                        break

    return out


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef height4Dvec(np.ndarray[DTYPEf_t, ndim=4] tau, np.ndarray[DTYPEf_t, ndim=1] z,
                     np.ndarray[DTYPEf_t, ndim=1] val):
    """
    Description:
        Computes the height-surface of tau=val. It assumes linear interpolation of tau along the last axis.
    input:
        tau: 4D ndarray (float64), Containing optical depth. Last dimension is height-dimension
        z: 1D ndarray (float64), Vertical axis along last dimension of tau.
        val: 1D ndarray (float64), Values of tau, at which geometrical height is to be computed.

    out: 4D ndarray (float32), heights of tau=val-level
    """
    cdef DTYPE_t i, j, k, l, m
    cdef DTYPE_t nt = tau.shape[0]
    cdef DTYPE_t nx = tau.shape[1]
    cdef DTYPE_t ny = tau.shape[2]
    cdef DTYPE_t nz = tau.shape[3]
    cdef DTYPE_t nv = val.shape[0]
    cdef np.ndarray[DTYPEf_t, ndim=4] out = np.empty((nt, nx, ny, nv), dtype=DTYPEf)
    for i in range(nt):
        # for j in range(nx):
        for j in prange(nx, nogil=True):
            for k in range(ny):
                for m in range(nv):
                    for l in range(1, nz):
                        if tau[i, j, k , l] - val[m] < 0:
                            out[i, j, k, m] = (z[l - 1] - z[l]) / (tau[i, j, k, l] - tau[i, j, k, l-1]) * tau[i, j, k, l] + z[l]
                            break

    return out


@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off bounds-checking for entire function
cpdef interp3d(np.ndarray[DTYPEf_t, ndim=1] x, np.ndarray[DTYPEf_t, ndim=3] y, np.ndarray[DTYPEf_t, ndim=2] new_x):
    """
    interp3d(x, y, new_x)

    Performs linear interpolation over the last dimension of a 3D array, according to new values from a 2D array new_x.
    Thus, interpolate y[i, j, :] for new_x[i, j]. Modified for tau-interpolation (tau as axis)!

    Parameters
    ----------
    x : 1-D ndarray (double type)
        Array containing the x (abcissa) values. Must be monotonically decreasing.
    y : 3-D ndarray (double type)
        Array containing the y values to interpolate.
    x_new: 2-D ndarray (double type)
        Array with new abcissas to interpolate.

    Returns
    -------
    new_y : 3-D ndarray
        Interpolated values.
    """
    cdef DTYPE_t nx = y.shape[0]
    cdef DTYPE_t ny = y.shape[1]
    cdef DTYPE_t nz = y.shape[2]
    cdef DTYPE_t i, j, k
    cdef np.ndarray[DTYPEf_t, ndim=2] new_y = np.zeros((nx, ny), dtype=DTYPEf)

    # for i in range(nx):
    for i in prange(nx, nogil=True):
        for j in range(ny):
            for k in range(1, nz):
                 if x[k] < new_x[i, j]:
                     new_y[i, j] = (y[i, j, k] - y[i, j, k - 1]) * (new_x[i, j] - x[k-1]) /\
                                   (x[k] - x[k - 1]) + y[i, j, k - 1]
                     break
    return new_y

@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off bounds-checking for entire function
cpdef interp4d(np.ndarray[DTYPEf_t, ndim=1] x, np.ndarray[DTYPEf_t, ndim=4] y, np.ndarray[DTYPEf_t, ndim=3] new_x):
    """
    interp4d(x, y, new_x)

    Performs linear interpolation over the last dimension of a 4D array, according to new values from a 3D array new_x.
    Thus, interpolate y[i, j, :] for new_x[i, j]. Modified for tau-interpolation (tau as axis)!

    Parameters
    ----------
    x : 1D ndarray (double type)
        Array containg the x (abcissa) values. Must be monotonically decreasing.
    y : 4D ndarray (double type)
        Array containing the y values to interpolate.
    x_new: 3D ndarray (double type)
        Array with new abcissas to interpolate.

    Returns
    -------
    new_y : 3D ndarray
        Interpolated values.
    """
    cdef DTYPE_t nt = y.shape[0]
    cdef DTYPE_t nx = y.shape[1]
    cdef DTYPE_t ny = y.shape[2]
    cdef DTYPE_t nz = y.shape[3]
    cdef DTYPE_t i, j, k, l
    cdef np.ndarray[DTYPEf_t, ndim=3] new_y = np.zeros((nt, nx, ny), dtype=DTYPEf)

    for i in range(nt):
        # for j in range(nx):
        for j in prange(nx, nogil=True):
            for k in range(ny):
                for l in range(1, nz):
                     if x[l] < new_x[i, j, k]:
                         new_y[i, j, k] = (y[i, j, k, l] - y[i, j, k, l - 1]) * (new_x[i, j, k] - x[l-1]) /\
                                          (x[l] - x[l - 1]) + y[i, j, k, l - 1]
                         break
    return new_y


@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off bounds-checking for entire function
cpdef interp3dcube(np.ndarray[DTYPEf_t, ndim=1] x, np.ndarray[DTYPEf_t, ndim=3] y,
                   np.ndarray[DTYPEf_t, ndim=3] new_x):
    """
    interp3d(x, y, new_x)

    Performs linear interpolation over the last dimension of a 3D array, according to new values from a 2D array new_x.
    Thus, interpolate y[i, j, :] for new_x[i, j]. Modified for tau-interpolation (tau as axis)!

    Parameters
    ----------
    x : 1-D ndarray (double type)
        Array containg the x (abcissa) values. Must be monotonically decreasing.
    y : 3-D ndarray (double type)
        Array containing the y values to interpolate.
    x_new: 2-D ndarray (double type)
        Array with new abcissas to interpolate.

    Returns
    -------
    new_y : 3-D ndarray
        Interpolated values.
    """
    cdef DTYPE_t nx = y.shape[0]
    cdef DTYPE_t ny = y.shape[1]
    cdef DTYPE_t nz = y.shape[2]
    cdef DTYPE_t nxn = new_x.shape[2]
    cdef DTYPE_t i, j, k, l
    cdef np.ndarray[DTYPEf_t, ndim=3] new_y = np.zeros((nx, ny, nxn), dtype=DTYPEf)

    # for i in range(nx):
    for i in prange(nx, nogil=True):
        for j in range(ny):
            for l in range(nxn):
                for k in range(1, nz):
                     if x[k] < new_x[i, j, l]:
                         new_y[i, j, l] = (y[i, j, k] - y[i, j, k - 1]) * (new_x[i, j, l] - x[k-1]) /\
                                          (x[k] - x[k - 1]) + y[i, j, k - 1]
                         break
    return new_y

@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off bounds-checking for entire function
cpdef interp4dcube(np.ndarray[DTYPEf_t, ndim=1] x, np.ndarray[DTYPEf_t, ndim=4] y,
                   np.ndarray[DTYPEf_t, ndim=4] new_x):
    """
    interp4d(x, y, new_x)

    Performs linear interpolation over the last dimension of a 4D array, according to new values from a 3D array new_x.
    Thus, interpolate y[i, j, :] for new_x[i, j]. Modified for tau-interpolation (tau as axis)!

    Parameters
    ----------
    x : 1D ndarray (double type)
        Array containg the x (abcissa) values. Must be monotonically decreasing.
    y : 4D ndarray (double type)
        Array containing the y values to interpolate.
    x_new: 3D ndarray (double type)
        Array with new abcissas to interpolate.

    Returns
    -------
    new_y : 3D ndarray
        Interpolated values.
    """
    cdef DTYPE_t nt = y.shape[0]
    cdef DTYPE_t nx = y.shape[1]
    cdef DTYPE_t ny = y.shape[2]
    cdef DTYPE_t nz = y.shape[3]
    cdef DTYPE_t nxn = new_x.shape[3]
    cdef DTYPE_t i, j, k, l, m
    cdef np.ndarray[DTYPEf_t, ndim=4] new_y = np.zeros((nt, nx, ny, nxn), dtype=DTYPEf)

    for i in range(nt):
        # for j in range(nx):
        for j in prange(nx, nogil=True):
            for k in range(ny):
                for m in range(nxn):
                    for l in range(1, nz):
                         if x[l] < new_x[i, j, k, m]:
                             new_y[i, j, k, m] = (y[i, j, k, l] - y[i, j, k, l - 1]) * (new_x[i, j, k, m] - x[l-1]) /\
                                                 (x[l] - x[l - 1]) + y[i, j, k, l - 1]
                             break
    return new_y


@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off bounds-checking for entire function
cpdef cubeinterp3dcube(np.ndarray[DTYPEf_t, ndim=3] x, np.ndarray[DTYPEf_t, ndim=3] y,
                       np.ndarray[DTYPEf_t, ndim=3] new_x):
    """
    interp3d(x, y, new_x)

    Performs linear interpolation over the last dimension of a 3D array,
    according to new values from a 2D array new_x. Thus, interpolate
    y[i, j, :] for new_x[i, j].

    Parameters
    ----------
    x : 1-D ndarray (double type)
        Array containg the x (abcissa) values. Must be monotonically
        increasing.
    y : 3-D ndarray (double type)
        Array containing the y values to interpolate.
    x_new: 2-D ndarray (double type)
        Array with new abcissas to interpolate.

    Returns
    -------
    new_y : 3-D ndarray
        Interpolated values.
    """
    cdef DTYPE_t nx = y.shape[0]
    cdef DTYPE_t ny = y.shape[1]
    cdef DTYPE_t nz = y.shape[2]
    cdef DTYPE_t nxn = new_x.shape[2]
    cdef DTYPE_t i, j, k, l
    cdef np.ndarray[DTYPEf_t, ndim=3] new_y = np.zeros((nx, ny, nxn), dtype=DTYPEf)

    # for i in range(nx):
    for i in prange(nx, nogil=True):
        for j in range(ny):
            for l in range(nxn):
                for k in range(1, nz):
                     if x[i, j, k] < new_x[i, j, l]:
                         new_y[i, j, l] = (y[i, j, k] - y[i, j, k - 1]) * (new_x[i, j, l] - x[i, j, k-1]) /\
                                          (x[i, j, k] - x[i, j, k - 1]) + y[i, j, k - 1]
                         break
    return new_y


@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off bounds-checking for entire function
cpdef cubeinterp4dcube(np.ndarray[DTYPEf_t, ndim=4] x, np.ndarray[DTYPEf_t, ndim=4] y,
                   np.ndarray[DTYPEf_t, ndim=4] new_x):
    """
    interp4d(x, y, new_x)

    Performs linear interpolation over the last dimension of a 4D array,
    according to new values from a 3D array new_x. Thus, interpolate
    y[i, j, :] for new_x[i, j].

    Parameters
    ----------
    x : 1D ndarray (double type)
        Array containg the x (abcissa) values. Must be monotonically
        increasing.
    y : 4D ndarray (double type)
        Array containing the y values to interpolate.
    x_new: 3D ndarray (double type)
        Array with new abcissas to interpolate.

    Returns
    -------
    new_y : 3D ndarray
        Interpolated values.
    """
    cdef DTYPE_t nt = y.shape[0]
    cdef DTYPE_t nx = y.shape[1]
    cdef DTYPE_t ny = y.shape[2]
    cdef DTYPE_t nz = y.shape[3]
    cdef DTYPE_t nxn = new_x.shape[3]
    cdef DTYPE_t i, j, k, l, m
    cdef np.ndarray[DTYPEf_t, ndim=4] new_y = np.zeros((nt, nx, ny, nxn), dtype=DTYPEf)

    for i in range(nt):
        # for j in range(nx):
        for j in prange(nx, nogil=True):
            for k in range(ny):
                for m in range(nxn):
                    for l in range(1, nz):
                         if x[i, j, k, l] < new_x[i, j, k, m]:
                             new_y[i, j, k, m] = (y[i, j, k, l] - y[i, j, k, l - 1]) * (new_x[i, j, k, m] - x[i, j, k, l-1]) /\
                                                 (x[i, j, k, l] - x[i, j,k, l - 1]) + y[i, j, k, l - 1]
                             break
    return new_y


@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off bounds-checking for entire function
cpdef cubeinterp3d(np.ndarray[DTYPEf_t, ndim=3] x, np.ndarray[DTYPEf_t, ndim=3] y,
                       np.ndarray[DTYPEf_t, ndim=2] new_x):
    """
    interp3d(x, y, new_x)

    Performs linear interpolation over the last dimension of a 3D array,
    according to new values from a 2D array new_x. Thus, interpolate
    y[i, j, :] for new_x[i, j].

    Parameters
    ----------
    x : 1-D ndarray (double type)
        Array containg the x (abcissa) values. Must be monotonically
        increasing.
    y : 3-D ndarray (double type)
        Array containing the y values to interpolate.
    x_new: 2-D ndarray (double type)
        Array with new abcissas to interpolate.

    Returns
    -------
    new_y : 2D-ndarray, y-values at new x-positions.
    """
    cdef DTYPE_t nx = y.shape[0]
    cdef DTYPE_t ny = y.shape[1]
    cdef DTYPE_t nz = y.shape[2]
    cdef DTYPE_t i, j, k
    cdef np.ndarray[DTYPEf_t, ndim=2] new_y = np.zeros((nx, ny), dtype=DTYPEf)

    # for i in range(nx):
    for i in prange(nx, nogil=True):
        for j in range(ny):
            for k in range(1, nz):
                 if x[i, j, k] < new_x[i, j]:
                     new_y[i, j] = (y[i, j, k] - y[i, j, k - 1]) * (new_x[i, j] - x[i, j, k-1]) /\
                                   (x[i, j, k] - x[i, j, k - 1]) + y[i, j, k - 1]
                     break
    return new_y


@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off bounds-checking for entire function
cpdef cubeinterp4d(np.ndarray[DTYPEf_t, ndim=4] x, np.ndarray[DTYPEf_t, ndim=4] y,
                   np.ndarray[DTYPEf_t, ndim=3] new_x):
    """
    interp4d(x, y, new_x)

    Performs linear interpolation over the last dimension of a 4D array, according to new values from a 3D array new_x.
    Thus, interpolate y[i, j, :] for new_x[i, j].

    Parameters
    ----------
    x : 1D ndarray (double type), positions of y values. Must be strictly monotonically increasing.
    y : 4D ndarray (double type), y values to interpolate.
    x_new: 3D ndarray (double type),  new x-positions to interpolate.

    Returns
    -------
    new_y : 3D-ndarray, y-values at new x-positions.
    """
    cdef DTYPE_t nt = y.shape[0]
    cdef DTYPE_t nx = y.shape[1]
    cdef DTYPE_t ny = y.shape[2]
    cdef DTYPE_t nz = y.shape[3]
    cdef DTYPE_t i, j, k, l
    cdef np.ndarray[DTYPEf_t, ndim=3] new_y = np.zeros((nt, nx, ny), dtype=DTYPEf)

    for i in range(nt):
        # for j in range(nx):
        for j in prange(nx, nogil=True):
            for k in range(ny):
                for l in range(1, nz):
                     if x[i, j, k, l] < new_x[i, j, k]:
                         new_y[i, j, k] = (y[i, j, k, l] - y[i, j, k, l - 1]) * (new_x[i, j, k] - x[i, j, k, l-1]) /\
                                          (x[i, j, k, l] - x[i, j,k, l - 1]) + y[i, j, k, l - 1]
                         break
    return new_y


@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off bounds-checking for entire function
cpdef cubeinterp3dvec(np.ndarray[DTYPEf_t, ndim=3] x, np.ndarray[DTYPEf_t, ndim=3] y,
                       np.ndarray[DTYPEf_t, ndim=1] new_x):
    """
    interp3d(x, y, new_x)

    Performs linear interpolation over the last dimension of a 3D array,
    according to new values from a 2D array new_x. Thus, interpolate
    y[i, j, :] for new_x[i, j].

    Parameters
    ----------
    x : 1-D ndarray (double type)
        Array containg the x (abcissa) values. Must be monotonically
        increasing.
    y : 3-D ndarray (double type)
        Array containing the y values to interpolate.
    x_new: 2-D ndarray (double type)
        Array with new abcissas to interpolate.

    Returns
    -------
    new_y : 2D-ndarray, y-values at new x-positions.
    """
    cdef DTYPE_t nx = y.shape[0]
    cdef DTYPE_t ny = y.shape[1]
    cdef DTYPE_t nz = y.shape[2]
    cdef DTYPE_t nxn = new_x.shape[0]
    cdef DTYPE_t i, j, k, l
    cdef np.ndarray[DTYPEf_t, ndim=3] new_y = np.zeros((nx, ny, nxn), dtype=DTYPEf)

    # for i in range(nx):
    for i in prange(nx, nogil=True):
        for j in range(ny):
            for l in range(nxn):
                for k in range(1, nz):
                     if x[i, j, k] < new_x[l]:
                         new_y[i, j, l] = (y[i, j, k] - y[i, j, k - 1]) * (new_x[l] - x[i, j, k-1]) /\
                                          (x[i, j, k] - x[i, j, k - 1]) + y[i, j, k - 1]
                         break
    return new_y


@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off bounds-checking for entire function
cpdef cubeinterp4dvec(np.ndarray[DTYPEf_t, ndim=4] x, np.ndarray[DTYPEf_t, ndim=4] y,
                      np.ndarray[DTYPEf_t, ndim=1] new_x):
    """
    interp4d(x, y, new_x)

    Performs linear interpolation over the last dimension of a 4D array, according to new values from a 3D array new_x.
    Thus, interpolate y[i, j, :] for new_x[i, j].

    Parameters
    ----------
    x : 1D ndarray (double type), positions of y values. Must be strictly monotonically increasing.
    y : 4D ndarray (double type), y values to interpolate.
    x_new: 3D ndarray (double type),  new x-positions to interpolate.

    Returns
    -------
    new_y : 4D-ndarray, y-values at new x-positions.
    """
    cdef DTYPE_t nt = y.shape[0]
    cdef DTYPE_t nx = y.shape[1]
    cdef DTYPE_t ny = y.shape[2]
    cdef DTYPE_t nz = y.shape[3]
    cdef DTYPE_t nxn = new_x.shape[0]
    cdef DTYPE_t i, j, k, l, m
    cdef np.ndarray[DTYPEf_t, ndim=4] new_y = np.zeros((nt, nx, ny, nxn), dtype=DTYPEf)

    for i in range(nt):
        # for j in range(nx):
        for j in prange(nx, nogil=True):
            for k in range(ny):
                for m in range(nxn):
                    for l in range(1, nz):
                         if x[i, j, k, l] < new_x[m]:
                             new_y[i, j, k, m] = (y[i, j, k, l] - y[i, j, k, l - 1]) * (new_x[m] - x[i, j, k, l-1]) /\
                                                 (x[i, j, k, l] - x[i, j,k, l - 1]) + y[i, j, k, l - 1]
                             break
    return new_y


@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off bounds-checking for entire function
cpdef cubeinterp3dval(np.ndarray[DTYPEf_t, ndim=3] x, np.ndarray[DTYPEf_t, ndim=3] y, DTYPEf_t new_x):
    """
    cubeinterp3dval(x, y, new_x)

    Performs linear interpolation over the last dimension of a 3D array, according to new values from a 2D array new_x.
    Thus, interpolate y[i, j, :] defined at x[i, j, :] onto new_x[i, j].

    Parameters
    ----------
    x : 3D ndarray (float), positions of y values. Must be strictly monotonically increasing along last axis.
    y : 3D ndarray (float), y values.
    x_new: float,  new x-positions of interpolation. Will lead to plane perpendicular to last axis.

    Returns
    -------
    new_y : 2D-ndarray, y-values at new x-positions.
    """
    cdef DTYPE_t nx = y.shape[0]
    cdef DTYPE_t ny = y.shape[1]
    cdef DTYPE_t nz = y.shape[2]
    cdef DTYPE_t i, j, k
    cdef np.ndarray[DTYPEf_t, ndim=2] new_y = np.zeros((nx, ny), dtype=DTYPEf)

    # for i in range(nx):
    for i in prange(nx, nogil=True):
        for j in range(ny):
            for k in range(1, nz):
                 if x[i, j, k] < new_x:
                     new_y[i, j] = (y[i, j, k] - y[i, j, k - 1]) * (new_x - x[i, j, k-1]) /\
                                   (x[i, j, k] - x[i, j, k - 1]) + y[i, j, k - 1]
                     break
    return new_y


@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off bounds-checking for entire function
cpdef cubeinterp4dval(np.ndarray[DTYPEf_t, ndim=4] x, np.ndarray[DTYPEf_t, ndim=4] y, DTYPEf_t new_x):
    """
    cubeinterp4dval(x, y, new_x)

    Performs linear interpolation over the last dimension of a 4D array, according to new values from a 3D array new_x.
    Thus, interpolate y[i, j, k, :] defined at x[i, j, k, :] onto new_x[i, j, k].

    Parameters
    ----------
    x : 4D ndarray (float), positions of y values. Must be strictly monotonically increasing along last axis.
    y : 4D ndarray (float), y values.
    x_new: float,  new x-positions of interpolation. Will lead to plane perpendicular to last axis.

    Returns
    -------
    new_y : 3D-ndarray, y-values at new x-positions.
    """
    cdef DTYPE_t nt = y.shape[0]
    cdef DTYPE_t nx = y.shape[1]
    cdef DTYPE_t ny = y.shape[2]
    cdef DTYPE_t nz = y.shape[3]
    cdef DTYPE_t i, j, k, l
    cdef np.ndarray[DTYPEf_t, ndim=3] new_y = np.zeros((nt, nx, ny), dtype=DTYPEf)

    for i in range(nt):
        # for j in range(nx):
        for j in prange(nx, nogil=True):
            for k in range(ny):
                for l in range(1, nz):
                     if x[i, j, k, l] < new_x:
                         new_y[i, j, k] = (y[i, j, k, l] - y[i, j, k, l - 1]) * (new_x - x[i, j, k, l-1]) /\
                                          (x[i, j, k, l] - x[i, j,k, l - 1]) + y[i, j, k, l - 1]
                         break
    return new_y

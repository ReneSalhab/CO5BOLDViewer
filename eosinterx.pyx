# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 14:21:59 2016

@author: René Georg Salhab
"""

from __future__ import division, print_function

import numpy as np
import numexpr as ne
from cython.parallel import prange

cimport cython
cimport numpy as np
from libc.math cimport exp, log10

DTYPE = np.int32
DTYPEf = np.float32
DTYPEf64 = np.float64

ctypedef np.int32_t DTYPE_t
ctypedef np.float32_t DTYPEf_t
ctypedef np.float64_t DTYPEf64_t

cdef inline DTYPE_t int_max(DTYPE_t a, DTYPE_t b): return a if a >= b else b
cdef inline DTYPE_t int_min(DTYPE_t a, DTYPE_t b): return a if a <= b else b

cdef inline DTYPEf64_t float_max(DTYPEf64_t a, DTYPEf64_t b): return a if a >= b else b
cdef inline DTYPEf64_t float_min(DTYPEf64_t a, DTYPEf64_t b): return a if a <= b else b


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef STP3D(np.ndarray[DTYPEf64_t, ndim=3] rho, np.ndarray[DTYPEf64_t, ndim=3] ei, np.ndarray[DTYPEf64_t, ndim=3] C,
            np.ndarray[DTYPE_t, ndim=3] i1, np.ndarray[DTYPE_t, ndim=3] i2, np.ndarray[DTYPEf64_t, ndim=3] x1ta,
            np.ndarray[DTYPEf64_t, ndim=3] x2ta):
    """
        Description:
            Computes the entropy, pressure, or temperature with the hekp of the eos-file.
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
    cdef unsigned int i, j, k
    cdef unsigned int nx = rho.shape[0]
    cdef unsigned int ny = rho.shape[1]
    cdef unsigned int nz = rho.shape[2]
    cdef np.ndarray[DTYPEf64_t, ndim = 3] out = np.empty((nx, ny, nz), dtype=DTYPEf64)

    for i in prange(nx, nogil=True):
        for j in range(ny):
            for k in range(nz):
                out[i,j,k] = C[0,i1[i,j,k],i2[i,j,k]] + x1ta[i,j,k] * (C[1,i1[i,j,k],i2[i,j,k]] + x1ta[i,j,k] *
                                (C[2,i1[i,j,k],i2[i,j,k]] + x1ta[i,j,k] * C[3,i1[i,j,k],i2[i,j,k]])) +\
                             x2ta[i,j,k] * (C[4,i1[i,j,k],i2[i,j,k]] + x1ta[i,j,k] * (C[5,i1[i,j,k], i2[i,j,k]] +
                                 x1ta[i,j,k] * (C[6,i1[i,j,k],i2[i,j,k]] + x1ta[i,j,k] * C[7,i1[i,j,k],i2[i,j,k]])) +
                                 x2ta[i,j,k] * (C[8,i1[i,j,k],i2[i,j,k]] + x1ta[i,j,k] * (C[9,i1[i,j,k], i2[i,j,k]] +
                                 x1ta[i,j,k] * (C[10,i1[i,j,k],i2[i,j,k]] + x1ta[i,j,k] * C[11,i1[i,j,k], i2[i,j,k]])) +
                                 x2ta[i,j,k] * (C[12,i1[i,j,k],i2[i,j,k]] + x1ta[i,j,k] * (C[13,i1[i,j,k],i2[i,j,k]] +
                                 x1ta[i,j,k] * (C[14,i1[i,j,k], i2[i,j,k]] + x1ta[i,j,k] * C[15,i1[i,j,k],i2[i,j,k]])))))
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef STP4D(np.ndarray[DTYPEf64_t, ndim=4] rho, np.ndarray[DTYPEf64_t, ndim=4] ei, np.ndarray[DTYPEf64_t, ndim=3] C,
            np.ndarray[DTYPE_t, ndim=4] i1, np.ndarray[DTYPE_t, ndim=4] i2, np.ndarray[DTYPEf64_t, ndim=4] x1ta,
            np.ndarray[DTYPEf64_t, ndim=4] x2ta):
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
    cdef unsigned int i, j, k, l
    cdef unsigned int nt = rho.shape[0]
    cdef unsigned int nx = rho.shape[1]
    cdef unsigned int ny = rho.shape[2]
    cdef unsigned int nz = rho.shape[3]
    cdef np.ndarray[DTYPEf64_t, ndim = 4] out = np.empty((nt, nx, ny, nz), dtype=DTYPEf64)

    for i in range(nt):
        for j in prange(nx, nogil=True):
            for k in range(ny):
                for l in range(nz):
                    out[i,j,k,l] = C[0,i1[i,j,k,l],i2[i, j, k, l]] + x1ta[i, j, k, l] *\
                                      (C[1, i1[i, j, k, l], i2[i, j, k, l]] + x1ta[i, j, k, l] *
                                        (C[2, i1[i, j, k, l], i2[i, j, k, l]] + x1ta[i, j, k, l] *
                                         C[3, i1[i, j, k, l], i2[i, j, k, l]])) +\
                                      x2ta[i, j, k, l] * (C[4, i1[i, j, k, l], i2[i, j, k, l]] + x1ta[i, j, k, l] *
                                       (C[5, i1[i, j, k, l], i2[i, j, k, l]] + x1ta[i, j, k, l] *
                                        (C[6, i1[i, j, k, l], i2[i, j, k, l]] + x1ta[i, j, k, l] *
                                         C[7, i1[i, j, k, l], i2[i, j, k ,l]])) +
                                       x2ta[i, j, k, l] * (C[8, i1[i, j, k, l], i2[i, j, k, l]] + x1ta[i, j, k, l] *
                                        (C[9, i1[i, j, k, l], i2[i, j, k, l]] + x1ta[i, j, k, l] *
                                         (C[10, i1[i, j, k, l], i2[i, j, k, l]] + x1ta[i, j, k, l] *
                                          C[11, i1[i, j, k, l], i2[i, j, k, l]])) + x2ta[i, j, k ,l] *
                                          (C[12, i1[i, j, k, l], i2[i, j, k, l]] + x1ta[i, j, k, l] *
                                           (C[13, i1[i, j, k, l], i2[i, j, k, l]] + x1ta[i, j, k, l] *
                                            (C[14, i1[i, j, k, l], i2[i, j, k, l]] + x1ta[i, j, k, l] *
                                             C[15, i1[i, j, k, l], i2[i, j, k, l]])))))
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef PandT3D(np.ndarray[DTYPEf64_t, ndim=3] rho, np.ndarray[DTYPEf64_t, ndim=3] ei, np.ndarray[DTYPEf64_t, ndim=3] CP,
              np.ndarray[DTYPEf64_t, ndim=3] CT, np.ndarray[DTYPE_t, ndim=3] i1, np.ndarray[DTYPE_t, ndim=3] i2,
              np.ndarray[DTYPEf64_t, ndim=3] x1ta, np.ndarray[DTYPEf64_t, ndim=3] x2ta):
    cdef unsigned int i, j, k
    cdef unsigned int nx = rho.shape[0]
    cdef unsigned int ny = rho.shape[1]
    cdef unsigned int nz = rho.shape[2]
    cdef np.ndarray[DTYPEf64_t, ndim = 3] P = np.empty((nx, ny, nz), dtype=DTYPEf64)
    cdef np.ndarray[DTYPEf64_t, ndim = 3] T = np.empty((nx, ny, nz), dtype=DTYPEf64)

    for i in prange(nx, nogil=True):
        for j in range(ny):
            for k in range(nz):
                P[i, j, k] = exp(CP[0, i1[i, j, k], i2[i, j, k]] + x1ta[i, j, k] *\
                               (CP[1, i1[i, j, k], i2[i, j, k]] + x1ta[i, j, k] * (CP[2, i1[i, j, k], i2[i, j, k]] +
                                    x1ta[i, j, k] * CP[3, i1[i, j, k], i2[i, j, k]])) +\
                               x2ta[i, j, k] * (CP[4, i1[i, j, k], i2[i, j, k]] + x1ta[i, j, k] *
                                    (CP[5, i1[i, j, k], i2[i, j, k]] + x1ta[i, j, k] *
                                     (CP[6, i1[i, j, k], i2[i, j, k]] + x1ta[i, j, k] * CP[7, i1[i, j, k], i2[i, j, k]])) +
                               x2ta[i, j, k] * (CP[8, i1[i, j, k], i2[i, j, k]] + x1ta[i, j, k] *
                                    (CP[9, i1[i, j, k], i2[i, j, k]] + x1ta[i, j, k] * (CP[10, i1[i, j, k], i2[i, j, k]] +
                                        x1ta[i, j, k] * CP[11, i1[i, j, k], i2[i, j, k]])) + x2ta[i, j, k] *
                                            (CP[12, i1[i, j, k], i2[i, j, k]] + x1ta[i, j, k] *
                                             (CP[13, i1[i, j, k], i2[i, j, k]] + x1ta[i, j, k] *
                                              (CP[14, i1[i, j, k], i2[i, j, k]] + x1ta[i, j, k] *
                                               CP[15, i1[i, j, k], i2[i, j, k]]))))))
                T[i, j, k] = exp(CT[0, i1[i, j, k], i2[i, j, k]] + x1ta[i, j, k] *\
                               (CT[1, i1[i, j, k], i2[i, j, k]] + x1ta[i, j, k] * (CT[2, i1[i, j, k], i2[i, j, k]] +
                                    x1ta[i, j, k] * CT[3, i1[i, j, k], i2[i, j, k]])) +\
                               x2ta[i, j, k] * (CT[4, i1[i, j, k], i2[i, j, k]] + x1ta[i, j, k] *
                                    (CT[5, i1[i, j, k], i2[i, j, k]] + x1ta[i, j, k] *
                                     (CT[6, i1[i, j, k], i2[i, j, k]] + x1ta[i, j, k] * CT[7, i1[i, j, k], i2[i, j, k]])) +
                               x2ta[i, j, k] * (CT[8, i1[i, j, k], i2[i, j, k]] + x1ta[i, j, k] *
                                    (CT[9, i1[i, j, k], i2[i, j, k]] + x1ta[i, j, k] * (CT[10, i1[i, j, k], i2[i, j, k]] +
                                        x1ta[i, j, k] * CT[11, i1[i, j, k], i2[i, j, k]])) + x2ta[i, j, k] *
                                            (CT[12, i1[i, j, k], i2[i, j, k]] + x1ta[i, j, k] *
                                             (CT[13, i1[i, j, k], i2[i, j, k]] + x1ta[i, j, k] *
                                              (CT[14, i1[i, j, k], i2[i, j, k]] + x1ta[i, j, k] *
                                               CT[15, i1[i, j, k], i2[i, j, k]]))))))
    return P, T #ne.evaluate("exp(P)"), ne.evaluate("exp(T)")


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef PandT4D(np.ndarray[DTYPEf64_t, ndim=4] rho, np.ndarray[DTYPEf64_t, ndim=4] ei, np.ndarray[DTYPEf64_t, ndim=3] CP,
              np.ndarray[DTYPEf64_t, ndim=3] CT, np.ndarray[DTYPE_t, ndim=4] i1, np.ndarray[DTYPE_t, ndim=4] i2,
              np.ndarray[DTYPEf64_t, ndim=4] x1ta, np.ndarray[DTYPEf64_t, ndim=4] x2ta):
    cdef unsigned int i, j, k, l
    cdef unsigned int nt = rho.shape[0]
    cdef unsigned int nx = rho.shape[1]
    cdef unsigned int ny = rho.shape[2]
    cdef unsigned int nz = rho.shape[3]
    cdef np.ndarray[DTYPEf64_t, ndim = 4] P = np.empty((nt, nx, ny, nz), dtype=DTYPEf64)
    cdef np.ndarray[DTYPEf64_t, ndim = 4] T = np.empty((nt, nx, ny, nz), dtype=DTYPEf64)

    for i in range(nt):
        for j in prange(nx, nogil=True):
            for k in range(ny):
                for l in range(nz):
                    P[i, j, k, l] = exp(CP[0, i1[i, j, k, l], i2[i, j, k, l]] + x1ta[i, j, k, l] *\
                                    (CP[1, i1[i, j, k, l], i2[i, j, k, l]] + x1ta[i, j, k, l] *
                                     (CP[2, i1[i, j, k, l], i2[i, j, k, l]] + x1ta[i, j, k, l] *
                                      CP[3, i1[i, j, k, l], i2[i, j, k, l]])) +\
                                    x2ta[i, j, k, l] * (CP[4, i1[i, j, k, l], i2[i, j, k, l]] + x1ta[i, j, k, l] *
                                     (CP[5, i1[i, j, k, l], i2[i, j, k, l]] + x1ta[i, j, k, l] *
                                      (CP[6, i1[i, j, k, l], i2[i, j, k, l]] + x1ta[i, j, k, l] *
                                       CP[7, i1[i, j, k, l], i2[i, j, k ,l]])) +
                                     x2ta[i, j, k, l] * (CP[8, i1[i, j, k, l], i2[i, j, k, l]] + x1ta[i, j, k, l] *
                                      (CP[9, i1[i, j, k, l], i2[i, j, k, l]] + x1ta[i, j, k, l] *
                                       (CP[10, i1[i, j, k, l], i2[i, j, k, l]] + x1ta[i, j, k, l] *
                                        CP[11, i1[i, j, k, l], i2[i, j, k, l]])) + x2ta[i, j, k ,l] *
                                        (CP[12, i1[i, j, k, l], i2[i, j, k, l]] + x1ta[i, j, k, l] *
                                         (CP[13, i1[i, j, k, l], i2[i, j, k, l]] + x1ta[i, j, k, l] *
                                          (CP[14, i1[i, j, k, l], i2[i, j, k, l]] + x1ta[i, j, k, l] *
                                           CP[15, i1[i, j, k, l], i2[i, j, k, l]]))))))
                    T[i, j, k, l] = exp(CT[0, i1[i, j, k, l], i2[i, j, k, l]] + x1ta[i, j, k, l] *\
                                    (CT[1, i1[i, j, k, l], i2[i, j, k, l]] + x1ta[i, j, k, l] *
                                     (CT[2, i1[i, j, k, l], i2[i, j, k, l]] + x1ta[i, j, k, l] *
                                      CT[3, i1[i, j, k, l], i2[i, j, k, l]])) +\
                                    x2ta[i, j, k, l] * (CT[4, i1[i, j, k, l], i2[i, j, k, l]] + x1ta[i, j, k, l] *
                                    (CT[5, i1[i, j, k, l], i2[i, j, k, l]] + x1ta[i, j, k, l] *
                                     (CT[6, i1[i, j, k, l], i2[i, j, k, l]] + x1ta[i, j, k, l] *
                                      CT[7, i1[i, j, k, l], i2[i, j, k ,l]])) +
                                     x2ta[i, j, k, l] * (CT[8, i1[i, j, k, l], i2[i, j, k, l]] + x1ta[i, j, k, l] *
                                      (CT[9, i1[i, j, k, l], i2[i, j, k, l]] + x1ta[i, j, k, l] *
                                       (CT[10, i1[i, j, k, l], i2[i, j, k, l]] + x1ta[i, j, k, l] *
                                        CT[11, i1[i, j, k, l], i2[i, j, k, l]])) + x2ta[i, j, k ,l] *
                                        (CT[12, i1[i, j, k, l], i2[i, j, k, l]] + x1ta[i, j, k, l] *
                                         (CT[13, i1[i, j, k, l], i2[i, j, k, l]] + x1ta[i, j, k, l] *
                                          (CT[14, i1[i, j, k, l], i2[i, j, k, l]] + x1ta[i, j, k, l] *
                                           CT[15, i1[i, j, k, l], i2[i, j, k, l]]))))))
    return P, T #ne.evaluate("exp(P)"), ne.evaluate("exp(T)")


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef Pall3D(np.ndarray[DTYPEf64_t, ndim=3] rho, np.ndarray[DTYPEf64_t, ndim=3] ei, np.ndarray[DTYPEf64_t, ndim=3] C,
             np.ndarray[DTYPE_t, ndim=3] i1, np.ndarray[DTYPE_t, ndim=3] i2, np.ndarray[DTYPEf64_t, ndim=3] x1ta,
             np.ndarray[DTYPEf64_t, ndim=3] x2ta, DTYPEf64_t x2shift):
    cdef unsigned int i, j, k
    cdef unsigned int nx = rho.shape[0]
    cdef unsigned int ny = rho.shape[1]
    cdef unsigned int nz = rho.shape[2]
    cdef np.ndarray[DTYPEf64_t, ndim = 3] P = np.empty((nx, ny, nz), dtype=DTYPEf64)
    cdef np.ndarray[DTYPEf64_t, ndim = 3] dPdrho = np.empty((nx, ny, nz), dtype=DTYPEf64)
    cdef np.ndarray[DTYPEf64_t, ndim = 3] dPde = np.empty((nx, ny, nz), dtype=DTYPEf64)

    for i in prange(nx, nogil=True):
        for j in range(ny):
            for k in range(nz):
                P[i, j, k] = exp(C[0, i1[i, j, k], i2[i, j, k]] + x1ta[i, j, k] * (C[1, i1[i, j, k], i2[i, j, k]] +
                    x1ta[i, j, k] * (C[2, i1[i, j, k], i2[i, j, k]] + x1ta[i, j, k] * C[3, i1[i, j, k], i2[i, j, k]])) +\
                  x2ta[i, j, k] * (C[4, i1[i, j, k], i2[i, j, k]] + x1ta[i, j, k] * (C[5, i1[i, j, k], i2[i, j, k]] +
                    x1ta[i, j, k] * (C[6, i1[i, j, k], i2[i, j, k]] + x1ta[i, j, k] * C[7, i1[i, j, k], i2[i, j, k]])) +
                  x2ta[i, j, k] * (C[8, i1[i, j, k], i2[i, j, k]] + x1ta[i, j, k] * (C[9, i1[i, j, k], i2[i, j, k]] +
                    x1ta[i, j, k] * (C[10, i1[i, j, k], i2[i, j, k]] + x1ta[i, j, k] * C[11, i1[i, j, k], i2[i, j, k]])) +
                  x2ta[i, j, k] * (C[12, i1[i, j, k], i2[i, j, k]] + x1ta[i, j, k] * (C[13, i1[i, j, k], i2[i, j, k]] +
                    x1ta[i, j, k] * (C[14, i1[i, j, k], i2[i, j, k]] + x1ta[i, j, k] * C[15, i1[i, j, k], i2[i, j, k]]))))))

    # P = ne.avaluate("exp(P)")
    # for i in range(nx):
    #     for j in range(ny):
    #         for k in range(nz):
                dPdrho[i,j,k] = P[i,j,k] / rho[i,j,k] * (C[1,i1[i,j,k],i2[i,j,k]] + x1ta[i,j,k] *
                                    (2 * C[2,i1[i,j,k],i2[i,j,k]] + x1ta[i,j,k] * 3 * C[3,i1[i,j,k],i2[i,j,k]]) +
                                  x2ta[i,j,k]*(C[5,i1[i,j,k],i2[i,j,k]] + x1ta[i,j,k] * (2 * C[6,i1[i,j,k],i2[i,j,k]]+
                                    x1ta[i,j,k] * 3 * C[7,i1[i,j,k],i2[i,j,k]]) + x2ta[i,j,k] *
                                      (C[9,i1[i,j,k],i2[i,j,k]] + x1ta[i,j,k] * (2 * C[10,i1[i,j,k],i2[i,j,k]] +
                                        x1ta[i,j,k] * 3 * C[11,i1[i,j,k],i2[i,j,k]]) + x2ta[i,j,k] *
                                       (C[13,i1[i,j,k],i2[i,j,k]] + x1ta[i,j,k] * (2 * C[14,i1[i,j,k],i2[i,j,k]] +
                                        x1ta[i,j,k] * 3 * C[15,i1[i,j,k],i2[i,j,k]])))))

                dPde[i,j,k] = P[i,j,k] / (ei[i,j,k] + x2shift) * (C[4,i1[i,j,k],i2[i,j,k]] + x1ta[i,j,k] *
                                (C[5,i1[i,j,k],i2[i,j,k]] + x1ta[i,j,k] * (C[6,i1[i,j,k],i2[i,j,k]] + x1ta[i,j,k] *
                                 C[7,i1[i,j,k],i2[i,j,k]])) + 2 * x2ta[i,j,k] * (C[8,i1[i,j,k],i2[i,j,k]] +
                               x1ta[i,j,k] * (C[9,i1[i,j,k],i2[i,j,k]] + x1ta[i,j,k] * (C[10,i1[i,j,k],i2[i,j,k]] +
                                x1ta[i,j,k] * C[11,i1[i,j,k],i2[i,j,k]])) + 1.5 * x2ta[i,j,k] *
                                (C[12,i1[i,j,k],i2[i,j,k]] + x1ta[i,j,k] * (C[13,i1[i,j,k],i2[i,j,k]] + x1ta[i,j,k] *
                                (C[14,i1[i,j,k],i2[i,j,k]] + x1ta[i,j,k] * C[15,i1[i,j,k],i2[i,j,k]])))))
    return P, dPdrho, dPde


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef Pall4D(np.ndarray[DTYPEf64_t, ndim=4] rho, np.ndarray[DTYPEf64_t, ndim=4] ei, np.ndarray[DTYPEf64_t, ndim=3] C,
             np.ndarray[DTYPE_t, ndim=4] i1, np.ndarray[DTYPE_t, ndim=4] i2, np.ndarray[DTYPEf64_t, ndim=4] x1ta,
             np.ndarray[DTYPEf64_t, ndim=4] x2ta, DTYPEf64_t x2shift):
    cdef unsigned int i, j, k, l
    cdef unsigned int nt = rho.shape[0]
    cdef unsigned int nx = rho.shape[1]
    cdef unsigned int ny = rho.shape[2]
    cdef unsigned int nz = rho.shape[3]
    cdef np.ndarray[DTYPEf64_t, ndim = 4] P = np.empty((nt, nx, ny, nz), dtype=DTYPEf64)
    cdef np.ndarray[DTYPEf64_t, ndim = 4] dPdrho = np.empty((nt, nx, ny, nz), dtype=DTYPEf64)
    cdef np.ndarray[DTYPEf64_t, ndim = 4] dPde = np.empty((nt, nx, ny, nz), dtype=DTYPEf64)

    for i in range(nt):
        for j in prange(nx, nogil=True):
            for k in range(ny):
                for l in range(nz):
                    P[i,j,k,l] = exp(C[0,i1[i,j,k,l], i2[i,j,k,l]] + x1ta[i,j,k,l] * (C[1,i1[i,j,k,l],i2[i,j,k,l]] +
                        x1ta[i,j,k,l] * (C[2,i1[i,j,k,l],i2[i,j,k,l]] + x1ta[i,j,k,l] * C[3,i1[i,j,k,l],i2[i,j,k,l]])) +\
                      x2ta[i,j,k,l] * (C[4,i1[i,j,k,l],i2[i,j,k,l]] + x1ta[i,j,k,l] * (C[5,i1[i,j,k,l],i2[i,j,k,l]] +
                        x1ta[i,j,k,l] * (C[6,i1[i,j,k,l],i2[i,j,k,l]] + x1ta[i,j,k,l] * C[7,i1[i,j,k,l],i2[i,j,k,l]])) +
                      x2ta[i,j,k,l] * (C[8,i1[i,j,k,l],i2[i,j,k,l]] + x1ta[i,j,k,l] * (C[9,i1[i,j,k,l],i2[i,j,k,l]] +
                        x1ta[i,j,k,l] * (C[10,i1[i,j,k,l],i2[i,j,k,l]] + x1ta[i,j,k,l] * C[11,i1[i,j,k,l], i2[i,j,k,l]])) +
                      x2ta[i,j,k,l] * (C[12, i1[i,j,k,l], i2[i,j,k,l]] + x1ta[i,j,k,l] * (C[13, i1[i,j,k,l], i2[i,j,k,l]] +
                        x1ta[i,j,k,l] * (C[14, i1[i,j,k,l], i2[i,j,k,l]] + x1ta[i,j,k,l] * C[15, i1[i,j,k,l], i2[i,j,k,l]]))))))

    # P = ne.avaluate("exp(P)")
    # for i in range(nt):
    #     for j in range(nx):
    #         for k in range(ny):
    #             for l in range(nz):
                    dPdrho[i,j,k,l] = P[i,j,k,l] / rho[i,j,k,l] * (C[1,i1[i,j,k,l],i2[i,j,k,l]] + x1ta[i,j,k,l] *
                                        (2 * C[2,i1[i,j,k,l],i2[i,j,k,l]] + x1ta[i,j,k,l] * 3 * C[3,i1[i,j,k,l],i2[i,j,k,l]]) +
                                      x2ta[i,j,k,l]*(C[5,i1[i,j,k,l],i2[i,j,k,l]] + x1ta[i,j,k,l] * (2 * C[6,i1[i,j,k,l],i2[i,j,k,l]]+
                                        x1ta[i,j,k,l] * 3 * C[7,i1[i,j,k,l],i2[i,j,k,l]]) + x2ta[i,j,k,l] *
                                          (C[9,i1[i,j,k,l],i2[i,j,k,l]] + x1ta[i,j,k,l] * (2 * C[10,i1[i,j,k,l],i2[i,j,k,l]] +
                                            x1ta[i,j,k,l] * 3 * C[11,i1[i,j,k,l],i2[i,j,k,l]]) + x2ta[i,j,k,l] *
                                           (C[13,i1[i,j,k,l],i2[i,j,k,l]] + x1ta[i,j,k,l] * (2 * C[14,i1[i,j,k,l],i2[i,j,k,l]] +
                                            x1ta[i,j,k,l] * 3 * C[15,i1[i,j,k,l],i2[i,j,k,l]])))))

                    dPde[i,j,k,l] = P[i,j,k,l] / (ei[i,j,k,l] + x2shift) * (C[4,i1[i,j,k,l],i2[i,j,k,l]] + x1ta[i,j,k,l] *
                                    (C[5,i1[i,j,k,l],i2[i,j,k,l]] + x1ta[i,j,k,l] * (C[6,i1[i,j,k,l],i2[i,j,k,l]] + x1ta[i,j,k,l] *
                                     C[7,i1[i,j,k,l],i2[i,j,k,l]])) + 2 * x2ta[i,j,k,l] * (C[8,i1[i,j,k,l],i2[i,j,k,l]] +
                                   x1ta[i,j,k,l] * (C[9,i1[i,j,k,l],i2[i,j,k,l]] + x1ta[i,j,k,l] * (C[10,i1[i,j,k,l],i2[i,j,k,l]] +
                                    x1ta[i,j,k,l] * C[11,i1[i,j,k,l],i2[i,j,k,l]])) + 1.5 * x2ta[i,j,k,l] *
                                    (C[12,i1[i,j,k,l],i2[i,j,k,l]] + x1ta[i,j,k,l] * (C[13,i1[i,j,k,l],i2[i,j,k,l]] + x1ta[i,j,k,l] *
                                    (C[14,i1[i,j,k,l],i2[i,j,k,l]] + x1ta[i,j,k,l] * C[15,i1[i,j,k,l],i2[i,j,k,l]])))))
    return P, dPdrho, dPde


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef Tall3D(np.ndarray[DTYPEf64_t, ndim=3] rho, np.ndarray[DTYPEf64_t, ndim=3] ei, np.ndarray[DTYPEf64_t, ndim=3] C,
             np.ndarray[DTYPE_t, ndim=3] i1, np.ndarray[DTYPE_t, ndim=3] i2, np.ndarray[DTYPEf64_t, ndim=3] x1ta,
             np.ndarray[DTYPEf64_t, ndim=3] x2ta, DTYPEf64_t x2shift):
    cdef unsigned int i, j, k
    cdef unsigned int nx = rho.shape[0]
    cdef unsigned int ny = rho.shape[1]
    cdef unsigned int nz = rho.shape[2]
    cdef np.ndarray[DTYPEf64_t, ndim=3] T = np.empty((nx, ny, nz), dtype=DTYPEf64)
    cdef np.ndarray[DTYPEf64_t, ndim=3] dTde = np.empty((nx, ny, nz), dtype=DTYPEf64)

    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                T[i, j, k] = exp(C[0, i1[i, j, k], i2[i, j, k]] + x1ta[i, j, k] * (C[1, i1[i, j, k], i2[i, j, k]] +
                    x1ta[i, j, k] * (C[2, i1[i, j, k], i2[i, j, k]] + x1ta[i, j, k] * C[3, i1[i, j, k], i2[i, j, k]])) +\
                  x2ta[i, j, k] * (C[4, i1[i, j, k], i2[i, j, k]] + x1ta[i, j, k] * (C[5, i1[i, j, k], i2[i, j, k]] +
                    x1ta[i, j, k] * (C[6, i1[i, j, k], i2[i, j, k]] + x1ta[i, j, k] * C[7, i1[i, j, k], i2[i, j, k]])) +
                  x2ta[i, j, k] * (C[8, i1[i, j, k], i2[i, j, k]] + x1ta[i, j, k] * (C[9, i1[i, j, k], i2[i, j, k]] +
                    x1ta[i, j, k] * (C[10, i1[i, j, k], i2[i, j, k]] + x1ta[i, j, k] * C[11, i1[i, j, k], i2[i, j, k]])) +
                  x2ta[i, j, k] * (C[12, i1[i, j, k], i2[i, j, k]] + x1ta[i, j, k] * (C[13, i1[i, j, k], i2[i, j, k]] +
                    x1ta[i, j, k] * (C[14, i1[i, j, k], i2[i, j, k]] + x1ta[i, j, k] * C[15, i1[i, j, k], i2[i, j, k]]))))))

    # T = ne.avaluate("exp(T)")
    # for i in range(nx):
    #     for j in range(ny):
    #         for k in range(nz):
                dTde[i,j,k] = T[i,j,k] / (ei[i,j,k] + x2shift) * (C[4,i1[i,j,k],i2[i,j,k]] + x1ta[i,j,k] *
                                (C[5,i1[i,j,k],i2[i,j,k]] + x1ta[i,j,k] * (C[6,i1[i,j,k],i2[i,j,k]] + x1ta[i,j,k] *
                                 C[7,i1[i,j,k],i2[i,j,k]])) + 2 * x2ta[i,j,k] * (C[8,i1[i,j,k],i2[i,j,k]] +
                               x1ta[i,j,k] * (C[9,i1[i,j,k],i2[i,j,k]] + x1ta[i,j,k] * (C[10,i1[i,j,k],i2[i,j,k]] +
                                x1ta[i,j,k] * C[11,i1[i,j,k],i2[i,j,k]])) + 1.5 * x2ta[i,j,k] *
                                (C[12,i1[i,j,k],i2[i,j,k]] + x1ta[i,j,k] * (C[13,i1[i,j,k],i2[i,j,k]] + x1ta[i,j,k] *
                                (C[14,i1[i,j,k],i2[i,j,k]] + x1ta[i,j,k] * C[15,i1[i,j,k],i2[i,j,k]])))))
    return T, dTde


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef Tall4D(np.ndarray[DTYPEf64_t, ndim=4] rho, np.ndarray[DTYPEf64_t, ndim=4] ei, np.ndarray[DTYPEf64_t, ndim=3] C,
             np.ndarray[DTYPE_t, ndim=4] i1, np.ndarray[DTYPE_t, ndim=4] i2, np.ndarray[DTYPEf64_t, ndim=4] x1ta,
             np.ndarray[DTYPEf64_t, ndim=4] x2ta, DTYPEf64_t x2shift):
    cdef unsigned int i, j, k, l
    cdef unsigned int nt = rho.shape[0]
    cdef unsigned int nx = rho.shape[1]
    cdef unsigned int ny = rho.shape[2]
    cdef unsigned int nz = rho.shape[3]
    cdef np.ndarray[DTYPEf64_t, ndim=4] T = np.empty((nx, ny, nz), dtype=DTYPEf64)
    cdef np.ndarray[DTYPEf64_t, ndim=4] dTde = np.empty((nx, ny, nz), dtype=DTYPEf64)

    for i in range(nt):
        for j in prange(nx):
            for k in range(ny):
                for l in range(nz):
                    T[i,j,k,l] = exp(C[0,i1[i,j,k,l], i2[i,j,k,l]] + x1ta[i,j,k,l] * (C[1,i1[i,j,k,l],i2[i,j,k,l]] +
                        x1ta[i,j,k,l] * (C[2,i1[i,j,k,l],i2[i,j,k,l]] + x1ta[i,j,k,l] * C[3,i1[i,j,k,l],i2[i,j,k,l]])) +\
                      x2ta[i,j,k,l] * (C[4,i1[i,j,k,l],i2[i,j,k,l]] + x1ta[i,j,k,l] * (C[5,i1[i,j,k,l],i2[i,j,k,l]] +
                        x1ta[i,j,k,l] * (C[6,i1[i,j,k,l],i2[i,j,k,l]] + x1ta[i,j,k,l] * C[7,i1[i,j,k,l],i2[i,j,k,l]])) +
                      x2ta[i,j,k,l] * (C[8,i1[i,j,k,l],i2[i,j,k,l]] + x1ta[i,j,k,l] * (C[9,i1[i,j,k,l],i2[i,j,k,l]] +
                        x1ta[i,j,k,l] * (C[10,i1[i,j,k,l],i2[i,j,k,l]] + x1ta[i,j,k,l] * C[11,i1[i,j,k,l], i2[i,j,k,l]])) +
                      x2ta[i,j,k,l] * (C[12, i1[i,j,k,l], i2[i,j,k,l]] + x1ta[i,j,k,l] * (C[13, i1[i,j,k,l], i2[i,j,k,l]] +
                        x1ta[i,j,k,l] * (C[14, i1[i,j,k,l], i2[i,j,k,l]] + x1ta[i,j,k,l] * C[15, i1[i,j,k,l], i2[i,j,k,l]]))))))

                    dTde[i,j,k,l] = T[i,j,k,l] / (ei[i,j,k,l] + x2shift) * (C[4,i1[i,j,k,l],i2[i,j,k,l]] + x1ta[i,j,k,l] *
                                    (C[5,i1[i,j,k,l],i2[i,j,k,l]] + x1ta[i,j,k,l] * (C[6,i1[i,j,k,l],i2[i,j,k,l]] + x1ta[i,j,k,l] *
                                     C[7,i1[i,j,k,l],i2[i,j,k,l]])) + 2 * x2ta[i,j,k,l] * (C[8,i1[i,j,k,l],i2[i,j,k,l]] +
                                   x1ta[i,j,k,l] * (C[9,i1[i,j,k,l],i2[i,j,k,l]] + x1ta[i,j,k,l] * (C[10,i1[i,j,k,l],i2[i,j,k,l]] +
                                    x1ta[i,j,k,l] * C[11,i1[i,j,k,l],i2[i,j,k,l]])) + 1.5 * x2ta[i,j,k,l] *
                                    (C[12,i1[i,j,k,l],i2[i,j,k,l]] + x1ta[i,j,k,l] * (C[13,i1[i,j,k,l],i2[i,j,k,l]] + x1ta[i,j,k,l] *
                                    (C[14,i1[i,j,k,l],i2[i,j,k,l]] + x1ta[i,j,k,l] * C[15,i1[i,j,k,l],i2[i,j,k,l]])))))
    return T, dTde


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef logPT2kappa3D(np.ndarray[DTYPEf64_t, ndim=3] logP, np.ndarray[DTYPEf64_t, ndim=3] T,
                    np.ndarray[DTYPEf64_t, ndim=1] tabP, np.ndarray[DTYPEf64_t, ndim=1] tabT,
                    np.ndarray[DTYPEf64_t, ndim=3] tabKap, np.ndarray[DTYPEf64_t, ndim=1] tabTBN,
                    np.ndarray[DTYPEf64_t, ndim=1] tabDTB, np.ndarray[DTYPE_t, ndim=1] idxTBN,
                    np.ndarray[DTYPEf64_t, ndim=1] tabPBN, np.ndarray[DTYPEf64_t, ndim=1] tabDPB,
                    np.ndarray[DTYPE_t, ndim=1] idxPBN, int iband):
    cdef unsigned int i, j, k, iTx, iPx, nTx, nPx
    cdef unsigned int nx = T.shape[0]
    cdef unsigned int ny = T.shape[1]
    cdef unsigned int nz = T.shape[2]
    cdef unsigned int NT = tabT.size
    cdef unsigned int NP = tabP.size
    cdef DTYPEf64_t aln10 = 2.30258509299405
    cdef DTYPEf64_t Tx, Px, gT1, gT2, gT3, gT4, gP1, gP2, gP3, gP4, fP1, fP2, fP3, fP4
    cdef np.ndarray[DTYPEf64_t, ndim=3] xkaros = np.empty((nx, ny, nz), dtype=DTYPEf64)
    idxTBN -= 1
    idxPBN -= 1

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                Tx = float_min(tabT[NT-2], float_max(tabT[1], log10(T[i,j,k])))
                Px = float_min(tabP[NP-2], float_max(tabP[1], 0.434294481903 * logP[i,j,k]))

                if Tx < tabTBN[1]:
                    nTx = int((Tx - tabTBN[0]) / tabDTB[0]) + idxTBN[0]
                elif Tx < tabTBN[2]:
                    nTx = int((Tx - tabTBN[1]) / tabDTB[1]) + idxTBN[1]
                elif Tx < tabTBN[3]:
                    nTx = int((Tx - tabTBN[2]) / tabDTB[2]) + idxTBN[2]
                else:
                    nTx = NT - 1
                iTx = int_min(int_max(1, nTx), NT - 1)

                if Px < tabPBN[1]:
                    nPx = int((Px - tabPBN[0]) / tabDPB[0]) + idxPBN[0]
                elif Px < tabPBN[2]:
                    nPx=int((Px - tabPBN[1]) / tabDPB[1]) + idxPBN[1]
                elif Px < tabPBN[3]:
                    nPx=int((Px - tabPBN[2]) / tabDPB[2]) + idxPBN[2]
                else:
                    nPx = NP - 1
                iPx = int_min(int_max(1, nPx), NP - 1)

                gT1 =   (Tx            - tabT[iTx]    ) * (Tx            - tabT[iTx + 1]) * (Tx        - tabT[iTx + 1]) /\
                      ( (tabT[iTx-1]   - tabT[iTx]    ) * (tabT[iTx - 1] - tabT[iTx + 1]) * (tabT[iTx] - tabT[iTx + 1]))
                gT2 =   (Tx            - tabT[iTx + 1]) * \
                      (-(tabT[iTx - 1] - tabT[iTx + 2]) * (Tx            - tabT[iTx]    ) * (Tx        - tabT[iTx]    ) -
                        (tabT[iTx]     - tabT[iTx + 1]) * (tabT[iTx]     - tabT[iTx + 2]) * (Tx        - tabT[iTx - 1])) /\
                      ( (tabT[iTx - 1] - tabT[iTx]    ) * (tabT[iTx]     - tabT[iTx + 1]) * (tabT[iTx] - tabT[iTx + 1]) *
                        (tabT[iTx]     - tabT[iTx + 2]))
                gT3 =   (Tx            - tabT[iTx]    ) * \
                      ( (tabT[iTx - 1] - tabT[iTx + 2]) * (Tx        - tabT[iTx + 1]) * (Tx        - tabT[iTx + 1]) -
                        (tabT[iTx - 1] - tabT[iTx + 1]) * (tabT[iTx] - tabT[iTx + 1]) * (Tx        - tabT[iTx + 2])) /\
                      ( (tabT[iTx - 1] - tabT[iTx + 1]) * (tabT[iTx] - tabT[iTx + 1]) * (tabT[iTx] - tabT[iTx + 1]) *
                        (tabT[iTx + 1] - tabT[iTx + 2]))
                gT4 =  -(Tx            - tabT[iTx]    ) * (Tx        - tabT[iTx]    ) * (Tx        - tabT[iTx + 1]) /\
                      ( (tabT[iTx]     - tabT[iTx + 1]) * (tabT[iTx] - tabT[iTx + 2]) * (tabT[iTx + 1] - tabT[iTx + 2]))

                gP1 = (Px - tabP[iPx]) * (Px - tabP[iPx+1]) * (Px - tabP[iPx+1]) / ((tabP[iPx-1] - tabP[iPx]) *
                        (tabP[iPx-1] - tabP[iPx+1]) * (tabP[iPx] - tabP[iPx + 1]))
                gP2 = (Px - tabP[iPx + 1]) * (-(tabP[iPx - 1] - tabP[iPx + 2]) * (Px - tabP[iPx]) * (Px - tabP[iPx]) -
                        (tabP[iPx] - tabP[iPx + 1]) * (tabP[iPx] - tabP[iPx + 2]) * (Px - tabP[iPx - 1])) / (
                        (tabP[iPx - 1] - tabP[iPx]) * (tabP[iPx] - tabP[iPx + 1]) * (tabP[iPx] - tabP[iPx + 1]) *
                        (tabP[iPx] - tabP[iPx + 2]))
                gP3 = (Px - tabP[iPx]) * ((tabP[iPx - 1] - tabP[iPx + 2]) * (Px - tabP[iPx + 1]) * (Px - tabP[iPx + 1]) -
                        (tabP[iPx - 1] - tabP[iPx + 1]) * (tabP[iPx] - tabP[iPx + 1]) * (Px - tabP[iPx + 2])) / (
                        (tabP[iPx - 1] - tabP[iPx + 1]) * (tabP[iPx] - tabP[iPx + 1]) * (tabP[iPx] - tabP[iPx + 1]) *
                        (tabP[iPx + 1] - tabP[iPx + 2]))
                gP4 = -(Px - tabP[iPx]) * (Px - tabP[iPx]) * (Px - tabP[iPx + 1]) / ((tabP[iPx] - tabP[iPx + 1]) *
                        (tabP[iPx] - tabP[iPx + 2]) * (tabP[iPx + 1] - tabP[iPx + 2]))

                fP1 = tabKap[iTx - 1, iPx - 1, iband] * gT1 + tabKap[iTx, iPx - 1, iband] * gT2 + \
                      tabKap[iTx + 1, iPx - 1, iband] * gT3 + tabKap[iTx + 2, iPx - 1, iband] * gT4
                fP2 = tabKap[iTx - 1, iPx, iband] * gT1 + tabKap[iTx, iPx, iband] * gT2 + \
                      tabKap[iTx + 1, iPx, iband] * gT3 + tabKap[iTx + 2, iPx, iband] * gT4
                fP3 = tabKap[iTx - 1, iPx + 1, iband] * gT1 + tabKap[iTx, iPx + 1, iband] * gT2 + \
                      tabKap[iTx + 1, iPx + 1, iband] * gT3 + tabKap[iTx + 2, iPx + 1, iband] * gT4
                fP4 = tabKap[iTx - 1, iPx + 2, iband] * gT1 + tabKap[iTx, iPx + 2, iband] * gT2 + \
                      tabKap[iTx + 1, iPx + 2, iband] * gT3 + tabKap[iTx + 2, iPx + 2, iband] * gT4

                xkaros[i, j, k] = exp(aln10 * (fP1 * gP1 + fP2 * gP2 + fP3 * gP3 + fP4 * gP4))
    return xkaros


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef logPT2kappa4D(np.ndarray[DTYPEf64_t, ndim=4] logP, np.ndarray[DTYPEf64_t, ndim=4] T,
                    np.ndarray[DTYPEf64_t, ndim=1] tabP, np.ndarray[DTYPEf64_t, ndim=1] tabT,
                    np.ndarray[DTYPEf64_t, ndim=3] tabKap, np.ndarray[DTYPEf64_t, ndim=1] tabTBN,
                    np.ndarray[DTYPEf64_t, ndim=1] tabDTB, np.ndarray[DTYPE_t, ndim=1] idxTBN,
                    np.ndarray[DTYPEf64_t, ndim=1] tabPBN, np.ndarray[DTYPEf64_t, ndim=1] tabDPB,
                    np.ndarray[DTYPE_t, ndim=1] idxPBN, int iband):
    cdef unsigned int i, j, k, l, iTx, iPx, nTx, nPx
    cdef unsigned int nt = T.shape[0]
    cdef unsigned int nx = T.shape[1]
    cdef unsigned int ny = T.shape[2]
    cdef unsigned int nz = T.shape[2]
    cdef unsigned int NT = tabT.size
    cdef unsigned int NP = tabP.size
    cdef DTYPEf64_t aln10 = 2.30258509299405
    cdef DTYPEf64_t Tx, Px, gT1, gT2, gT3, gT4, gP1, gP2, gP3, gP4, fP1, fP2, fP3, fP4
    cdef np.ndarray[DTYPEf64_t, ndim=4] xkaros = np.empty((nt, nx, ny, nz), dtype=DTYPEf64)
    idxTBN -= 1
    idxPBN -= 1

    for i in range(nt):
        for j in range(nx):
            for k in range(ny):
                for l in range(nz):
                    Tx = float_min(tabT[NT-2], float_max(tabT[1], log10(T[i,j,k,l])))
                    Px = float_min(tabP[NP-2], float_max(tabP[1], 0.434294481903 * logP[i,j,k,l]))

                    if Tx < tabTBN[1]:
                        nTx = int((Tx - tabTBN[0]) / tabDTB[0]) + idxTBN[0]
                    elif Tx < tabTBN[2]:
                        nTx = int((Tx - tabTBN[1]) / tabDTB[1]) + idxTBN[1]
                    elif Tx < tabTBN[3]:
                        nTx = int((Tx - tabTBN[2]) / tabDTB[2]) + idxTBN[2]
                    else:
                        nTx = NT - 1
                    iTx = int_min(int_max(1, nTx), NT - 1)

                    if Px < tabPBN[1]:
                        nPx = int((Px - tabPBN[0]) / tabDPB[0]) + idxPBN[0]
                    elif Px < tabPBN[2]:
                        nPx=int((Px - tabPBN[1]) / tabDPB[1]) + idxPBN[1]
                    elif Px < tabPBN[3]:
                        nPx=int((Px - tabPBN[2]) / tabDPB[2]) + idxPBN[2]
                    else:
                        nPx = NP - 1
                    iPx = int_min(int_max(1, nPx), NP - 1)

                    gT1 =   (Tx            - tabT[iTx]    ) * (Tx            - tabT[iTx + 1]) * (Tx        - tabT[iTx + 1]) /\
                          ( (tabT[iTx-1]   - tabT[iTx]    ) * (tabT[iTx - 1] - tabT[iTx + 1]) * (tabT[iTx] - tabT[iTx + 1]))
                    gT2 =   (Tx            - tabT[iTx + 1]) * \
                          (-(tabT[iTx - 1] - tabT[iTx + 2]) * (Tx            - tabT[iTx]    ) * (Tx        - tabT[iTx]    ) -
                            (tabT[iTx]     - tabT[iTx + 1]) * (tabT[iTx]     - tabT[iTx + 2]) * (Tx        - tabT[iTx - 1])) /\
                          ( (tabT[iTx - 1] - tabT[iTx]    ) * (tabT[iTx]     - tabT[iTx + 1]) * (tabT[iTx] - tabT[iTx + 1]) *
                            (tabT[iTx]     - tabT[iTx + 2]))
                    gT3 =   (Tx            - tabT[iTx]    ) * \
                          ( (tabT[iTx - 1] - tabT[iTx + 2]) * (Tx        - tabT[iTx + 1]) * (Tx        - tabT[iTx + 1]) -
                            (tabT[iTx - 1] - tabT[iTx + 1]) * (tabT[iTx] - tabT[iTx + 1]) * (Tx        - tabT[iTx + 2])) /\
                          ( (tabT[iTx - 1] - tabT[iTx + 1]) * (tabT[iTx] - tabT[iTx + 1]) * (tabT[iTx] - tabT[iTx + 1]) *
                            (tabT[iTx + 1] - tabT[iTx + 2]))
                    gT4 =  -(Tx            - tabT[iTx]    ) * (Tx        - tabT[iTx]    ) * (Tx        - tabT[iTx + 1]) /\
                          ( (tabT[iTx]     - tabT[iTx + 1]) * (tabT[iTx] - tabT[iTx + 2]) * (tabT[iTx + 1] - tabT[iTx + 2]))

                    gP1 = (Px - tabP[iPx]) * (Px - tabP[iPx+1]) * (Px - tabP[iPx+1]) / ((tabP[iPx-1] - tabP[iPx]) *
                            (tabP[iPx-1] - tabP[iPx+1]) * (tabP[iPx] - tabP[iPx + 1]))
                    gP2 = (Px - tabP[iPx + 1]) * (-(tabP[iPx - 1] - tabP[iPx + 2]) * (Px - tabP[iPx]) * (Px - tabP[iPx]) -
                            (tabP[iPx] - tabP[iPx + 1]) * (tabP[iPx] - tabP[iPx + 2]) * (Px - tabP[iPx - 1])) / (
                            (tabP[iPx - 1] - tabP[iPx]) * (tabP[iPx] - tabP[iPx + 1]) * (tabP[iPx] - tabP[iPx + 1]) *
                            (tabP[iPx] - tabP[iPx + 2]))
                    gP3 = (Px - tabP[iPx]) * ((tabP[iPx - 1] - tabP[iPx + 2]) * (Px - tabP[iPx + 1]) * (Px - tabP[iPx + 1]) -
                            (tabP[iPx - 1] - tabP[iPx + 1]) * (tabP[iPx] - tabP[iPx + 1]) * (Px - tabP[iPx + 2])) / (
                            (tabP[iPx - 1] - tabP[iPx + 1]) * (tabP[iPx] - tabP[iPx + 1]) * (tabP[iPx] - tabP[iPx + 1]) *
                            (tabP[iPx + 1] - tabP[iPx + 2]))
                    gP4 = -(Px - tabP[iPx]) * (Px - tabP[iPx]) * (Px - tabP[iPx + 1]) / ((tabP[iPx] - tabP[iPx + 1]) *
                            (tabP[iPx] - tabP[iPx + 2]) * (tabP[iPx + 1] - tabP[iPx + 2]))

                    fP1 = tabKap[iTx - 1, iPx - 1, iband] * gT1 + tabKap[iTx, iPx - 1, iband] * gT2 + \
                          tabKap[iTx + 1, iPx - 1, iband] * gT3 + tabKap[iTx + 2, iPx - 1, iband] * gT4
                    fP2 = tabKap[iTx - 1, iPx, iband] * gT1 + tabKap[iTx, iPx, iband] * gT2 + \
                          tabKap[iTx + 1, iPx, iband] * gT3 + tabKap[iTx + 2, iPx, iband] * gT4
                    fP3 = tabKap[iTx - 1, iPx + 1, iband] * gT1 + tabKap[iTx, iPx + 1, iband] * gT2 + \
                          tabKap[iTx + 1, iPx + 1, iband] * gT3 + tabKap[iTx + 2, iPx + 1, iband] * gT4
                    fP4 = tabKap[iTx - 1, iPx + 2, iband] * gT1 + tabKap[iTx, iPx + 2, iband] * gT2 + \
                          tabKap[iTx + 1, iPx + 2, iband] * gT3 + tabKap[iTx + 2, iPx + 2, iband] * gT4

                    xkaros[i, j, k, l] = exp(aln10 * (fP1 * gP1 + fP2 * gP2 + fP3 * gP3 + fP4 * gP4))
    return xkaros



@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tau3D(np.ndarray[DTYPEf64_t, ndim=3] kaprho, np.ndarray[DTYPEf64_t, ndim=1] dz, DTYPE_t radHtautop):
    cdef unsigned int i, j, k, kt
    cdef unsigned int nx = kaprho.shape[0]
    cdef unsigned int ny = kaprho.shape[1]
    cdef unsigned int nz = kaprho.shape[2]
    cdef unsigned int k1 = nz - 1
    cdef unsigned int k2 = nz - 2
    cdef unsigned int k3 = nz - 3
    cdef DTYPEf64_t s3, s4, s5
    cdef np.ndarray[DTYPEf64_t, ndim=1] dkds = np.empty((nz), dtype=DTYPEf64)
    cdef np.ndarray[DTYPEf64_t, ndim=1] dxds = np.empty((nz), dtype=DTYPEf64)
    cdef np.ndarray[DTYPEf64_t, ndim=3] tau = np.empty((nx, ny, nz), dtype=DTYPEf64)

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
                dxds[k] = 2.0 * (float_max(s4, 0.0 ) + float_min(s5, 0.0))

            dxds[0] = 1.5 * dkds[0] - 0.5 * dxds[1]

            for k in range(k2, -1, -1):
                kt = k + 1
                tau[i, j, k] = tau[i, j, kt] + dz[k] * (0.5 * (kaprho[i, j, kt] + kaprho[i, j, k]) +
                                                        dz[k] * (dxds[kt] - dxds[k]) / 12.0)
    return tau


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tau4D(np.ndarray[DTYPEf64_t, ndim=4] kaprho, np.ndarray[DTYPEf64_t, ndim=1] dz, DTYPE_t radHtautop):
    cdef unsigned int i, j, k, l, lt
    cdef unsigned int nt = kaprho.shape[0]
    cdef unsigned int nx = kaprho.shape[1]
    cdef unsigned int ny = kaprho.shape[2]
    cdef unsigned int nz = kaprho.shape[3]
    cdef unsigned int l1 = nz - 1
    cdef unsigned int l2 = nz - 2
    cdef unsigned int l3 = nz - 3
    cdef DTYPEf64_t s3, s4, s5
    cdef np.ndarray[DTYPEf64_t, ndim=1] dkds = np.empty((nz), dtype=DTYPEf64)
    cdef np.ndarray[DTYPEf64_t, ndim=1] dxds = np.empty((nz), dtype=DTYPEf64)
    cdef np.ndarray[DTYPEf64_t, ndim=4] tau = np.empty((nt, nx, ny, nz), dtype=DTYPEf64)

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
                    dxds[l] = 2.0 * (float_max(s4, 0.0 ) + float_min(s5, 0.0))

                dxds[0] = 1.5 * dkds[0] - 0.5 * dxds[1]

                for l in range(l2, -1, -1):
                    lt = l + 1
                    tau[i, j, k, l] = tau[i, j, k, lt] + dz[l] * (0.5 * (kaprho[i, j, k, lt] + kaprho[i, j, k, l]) +
                                                                  dz[l] * (dxds[lt] - dxds[l]) / 12.0)
    return tau

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 06 12:09:39 2014

@author: salhab
"""

from scipy import integrate as integ
from scipy import interpolate as ip
import numpy as np
import numexpr as ne
import re

try:
    import eosinterx as eosx
    eosx_available = True
except:
    eosx_available = False
print("eosx available (opta):", eosx_available)

class Opac:
    def __init__(self, fname):
        """
            Description
            -----------
                Opens an .opa-file given by :param fname.
            Input
            -----
                :param fname: string, path and name of file with opacity table.
        """

        hsep = "\*{2,}"
        header = []
        tabKap = []
        with open(fname, "r") as self.f:
            for self.line in self.f:
                if re.match(hsep, self.line):
                    # read header
                    for self.line in self.f:
                        if re.match(hsep, self.line):
                            break
                        header.append(self.line)
                if "NT\n" in self.line:
                    dimT = self._fileiter(True)
                if "NP\n" in self.line:
                    dimP = self._fileiter(True)
                if " NBAND\n" in self.line:
                    dimBAND = self._fileiter(True) + 1
                if "TABT\n" in self.line:
                    temptb = self._fileiter()
                if "TABTBN" in self.line:
                    tabtbn = self._fileiter()
                if "IDXTBN" in self.line:
                    idxtbn = self._fileiter()
                if "TABDTB" in self.line:
                    tabdtb = self._fileiter()

                if "TABP\n" in self.line:
                    presstb = self._fileiter()
                if "TABPBN" in self.line:
                    tabpbn = self._fileiter()
                if "IDXPBN" in self.line:
                    idxpbn = self._fileiter()
                if "TABDPB" in self.line:
                    tabdpb = self._fileiter()
                if "log10 P" in self.line:
                    next(self.f)
                    tabKap = self._fileiter()

        self.header = header

        self.tabT = np.array([item for sublist in temptb for item in sublist if item != '']).astype(np.float32)
        self.tabTBN = np.array([item for sublist in tabtbn for item in sublist if item != '']).astype(np.float32)
        self.idxTBN = np.array([item for sublist in idxtbn for item in sublist if item != '']).astype(np.int32)
        self.tabDTB = np.array([item for sublist in tabdtb for item in sublist if item != '']).astype(np.float32)

        self.tabP = np.array([item for sublist in presstb for item in sublist if item != '']).astype(np.float32)
        self.tabPBN = np.array([item for sublist in tabpbn for item in sublist if item != '']).astype(np.float32)
        self.idxPBN = np.array([item for sublist in idxpbn for item in sublist if item != '']).astype(np.int32)
        self.tabDPB = np.array([item for sublist in tabdpb for item in sublist if item != '']).astype(np.float32)

        tabKap = np.array([a.strip() for sublist in tabKap for a in sublist if a != '']).astype(np.float32)
        tabKap = tabKap.reshape(dimP, dimBAND, dimT)
        self.tabKap = np.transpose(tabKap, axes=(2, 0, 1))

    def _fileiter(self, conv=False):
        val = []
        for self.line in self.f:
            if "log10 P" in self.line:
                next(self.f)
                continue
            elif "*\n" in self.line:
                continue
            elif "*" in self.line:
                break
            if conv:
                return np.int16(np.char.strip(self.line))
            val.append(re.split("  | |\n", self.line))

        return np.array(val)

    def _get_ind(self, axis, dim, start=None, stop=None, step=None, cut=None, expand=False):
        if expand:
            ind = [None for _ in range(dim)]
            start = stop = step = cut = None
        else:
            ind = [slice(None) for _ in range(dim)]
        if cut is None:
            ind[axis] = slice(start, stop, step)
        else:
            ind[axis] = cut
        return tuple(ind)


    def kappa(self, T, P, iBand=0):
        """
            Description
            -----------
                Computes opacity.

            Notes
            -----
                When opta is imported, it checks, if the compiled functions for computing opacity and optical depth are
                available. The availability is printed into the console ("eosx is available: True/False").

                If they are available, the computation might be much faster and less memory-consuming.

                If they are not available, numpy functions will be used. The output will ``not`` be extactly like the
                values internally computed in CO5BOLD!

            Input
            -----
                :param T: ndarray, temperature. Must be of same shape as P (pressure).
                :param P: ndarray, pressure. Must be of same shape as T (temperature).
                :param iBand: int, optional, opacity-band. Default: 0

            Output
            ------
                :return: ndarray, opacity (kappa) of desired band.

            Example
            -------
                    >>> from opta import Opac
                    >>>
                    >>> opan = r"directory/to/opacity-file/file.opta"
                    >>> opa = Opac(opan)
                    >>> [...]                       # rho, z, T and P are prepared
                    >>> kappa = opa.kappa(T, P[, iBand=1])
                """
        if eosx_available:
            log10P = ne.evaluate("log10(P)")
            del P
            log10T = ne.evaluate("log10(T)")
            del T
            if log10T.ndim == 3:
                return ne.evaluate("10**kap",
                                   local_dict={'kap': eosx.logPT2kappa3D(log10P, log10T, self.tabP, self.tabT,
                                                                         self.tabKap, self.tabTBN, self.tabDTB,
                                                                         self.idxTBN, self.tabPBN, self.tabDPB,
                                                                         self.idxPBN, iBand)})
            elif log10T.ndim == 4:
                return ne.evaluate("10**kap",
                                   local_dict={'kap': eosx.logPT2kappa4D(log10P, log10T, self.tabP, self.tabT,
                                                                         self.tabKap, self.tabTBN, self.tabDTB,
                                                                         self.idxTBN, self.tabPBN, self.tabDPB,
                                                                         self.idxPBN, iBand)})
        logP = ne.evaluate("log10(P)")
        logT = ne.evaluate("log10(T)")
        kap = ip.RectBivariateSpline(self.tabT, self.tabP, self.tabKap[:, :, iBand], kx=2, ky=2).ev(logT, logP)

        return ne.evaluate("10**kap")

    def tau(self, rho, z, axis=-1, mode=0, **kwargs):
        """
            Description
            -----------
                Computes optical depth. Either opacity (kappa), or temperature and pressure have to be provided. If
                temperature and pressure are provided, the opacity is computed first.

            Notes
            -----
                When opta is imported, it checks, if the compiled functions for computing opacity and optical depth are
                available. The availability is printed into the console ("eosx is available: True/False").

                If they are available, the computation might be much faster and less memory-consuming.

                If they are not available, numpy functions will be used. The output will ``not`` be extactly like the
                values internally computed in CO5BOLD!

            Input
            -----
                :param rho: ndarray, mass-density
                :param z: 1D ndarray, positions of desired values. Has to have the same length like the axis of rho and
                          the other provided values that have to be integrated (rho and kappa, or T and P).
                :param axis: int, optional, axis along the integration will take place. Default: -1
                :param mode: int, optional, if mode=0: cubic integration, else linear integration (see CAT). Default: 0
                :param kwargs:
                    :param kappa: ndarray, opacity. If provided, kappa times rho will be integrated directly.
                    :param T: ndarray, temperature. If kappa is not provided, but T and P, the opacity will be computed
                              first. Will be ignored, if kappa is provided.
                    :param P: ndarray, pressure. If kappa is not provided, but T and P, the opacity will be computed
                              first. Will be ignored, if kappa is provided.
                    :param iBand: int, optional, opacity-band. If kappa is not provided, but T and P, the opacity will
                                  be computed first. Will be ignored, if kappa is provided. Default: 0
                    :param zb: 1D ndarray, optional, boundary-centered z-positions. If not provided, cell-heights will
                               be computed with z.

            Output
            ------
                :return: ndarray, optical depth (tau) of desired band.

            Example
            -------
                If kappa is not pre-computed use:

                    >>> from opta import Opac
                    >>>
                    >>> opan = r"directory/to/opacity-file/file.opta"
                    >>> opa = Opac(opan)
                    >>> [...]                       # rho, z, T and P are prepared
                    >>> tau = opa.tau(rho, z[, axis=1[, iBand=1]], T=T, P=P)   # computes kappa and tau. Returns tau

                If you want to pre-compute kappa use:

                    >>> from opta import Opac
                    >>>
                    >>> opan = r"directory/to/opacity-file/file.opta"
                    >>> opa = Opac(opan)
                    >>> [...]                       # rho, z, T and P are prepared
                    >>> kappa = opa.kappa(T, P[, iBand=1])
                    >>> tau = opa.tau(rho, z[, axis=1[, iBand=1]], kappa=kappa)   # computes and returns tau.
        """
        if 'kappa' in kwargs:
            kaprho = ne.evaluate("kappa*rho", local_dict={'rho': rho, 'kappa': kwargs['kappa']})
        elif 'T' in kwargs and 'P' in kwargs:
            if 'iBand' in kwargs:
                kappa = self.kappa(kwargs['T'], kwargs['P'], kwargs['iBand'])
            else:
                kappa = self.kappa(kwargs['T'], kwargs['P'])
            kaprho = ne.evaluate("kappa * rho")
            del kappa
        else:
            raise ValueError("Either the keyword-argument 'kappa', or 'T' (temperature) and 'P' (pressure) have to be "
                             "provided")

        if 'zb' in kwargs:
            dz = np.diff(kwargs['zb'])
        else:
            dz = np.diff(z)
            dz = np.append(dz, dz[-1])

        if 'radHtautop' in kwargs:
            radHtautop = np.float32(kwargs['radHtautop'])
        else:
            radHtautop = np.float32(-1.0)

        dim = rho.ndim
        if mode == 0:
            if eosx_available:
                trans = list(range(dim))
                trans[-1], trans[axis] = trans[axis], trans[-1]
                kaprho = np.transpose(kaprho, axes=trans)
                dz = dz

                if dim == 3:
                    return np.transpose(eosx.tau3D(kaprho, dz, radHtautop), axes=trans)
                elif dim == 4:
                    return np.transpose(eosx.tau4D(kaprho, dz, radHtautop), axes=trans)
            kaprho = np.transpose(kaprho, axes=trans)
            ind3 = self._get_ind(axis, dim, cut=-1)
            dz = dz[self._get_ind(axis, dim, expand=True)]
            tau = np.empty(rho.shape)
            dxds = np.empty(rho.shape)
            dkds = np.empty(rho.shape)
            tau[ind3] = kaprho[ind3] * radHtautop

            # Top
            ind = self._get_ind(axis, dim, start=1)
            ind2 = self._get_ind(axis, dim, stop=-1)
            dkds[ind2] = -(kaprho[ind] - kaprho[ind2]) / dz[ind]

            ind = self._get_ind(axis, dim, cut=-2)
            ind2 = self._get_ind(axis, dim, cut=-3)
            s3 = (dkds[ind] * dz[ind2] + dkds[ind2] * dz[ind]) / (2 * (dz[ind] + dz[ind2]))
            s4 = np.minimum(s3, dkds[ind], dkds[ind2])
            s5 = np.maximum(s3, dkds[ind], dkds[ind2])
            dxds[ind3] = 1.5 * dkds[ind] - (np.select([s4 <= 0, s4 > 0], [0, s4]) + np.select([s5 <= 0, s5 > 0], [s5, 0]))

            # Interior
            ind = self._get_ind(axis, dim, start=-2, stop=0, step=-1)
            ind2 = self._get_ind(axis, dim, start=-3, step=-1)
            s3 = (dkds[ind] * dz[ind2] + dkds[ind2] * dz[ind]) / (2 * (dz[ind] + dz[ind2]))
            s4 = np.minimum(s3, dkds[ind], dkds[ind2])
            s5 = np.maximum(s3, dkds[ind], dkds[ind2])
            dxds[ind] = 2.0 * (np.select([s4 <= 0, s4 > 0], [0, s4]) + np.select([s5 <= 0, s5 > 0], [s5, 0]))

            # Bottom
            ind = self._get_ind(axis, dim, cut=0)
            ind2 = self._get_ind(axis, dim, cut=1)
            dxds[ind] = 1.5 * dkds[ind] - 0.5 * dxds[ind2]

            ind = self._get_ind(axis, dim, start=-2, step=-1)
            ind2 = self._get_ind(axis, dim, stop=0, step=-1)
            tau[ind] = tau[ind2] + dz[ind] * (0.5 * (kaprho[ind2] + kaprho[ind]) + dz[ind] * (dxds[ind2] - dxds[ind]) /
                                                                                              12.0)
            return tau
        else:
            init = radHtautop * kaprho[-1].min()
            ind = self._get_ind(axis, dim, step=-1)
            return integ.cumtrapz(kaprho[ind], z, axis=axis, initial=init)[ind]

    def height(self, z, value=1.0, axis=-1, **kwargs):
        """
            Description
            -----------
                Computes the geometrical height of optical depth. Either optical depth, opacity (kappa) and density, or
                temperature and pressure have to be provided. If kappa is provided, optical depth will be computed
                first. If temperature and pressure are provided, the opacity is computed first and then optical depth.

            Notes
            -----
                When opta is imported, it checks, if the compiled functions for computing opacity and optical depth are
                available. The availability is printed into the console ("eosx is available: True/False").

                If they are available, the computation might be much faster and less memory-consuming.

                If they are not available, numpy functions will be used. The output will ``not`` be extactly like the
                values internally computed in CO5BOLD!

            Input
            -----
                :param z: 1D ndarray, positions of desired values. Has to have the same length like the axis of rho and
                          the other provided values that have to be integrated (optical depth, rho and kappa, or T and
                          P).
                :param value: float, or 1D-ndarray, value(s) of tau at which the geometrical height is/are to be computed.
                              Default: 1.0
                :param axis: int, optional, axis along the root-seeking will take place. Default: -1
                :param kwargs:
                    :param tau: ndarray, optical depth. If provided, the height will be computed directly.
                    :param rho: ndarray, mass density. If provided, along with kappa, optical depth will be integrated
                                first. Will be ignored, if tau is provided.
                    :param kappa: ndarray, opacity. If provided, along with rho, optical depth will be integrated first.
                                  Will be ignored, if tau is provided.
                    :param T: ndarray, temperature. If provided, along with rho and P, kappa and optical depth will be
                              computed first. Will be ignored, if tau, or kappa and rho are provided.
                    :param P: ndarray, pressure. If provided, along with rho and T, kappa and optical depth will be
                              computed first. Will be ignored, if tau, or kappa and rho are provided.
                    :param iBand: int, optional, opacity-band. If T, rho, and P, or kappa and rho are provdided, kappa
                                  and optical depth will be computed first. Will be ignored, if tau is provided.
                                  Default: 0
                    :param zb: 1D ndarray, optional, boundary-centered z-positions. If not provided, cell-heights will
                               be computed with z.

            Output
            ------
                :return: ndarray, optical depth (tau) of desired band.

            Example
            -------
                If kappa is not pre-computed use:

                    >>> from opta import Opac
                    >>>
                    >>> opan = r"directory/to/opacity-file/file.opta"
                    >>> opa = Opac(opan)
                    >>> [...]                       # rho, z, T and P are prepared
                    >>> tau = opa.tau(rho, z[, axis=1[, iBand=1]], T=T, P=P)   # computes kappa and tau. Returns tau

                If you want to pre-compute kappa use:

                    >>> from opta import Opac
                    >>>
                    >>> opan = r"directory/to/opacity-file/file.opta"
                    >>> opa = Opac(opan)
                    >>> [...]                       # rho, z, T and P are prepared
                    >>> kappa = opa.kappa(T, P[, iBand=1])
                    >>> tau = opa.tau(rho, z[, axis=1[, iBand=1]], kappa=kappa)   # computes and returns tau.
        """
        if not eosx_available:
            raise ImportError("Compiled functions of eosx not found!")

        if 'tau' in kwargs:
            dim = kwargs['tau'].ndim
            trans = list(range(dim))
            trans[-1], trans[axis] = trans[axis], trans[-1]
            kwargs['tau'] = np.transpose(kwargs['tau'], axes=trans)
            if dim == 3:
                if type(value) == np.ndarray and value.ndim == 1:
                    return eosx.height3Dvec(kwargs['tau'], z, value)
                else:
                    return eosx.height3D(kwargs['tau'], z, value)
            elif dim == 4:
                if type(value) == np.ndarray and value.ndim == 1:
                    return eosx.height4Dvec(kwargs['tau'], z, value)
                else:
                    return eosx.height4D(kwargs['tau'], z, value)

        if 'zb' in kwargs:
            dz = np.diff(kwargs['zb'])
        else:
            dz = np.diff(z)
            zb = z - dz/2
            zb = np.append(zb, zb[-1])
            zb[-1] += dz/2

        if 'rho' in kwargs:
            if 'kappa' in kwargs:
                if 'iBand' in kwargs:
                    tau = self.tau(kwargs['rho'], z, axis=axis, kappa=kwargs['kappa'], zb=zb, iBand=kwargs['iBand'])
                else:
                    tau = self.tau(kwargs['rho'], z, axis=axis, kappa=kwargs['kappa'], zb=zb)
            elif 'T' in kwargs and 'P' in kwargs:
                if 'iBand' in kwargs:
                    tau = self.tau(kwargs['rho'], z, axis=axis, P=kwargs['P'], T=kwargs['T'], iBand=kwargs['iBand'])
                else:
                    tau = self.tau(kwargs['rho'], z, axis=axis, P=kwargs['P'], T=kwargs['T'])
            else:
                raise ValueError("Either the keyword-argument 'kappa', or 'T' (temperature) and 'P' (pressure) have to"
                                 " be provided.")
        else:
            raise ValueError("The keyword-argument 'rho' has to be provided.")

        dim = kwargs['rho'].ndim
        trans = list(range(dim))
        trans[-1], trans[axis] = trans[axis], trans[-1]
        tau = np.transpose(tau, axes=trans)

        if dim == 3:
            if type(value) == np.ndarray:
                return eosx.height3Dvec(kwargs['tau'], z, value)
            else:
                return eosx.height3D(kwargs['tau'], z, value)
        elif dim == 4:
            if type(value) == np.ndarray:
                return eosx.height4Dvec(kwargs['tau'], z, value)
            else:
                return eosx.height4D(kwargs['tau'], z, value)
        raise ValueError("rho has to be 3D, or 4D.")


    def quant_at_tau(self, quant, new_tau, axis=-1, **kwargs):
        """
            Description
            -----------
                Computes the the field of a specified quantity at given optical depth.

                Either optical depth, opacity (kappa) and density, or temperature and pressure have to be provided. If
                kappa is provided, optical depth will be computed first. If temperature and pressure are provided, the
                opacity is computed first and then optical depth.

            Notes
            -----
                When opta is imported, it checks, if the compiled functions for computing opacity and optical depth are
                available. The availability is printed into the console ("eosx is available: True/False").

                If they are available, the computation might be much faster and less memory-consuming.

                If they are not available, numpy functions will be used. The output will ``not`` be extactly like the
                values internally computed in CO5BOLD!

            Input
            -----
                :param quant: ndarray, 3D or 4D, quantity defined at tau-values.
                :param new_tau: , value of tau at which the geometrical height is to be computed. Default: 1.0
                :param axis: int, optional, axis along the root-seeking will take place. Default: -1
                :param kwargs:
                    :param tau: ndarray, optical depth. If provided, the height will be computed directly.
                    :param rho: ndarray, mass density. If provided, along with kappa, optical depth will be integrated
                                first. Will be ignored, if tau is provided.
                    :param kappa: ndarray, opacity. If provided, along with rho, optical depth will be integrated first.
                                  Will be ignored, if tau is provided.
                    :param T: ndarray, temperature. If provided, along with rho and P, kappa and optical depth will be
                              computed first. Will be ignored, if tau, or kappa and rho are provided.
                    :param P: ndarray, pressure. If provided, along with rho and T, kappa and optical depth will be
                              computed first. Will be ignored, if tau, or kappa and rho are provided.
                    :param iBand: int, optional, opacity-band. If T, rho, and P, or kappa and rho are provdided, kappa
                                  and optical depth will be computed first. Will be ignored, if tau is provided.
                                  Default: 0
                    :param zb: 1D ndarray, optional, boundary-centered z-positions. If not provided, cell-heights will
                               be computed with z.

            Output
            ------
                :return: ndarray, optical depth (tau) of desired band.

            Example
            -------
                If kappa is not pre-computed use:

                    >>> from opta import Opac
                    >>>
                    >>> opan = r"directory/to/opacity-file/file.opta"
                    >>> opa = Opac(opan)
                    >>> [...]                       # rho, z, T and P are prepared
                    >>> tau = opa.tau(rho, z[, axis=1[, iBand=1]], T=T, P=P)   # computes kappa and tau. Returns tau

                If you want to pre-compute kappa use:

                    >>> from opta import Opac
                    >>>
                    >>> opan = r"directory/to/opacity-file/file.opta"
                    >>> opa = Opac(opan)
                    >>> [...]                       # rho, z, T and P are prepared
                    >>> kappa = opa.kappa(T, P[, iBand=1])
                    >>> tau = opa.tau(rho, z[, axis=1[, iBand=1]], kappa=kappa)   # computes and returns tau.
        """
        if not eosx_available:
            raise ImportError("Compiled functions of eosx not found!")

        if 'tau' in kwargs:
            dim = kwargs['tau'].ndim
            trans = list(range(dim))
            trans[-1], trans[axis] = trans[axis], trans[-1]
            kwargs['tau'] = np.transpose(kwargs['tau'], axes=trans)
            quant = np.transpose(quant, axes=trans)
            if dim == 3:
                if type(new_tau) == np.ndarray:
                    if new_tau.ndim == 1:
                        if np.all(np.diff(new_tau) > 0):
                            new_tau = new_tau[::-1]
                            inv = True
                            return np.transpose(eosx.cubeinterp3dvec(kwargs['tau'], quant, new_tau)[::-1], axes=trans)
                        return np.transpose(eosx.cubeinterp3dvec(kwargs['tau'], quant, new_tau), axes=trans)
                    elif new_tau.ndim == 2:
                        if axis == 0:
                            trans = [1, 0]
                        else:
                            trans = [0, 1]
                        return np.transpose(eosx.cubeinterp3d(kwargs['tau'], quant, new_tau), axes=trans)
                    elif new_tau.ndim == 3:
                        new_tau = np.transpose(new_tau, axes=trans)
                        if np.all(np.diff(new_tau, axis=-1) > 0):
                            new_tau = new_tau[:, :, ::-1]
                            return np.transpose(eosx.cubeinterp3dcube(kwargs['tau'], quant, new_tau)[:, :, ::-1],
                                                axes=trans)
                        return np.transpose(eosx.cubeinterp3dcube(kwargs['tau'], quant, new_tau), axes=trans)
                    else:
                        raise ValueError("new_tau has wrong dimension. new_tau.ndim must be 1, 2, or 3")
                else:
                    if axis == 0:
                        trans = [1, 0]
                    else:
                        trans = [0, 1]
                    return np.transpose(eosx.cubeinterp3dval(kwargs['tau'], quant, np.float64(new_tau)), axes=trans)
            elif dim == 4:
                if type(new_tau) == np.ndarray:
                    if new_tau.ndim == 1:
                        if np.all(np.diff(new_tau) > 0):
                            new_tau = new_tau[::-1]
                            return np.transpose(eosx.cubeinterp4dvec(kwargs['tau'], quant, new_tau)[::-1], axes=trans)
                        return np.transpose(eosx.cubeinterp4dvec(kwargs['tau'], quant, new_tau), axes=trans)
                    elif new_tau.ndim == 3:
                        if axis == 1:
                            trans = [0, 2, 1]
                        elif axis == 0:
                            trans = [2, 0, 1]
                        else:
                            trans = [0, 1, 2]
                        return np.transpose(eosx.cubeinterp4d(kwargs['tau'], quant, new_tau), axes=trans)
                    elif new_tau.ndim == 4:
                        new_tau = np.transpose(new_tau, axes=trans)
                        if np.all(np.diff(new_tau, axis=-1) > 0):
                            new_tau = new_tau[:, :, :, ::-1]
                            return np.transpose(eosx.cubeinterp4dcube(kwargs['tau'], quant, new_tau)[:, :, :, ::-1],
                                                axes=trans)
                        return np.transpose(eosx.cubeinterp4dcube(kwargs['tau'], quant, new_tau), axes=trans)
                    else:
                        raise ValueError("new_tau has wrong dimension. new_tau.ndim must be 1, 3, or 4")
                else:
                    if axis == 1:
                        trans = [0, 2, 1]
                    elif axis == 0:
                        trans = [2, 0, 1]
                    else:
                        trans = [0, 1, 2]
                    return np.transpose(eosx.cubeinterp4dval(kwargs['tau'], quant, new_tau), axes=trans)
            else:
                raise ValueError("tau has wrong dimension. new_tau.ndim must be 1, 3, or 4")

        if 'zb' in kwargs:
            dz = np.diff(kwargs['zb'])
        else:
            dz = np.diff(z)
            zb = z - dz/2
            zb = np.append(zb, zb[-1])
            zb[-1] += dz/2

        if 'rho' in kwargs:
            if 'kappa' in kwargs:
                tau = self.tau(kwargs['rho'], z, axis=axis, kappa=kwargs['kappa'], zb=zb)
            elif 'T' in kwargs and 'P' in kwargs:
                if 'iBand' in kwargs:
                    tau = self.tau(kwargs['rho'], z, axis=axis, P=kwargs['P'], T=kwargs['T'], iBand=kwargs['iBand'])
                else:
                    tau = self.tau(kwargs['rho'], z, axis=axis, P=kwargs['P'], T=kwargs['T'])
            else:
                raise ValueError("Either the keyword-argument 'kappa', or 'T' (temperature) and 'P' (pressure) have to"
                                 " be provided.")
        else:
            raise ValueError("Either the keyword-argument 'tau' and 'rho' has to be provided.")

        dim = kwargs['rho'].ndim
        trans = list(range(dim))
        trans[-1], trans[axis] = trans[axis], trans[-1]
        tau = np.transpose(tau, axes=trans)
        quant = np.transpose(quant, axes=trans)

        if dim == 3:
            if type(new_tau) == np.ndarray:
                if new_tau.ndim == 1:
                    return np.transpose(eosx.cubeinterp3dvec(tau.astype(np.float64), quant.astype(np.float64),
                                                             new_tau.astype(np.float64)), axes=trans)
                elif new_tau.ndim == 2:
                    if axis == 0:
                        trans = [1, 0]
                    else:
                        trans = [0, 1]
                    return np.transpose(eosx.cubeinterp3d(tau.astype(np.float64), quant.astype(np.float64),
                                                          new_tau.astype(np.float64)), axes=trans)
                elif new_tau.ndim == 3:
                    new_tau = np.transpose(new_tau, axes=trans)
                    return np.transpose(eosx.cubeinterp3dcube(tau.astype(np.float64), quant.astype(np.float64),
                                                              new_tau.astype(np.float64)), axes=trans)
                else:
                    raise ValueError("new_tau has wrong dimension. new_tau.ndim must be 1, 2, or 3")
            else:
                if axis == 0:
                    trans = [1, 0]
                else:
                    trans = [0, 1]
                return np.transpose(eosx.cubeinterp3dval(tau.astype(np.float64), quant.astype(np.float64),
                                                         new_tau.astype(np.float64)), axes=trans)
        elif dim == 4:
            if type(new_tau) == np.ndarray:
                if new_tau.ndim == 1:
                    return np.transpose(eosx.cubeinterp4dvec(tau.astype(np.float64), quant.astype(np.float64),
                                                             new_tau.astype(np.float64)), axes=trans)
                elif new_tau.ndim == 3:
                    if axis == 1:
                        trans = [0, 2, 1]
                    elif axis == 0:
                        trans = [2, 0, 1]
                    else:
                        trans = [0, 1, 2]
                    return np.transpose(eosx.cubeinterp4d(tau.astype(np.float64), quant.astype(np.float64),
                                                          new_tau.astype(np.float64)), axes=trans)
                elif new_tau.ndim == 4:
                    new_tau = np.transpose(new_tau, axes=trans)
                    return np.transpose(eosx.cubeinterp4dcube(tau.astype(np.float64),
                                                              quant.astype(np.float64), new_tau.astype(np.float64)),
                                        axes=trans)
                else:
                    raise ValueError("new_tau has wrong dimension. new_tau.ndim must be 1, 3, or 4")
            else:
                if axis == 1:
                    trans = [0, 2, 1]
                elif axis == 0:
                    trans = [2, 0, 1]
                else:
                    trans = [0, 1, 2]
                return np.transpose(eosx.cubeinterp4dval(tau.astype(np.float64), quant.astype(np.float64),
                                                         new_tau.astype(np.float64)), axes=trans)
        else:
            raise ValueError("tau has wrong dimension. new_tau.ndim must be 1, 3, or 4")


if __name__ == '__main__':
    from eosinter import EosInter
    import uio

    dpath = r"X:\pluto_2\salhab\cobold\scratchy\job_d3gt57g44rsn01_400x400x188\rhd020.full"
    eosn = r"H:\Documents\cobold\eos\dat\eos_cifist2006_m00_a00_l5.eos"
    fname = r"N:\Python\Analyse\opa\g2v_marcs_idmean3xRT3.opta"

    eos = EosInter(eosn)
    opa = Opac(fname)
    f = uio.File(dpath)

    z = f.dataset[0].box[0]['xc3'].data.squeeze()
    zb = f.dataset[0].box[0]['xb3'].data.squeeze()
    rho = f.dataset[0].box[0]['rho'].data.astype(np.float64)
    ei = f.dataset[0].box[0]['ei'].data.astype(np.float64)

    P, T = eos.PandT(rho, ei)
    kappa = opa.kappa(T, P)

    tau = opa.tau(rho, z, axis=0, zb=zb, kappa=kappa)
    del P, T
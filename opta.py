# -*- coding: utf-8 -*-
"""
Created on Fri Jun 06 12:09:39 2014

@author: René Georg Salhab
"""

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

        self.tabT = self.enlist(temptb, np.float32)
        self.tabTBN = self.enlist(tabtbn, np.float32)
        self.idxTBN = self.enlist(idxtbn, np.int32)
        self.idxTBN -= 1
        self.tabDTB = self.enlist(tabdtb, np.float32)

        self.tabP = self.enlist(presstb, np.float32)
        self.tabPBN = self.enlist(tabpbn, np.float32)
        self.idxPBN = self.enlist(idxpbn, np.int32)
        self.idxPBN -= 1
        self.tabDPB = self.enlist(tabdpb, np.float32)

        tabKap = np.array([a.strip() for sublist in tabKap for a in sublist if a != '']).astype(np.float32)
        tabKap = tabKap.reshape(dimP, dimBAND, dimT)
        self.tabKap = np.transpose(tabKap, axes=(2, 0, 1))

    def enlist(self, inList, datatype):
        return np.array([item for sublist in inList for item in sublist if item != '']).astype(datatype)

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
                available. The availability is printed into the console ("eosx is available: True/False"). If they are
                not available, the methods for computing kappa, tau, height and quant_at_tau cannot be used.

                All quantities must be float32.

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
        if not eosx_available:
            raise IOError("Compilation of eosinterx.pyx necessary.")
        log10P = ne.evaluate("log10(P)")
        log10T = ne.evaluate("log10(T)")

        if T.ndim == 3:
            func = eosx.logPT2kappa3D
        elif T.ndim == 4:
            func = eosx.logPT2kappa4D
        else:
            raise ValueError("Wrong dimension. Only 3D- and 4D-arrays supported.")

        return ne.evaluate("10**kap", local_dict={'kap': func(log10P, log10T, self.tabP, self.tabT, self.tabKap,
                                                              self.tabTBN, self.tabDTB, self.idxTBN, self.tabPBN,
                                                              self.tabDPB, self.idxPBN, iBand)})

    def tau(self, rho, axis=-1, **kwargs):
        """
            Description
            -----------
                Computes optical depth. Either opacity (kappa), or temperature and pressure have to be provided. If
                temperature and pressure are provided, the opacity is computed first.

            Notes
            -----
                When opta is imported, it checks, if the compiled functions for computing opacity and optical depth are
                available. The availability is printed into the console ("eosx is available: True/False"). If they are
                not available, the methods for computing kappa, tau, height and quant_at_tau cannot be used.

                All quantities must be float32.

            Input
            -----
                :param rho: ndarray, mass-density
                :param axis: int, optional, axis along the integration will take place. Default: -1
                :param kwargs:
                    :param kappa: ndarray, opacity. If provided, kappa times rho will be integrated directly.
                    :param T: ndarray, temperature. If kappa is not provided, but T and P, the opacity will be computed
                              first. Will be ignored, if kappa is provided.
                    :param P: ndarray, pressure. If kappa is not provided, but T and P, the opacity will be computed
                              first. Will be ignored, if kappa is provided.
                    :param radhtautop: float, Scale height of optical depth at top (from .par-file). Default: -1
                    :param iBand: int, optional, opacity-band. If kappa is not provided, but T and P, the opacity will
                                  be computed first. Will be ignored, if kappa is provided. Default: 0
                    :param z: 1D ndarray, height-scale. Has to have the same length like the vertical axis of rho and
                              the other provided values (kappa, or T and P).
                              Attention: Must be in cgs!
                    :param zb: 1D ndarray, optional, boundary-centered z-positions. If not provided, cell-heights will
                               be computed with z. Attention: Must be in cgs!

            Output
            ------
                :return: ndarray, optical depth (tau) of desired band (default band: "0" (bolometric).

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
            raise IOError("Compilation of eosinterx.pyx necessary.")

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
            dz = np.diff(kwargs['zb']).astype(np.float32)
        elif 'z' in kwargs:
            dz = np.diff(kwargs['z']).astype(np.float32)
            dz = np.append(dz, dz[-1])
        else:
            raise ValueError("Either 'zb' or 'z' has to be provided.")

        if 'radhtautop' in kwargs:
            radHtautop = np.float32(kwargs['radhtautop'])
        else:
            radHtautop = np.float32(-1.0)

        dim = rho.ndim

        trans = list(range(dim))
        trans[-1], trans[axis] = trans[axis], trans[-1]
        kaprho = np.transpose(kaprho, axes=trans)

        if dim == 3:
            return np.transpose(eosx.tau3D(kaprho, dz, radHtautop), axes=trans)
        elif dim == 4:
            return np.transpose(eosx.tau4D(kaprho, dz, radHtautop), axes=trans)
        else:
            raise ValueError("Wrong dimension. Only 3D- and 4D-arrays supported.")

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
                available. The availability is printed into the console ("eosx is available: True/False"). If they are
                not available, the methods for computing kappa, tau, height and quant_at_tau cannot be used.

                All quantities must be float32.

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
                    :param radhtautop: float, Scale height of optical depth at top (from .par-file). Default: -1
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
            zb = kwargs['zb'].astype(np.float32)
        else:
            dz = np.diff(z).astype(np.float32)
            zb = z - dz/2
            zb = np.append(zb, zb[-1] + dz[-1])

        if 'radhtautop' in kwargs:
            radhtautop = np.float32(kwargs['radhtautop'])
        else:
            radhtautop = np.float32(-1.0)

        if 'rho' in kwargs:
            if 'kappa' in kwargs:
                if 'iBand' in kwargs:
                    tau = self.tau(kwargs['rho'], axis=axis, kappa=kwargs['kappa'], zb=zb, radhtautop=radhtautop,
                                   iBand=kwargs['iBand'])
                else:
                    tau = self.tau(kwargs['rho'], axis=axis, kappa=kwargs['kappa'], zb=zb)
            elif 'T' in kwargs and 'P' in kwargs:
                if 'iBand' in kwargs:
                    tau = self.tau(kwargs['rho'], axis=axis, P=kwargs['P'], T=kwargs['T'], zb=zb, iBand=kwargs['iBand'])
                else:
                    tau = self.tau(kwargs['rho'], axis=axis, P=kwargs['P'], T=kwargs['T'], zb=zb)
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
                return eosx.height3Dvec(tau, z, value)
            else:
                return eosx.height3D(tau, z, value)
        elif dim == 4:
            if type(value) == np.ndarray:
                return eosx.height4Dvec(tau, z, value)
            else:
                return eosx.height4D(tau, z, value)
        raise ValueError("rho has to be 3D, or 4D.")


    def quant_at_tau(self, quant, new_tau=1, axis=-1, **kwargs):
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
                available. The availability is printed into the console ("eosx is available: True/False"). If they are
                not available, the methods for computing kappa, tau, height and quant_at_tau cannot be used.

                All quantities must be float32.

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
                    :param radhtautop: float, Scale height of optical depth at top (from .par-file). Default: -1
                    :param iBand: int, optional, opacity-band. If T, rho, and P, or kappa and rho are provdided, kappa
                                  and optical depth will be computed first. Will be ignored, if tau is provided.
                                  Default: 0
                    :param z: 1D ndarray, optional, cell-centered z-positions. Is mandatory, if tau has to be computed
                                (only rho, P and T, or rho and kappa provided). Has to be in cgs.
                    :param zb: 1D ndarray, optional, boundary-centered z-positions. If not provided, cell-heights will
                               be computed with z. Has to be in cgs.

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

        # define dimension of input arrays and transpose-tuple, which changes the order of axis thus that last axis is
        # interpolation-axis (necessary for cython-functions.
        dim = quant.ndim
        trans = list(range(dim))
        trans[-1], trans[axis] = trans[axis], trans[-1]

        if not eosx_available:
            raise ImportError("Compiled functions of eosx not found!")

        if 'tau' in kwargs:
            tau = kwargs['tau'].astype(np.float32)
        else:
            if 'rho' in kwargs:

                if 'zb' in kwargs:
                    zb = kwargs['zb'].astype(np.float32)
                elif 'z' in kwargs:
                    dz = np.diff(z)
                    zb = z - dz / 2
                    zb = np.append(zb, zb[-1] + dz[-1])
                else:
                    raise ValueError("Either 'zb' or 'z' has to be provided, if 'tau' is not available.")

                if 'radhtautop' in kwargs:
                    radhtautop = np.float32(kwargs['radhtautop'])
                else:
                    radhtautop = np.float32(-1.0)

                if 'kappa' in kwargs:
                    kappa = kwargs['kappa'].astype(np.float32)
                    tau = self.tau(kwargs['rho'], axis=axis, kappa=kappa, zb=zb, radhtautop=radhtautop)
                elif 'T' in kwargs and 'P' in kwargs:
                    T = kwargs['T'].astype(np.float32)
                    P = kwargs['P'].astype(np.float32)
                    if 'iBand' in kwargs:
                        tau = self.tau(kwargs['rho'], axis=axis, P=P, T=T, zb=zb, radhtautop=radhtautop,
                                       iBand=kwargs['iBand'])
                    else:
                        tau = self.tau(kwargs['rho'], axis=axis, P=P, T=T, zb=zb, radhtautop=radhtautop)
                else:
                    raise ValueError("Either the keyword-argument 'kappa', or 'T' (temperature) and 'P' (pressure) have"
                                     " to be provided.")
            else:
                raise ValueError("Either the keyword-argument 'tau' or 'rho' has to be provided.")

        tau = np.transpose(tau, axes=trans)
        quantity = np.transpose(quant, axes=trans).astype(np.float32)
        ntau = np.float32(new_tau)

        if dim == 3:
            if type(ntau) == np.ndarray:
                if ntau.ndim == 1:
                    if np.all(np.diff(ntau) > 0):
                        ntau = ntau[::-1]
                    return np.transpose(eosx.cubeinterp3dvec(tau, quantity, ntau), axes=trans)
                elif ntau.ndim == 2:
                    # result will be reduced by 1 dimension (surface). Trans has to be redefined.
                    if axis == 0:
                        trans = [1, 0]
                    else:
                        trans = [0, 1]
                    return np.transpose(eosx.cubeinterp3d(tau, quantity, ntau), axes=trans)
                elif ntau.ndim == 3:
                    ntau = np.transpose(ntau, axes=trans)
                    if np.all(np.diff(ntau, axis=-1) > 0):
                        ntau = ntau[:, :, ::-1]
                    return np.transpose(eosx.cubeinterp3dcube(tau, quantity, ntau), axes=trans)
                else:
                    raise ValueError("new_tau has wrong dimension. new_tau.ndim must be 1, 2, or 3")
            else:
                # result will be reduced by 1 dimension (plane). Trans has to be redefined.
                if axis == 0:
                    trans = [1, 0]
                else:
                    trans = [0, 1]
                return np.transpose(eosx.cubeinterp3dval(tau, quantity, ntau), axes=trans)
        elif dim == 4:
            if type(ntau) == np.ndarray:
                if ntau.ndim == 1:
                    if np.all(np.diff(ntau) > 0):
                        ntau = ntau[::-1]
                    return np.transpose(eosx.cubeinterp4dvec(tau, quantity, ntau), axes=trans)
                elif ntau.ndim == 3:
                    # result will be reduced by 1 dimension (surface). Trans has to be redefined.
                    if axis == 0:
                        trans = [2, 0, 1]
                    elif axis == 1:
                        trans = [0, 2, 1]
                    else:
                        trans = [0, 1, 2]
                    return np.transpose(eosx.cubeinterp4d(tau, quantity, ntau), axes=trans)
                elif ntau.ndim == 4:
                    ntau = np.transpose(ntau, axes=trans)
                    if np.all(np.diff(ntau, axis=-1) > 0):
                        ntau = ntau[:, :, :, ::-1]
                    return np.transpose(eosx.cubeinterp4dcube(tau, quantity, ntau), axes=trans)
                else:
                    raise ValueError("new_tau has wrong dimension. new_tau.ndim must be 1, 3, or 4")
            else:
                # result will be reduced by 1 dimension (plane). Trans has to be redefined.
                if axis == 0:
                    trans = [2, 0, 1]
                elif axis == 1:
                    trans = [0, 2, 1]
                else:
                    trans = [0, 1, 2]
                return np.transpose(eosx.cubeinterp4dval(tau, quantity, ntau), axes=trans)
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

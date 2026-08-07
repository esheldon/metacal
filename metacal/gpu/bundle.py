"""
PsfBundle: everything the gpu engine needs that is derived from
one psf, computed on the CPU by the existing validated code
paths (FusionFilter, AZGauss, galsim).  This module imports no
cupy, so pipeline worker processes can build bundles without any
device involvement; the engine assembles its per-psf device
factors from a bundle in milliseconds.

Per-image psfs are the fundamental case (real coadds): build one
bundle per [image, psf, noise] input.  The engine keeps a small
LRU of assembled factors keyed on the psf content, so repeated
psfs (e.g. a fixed-psf simulation) cost nothing after the first.
"""
import numpy as np

from ..metacalibration import SHEAR_STEP, LANCZOS
from .._util import _wcs_and_matrix
from ..azgauss_target_psf import AZGauss
from ..fusion_filter import FusionFilter

FWHM_FAC = 2.0 * np.sqrt(2.0 * np.log(2.0))


def _diagonal_scale(wcs, rtol=1.0e-9):
    """the scalar pixel scale of a diagonal isotropic wcs; a
    relative tolerance absorbs ulp-level arithmetic noise in wcs
    chains (a sim coadd wcs can carry dudx != dvdy at 1e-16)"""
    wcs, mat = _wcs_and_matrix(wcs)
    s = 0.5 * (mat[0, 0] + mat[1, 1])
    if (abs(mat[0, 0] - mat[1, 1]) > rtol * abs(s)
            or abs(mat[0, 1]) > rtol * abs(s)
            or abs(mat[1, 0]) > rtol * abs(s)):
        raise NotImplementedError(
            f'the gpu engine requires a diagonal isotropic pixel '
            f'scale, got jacobian {mat}'
        )
    return wcs, float(s)


class PsfBundle:
    """
    Parameters
    ----------
    psf_image: array
        the psf image.  galsim draws stamps in float32; the image
        is upcast to float64 before any accumulation (this
        matters at the 1e-6 level; see the gpu-port document)
    wcs: galsim wcs or 2x2 jacobian matrix
        must be diagonal isotropic
    types: sequence of str
        the metacal type set; the fusion filter couples the set,
        so one bundle serves exactly this set (build a second,
        noshear-only bundle for the noise-observation run)
    dim: int
        image dimension of the cells this bundle will serve
    target_psf: AZGauss, optional
        validated to be an AZGauss (the engine's math is specific
        to the round gaussian target)
    noise_filter: FusionFilter, optional
        validated to be a FusionFilter
    """

    def __init__(self, psf_image, wcs, types, dim,
                 target_psf=None, noise_filter=None):
        import galsim

        if target_psf is None:
            target_psf = AZGauss()
        if not isinstance(target_psf, AZGauss):
            raise NotImplementedError(
                f'the gpu engine implements the AZGauss target '
                f'only, got {type(target_psf).__name__}'
            )
        if noise_filter is None:
            noise_filter = FusionFilter()
        if not isinstance(noise_filter, FusionFilter):
            raise NotImplementedError(
                f'the gpu engine implements the FusionFilter '
                f'noise correction only, got '
                f'{type(noise_filter).__name__}'
            )

        # upcast FIRST: float32 stamp accumulations shift the psf
        # flux, which scales the target, at the 3.5e-8 level
        psf_image = np.ascontiguousarray(
            np.asarray(psf_image, dtype=float))
        self.psf_image = psf_image
        self.psf_flux = float(np.sum(psf_image))
        self.types = list(types)
        self.dim = int(dim)

        wcs, scale = _diagonal_scale(wcs)
        self.wcs = wcs
        self.scale = scale

        self.hfilt, self.noise_var_factor = noise_filter(
            psf_image=psf_image, wcs=wcs, target_psf=target_psf,
            dim=self.dim, types=self.types,
        )
        self.npix = self.hfilt[self.types[0]].shape[0]

        # the target and psf-side conventions, exactly as Metacal
        # builds them
        psf_int = galsim.InterpolatedImage(
            galsim.Image(psf_image, wcs=wcs),
            x_interpolant=LANCZOS,
        )
        tp0 = target_psf(psf_int, flux=self.psf_flux)
        self.sigma_pix = (
            tp0.fwhm * (1.0 + 2.0 * SHEAR_STEP) / FWHM_FAC / scale
        )
        ny, nx = psf_image.shape
        self.target_psf_image = tp0.dilate(
            1.0 + 2.0 * SHEAR_STEP
        ).drawImage(
            nx=nx, ny=ny, wcs=wcs, method='no_pixel'
        ).array

        # galsim SBGaussian truncates the target at this radius
        # in its general (jacobian) fillKImage branch, which the
        # rotated noise pass goes through; the engine replicates
        # the cut (kvalue_accuracy = 1e-5)
        self.gauss_cut = (np.sqrt(-2.0 * np.log(1.0e-5))
                          / self.sigma_pix)
        self.psf_maxk_pix = scale * psf_int.maxk

    def key(self):
        """the engine's factor-cache key: the psf content plus
        everything else the factors depend on"""
        return (self.psf_image.tobytes(), self.psf_image.shape,
                tuple(self.types), self.dim, self.npix,
                round(self.scale, 12))


class RawBundle:
    """
    a bundle reconstructed from shipped arrays and scalars — for
    a device-owning process receiving the psf-dependent pieces
    from workers that built the real PsfBundle.  Carries exactly
    what the engine's factor assembly needs; the packaging pieces
    (target psf image, noise_var_factor) stay with the worker.
    """

    def __init__(self, psf_image, hfilt, types, dim, npix, scale,
                 psf_flux, sigma_pix, gauss_cut, psf_maxk_pix):
        self.psf_image = np.ascontiguousarray(
            np.asarray(psf_image, dtype=float))
        self.hfilt = hfilt
        self.types = list(types)
        self.dim = int(dim)
        self.npix = int(npix)
        self.scale = float(scale)
        self.psf_flux = float(psf_flux)
        self.sigma_pix = float(sigma_pix)
        self.gauss_cut = float(gauss_cut)
        self.psf_maxk_pix = float(psf_maxk_pix)

    key = PsfBundle.key

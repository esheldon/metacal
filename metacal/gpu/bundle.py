"""
Per-psf CPU-side derivations for the gpu engine.  This module
imports no cupy, so pipeline worker processes can build these
without any device involvement.

PsfCore: the cheap per-psf pieces — the AZGauss target and its
drawn image, the scalar conventions (sigma, cuts), and the exact
galsim draw-grid size npix (via a build-only Metacal on a delta
image, ~10-30 ms).  No fusion filter: the engine builds the
filter from device impulse transfers (a delta image's k-table is
exactly ones, so the transfers are the factor magnitudes the
engine assembles anyway).

PsfBundle: PsfCore plus the CPU FusionFilter build (~0.4 s per
psf).  This is the stage-1 path, kept as the validation
reference for the device-built filter; the two agree at the
1e-5 level (the difference is galsim's gaussian jac-branch fill
on the rotated transfer's inner k rings).

Per-image psfs are the fundamental case (real coadds): build one
core per [image, psf, noise] input.  The engine keeps a small
LRU of assembled per-psf state keyed on the psf content, so
repeated psfs (e.g. a fixed-psf simulation) cost nothing after
the first.
"""
import numpy as np

from ..metacalibration import SHEAR_STEP, LANCZOS, Metacal
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


class PsfCore:
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
        so one core serves exactly this set (build a second,
        noshear-only core for the noise-observation run)
    dim: int
        image dimension of the cells this core will serve
    target_psf: AZGauss, optional
        validated to be an AZGauss (the engine's math is specific
        to the round gaussian target)
    noise_filter: FusionFilter, optional
        validated to be a FusionFilter; its configuration (the
        diagnostic ``full`` flag) is honored by the device filter
        build
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
        self.filter_full = bool(noise_filter.full)
        self._target_psf_obj = target_psf

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

        # the exact galsim draw-grid size: a build-only Metacal
        # on a delta image (the same object whose khats the CPU
        # filter would draw); psf-dependent in principle, though
        # the good-fft-size quantization makes it stable across
        # realistic psfs
        delta = np.zeros((self.dim, self.dim))
        delta[self.dim // 2, self.dim // 2] = 1.0
        self.npix = Metacal(
            image=delta, psf_image=psf_image, wcs=wcs,
            target_psf=target_psf, types=self.types,
        ).npix

        # filled by the CPU filter (PsfBundle) or by the engine's
        # device filter build
        self.hfilt = None
        self.noise_var_factor = None

    def key(self):
        """the engine's factor-cache key: the psf content plus
        everything else the factors depend on.  'cpu'/'gpu'
        marks which filter build the factors carry."""
        fsrc = 'cpu' if self.hfilt is not None else 'gpu'
        return (self.psf_image.tobytes(), self.psf_image.shape,
                tuple(self.types), self.dim, self.npix,
                round(self.scale, 12), fsrc, self.filter_full)

    def with_types(self, types):
        """a sibling core for a different type set on the same
        psf (e.g. the noshear-only noise-observation run): the
        per-psf derivations are type-independent and carried
        over, so this costs nothing"""
        import copy

        out = copy.copy(self)
        out.types = list(types)
        out.hfilt = None
        out.noise_var_factor = None
        return out


class PsfBundle(PsfCore):
    """
    PsfCore plus the CPU FusionFilter build: the stage-1 path and
    the validation reference for the engine's device-built
    filter.  Parameters as for PsfCore.
    """

    def __init__(self, psf_image, wcs, types, dim,
                 target_psf=None, noise_filter=None):
        super().__init__(
            psf_image, wcs, types, dim,
            target_psf=target_psf, noise_filter=noise_filter,
        )
        if noise_filter is None:
            noise_filter = FusionFilter()
        self.hfilt, self.noise_var_factor = noise_filter(
            psf_image=self.psf_image, wcs=self.wcs,
            target_psf=self._target_psf_obj, dim=self.dim,
            types=self.types,
        )
        npix_filt = self.hfilt[self.types[0]].shape[0]
        assert npix_filt == self.npix, (
            f'filter grid {npix_filt} != delta-Metacal npix '
            f'{self.npix}'
        )


class RawBundle:
    """
    a core reconstructed from shipped arrays and scalars — for a
    device-owning process receiving the psf-dependent pieces from
    workers that built the real PsfCore.  With hfilt=None (the
    stage-2 protocol) the engine builds the filter on device and
    fills noise_var_factor; with hfilt arrays it behaves as a
    shipped stage-1 bundle.
    """

    def __init__(self, psf_image, types, dim, npix, scale,
                 psf_flux, sigma_pix, gauss_cut, psf_maxk_pix,
                 hfilt=None, filter_full=False):
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
        self.filter_full = bool(filter_full)
        self.noise_var_factor = None

    key = PsfCore.key

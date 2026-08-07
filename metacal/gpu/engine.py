"""
FusionEngine: the fusion/AZGauss metacal pipeline on the GPU,
with per-image psfs as the fundamental case.

The engine itself holds only psf-INDEPENDENT device state: the
composed coordinate grids, phase ramps and argument-norm grids
for a given (dim, psf_dim, npix, types, precision), cached
per configuration.  Everything derived from a psf arrives in a
PsfBundle (built on the CPU by the validated production code;
see bundle.py) and is assembled into device factor tables in a
few milliseconds per psf — psf table FFT, the psf-side gathers,
the analytic target — with a small LRU keyed on the psf content
so repeated psfs cost nothing.

Per observation the work is two table FFTs, two gathers, an
elementwise multiply and a batched inverse FFT, with the
content-dependent maxk cuts measured on device by an exact port
of galsim's calculateMaxK — zero per-cell galsim calls.

Scope: the production composite (InterpolatedImage/lanczos15
deconvolution, reduced-shear metacal, AZGauss round target,
fusion noise correction, the rotated-noise pass) — not the
general Metacal API.  The wcs must be a diagonal isotropic pixel
scale (enforced by PsfBundle).

Precision: fp32=True is the production configuration (validated
at ~1e-5 of the noise sigma against the CPU path and certified
at ensemble level); fp32=False tracks the CPU reference at
~1e-14 and production at ~1e-8 of sigma.
"""
from collections import OrderedDict

import numpy as np

from ..metacalibration import SHEAR_STEP
from ..result import MetacalResult
from ..obs import package_metacal_obs
from . import _kernels
from ._maxk import maxk_scan, lanczos15_krange_pix
from .bundle import PsfBundle, _diagonal_scale

R90 = np.array([[0.0, -1.0], [1.0, 0.0]])

SHEARS = {
    'noshear': (0.0, 0.0),
    '1p': (SHEAR_STEP, 0.0), '1m': (-SHEAR_STEP, 0.0),
    '2p': (0.0, SHEAR_STEP), '2m': (0.0, -SHEAR_STEP),
}

DEFAULT_FACTOR_CACHE = 8


def shear_matrix(g1, g2):
    """the reduced-shear distortion matrix"""
    fac = 1.0 / np.sqrt(1.0 - g1 * g1 - g2 * g2)
    return fac * np.array([[1.0 + g1, g2], [g2, 1.0 - g1]])


class _KGrids:
    """the psf-independent device state for one
    (dim, psf_dim, npix, types) configuration: composed gather
    coordinates (fp64 for the psf-side gathers plus a cast copy
    for the per-cell gathers), argument-norm grids, the k^2 grid
    and the centering/fractional phase grids"""

    def __init__(self, cp, dim, psf_dim, npix, types, fdt):
        frac_img = (dim - 1) / 2.0 - dim // 2
        frac_psf = (psf_dim - 1) / 2.0 - psf_dim // 2
        k1 = 2.0 * np.pi * np.fft.fftfreq(npix)
        KX, KY = np.meshgrid(k1, k1)
        self.k2 = cp.asarray((KX ** 2 + KY ** 2).ravel())
        self.ramp = cp.asarray(
            np.exp(-1j * (KX + KY) * (npix - 1) / 2.0).ravel())
        nt = len(types)
        npts = npix * npix
        self.npts = npts

        cAx = np.empty((nt, npts))
        cAy = np.empty((nt, npts))
        cNx = np.empty((nt, npts))
        cNy = np.empty((nt, npts))
        pNx = np.empty((nt, npts))
        pNy = np.empty((nt, npts))
        arg2A = np.empty((nt, npts))
        arg2N = np.empty((nt, npts))
        phA = np.empty((nt, npts), dtype=complex)
        phN = np.empty((nt, npts), dtype=complex)
        for i, t in enumerate(types):
            # image pass: gather argument M^T k
            M = shear_matrix(*SHEARS[t])
            kx = (M[0, 0] * KX + M[1, 0] * KY).ravel()
            ky = (M[0, 1] * KX + M[1, 1] * KY).ravel()
            cAx[i] = kx
            cAy[i] = ky
            arg2A[i] = kx ** 2 + ky ** 2
            phA[i] = np.exp(
                1j * (kx + ky) * (frac_img - frac_psf))

            # rotated noise pass: image argument R(-90) M^T R90 k,
            # psf argument M^T R90 k; round target unrotated
            Bi = R90.T @ M.T @ R90
            Bp = M.T @ R90
            kxi = (Bi[0, 0] * KX + Bi[0, 1] * KY).ravel()
            kyi = (Bi[1, 0] * KX + Bi[1, 1] * KY).ravel()
            kxp = (Bp[0, 0] * KX + Bp[0, 1] * KY).ravel()
            kyp = (Bp[1, 0] * KX + Bp[1, 1] * KY).ravel()
            cNx[i] = kxi
            cNy[i] = kyi
            pNx[i] = kxp
            pNy[i] = kyp
            arg2N[i] = kxp ** 2 + kyp ** 2
            phN[i] = np.exp(1j * ((kxi + kyi) * frac_img
                                  - (kxp + kyp) * frac_psf))

        # fp64 coordinates for the psf-side gathers (per-psf
        # factor assembly), cast copies for the per-cell gathers
        self.cAx64 = cp.asarray(cAx)
        self.cAy64 = cp.asarray(cAy)
        self.pNx64 = cp.asarray(pNx)
        self.pNy64 = cp.asarray(pNy)
        self.cAx = cp.asarray(cAx.ravel()).astype(fdt)
        self.cAy = cp.asarray(cAy.ravel()).astype(fdt)
        self.cNx = cp.asarray(cNx.ravel()).astype(fdt)
        self.cNy = cp.asarray(cNy.ravel()).astype(fdt)
        self.arg2A = cp.asarray(arg2A)
        self.arg2N = cp.asarray(arg2N)
        self.phA = cp.asarray(phA)
        self.phN = cp.asarray(phN)


class FusionEngine:
    """
    Parameters
    ----------
    dim: int
        image dimension of the cells this engine processes
    types: sequence of str
        the metacal type set (one engine serves exactly one set;
        the fusion filter couples the set)
    fp32: bool
        run the per-cell work in single precision (production
        configuration).  Filters and factors are always built in
        fp64 and cast.
    factor_cache: int
        how many per-psf factor sets to keep on the device
        (LRU keyed on psf content).  Repeated psfs — a fixed-psf
        simulation, or reprocessing the same cell — hit the
        cache; a stream of distinct psfs costs a few ms each.
    """

    def __init__(self, dim, types, fp32=True,
                 factor_cache=DEFAULT_FACTOR_CACHE):
        import cupy as cp

        self.cp = cp
        self.dim = int(dim)
        self.types = list(types)
        self.fp32 = bool(fp32)
        self._cdt = cp.complex64 if fp32 else cp.complex128
        self._fdt = cp.float32 if fp32 else cp.float64
        self._kern = _kernels.gather_kernel(fp32)
        self._kern64 = _kernels.gather_kernel(False)
        self._krange_pix = lanczos15_krange_pix()
        self._grids = {}
        self._factors = OrderedDict()
        self._bundles = OrderedDict()
        self._factor_cache = int(factor_cache)

    def _get_grids(self, psf_dim, npix):
        key = (psf_dim, npix)
        if key not in self._grids:
            self._grids[key] = _KGrids(
                self.cp, self.dim, psf_dim, npix, self.types,
                self._fdt,
            )
        return self._grids[key]

    def _get_factors(self, bundle):
        """the per-psf device factor tables, assembled from the
        bundle via the cached psf-independent grids; LRU on the
        psf content"""
        cp = self.cp
        key = bundle.key()
        if key in self._factors:
            self._factors.move_to_end(key)
            return self._factors[key]

        if list(bundle.types) != self.types:
            raise ValueError(
                f'bundle types {bundle.types} != engine types '
                f'{self.types}'
            )
        if bundle.dim != self.dim:
            raise ValueError(
                f'bundle dim {bundle.dim} != engine dim '
                f'{self.dim}'
            )
        psf_dim = bundle.psf_image.shape[0]
        g = self._get_grids(psf_dim, bundle.npix)

        t_psf, npad_psf, _ = _kernels.build_table(
            bundle.psf_image)
        that = bundle.psf_flux * cp.exp(
            -0.5 * bundle.sigma_pix ** 2 * g.k2)
        tr = that * g.ramp
        gmask = (g.k2 <= bundle.gauss_cut ** 2).astype(
            cp.float64)
        pk2 = bundle.psf_maxk_pix ** 2

        nt = len(self.types)
        facA = []
        facN = []
        for i in range(nt):
            den = _kernels.gather(
                self._kern64, t_psf, npad_psf,
                g.cAx64[i], g.cAy64[i], cp.complex128)
            fA = tr * g.phA[i] / den
            fA = fA * (g.arg2A[i] <= pk2)
            facA.append(fA.astype(self._cdt))

            den = _kernels.gather(
                self._kern64, t_psf, npad_psf,
                g.pNx64[i], g.pNy64[i], cp.complex128)
            fN = (tr * g.phN[i] * gmask / den
                  * cp.asarray(
                      bundle.hfilt[self.types[i]].ravel()))
            fN = fN * (g.arg2N[i] <= pk2)
            facN.append(fN.astype(self._cdt))

        out = dict(
            facA=facA, facN=facN, grids=g, npix=bundle.npix,
            psf_maxk_pix=bundle.psf_maxk_pix,
        )
        self._factors[key] = out
        while len(self._factors) > self._factor_cache:
            self._factors.popitem(last=False)
        return out

    def _table(self, stamp):
        if self.fp32:
            return _kernels.build_table_fp32(stamp)
        return _kernels.build_table(stamp)

    def metacal_images(self, image, noise, bundle):
        """
        The numerics core: metacal the image with the fusion
        rotated-noise correction, using the psf described by the
        bundle; returns dict type -> float64 image array (the
        final images, not yet packed into observations)
        """
        cp = self.cp
        if image.shape != (self.dim, self.dim):
            raise ValueError(
                f'image shape {image.shape} != engine dim '
                f'{self.dim}'
            )
        if noise.shape != image.shape:
            raise ValueError(
                f'noise shape mismatch, {noise.shape} != '
                f'{image.shape}'
            )
        fac = self._get_factors(bundle)
        g = fac['grids']
        nt = len(self.types)
        npts = g.npts
        npix = fac['npix']
        lo = (npix - self.dim) // 2
        N = self.dim

        T_img, npad, _ = self._table(image)
        T_noi, _, _ = self._table(noise)
        pm = fac['psf_maxk_pix']
        kcut_img = min(
            maxk_scan(T_img, npad, float(np.sum(image)),
                      self._krange_pix, cp), pm)
        kcut_noi = min(
            maxk_scan(T_noi, npad, float(np.sum(noise)),
                      self._krange_pix, cp), pm)

        gA = _kernels.gather(
            self._kern, T_img, npad, g.cAx, g.cAy,
            self._cdt).reshape(nt, npts)
        gN = _kernels.gather(
            self._kern, T_noi, npad, g.cNx, g.cNy,
            self._cdt).reshape(nt, npts)
        khs = cp.empty((nt, npix, npix), dtype=self._cdt)
        for i in range(nt):
            kh = (gA[i] * fac['facA'][i]
                  * (g.arg2A[i] <= kcut_img ** 2)
                  + gN[i] * fac['facN'][i]
                  * (g.arg2N[i] <= kcut_noi ** 2))
            khs[i] = kh.reshape(npix, npix)
        ims = cp.fft.ifft2(khs, axes=(1, 2)).real
        return {
            t: np.ascontiguousarray(
                cp.asnumpy(ims[i])[lo:lo + N, lo:lo + N]
            ).astype(np.float64)
            for i, t in enumerate(self.types)
        }

    def get_bundle(self, obs):
        """a PsfBundle for this observation's psf, LRU-cached on
        the psf content: repeated psfs skip the CPU derivation"""
        psf_image = np.asarray(obs.psf.image, dtype=float)
        key = (psf_image.tobytes(), psf_image.shape)
        if key in self._bundles:
            self._bundles.move_to_end(key)
            return self._bundles[key]
        bundle = PsfBundle(
            psf_image=psf_image,
            wcs=obs.jacobian.get_galsim_wcs(),
            types=self.types, dim=self.dim,
        )
        self._bundles[key] = bundle
        while len(self._bundles) > self._factor_cache:
            self._bundles.popitem(last=False)
        return bundle

    def metacal_obs(self, obs, rng, bundle=None):
        """
        The metacal_obs contract on the gpu: metacal the
        observation (image pass plus filtered rotated-noise pass)
        with its own psf, pack each type into a copy of the input
        observation with the weight rescaled by the
        noise-variance factor and the reconvolution psf installed
        with its dither.  The packaging is the same code as the
        CPU path, so the rng stream matches production draw for
        draw.  When `bundle` is omitted it is derived on the CPU
        from the observation's psf (cached on psf content).
        """
        if not obs.has_psf():
            raise ValueError('observation must have a .psf')
        if not obs.has_noise():
            raise ValueError('observation must have a .noise set')
        if bundle is None:
            bundle = self.get_bundle(obs)
        else:
            if not np.array_equal(
                    np.asarray(obs.psf.image, dtype=float),
                    bundle.psf_image):
                raise ValueError(
                    'observation psf does not match the bundle '
                    'psf'
                )
        _, scale = _diagonal_scale(obs.jacobian.get_galsim_wcs())
        if abs(scale - bundle.scale) > 1.0e-9 * abs(bundle.scale):
            raise ValueError(
                f'observation pixel scale {scale} != bundle '
                f'scale {bundle.scale}'
            )

        images = self.metacal_images(obs.image, obs.noise,
                                     bundle)
        res = MetacalResult(
            images=images,
            psf_image=bundle.target_psf_image,
            noise_var_factor=bundle.noise_var_factor,
        )
        return package_metacal_obs(res=res, obs=obs, rng=rng)

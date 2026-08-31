"""
Geometry-cached fusion-filter solve: the same math as
fusion_filter._common_harmonic_deficits and
_make_fusion_filters_kspace, with everything that depends only on
the grid — the annulus index, the trapezoidal azimuthal weights,
the harmonic basis grids, the interpolation abscissae — computed
once per (npix, scale) and reused for every psf.  The per-psf
work is then a handful of bincounts and elementwise operations
(~5 ms instead of ~65 ms per type set).

Validated against the fusion_filter originals at the 1e-13 level
(tests/test_gpu.py); the reference implementation is untouched.
Diagonal isotropic wcs only (the sky projection is then the
identity), which the gpu engine enforces anyway.
"""
import numpy as np

from ..fusion_filter import _trapz_weights

# runaway guard (host memory, a few MB per (npix, scale)): sane
# workloads see a handful of entries
GEOM_CACHE_SIZE = 16

_GEOM_CACHE = {}


class SolveGeometry:
    """per-(npix, scale) precomputed pieces of the harmonic
    deficit projection, for ms = (2, 6)"""

    def __init__(self, npix, scale, ms=(2, 6)):
        kx = 2 * np.pi * np.fft.fftfreq(npix, d=scale)
        kxg, kyg = np.meshgrid(kx, kx)
        kmag = np.hypot(kxg, kyg)
        theta = np.arctan2(kyg, kxg)
        dk = 2 * np.pi / (npix * scale)
        ibin = np.rint(kmag / dk).astype(int)
        self.nb = ibin.max() + 1
        self.kprof = np.arange(self.nb) * dk
        wgrid = _trapz_weights(theta, ibin, self.nb)

        self.ms = tuple(ms)
        self.kmag = kmag.ravel()
        self.ibin = ibin.ravel()
        self.w0 = wgrid.ravel()
        self.trig = {}
        self.wtrig = {}
        for m in self.ms:
            c = np.cos(m * theta).ravel()
            s = np.sin(m * theta).ravel()
            self.trig[m] = (c, s)
            self.wtrig[m] = (self.w0 * c, self.w0 * s)

    def az_avg(self, x, w):
        return np.bincount(self.ibin, weights=w * x,
                           minlength=self.nb)


def get_geometry(npix, scale):
    key = (int(npix), round(float(scale), 12))
    if key not in _GEOM_CACHE:
        while len(_GEOM_CACHE) >= GEOM_CACHE_SIZE:
            _GEOM_CACHE.pop(next(iter(_GEOM_CACHE)))
        _GEOM_CACHE[key] = SolveGeometry(npix, scale)
    return _GEOM_CACHE[key]


def harmonic_deficits(pts, geom):
    """fusion_filter._common_harmonic_deficits on cached
    geometry; pts values may be flat or (npix, npix)"""
    means, coeffs = {}, {}
    for t, pt in pts.items():
        p = pt.ravel()
        means[t] = geom.az_avg(p, geom.w0)
        coeffs[t] = {}
        for m in geom.ms:
            wc, ws = geom.wtrig[m]
            coeffs[t][m] = (geom.az_avg(p, wc),
                            geom.az_avg(p, ws))

    common_mean = np.maximum.reduce([means[t] for t in pts])
    common_amp = {
        m: np.maximum.reduce(
            [2 * np.hypot(*coeffs[t][m]) for t in pts])
        for m in geom.ms
    }

    out = {}
    for t in pts:
        padd = np.interp(geom.kmag, geom.kprof,
                         common_mean - means[t])
        for m in geom.ms:
            cmk = np.interp(geom.kmag, geom.kprof,
                            coeffs[t][m][0])
            smk = np.interp(geom.kmag, geom.kprof,
                            coeffs[t][m][1])
            amk = np.interp(geom.kmag, geom.kprof,
                            common_amp[m])
            c, s = geom.trig[m]
            padd = padd + (amk - 2 * cmk * c - 2 * smk * s)
        out[t] = padd.clip(min=0)
    return out


def solve_filters(pts, pts_rot, npix, scale, types, ktol=1e-4,
                  full=False):
    """fusion_filter._make_fusion_filters_kspace on cached
    geometry; returns dict type -> (npix, npix) filter"""
    geom = get_geometry(npix, scale)
    deficits = harmonic_deficits(pts, geom)
    out = {}
    for t in types:
        pt = pts_rot[t].ravel()
        eps = ktol * pt.max()
        if full:
            h = pt / (pt + eps)
        else:
            hraw = np.sqrt(np.divide(
                deficits[t], pt,
                out=np.zeros_like(deficits[t]), where=pt > 0,
            ))
            h = np.minimum(hraw, 1.0) * pt / (pt + eps)
        out[t] = h.reshape(npix, npix)
    return out

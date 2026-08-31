"""
The CUDA gather kernel (quintic 6x6 separable interpolation on the
periodic k-table) and the k-table builders.

The quintic xval polynomial coefficients are fitted numerically from
galsim.Quintic() per piece; the function is piecewise degree 5, so
the fit is exact to conditioning (asserted < 1e-13).

Precision doctrine (measured, see the gpu-port document): tables and
factors are built in fp64 and cast once; the fp32 kernel then runs
the per-cell work in single precision.  fp32 image tables may also
be built natively in fp32 (build_table_fp32), which is faster and
safe for the image side.

The CUDA source lives in cuda/gather.cu as a real file;
gather_kernel loads it behind a generated #define prologue carrying
the precision (REAL), the per-precision kernel name (GATHER_NAME,
kept distinct so profiler traces tell fp32 from fp64), and the
fitted coefficient table.  See _load_cuda_source for why local
includes are stripped rather than resolved by nvrtc.
"""
import os
import numpy as np
import cupy as cp


def _fit_quintic():
    import galsim

    q = galsim.Quintic()
    coeffs = []
    for a in [0, 1, 2]:
        t = np.linspace(0.001, 0.999, 401)
        x = a + t
        y = np.array([q.xval(float(v)) for v in x])
        c = np.polynomial.polynomial.polyfit(t, y, 5)
        resid = np.abs(
            np.polynomial.polynomial.polyval(t, c) - y
        ).max()
        assert resid < 1.0e-13, resid
        coeffs.append(c)
    return np.array(coeffs)  # (3 pieces, 6 coeffs), in t = x - piece


_CUDA_DIR = os.path.join(os.path.dirname(__file__), 'cuda')

_CUDA_FILES = ('gather.cu',)


def _load_cuda_source():
    """read and concatenate the kernel sources, stripping the
    tooling-only local-include/#pragma lines (the <cupy/...>
    system include stays: nvrtc resolves it from cupy's own,
    version-pinned headers).  Everything local lands in the one
    source string handed to cupy so its on-disk compile cache
    stays keyed on actual content; a local #include resolved by
    nvrtc at compile time would not be hashed, and editing that
    file would silently reuse the stale cubin"""
    parts = []
    for fname in _CUDA_FILES:
        with open(os.path.join(_CUDA_DIR, fname)) as fobj:
            text = fobj.read()
        keep = [
            ln for ln in text.split('\n')
            if not ln.startswith('#include "')
            and not ln.startswith('#pragma once')
        ]
        parts.append('\n'.join(keep))
    return '\n'.join(parts)


_QCOEF = None
_MODULES = {}


def gather_kernel(fp32):
    """the compiled gather kernel for the requested precision;
    compiled once per process (NVRTC result is also disk-cached by
    cupy)"""
    global _QCOEF
    key = 'f' if fp32 else 'd'
    if key not in _MODULES:
        if _QCOEF is None:
            _QCOEF = _fit_quintic()
        vals = ', '.join(f'{v:.17e}' for v in _QCOEF.ravel())
        prologue = '\n'.join([
            f'#define GATHER_NAME gather_{key}',
            f"#define REAL {'float' if fp32 else 'double'}",
            f'#define QC_VALUES_H {vals}',
            '',
        ])
        src = prologue + _load_cuda_source()
        _MODULES[key] = cp.RawKernel(src, f'gather_{key}')
    return _MODULES[key]


def gather(kern, tab, npad, kx, ky, out_dtype):
    """launch the gather at coordinate arrays kx, ky (device);
    the dk scalar must match the kernel precision"""
    out = cp.empty(kx.size, dtype=out_dtype)
    block = 256
    grid = ((kx.size + block - 1) // block,)
    sdt = np.float32 if out_dtype == cp.complex64 else np.float64
    kern(grid, (block,), (
        tab, np.int32(npad), sdt(2.0 * np.pi / npad),
        kx, ky, np.int64(kx.size), out,
    ))
    return out


def pad_size(n):
    """galsim's pad size for the interpolated-image k table"""
    import galsim

    return galsim.Image.good_fft_size(4 * n)


def build_table(stamp):
    """fp64 device k-table of the rolled, zero-padded stamp;
    returns (table, npad, frac) with frac the fractional center
    offset of the stamp"""
    n = stamp.shape[0]
    npad = pad_size(n)
    r = n // 2
    frac = (n - 1) / 2.0 - r
    pad = cp.zeros((npad, npad), dtype=cp.float64)
    pad[:n, :n] = cp.asarray(np.asarray(stamp, dtype=np.float64))
    tab = cp.fft.fft2(cp.roll(pad, (-r, -r), axis=(0, 1)))
    return tab, npad, frac


def build_table_fp32(stamp):
    """native float32 table build (float32 pad + fft2); faster than
    casting an fp64 build and validated safe for image-side tables"""
    n = stamp.shape[0]
    npad = pad_size(n)
    r = n // 2
    frac = (n - 1) / 2.0 - r
    pad = cp.zeros((npad, npad), dtype=cp.float32)
    pad[:n, :n] = cp.asarray(
        np.asarray(stamp, dtype=np.float32))
    tab = cp.fft.fft2(cp.roll(pad, (-r, -r), axis=(0, 1)))
    return tab, npad, frac

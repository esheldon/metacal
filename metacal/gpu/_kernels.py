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
"""
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


_KERNEL_TEMPLATE = r'''
#include <cupy/complex.cuh>

__device__ inline REAL qxval(REAL x)
{
    x = fabs(x);
    int a = (int)floor(x);
    if (a > 2) return (REAL)0;
    REAL t = x - a;
    const double* c = QC + 6 * a;
    return (REAL)(c[0] + t*(c[1] + t*(c[2] + t*(c[3]
                   + t*(c[4] + t*c[5])))));
}

extern "C" __global__
void gather_SUF(const complex<REAL>* __restrict__ tab,
                const int npad, const REAL dk,
                const REAL* __restrict__ kx,
                const REAL* __restrict__ ky,
                const long n,
                complex<REAL>* __restrict__ out)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    REAL fx = kx[i] / dk;
    REAL fy = ky[i] / dk;
    int jx0 = (int)floor(fx) - 2;
    int jy0 = (int)floor(fy) - 2;
    REAL wx[6], wy[6];
    for (int j = 0; j < 6; j++) {
        wx[j] = qxval(fx - (REAL)(jx0 + j));
        wy[j] = qxval(fy - (REAL)(jy0 + j));
    }
    complex<REAL> acc(0, 0);
    for (int a = 0; a < 6; a++) {
        int iy = (jy0 + a) % npad; if (iy < 0) iy += npad;
        complex<REAL> row(0, 0);
        for (int b = 0; b < 6; b++) {
            int ix = (jx0 + b) % npad; if (ix < 0) ix += npad;
            row += tab[(long)iy * npad + ix] * wx[b];
        }
        acc += row * wy[a];
    }
    out[i] = acc;
}
'''

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
        src = (
            f'__constant__ double QC[18] = {{{vals}}};\n'
            + _KERNEL_TEMPLATE.replace(
                'REAL', 'float' if fp32 else 'double'
            ).replace('SUF', key)
        )
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

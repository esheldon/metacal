// The quintic 6x6 separable gather on the periodic k-table.
//
// The Python loader (metacal/gpu/_kernels.py) prepends a
// generated prologue defining REAL, GATHER_NAME and the fitted
// QC_VALUES_H coefficient table, then hands the concatenation to
// nvrtc.  The <cupy/complex.cuh> include is resolved by nvrtc
// from cupy's own headers and is kept; only local ("...")
// includes are stripped by the loader.

// editor/tooling fallback ONLY: real compiles always get these
// from the generated prologue, which defines them first
#ifndef GATHER_NAME
#define GATHER_NAME gather_d
#define REAL double
#define QC_VALUES_H 0.0
#endif

#include <cupy/complex.cuh>

// the quintic xval polynomial coefficients, fitted numerically
// per piece from galsim.Quintic (see _fit_quintic); always fp64,
// the cell math casts
__constant__ double QC[18] = {QC_VALUES_H};

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
void GATHER_NAME(const complex<REAL>* __restrict__ tab,
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

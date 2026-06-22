"""
the rotated-hybrid noise correction's deficit: the per-type k-space power
``P_add`` that fills each metacal type's m=2/m=6 azimuthal modulation up to a
common (over types) level, so the corrected noise is spin-2 isotropic and the
metacal response is unbiased.

The azimuthal projection uses the rotation-covariant TRAPEZOIDAL (arc-length)
quadrature, and the polar grid is mapped to the SKY frame via the wcs jacobian
before binning -- the two choices that make the correction orientation- and
distortion-correct (see the package README; full derivation in the
small-gauss-tests / mcal_hybrid ``docs/cos4phi-gain-error.tex``).

numpy only.
"""

import numpy as np


def jacobian_matrix(jacobian):
    """the 2x2 pixel->sky jacobian of an ngmix Jacobian in (col, row) = (x, y)
    order, M = [[dudcol, dudrow], [dvdcol, dvdrow]] -- the matrix to pass as
    ``jac`` to ``common_harmonic_deficits`` (a diagonal scale is the identity
    no-op there, so it can always be passed)."""
    return np.array(
        [
            [jacobian.dudcol, jacobian.dudrow],
            [jacobian.dvdcol, jacobian.dvdrow],
        ]
    )


def _trapz_weights(theta, ibin, nb):
    """per-mode azimuthal trapezoidal (Voronoi / arc-length) quadrature weights
    for the annular m-mode projection.

    Within each |k| annulus every mode is weighted by the angular gap it
    occupies, half the gap to each of its two cyclic neighbours, and the
    weights are normalized to sum to 1 over the annulus.  This is the
    trapezoidal rule on the non-uniform ring of FFT modes: it approximates the
    continuum azimuthal average and drives the square-lattice moments <cos
    4theta>, <cos 8theta>, ... (which bias the plain uniform mean and make the
    m=2 projection orientation dependent) down to the trapezoidal error; a
    rotation-covariant projection.

    Pure grid geometry (no power), so it could be precomputed once per grid.
    """
    th = theta.ravel()
    ib = ibin.ravel()
    w = np.zeros(th.size)
    for b in range(nb):
        sel = np.nonzero(ib == b)[0]
        n = sel.size
        if n == 0:
            continue
        if n <= 2:
            # one or two modes: the Voronoi cells are equal -> uniform weight
            w[sel] = 1.0 / n
            continue
        order = np.argsort(th[sel])
        a = th[sel[order]]
        gap = np.empty(n)  # gap to the next mode (cyclic: last wraps +2pi)
        gap[:-1] = np.diff(a)
        gap[-1] = a[0] + 2.0 * np.pi - a[-1]
        cell = 0.5 * (gap + np.roll(gap, 1))  # half the gap on either side
        w[sel[order]] = cell / cell.sum()  # sum(cell)=2pi -> normalized to 1
    return w.reshape(theta.shape)


def common_harmonic_deficits(pts, dim, scale, ms=(2, 6), jac=None):
    """
    the per-type deficit powers P_add that fill each type's m=2 (and m=6)
    modulation up to the COMMON (max over types) amplitude, and raise each
    type's isotropic mean to the common mean.

    This brings noshear/1p/1m to the same noise level (required for the metacal
    response to be unbiased; the metacal shear gives 1p/1m an m=2 even for a
    round PSF, so noshear is filled up to their level) and leaves the total
    power m=2 isotropic.  Only m=2/m=6 (the harmonics the 90-degree rotation
    flips, both lattice-aliasing immune) and the mean are filled; m=4 is left
    alone (rotation-invariant, spin-4, does not bias shear).  Filling harmonics
    rather than (annulus_max - P_t) avoids clipping at the m=4 peaks, which
    would leave a spurious m=2.

    The azimuthal average uses the rotation-covariant TRAPEZOIDAL quadrature
    (``_trapz_weights``): each mode is weighted by its angular Voronoi cell,
    which approximates the continuum integral and removes the square-lattice
    cos(4 phi) gain error of the plain discrete mean (that gain error is the
    rotation leak; it fills the metacal g1/g2 deficits, 45 deg apart, at
    unequal gains -> a response anisotropy).

    The m-mode projection is done in SKY angle via ``jac`` (the pixel->sky
    jacobian): the shear bias is a SKY-frame contraction Tr[Q C], so it is the
    SKY-frame spin-2 that must be cancelled.  A diagonal scale maps to the
    identity (no-op); a pure rotation rotates theta by the wcs angle (absorbed
    by the (c_m, s_m) reconstruction, also a no-op); only a non-conformal shear
    actually changes the projection, where it is the correct target.  So
    ``jac`` can always be passed.

    Parameters
    ----------
    pts: dict of type -> (dim, dim) array
        the per-type power on the full fft2 grid
    dim: int
        grid size (the padded k-grid Np for the k-space metacal)
    scale: float
        pixel scale [arcsec/pixel]
    ms: tuple of int
        harmonics filled to the common amplitude (default (2, 6))
    jac: (2, 2) array, optional
        the pixel->sky jacobian M = scale*R(theta)*S(g) in (col, row) order
        (``wcs.distortion_matrix``).  k_sky = scale * M^{-T} k_pix is used for
        the binning and projection.  Default None: the raw pixel frame (= a
        diagonal wcs).

    Returns
    -------
    dict of type -> (dim, dim) array
    """
    # 1d wavenumbers along one axis: 2*pi*fftfreq gives k in rad/arcsec
    kx = 2 * np.pi * np.fft.fftfreq(dim, d=scale)
    kxg, kyg = np.meshgrid(kx, kx)
    if jac is not None:
        # map the pixel-frame wavenumbers to the SKY frame: a pixel mode
        # exp(i k.x) with sky coords u = M x has sky wavevector M^{-T} k;
        # normalizing out the pixel scale (only the wcs SHAPE matters) gives
        # k_sky = (M/scale)^{-T} k_pix = scale * M^{-T} k_pix.
        minvt = scale * np.linalg.inv(np.asarray(jac, dtype=float)).T
        kxg, kyg = (
            minvt[0, 0] * kxg + minvt[0, 1] * kyg,
            minvt[1, 0] * kxg + minvt[1, 1] * kyg,
        )
    kmag = np.hypot(kxg, kyg)
    theta = np.arctan2(kyg, kxg)
    dk = 2 * np.pi / (dim * scale)
    ibin = np.rint(kmag / dk).astype(int)
    nb = ibin.max() + 1
    kprof = np.arange(nb) * dk

    # the rotation-covariant azimuthal average: each mode weighted by its
    # angular Voronoi cell (sums to 1 per annulus)
    wgrid = _trapz_weights(theta, ibin, nb)

    def az_avg(x):
        return np.bincount(
            ibin.ravel(), weights=(wgrid * x).ravel(), minlength=nb
        )

    # decompose each type's power azimuthally: the m=0 mean profile and the
    # (c_m, s_m) m-mode coefficients per annulus
    means, coeffs = {}, {}
    for t, pt in pts.items():
        means[t] = az_avg(pt)
        coeffs[t] = {}
        for m in ms:
            coeffs[t][m] = (
                az_avg(pt * np.cos(m * theta)),
                az_avg(pt * np.sin(m * theta)),
            )
    # common envelope over types (elementwise max per annulus): we can only ADD
    # power, so fill everyone UP to the tallest mean / m-mode amplitude
    common_mean = np.maximum.reduce([means[t] for t in pts])
    common_amp = {
        m: np.maximum.reduce([2 * np.hypot(*coeffs[t][m]) for t in pts])
        for m in ms
    }

    out = {}
    for t in pts:
        # m=0: raise this type's isotropic level up to the common mean
        padd = np.interp(kmag, kprof, common_mean - means[t])
        for m in ms:
            cmk = np.interp(kmag, kprof, coeffs[t][m][0])
            smk = np.interp(kmag, kprof, coeffs[t][m][1])
            amk = np.interp(kmag, kprof, common_amp[m])
            # cancel P_t's own m-modulation and top up to the common amplitude
            padd += (
                amk - 2 * cmk * np.cos(m * theta) - 2 * smk * np.sin(m * theta)
            )
        # power added through |H|^2 must be non-negative
        out[t] = padd.clip(min=0)
    return out

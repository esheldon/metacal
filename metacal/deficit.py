import numpy as np


def common_harmonic_deficits(pts, dim, scale, ms=(2, 6), jac=None):
    """
    Project the transfer function onto harmonic modes.

    To ensure that all types get the same level of noise, the max amplitude and
    mean is used over types.

    Parameters
    ----------
    pts: dict of type -> (dim, dim) array
        A dict holding the Pt=square of the transfer function for each type.
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
        # map the pixel-frame wavenumbers to the sky frame: a pixel mode
        # exp(i k.x) with sky coords u = M x has sky wavevector M^{-T} k;
        # normalizing out the pixel scale (only the wcs shape matters) gives
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

    # the rotation-covariant azimuthal average. Each mode gets weighted by its
    # angular Voronoi cell (sums to 1 per annulus)
    wgrid = _trapz_weights(theta, ibin, nb)

    def az_avg(x):
        return np.bincount(
            ibin.ravel(), weights=(wgrid * x).ravel(), minlength=nb
        )

    # the raw mode projection and mean for each type

    means, coeffs = {}, {}

    for t, pt in pts.items():
        means[t] = az_avg(pt)
        coeffs[t] = {}
        for m in ms:
            coeffs[t][m] = (
                az_avg(pt * np.cos(m * theta)),
                az_avg(pt * np.sin(m * theta)),
            )

    # Elementwise max per annulus to ensure all metacal types
    # get a common, maximum noise level added
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

            # minus sign to match the rotated noise field
            padd += (
                amk - 2 * cmk * np.cos(m * theta) - 2 * smk * np.sin(m * theta)
            )

        # power added through |H|^2 must be non-negative
        out[t] = padd.clip(min=0)

    return out


def _trapz_weights(theta, ibin, nb):
    """
    per-mode azimuthal trapezoidal (Voronoi / arc-length) quadrature weights
    for the annular m-mode projection.

    Within each |k| annulus every mode is weighted by the angular gap it
    occupies, half the gap to each of its two cyclic neighbours, and the
    weights are normalized to sum to 1 over the annulus.

    This is the trapezoidal rule on the non-uniform ring of FFT modes. It
    approximates the continuum azimuthal average and suppresses the moments
    <cos 4theta>, <cos 8theta>, etc. that bias the straight sum
    over pixels and make the m=2 projection orientation dependent.

    This is just geometry, so it can be precomputed once per grid
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
            # There is only one or two modes, so the Voronoi cells are equal
            w[sel] = 1.0 / n
            continue

        order = np.argsort(th[sel])

        a = th[sel[order]]

        # gap to the next mode (cyclic: last wraps +2pi)
        gap = np.empty(n)

        gap[:-1] = np.diff(a)
        gap[-1] = a[0] + 2.0 * np.pi - a[-1]

        # half the gap on either side
        cell = 0.5 * (gap + np.roll(gap, 1))

        w[sel[order]] = cell / cell.sum()  # sum(cell)=2pi -> normalized to 1

    return w.reshape(theta.shape)

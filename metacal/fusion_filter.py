"""
Metacalibration with noise correction that cancels spin-2 modes induced by by
the metacal process
"""

import numpy as np
import galsim

from .metacalibration import Metacal
from .defaults import DEFAULT_TYPES
from ._util import _wcs_and_matrix


class FusionFilter:
    """
    The fusion noise filter

    Uses the transfer function to create a filter for the full noise field
    that adds the minimal amount of noise to correct the spin-2 modes
    imprinted on the noise by the metacalibration process

    Parameters
    ----------
    psf_image: array
        the psf image
    wcs: galsim WCS
        A galsim local wcs/Jacobian.  May also be given as the 2x2 pixel->sky
        matrix (col, row order) instead of a galsim wcs
    target_psf: A callable that returns a galsim object
        This should be callable with target_psf(psf=psf, flux=flux),
        with psf a galsim object such as galsim.InterpolatedImage.  For
        an example see metacal.AZGauss
    dim: int
        Number of pixels in the image to be corrected.  Used to create the
        impulse that generates the transfer function.
    types: sequence of str
        any of 'noshear', '1p', '1m', '2p', '2m'.  Note that to get the noise
        correction right, you need to send the +/- as well as noshear to be
        processed all together.  This is because they are all needed to
        determine the overall level of noise to be used (max among the types).
    """
    def __call__(
        self,
        psf_image,
        wcs,
        target_psf,
        dim,
        types=DEFAULT_TYPES,
    ):
        wcs, jmat = _wcs_and_matrix(wcs)

        # one shared padded grid for the transfer, the galaxy and the noise
        pts, npix = _impulse_transfer_kspace(
            psf_image=psf_image,
            wcs=wcs,
            target_psf=target_psf,
            dim=dim,
            types=types,
        )
        pts_rot, _ = _impulse_transfer_kspace(
            psf_image=psf_image,
            wcs=wcs,
            target_psf=target_psf,
            dim=dim,
            types=types,
            Np=npix,
            rotation=90 * galsim.degrees,
        )

        scale, _, _, _ = wcs.getDecomposition()
        hfilt = _make_fusion_filters_kspace(
            pts, pts_rot, npix, scale, types, jmat=jmat
        )
        noise_var_factor = _predict_noise_var_factor(
            pts=pts, pts_rot=pts_rot, hfilt=hfilt, types=types,
        )

        return hfilt, noise_var_factor


def _impulse_transfer_kspace(
    psf_image, wcs, target_psf, dim, types, Np=None, rotation=None
):
    """
    the per-type metacal noise transfer P_t = |K_t|^2 on the padded Np x Np
    grid, computed k-natively: push a unit delta impulse through ``Metacal``
    and take |k-array|^2 directly (|fft2(delta)| = 1, so this is relative to a
    white input).  ``Np`` (default galsim's draw size) must match the Metacal
    whose noise this filter cancels.  ``rotation = 90 * galsim.degrees`` builds
    the rotated transfer P_t^rot = |rotateback(metacal(delta))|^2 (the
    corr-field transfer after the sky rotate-back).

    Returns the Np x Np power dict and the Np used.
    """
    delta = np.zeros((dim, dim))
    delta[dim // 2, dim // 2] = 1.0
    km = Metacal(
        image=delta,
        psf_image=psf_image,
        wcs=wcs,
        target_psf=target_psf,
        types=types,
        Np=Np,
        rotation=rotation
    )
    khats = km.get_khats()
    return {t: np.abs(khats[t]) ** 2 for t in types}, km.Np


def _make_fusion_filters_kspace(
    pts, pts_rot, npix, scale, types, ktol=1e-4, jmat=None
):
    """
    the per-type fusion filters H_t (on the padded Np grid), in the final
    (image-metacal) frame; to apply to the sky-rotated, metacal'd correction
    noise (``Metacal(rotation=90*galsim.degrees).get_filtered_images``).

    ``pts`` is the un-rotated image-metacal transfer (the noise to isotropize,
    in its own frame); ``pts_rot`` is the rotated-back corr-field transfer
    (|T_M(R90 k)|^2).  The common-level m=2/m=6 deficit D is built from ``pts``
    (sky-projected, trapz quadrature; ``_common_harmonic_deficits``) and
    divided by ``pts_rot``:

        H_t = min(sqrt(D_t / pts_rot_t), 1) * taper,

    so adding H_t * corr lands the total noise on the common isotropic level.
    The cap keeps the fusion from adding more than the full rotated field
    (fixnoise) in any mode; the taper rolls H smoothly to zero out of band.

    Parameters
    ----------
    pts, pts_rot: dict type -> (Np, Np) array
        the un-rotated and sky-rotated metacal transfers
        (_impulse_transfer_kspace
        with rotation None and 90*galsim.degrees)
    npix: int
        the grid size Np
    scale: float
        pixel scale [arcsec/pixel]
    ktol: float
        out-of-band roll-off relative to the peak power
    jmat: (2, 2) array, optional
        the pixel->sky jacobian for the sky-angle deficit projection (None =
        the pixel frame = a diagonal wcs)
    """
    deficits = _common_harmonic_deficits(pts, npix, scale, jac=jmat)
    out = {}
    for t in types:
        pt = pts_rot[t]
        padd = deficits[t]
        eps = ktol * pt.max()
        hraw = np.sqrt(
            np.divide(padd, pt, out=np.zeros_like(padd), where=pt > 0)
        )
        out[t] = np.minimum(hraw, 1.0) * pt / (pt + eps)
    return out


def _predict_noise_var_factor(pts, pts_rot, hfilt, types):
    """
    the factor by which the per-pixel noise variance increases due to the
    rotated-fusion correction, relative to plain (uncorrected) metacal.

    For a white input field the metacal'd image-noise variance is mean(pts)
    and the added correction variance is mean(H^2 * pts_rot) (both per unit
    input variance, by Parseval on the padded grid; the per-pixel variance of
    a filtered white field is the mean of the squared filter).  The corrected
    variance lands at a common level across types: the deficit fills every
    type's m=0 (azimuthal-mean) power to a common level, and only m=0
    contributes to the total variance (the m=2/m=6 fills integrate to zero over
    each annulus).  So one number suffices; it is anchored on the noshear
    baseline (the science type):

        noise_var_factor = 1 + mean(H_ns^2 * pts_rot_ns) / mean(pts_ns)

    ~1.04 for a round psf, rising toward 2.0 (full fixnoise) as the psf becomes
    more elliptical.
    """
    ref = 'noshear' if 'noshear' in types else list(types)[0]
    v_plain = pts[ref].mean()
    v_added = (hfilt[ref] ** 2 * pts_rot[ref]).mean()
    return 1.0 + v_added / v_plain


def _common_harmonic_deficits(pts, dim, scale, ms=(2, 6), jac=None):
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

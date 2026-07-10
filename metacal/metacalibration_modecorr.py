"""
Metacalibration
"""

import numpy as np
import galsim

from .metacalibration import Metacal
from .deficit import common_harmonic_deficits
from .wcs import galsim_wcs
from .metacal_result import MetacalResult
from .defaults import DEFAULT_TYPES
from ._util import _wcs_and_matrix


def metacal_modecorr(
    image,
    psf_image,
    noise_image,
    wcs,
    target_psf,
    step=0.01,
    types=DEFAULT_TYPES,

):
    """
    metacal an image with spin-2 mode correction of the noise

    Parameters
    ----------
    image: array
        the image to metacal
    psf_image: array
        the psf image
    noise_image: array
        A noise field for noise anisotropy correction. The noise field should
        match the noise in image.
    wcs: galsim WCS
        A galsim local wcs/Jacobian.  May also be given as the 2x2 pixel->sky
        matrix (col, row order) instead of a galsim wcs
    target_psf: A callable that returns a galsim object
        This should be callable with target_psf(psf=psf, flux=flux),
        with psf a galsim object such as galsim.InterpolatedImage.  For
        an example see metacal.AZGauss
    step: float
        metacal shear step
    types: sequence of str
        any of 'noshear', '1p', '1m', '2p', '2m'.  Note that to get the noise
        correction right, you need to send the +/- as well as noshear to be
        processed all together.  This is because they are all needed to
        determine the overall level of noise to be used (max among the types).

    Returns
    -------
    MetacalResult
        Keyed by type to the sheared images (``res['noshear']``) and carries
        ``.psf_image`` (the reconvolution psf) and ``.noise_var_factor`` (the
        per-pixel noise-variance increase from the correction, 1.0 when no
        ``noise_image`` is given).
    """

    if image.shape != noise_image.shape:
        raise ValueError(
            f'noise shape mistmatch, {noise_image.shape} != {image.shape}'
        )

    dim = image.shape[0]
    wcs, jmat = _wcs_and_matrix(wcs)

    # one shared padded grid for the transfer, the galaxy and the noise
    pts, npix = _impulse_transfer_kspace(
        psf_image=psf_image,
        wcs=wcs,
        target_psf=target_psf,
        dim=dim,
        step=step,
        types=types,
    )
    pts_rot, _ = _impulse_transfer_kspace(
        psf_image=psf_image,
        wcs=wcs,
        target_psf=target_psf,
        dim=dim,
        step=step,
        types=types,
        Np=npix,
        rotation=90 * galsim.degrees,
    )

    scale, _, _, _ = wcs.getDecomposition()
    hfilt = _make_hybrid_filters_kspace(
        pts, pts_rot, npix, scale, types, jmat=jmat
    )

    mcal = Metacal(
        image=image,
        psf_image=psf_image,
        wcs=wcs,
        target_psf=target_psf,
        step=step,
        types=types,
        Np=npix,
    )
    mcal_images = mcal.get_images()

    # the sky-rotated correction field, metacal'd, filtered,
    # and rotated back.  already in the final frame
    mcal_noise = Metacal(
        noise_image,
        psf_image,
        wcs,
        target_psf=target_psf,
        step=step,
        types=types,
        Np=npix,
        rotation=90 * galsim.degrees,
    )
    mcal_noise_images = mcal_noise.get_filtered_images(hfilt)

    odict = {
        t: mcal_images[t] + mcal_noise_images[t]
        for t in types
    }
    return MetacalResult(
        images=odict,
        psf_image=mcal.get_target_psf_image(),
        noise_var_factor=_predict_noise_var_factor(pts, pts_rot, hfilt, types),
    )


def _predict_noise_var_factor(pts, pts_rot, hfilt, types):
    """
    the factor by which the per-pixel noise variance increases due to the
    rotated-hybrid correction, relative to plain (uncorrected) metacal.

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


def _impulse_transfer_kspace(
    psf_image, wcs, target_psf, dim, step, types, Np=None, rotation=None
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
        step=step,
        types=types,
        Np=Np,
        rotation=rotation
    )
    khats = km.get_khats()
    return {t: np.abs(khats[t]) ** 2 for t in types}, km.Np


def _make_hybrid_filters_kspace(
    pts, pts_rot, npix, scale, types, ktol=1e-4, jmat=None
):
    """
    the per-type hybrid filters H_t (on the padded Np grid), in the final
    (image-metacal) frame; to apply to the sky-rotated, metacal'd correction
    noise (``Metacal(rotation=90*galsim.degrees).get_filtered_images``).

    ``pts`` is the un-rotated image-metacal transfer (the noise to isotropize,
    in its own frame); ``pts_rot`` is the rotated-back corr-field transfer
    (|T_M(R90 k)|^2).  The common-level m=2/m=6 deficit D is built from ``pts``
    (sky-projected, trapz quadrature; ``common_harmonic_deficits``) and
    divided by ``pts_rot``:

        H_t = min(sqrt(D_t / pts_rot_t), 1) * taper,

    so adding H_t * corr lands the total noise on the common isotropic level.
    The cap keeps the hybrid from adding more than the full rotated field
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
    deficits = common_harmonic_deficits(pts, npix, scale, jac=jmat)
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

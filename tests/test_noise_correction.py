"""
tests for the high-level metacal with hybrid correction
"""

import numpy as np
import galsim
import pytest

from metacal import metacal_image
from metacal import AZGauss
from metacal.metacalibration import (
    Metacal,
    _impulse_transfer_kspace,
    _make_hybrid_filters_kspace,
)
from metacal.wcs import distortion_matrix, galsim_wcs

DIM = 48
SCALE = 0.2
STEP = 0.01
TYPES = ['noshear', '1p', '1m', '2p', '2m']


def _scene(theta=0.0, g1=0.0, g2=0.0, psf_g1=0.03):
    """
    psf image, a noisy galaxy image, a separate noise field, and the wcs
    """
    wcs = galsim_wcs(
        distortion_matrix(SCALE, theta=theta * galsim.degrees, g1=g1, g2=g2)
    )
    dkw = {'wcs': wcs}
    psf = galsim.Moffat(beta=2.5, fwhm=0.9).shear(g1=psf_g1)
    psf_im = psf.drawImage(nx=DIM, ny=DIM, **dkw).array
    gal = (
        galsim.Convolve(
            galsim.Exponential(half_light_radius=0.5).shear(g1=0.02), psf
        )
        .drawImage(nx=DIM, ny=DIM, **dkw)
        .array
    )
    rng = np.random.RandomState(42)
    sigma = np.sqrt((gal**2).sum()) / 15.0
    image = gal + rng.normal(scale=sigma, size=gal.shape)
    noise = rng.normal(scale=sigma, size=gal.shape)
    return psf_im, image, noise, wcs


@pytest.mark.parametrize(
    'wcs_kw',
    [
        dict(),  # diagonal
        dict(theta=10.0, g1=0.01, g2=-0.005),  # distorted (rot + small shear)
    ],
)
def test_metacal_hybrid_runs(wcs_kw):
    """
    returns finite corrected images for every type, for a
    diagonal and a distorted wcs, including the full 5-type set
    """
    psf_image, image, noise, wcs = _scene(**wcs_kw)
    res = metacal_image(
        image=image,
        psf_image=psf_image,
        noise_image=noise,
        wcs=wcs,
        target_psf=AZGauss(),
        types=TYPES,
    )
    assert set(res) == set(TYPES)
    for t in TYPES:
        assert res[t].shape == (DIM, DIM)
        assert np.all(np.isfinite(res[t]))
    assert res.psf_image.shape == (DIM, DIM)


def test_wcs_can_be_a_matrix():
    """
    the wcs may be given as the 2x2 pixel->sky matrix instead of a galsim wcs
    """
    psf_image, image, noise, wcs = _scene(theta=10.0, g1=0.01, g2=-0.005)
    M = distortion_matrix(SCALE, theta=10 * galsim.degrees, g1=0.01, g2=-0.005)
    a = metacal_image(
        image=image,
        psf_image=psf_image,
        noise_image=noise,
        wcs=wcs,
        target_psf=AZGauss(),
    )
    b = metacal_image(
        image=image,
        psf_image=psf_image,
        noise_image=noise,
        wcs=M,
        target_psf=AZGauss(),
    )
    for t in a:
        assert np.array_equal(a[t], b[t])
    assert a.noise_var_factor == b.noise_var_factor


def test_metacal_convenience():
    """
    the plain metacal_image() convenience equals Metacal(...).get_images()
    """
    psf_image, image, noise, wcs = _scene()
    a = metacal_image(
        image=image,
        psf_image=psf_image,
        noise_image=None,
        wcs=wcs,
        target_psf=AZGauss(),
    )
    b = Metacal(
        image=image,
        psf_image=psf_image,
        wcs=wcs,
        target_psf=AZGauss(),
    ).get_images()
    for t in a:
        assert np.array_equal(a[t], b[t])


def test_hybrid_adds_only_the_filtered_deficit():
    target_psf = AZGauss()
    psf_image, image, noise, wcs = _scene()
    plain = metacal_image(
        image=image,
        psf_image=psf_image,
        noise_image=None,
        wcs=wcs,
        target_psf=target_psf,
    )
    hyb = metacal_image(
        image=image,
        psf_image=psf_image,
        noise_image=noise,
        wcs=wcs,
        target_psf=target_psf,
    )
    # the full counter-rotated metacal'd noise fixnoise would add
    full = Metacal(
        image=noise,
        psf_image=psf_image,
        wcs=wcs,
        target_psf=target_psf,
        rotation=90 * galsim.degrees
    ).get_images()
    for t in plain:
        corr = hyb[t] - plain[t]
        assert np.all(np.isfinite(corr))
        assert not np.allclose(corr, 0.0)
        assert corr.var() < full[t].var()


def test_noise_var_factor_is_one_without_correction():
    """
    plain metacal applies no correction -> the factor is exactly 1.0
    """
    psf_image, image, noise, wcs = _scene()
    assert metacal_image(
        image=image,
        psf_image=psf_image,
        noise_image=None,
        wcs=wcs,
        target_psf=AZGauss(),
    ).noise_var_factor == 1.0


def test_noise_var_factor_grows_with_ellipticity():
    """
    minimal for a round psf (well under fixnoise's factor of 2), and larger for
    an elliptical psf
    """
    target_psf = AZGauss()

    psf_r, im_r, nz_r, wcs = _scene(psf_g1=0.0)
    psf_e, im_e, nz_e, wcs = _scene(psf_g1=0.10)
    fac_round = metacal_image(
        image=im_r,
        psf_image=psf_r,
        noise_image=nz_r,
        wcs=wcs,
        target_psf=target_psf,
    ).noise_var_factor
    fac_ell = metacal_image(
        image=im_e,
        psf_image=psf_e,
        noise_image=nz_e,
        wcs=wcs,
        target_psf=target_psf,
    ).noise_var_factor
    assert 1.0 < fac_round < 1.15
    assert fac_round < fac_ell < 2.0


def test_noise_var_factor_predicts_variance_ratio():
    """
    the predicted noise_var_factor matches the measured ratio of the corrected
    to the plain metacal'd noise variance over white-noise realizations
    """
    target_psf = AZGauss()

    psf_image, image, noise, wcs = _scene(psf_g1=0.08)
    pred = metacal_image(
        image=image,
        psf_image=psf_image,
        noise_image=noise,
        wcs=wcs,
        target_psf=target_psf,
    ).noise_var_factor

    # the shared transfers and hybrid filter (the diagonal scene needs no
    # sky-angle projection, so jmat defaults to the pixel frame)
    types = ['noshear']
    pts, Np = _impulse_transfer_kspace(
        psf_image=psf_image,
        wcs=wcs,
        target_psf=target_psf,
        dim=DIM,
        step=STEP,
        types=types,
    )
    pts_rot, _ = _impulse_transfer_kspace(
        psf_image=psf_image,
        wcs=wcs,
        target_psf=target_psf,
        dim=DIM,
        step=STEP,
        types=types,
        Np=Np,
        rotation=90 * galsim.degrees
    )
    hfilt = _make_hybrid_filters_kspace(pts, pts_rot, Np, SCALE, types)

    # E[var(plain + correction)] / E[var(plain)] in the central region
    b = 10
    sl = slice(b, DIM - b)
    rng = np.random.RandomState(99)
    vplain = vcorr = 0.0
    nreal = 60
    for _ in range(nreal):
        n1 = rng.normal(size=(DIM, DIM))
        n2 = rng.normal(size=(DIM, DIM))
        plain = Metacal(
            image=n1,
            psf_image=psf_image,
            wcs=wcs,
            target_psf=target_psf,
            step=STEP,
            types=types,
            Np=Np
        ).get_images()['noshear']
        corr = Metacal(
            image=n2,
            psf_image=psf_image,
            wcs=wcs,
            target_psf=target_psf,
            step=STEP,
            types=types,
            Np=Np,
            rotation=90 * galsim.degrees,
        ).get_filtered_images(hfilt)['noshear']
        vplain += plain[sl, sl].var()
        vcorr += (plain + corr)[sl, sl].var()
    measured = vcorr / vplain
    assert abs(measured - pred) / pred < 0.05

"""
tests for metacal (metacal.metacal).
"""

import numpy as np
import galsim
import pytest

from metacal import AZGauss
from metacal.metacalibration import Metacal
from metacal.metacalibration_modecorr import (
    _impulse_transfer_kspace,
    _make_hybrid_filters_kspace,
)
from metacal.wcs import distortion_matrix, galsim_wcs

DIM = 48
SCALE = 0.2
STEP = 0.01
TYPES = ['noshear', '1p', '1m']


def _build(theta):
    """psf image, galaxy image and the galsim wcs for a diagonal
    (theta None) or pure-rotation wcs"""
    if theta is None:
        wcs, dkw = (
            galsim.JacobianWCS(SCALE, 0, 0, SCALE),
            {'scale': SCALE},
        )
    else:
        jacmat = distortion_matrix(SCALE, theta=theta * galsim.degrees)
        wcs = galsim_wcs(jacmat)
        dkw = {'wcs': wcs}

    psf = galsim.Moffat(beta=2.5, fwhm=0.9)
    psf_image = psf.drawImage(nx=DIM, ny=DIM, **dkw).array
    gal_im = (
        galsim.Convolve(
            galsim.Exponential(half_light_radius=0.5).shear(g1=0.02), psf
        )
        .drawImage(nx=DIM, ny=DIM, **dkw)
        .array
    )

    return psf_image, gal_im, wcs


@pytest.mark.parametrize('theta', [None, 45.0])
def test_delta_transfer_padded(theta):
    """the transfer is on galsim's padded draw grid Np (>= the stamp), finite
    and non-negative"""
    psf_image, gal_im, wcs = _build(theta)
    pts, Np = _impulse_transfer_kspace(
        psf_image=psf_image,
        wcs=wcs,
        target_psf=AZGauss(),
        dim=DIM,
        step=STEP,
        types=TYPES,
    )
    assert Np >= DIM and Np % 2 == 0
    for t in TYPES:
        assert pts[t].shape == (Np, Np)
        assert np.all(np.isfinite(pts[t])) and pts[t].min() >= 0.0


def test_rotation_none_is_noop():
    """
    rotation=None (the default) is the identity: byte-for-byte the default
    """
    psf_image, gal_im, wcs = _build(None)
    noise = np.random.RandomState(7).normal(size=(DIM, DIM))
    a = Metacal(
        image=noise,
        psf_image=psf_image,
        wcs=wcs,
        target_psf=AZGauss(),
        step=STEP,
        types=TYPES,
    ).get_images()
    b = Metacal(
        image=noise,
        psf_image=psf_image,
        wcs=wcs,
        target_psf=AZGauss(),
        step=STEP,
        types=TYPES,
        rotation=None
    ).get_images()
    for t in TYPES:
        assert np.array_equal(a[t], b[t])


def test_rotation_requires_angle():
    """rotation must be a galsim.Angle, not a bare number"""
    psf_image, gal_im, wcs = _build(None)
    noise = np.random.RandomState(7).normal(size=(DIM, DIM))
    with pytest.raises(TypeError):
        Metacal(
            image=noise,
            psf_image=psf_image,
            wcs=wcs,
            target_psf=AZGauss(),
            step=STEP,
            types=TYPES,
            rotation=90.0,
        )
    # a galsim.Angle is accepted
    Metacal(
        image=noise,
        psf_image=psf_image,
        wcs=wcs,
        target_psf=AZGauss(),
        step=STEP,
        types=TYPES,
        rotation=90 * galsim.degrees,
    )


@pytest.mark.parametrize('theta', [None, 30.0])
def test_sky_rot_conformal_matches_pixel(theta):
    """
    for a CONFORMAL wcs (diagonal or pure rotation) the sky 90 rotation
    reproduces the pixel np.rot90 correction field to interpolation precision
    """
    psf_image, gal_im, wcs = _build(theta)
    _, NP = _impulse_transfer_kspace(
        psf_image=psf_image,
        wcs=wcs,
        target_psf=AZGauss(),
        dim=DIM,
        step=STEP,
        types=TYPES,
    )
    n2 = np.random.RandomState(11).normal(size=(DIM, DIM))
    pix = {
        t: np.rot90(
            Metacal(
                image=np.rot90(n2, 1),
                psf_image=psf_image,
                wcs=wcs,
                target_psf=AZGauss(),
                step=STEP,
                types=TYPES,
                Np=NP
            ).get_images()[t],
            3,
        )
        for t in TYPES
    }
    sky = Metacal(
        image=n2,
        psf_image=psf_image,
        wcs=wcs,
        target_psf=AZGauss(),
        step=STEP,
        types=TYPES,
        Np=NP,
        rotation=90 * galsim.degrees,
    ).get_images()
    for t in TYPES:
        assert np.abs(sky[t] - pix[t]).max() / np.abs(pix[t]).max() < 5e-3


def test_sky_rot_diverges_under_distortion():
    """
    under a NON-conformal wcs the sky rotation must DIFFER materially from the
    pixel np.rot90 (else it could not fix the additive c)
    """
    M = distortion_matrix(SCALE, g1=0.10, g2=0.05)
    wcs = galsim_wcs(M)
    psf_image = (
        galsim.Moffat(beta=2.5, fwhm=0.9)
        .drawImage(nx=DIM, ny=DIM, wcs=wcs)
        .array
    )
    _, NP = _impulse_transfer_kspace(
        psf_image=psf_image,
        wcs=wcs,
        target_psf=AZGauss(),
        dim=DIM,
        step=STEP,
        types=TYPES,
    )
    n2 = np.random.RandomState(11).normal(size=(DIM, DIM))
    pix = np.rot90(
        Metacal(
            image=np.rot90(n2, 1),
            psf_image=psf_image,
            wcs=wcs,
            target_psf=AZGauss(),
            step=STEP,
            types=TYPES,
            Np=NP
        ).get_images()['noshear'],
        3,
    )
    sky = Metacal(
        image=n2,
        psf_image=psf_image,
        wcs=wcs,
        target_psf=AZGauss(),
        step=STEP,
        types=TYPES,
        Np=NP,
        rotation=90 * galsim.degrees,
    ).get_images()['noshear']
    assert np.abs(sky - pix).max() / np.abs(pix).max() > 0.05


def test_kspace_filters_finite_and_capped():
    """the hybrid filters (padded Np grid) are real, finite and capped at 1"""
    psf_image, gal_im, wcs = _build(45.0)
    jmat = distortion_matrix(SCALE, theta=45 * galsim.degrees)
    pts, Np = _impulse_transfer_kspace(
        psf_image=psf_image,
        wcs=wcs,
        target_psf=AZGauss(),
        dim=DIM,
        step=STEP,
        types=TYPES,
    )
    pts_rot, _ = _impulse_transfer_kspace(
        psf_image=psf_image,
        wcs=wcs,
        target_psf=AZGauss(),
        dim=DIM,
        step=STEP,
        types=TYPES,
        Np=Np,
        rotation=90 * galsim.degrees
    )
    G = _make_hybrid_filters_kspace(pts, pts_rot, Np, SCALE, TYPES, jmat=jmat)
    for t in TYPES:
        assert G[t].shape == (Np, Np)
        assert np.all(np.isfinite(G[t]))
        assert G[t].min() >= 0.0 and G[t].max() <= 1.0 + 1e-12

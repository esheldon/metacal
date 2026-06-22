"""tests for the high-level metacal / metacal_hybrid entry points."""
import numpy as np
import galsim
import pytest

from metacal import metacal, metacal_hybrid, KMetacal
from metacal.wcs import distortion_matrix, galsim_wcs

DIM = 48
SCALE = 0.2
TYPES5 = ['noshear', '1p', '1m', '2p', '2m']


def _scene(theta=0.0, g1=0.0, g2=0.0):
    """psf image, a noisy galaxy image, a separate noise field, and the wcs"""
    wcs = galsim_wcs(distortion_matrix(SCALE, theta_deg=theta, g1=g1, g2=g2))
    dkw = {'wcs': wcs}
    psf = galsim.Moffat(beta=2.5, fwhm=0.9).shear(g1=0.03)
    psf_im = psf.drawImage(nx=DIM, ny=DIM, **dkw).array
    gal = galsim.Convolve(
        galsim.Exponential(half_light_radius=0.5).shear(g1=0.02), psf
    ).drawImage(nx=DIM, ny=DIM, **dkw).array
    rng = np.random.RandomState(42)
    sigma = np.sqrt((gal**2).sum()) / 15.0
    image = gal + rng.normal(scale=sigma, size=gal.shape)
    noise = rng.normal(scale=sigma, size=gal.shape)
    return psf_im, image, noise, wcs


@pytest.mark.parametrize('wcs_kw', [
    dict(),                                     # diagonal
    dict(theta=10.0, g1=0.01, g2=-0.005),       # distorted (rot + small shear)
])
def test_metacal_hybrid_runs(wcs_kw):
    """metacal_hybrid returns finite corrected images for every type, for a
    diagonal and a distorted wcs, including the full 5-type set"""
    psf_im, image, noise, wcs = _scene(**wcs_kw)
    mcal = metacal_hybrid(image, psf_im, wcs, noise, types=TYPES5)
    assert set(mcal) == set(TYPES5)
    for t in TYPES5:
        assert mcal[t].shape == (DIM, DIM)
        assert np.all(np.isfinite(mcal[t]))


def test_metacal_plain_matches_kmetacal():
    """the plain metacal() convenience equals KMetacal(...).get_images()"""
    psf_im, image, noise, wcs = _scene()
    a = metacal(image, psf_im, wcs)
    b = KMetacal(image, psf_im, wcs).get_images()
    for t in a:
        assert np.array_equal(a[t], b[t])


def test_hybrid_adds_only_the_filtered_deficit():
    """metacal_hybrid = plain metacal + a correction field, and that correction
    has LOWER variance than the full (fixnoise) rotated metacal'd noise -- the
    point of the hybrid (it adds only the anisotropic deficit, not the whole
    field)"""
    psf_im, image, noise, wcs = _scene()
    plain = metacal(image, psf_im, wcs)
    hyb = metacal_hybrid(image, psf_im, wcs, noise)
    # the full counter-rotated metacal'd noise fixnoise would add
    full = KMetacal(noise, psf_im, wcs, rotation=90 * galsim.degrees).get_images()
    for t in plain:
        corr = hyb[t] - plain[t]
        assert np.all(np.isfinite(corr))
        assert not np.allclose(corr, 0.0)
        assert corr.var() < full[t].var()

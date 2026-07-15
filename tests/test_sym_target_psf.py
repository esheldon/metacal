"""
tests of the Symmetrize target psf
"""
import numpy as np
import galsim
import pytest

from metacal import Symmetrize
from metacal import metacal_image
from metacal.metacalibration import LANCZOS, SHEAR_STEP

pytest.importorskip('ngmix')

DIM = 48
PSF_DIM = 33
SCALE = 0.2
TYPES = ('noshear', '1p', '1m')


def _build_psf(e1=0.05, e2=0.0):
    psf = galsim.Moffat(beta=2.5, fwhm=0.9).shear(e1=e1, e2=e2)
    psf_im = psf.drawImage(nx=PSF_DIM, ny=PSF_DIM, scale=SCALE)
    psf_int = galsim.InterpolatedImage(psf_im, x_interpolant=LANCZOS)
    return psf, psf_im.array, psf_int


def _measure_e1(arr):
    im = galsim.Image(arr, scale=SCALE)
    return galsim.hsm.FindAdaptiveMom(im).observed_shape.e1


@pytest.mark.parametrize('nrot', [4, 8])
def test_symmetrize_target_is_round(nrot):
    """
    the target must carry no spin-2, whatever the psf ellipticity
    """
    _, psf_im, psf_int = _build_psf(e1=0.05, e2=0.02)

    rng = np.random.RandomState(991)
    target = Symmetrize(nrot=nrot, rng=rng)(
        psf_int, flux=psf_im.sum(),
    )

    tim = target.drawImage(
        nx=PSF_DIM, ny=PSF_DIM, scale=SCALE, method='no_pixel',
    )
    mom = galsim.hsm.FindAdaptiveMom(tim)
    assert abs(mom.observed_shape.e1) < 1.0e-4
    assert abs(mom.observed_shape.e2) < 1.0e-4

    # flux preserved
    assert np.allclose(target.flux, psf_im.sum())


def test_symmetrize_target_size():
    """
    the target sits at the (1 + e/2) dilation of the psf size, the
    fundamental floor for a round non-amplifying target
    """
    _, psf_im, psf_int = _build_psf(e1=0.05)
    flux = psf_im.sum()

    mom = galsim.hsm.FindAdaptiveMom(galsim.Image(psf_im, scale=SCALE))
    e = np.hypot(mom.observed_shape.e1, mom.observed_shape.e2)
    sig0 = mom.moments_sigma

    rng = np.random.RandomState(992)
    target = Symmetrize(nrot=4, rng=rng)(psf_int, flux=flux)
    tim = target.drawImage(
        nx=PSF_DIM, ny=PSF_DIM, scale=SCALE, method='no_pixel',
    )
    sig = galsim.hsm.FindAdaptiveMom(tim).moments_sigma

    assert abs(sig / sig0 - (1 + e / 2)) < 0.01


def test_symmetrize_leakage():
    """
    round galaxy, elliptical psf: the calibrated additive term
    c1 = e1(noshear)/R11 must vanish
    """
    psf, psf_im, psf_int = _build_psf(e1=0.05)

    gal_im = galsim.Convolve(
        galsim.Exponential(half_light_radius=0.5), psf,
    ).drawImage(nx=DIM, ny=DIM, scale=SCALE).array

    res = metacal_image(
        image=gal_im,
        psf_image=psf_im,
        wcs=np.diag([SCALE, SCALE]),
        target_psf=Symmetrize(nrot=4, rng=np.random.RandomState(993)),
        types=list(TYPES),
    )

    e1 = {t: _measure_e1(res[t]) for t in TYPES}
    r11 = (e1['1p'] - e1['1m']) / (2 * SHEAR_STEP)
    c1 = e1['noshear'] / r11

    assert r11 > 0.5
    assert abs(c1) < 5.0e-5


def test_symmetrize_bad_nrot():
    with pytest.raises(ValueError):
        Symmetrize(nrot=3, rng=np.random.RandomState(994))

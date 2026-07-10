"""
compare to ngmix
"""
import numpy as np
import galsim
import pytest

from metacal import AZGauss
from metacal import metacal_image
from metacal.wcs import distortion_matrix, galsim_wcs

ngmix = pytest.importorskip('ngmix')

DIM = 48
SCALE = 0.2
STEP = 0.01
TYPES = ['noshear', '1p', '1m']


def _build(theta):
    """psf image, galaxy image, ngmix obs and the galsim wcs for a diagonal
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
    psf_im = psf.drawImage(nx=DIM, ny=DIM, **dkw).array
    gal_im = (
        galsim.Convolve(
            galsim.Exponential(half_light_radius=0.5).shear(g1=0.02), psf
        )
        .drawImage(nx=DIM, ny=DIM, **dkw)
        .array
    )

    return psf_im, gal_im, wcs


def _build_obs(theta):
    from metacal.wcs import ngmix_jacobian

    psf_im, gal_im, wcs = _build(theta)

    jacmat = wcs.getMatrix()

    cen = (DIM - 1.0) / 2.0
    jac = ngmix_jacobian(cen, cen, jacmat)
    psf_obs = ngmix.Observation(
        image=psf_im, weight=np.ones_like(psf_im), jacobian=jac
    )
    return ngmix.Observation(
        image=gal_im, weight=np.ones_like(gal_im), jacobian=jac, psf=psf_obs
    )


@pytest.mark.parametrize('theta', [None, 30.0, 45.0])
def test_metacal_matches_ngmix(theta):
    """
    k-native metacal field equals ngmix's drawn metacal to ~machine precision,
    edge included, for diagonal/rotated wcs and a galaxy AND a noise field
    """

    obs = _build_obs(theta)
    obs.noise = obs.image * 0

    # ngmix metacal without noise correction
    ngmix_odict = ngmix.metacal.get_all_metacal(
        obs,
        psf='azgauss',
        step=STEP,
        rng=None,
        types=TYPES,
        fixnoise=False,
    )

    res = metacal_image(
        obs.image,
        psf_image=obs.psf.image,
        wcs=obs.jacobian.get_galsim_wcs(),
        target_psf=AZGauss(),
        types=TYPES,
    )

    for t in TYPES:
        ngmix_image = ngmix_odict[t].image
        image = res[t]

        msqdiff = np.sqrt(((ngmix_image - image) ** 2).mean())
        sq = np.sqrt((ngmix_image**2).mean())
        assert msqdiff / sq < 1e-6

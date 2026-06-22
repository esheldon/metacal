"""
tests for the k-space metacal (metacal.kmetacal).

The key correctness check is that with galsim's own padding the k-native metacal
field EQUALS ngmix's real-space drawn metacal to ~machine precision (it
replicates galsim's drawFFT), so it adds no edge/aliasing artifact; the intended
difference is only that the transfer and the hybrid filter live on the padded
k-grid rather than a second small fft.
"""
import numpy as np
import galsim
import ngmix
import pytest

from metacal.kmetacal import (
    KMetacal, delta_transfer_kspace, make_hybrid_filters_kspace,
)
from metacal.wcs import distortion_matrix, galsim_wcs, ngmix_jacobian

DIM = 48
SCALE = 0.2
STEP = 0.01
TYPES = ['noshear', '1p', '1m']


def _noise_obs(obs, noise):
    new = obs.copy()
    new.image = noise
    return new


def _build(theta):
    """psf image, galaxy image, ngmix obs and the galsim wcs for a diagonal
    (theta None) or pure-rotation wcs"""
    if theta is None:
        jacmat, wcs, dkw = None, galsim.JacobianWCS(SCALE, 0, 0, SCALE), \
            {'scale': SCALE}
    else:
        jacmat = distortion_matrix(SCALE, theta_deg=theta)
        wcs = galsim_wcs(jacmat)
        dkw = {'wcs': wcs}
    psf = galsim.Moffat(beta=2.5, fwhm=0.9)
    psf_im = psf.drawImage(nx=DIM, ny=DIM, **dkw).array
    gal_im = galsim.Convolve(
        galsim.Exponential(half_light_radius=0.5).shear(g1=0.02), psf
    ).drawImage(nx=DIM, ny=DIM, **dkw).array

    cen = (DIM - 1.0) / 2.0
    jac = (ngmix.DiagonalJacobian(row=cen, col=cen, scale=SCALE)
           if jacmat is None else ngmix_jacobian(cen, cen, jacmat))
    psf_obs = ngmix.Observation(
        image=psf_im, weight=np.ones_like(psf_im), jacobian=jac)
    obs = ngmix.Observation(
        image=gal_im, weight=np.ones_like(gal_im), jacobian=jac, psf=psf_obs)
    return psf_im, gal_im, obs, wcs


@pytest.mark.parametrize('theta', [None, 30.0, 45.0])
def test_kmetacal_matches_ngmix(theta):
    """k-native metacal field EQUALS ngmix's drawn metacal to ~machine precision,
    edge included, for diagonal/rotated wcs and a galaxy AND a noise field"""
    psf_im, gal_im, obs, wcs = _build(theta)
    rng = np.random.RandomState(3)
    noise = rng.normal(size=(DIM, DIM))
    for img, mkobs in [(gal_im, obs), (noise, _noise_obs(obs, noise))]:
        od = ngmix.metacal.get_all_metacal(
            mkobs, psf='azgauss', step=STEP, rng=None, types=TYPES,
            fixnoise=False)
        ki = KMetacal(img, psf_im, wcs, step=STEP, types=TYPES).get_images()
        for t in TYPES:
            a, b = od[t].image, ki[t]
            assert np.sqrt(((a - b)**2).mean()) / np.sqrt((a**2).mean()) < 1e-6


@pytest.mark.parametrize('theta', [None, 45.0])
def test_delta_transfer_padded(theta):
    """the transfer is on galsim's padded draw grid Np (>= the stamp), finite
    and non-negative"""
    psf_im, gal_im, obs, wcs = _build(theta)
    pts, Np = delta_transfer_kspace(psf_im, wcs, DIM, STEP, TYPES)
    assert Np >= DIM and Np % 2 == 0
    for t in TYPES:
        assert pts[t].shape == (Np, Np)
        assert np.all(np.isfinite(pts[t])) and pts[t].min() >= 0.0


def test_rotation_none_is_noop():
    """rotation=None (the default) is the identity: byte-for-byte the default"""
    psf_im, gal_im, obs, wcs = _build(None)
    noise = np.random.RandomState(7).normal(size=(DIM, DIM))
    a = KMetacal(noise, psf_im, wcs, step=STEP, types=TYPES).get_images()
    b = KMetacal(noise, psf_im, wcs, step=STEP, types=TYPES,
                 rotation=None).get_images()
    for t in TYPES:
        assert np.array_equal(a[t], b[t])


def test_rotation_requires_angle():
    """rotation must be a galsim.Angle, not a bare number"""
    psf_im, gal_im, obs, wcs = _build(None)
    noise = np.random.RandomState(7).normal(size=(DIM, DIM))
    with pytest.raises(TypeError):
        KMetacal(noise, psf_im, wcs, step=STEP, types=TYPES, rotation=90.0)
    # a galsim.Angle is accepted
    KMetacal(noise, psf_im, wcs, step=STEP, types=TYPES,
             rotation=90 * galsim.degrees)


@pytest.mark.parametrize('theta', [None, 30.0])
def test_sky_rot_conformal_matches_pixel(theta):
    """for a CONFORMAL wcs (diagonal or pure rotation) the sky 90 rotation
    reproduces the pixel np.rot90 correction field to interpolation precision"""
    psf_im, gal_im, obs, wcs = _build(theta)
    _, NP = delta_transfer_kspace(psf_im, wcs, DIM, STEP, TYPES)
    n2 = np.random.RandomState(11).normal(size=(DIM, DIM))
    pix = {t: np.rot90(KMetacal(np.rot90(n2, 1), psf_im, wcs, step=STEP,
                                types=TYPES, Np=NP).get_images()[t], 3)
           for t in TYPES}
    sky = KMetacal(n2, psf_im, wcs, step=STEP, types=TYPES, Np=NP,
                   rotation=90 * galsim.degrees).get_images()
    for t in TYPES:
        assert np.abs(sky[t] - pix[t]).max() / np.abs(pix[t]).max() < 5e-3


def test_sky_rot_diverges_under_distortion():
    """under a NON-conformal wcs the sky rotation must DIFFER materially from the
    pixel np.rot90 (else it could not fix the additive c)"""
    M = distortion_matrix(SCALE, theta_deg=0.0, g1=0.10, g2=0.05)
    wcs = galsim_wcs(M)
    psf_im = galsim.Moffat(beta=2.5, fwhm=0.9).drawImage(
        nx=DIM, ny=DIM, wcs=wcs).array
    _, NP = delta_transfer_kspace(psf_im, wcs, DIM, STEP, TYPES)
    n2 = np.random.RandomState(11).normal(size=(DIM, DIM))
    pix = np.rot90(KMetacal(np.rot90(n2, 1), psf_im, wcs, step=STEP,
                            types=TYPES, Np=NP).get_images()['noshear'], 3)
    sky = KMetacal(n2, psf_im, wcs, step=STEP, types=TYPES, Np=NP,
                   rotation=90 * galsim.degrees).get_images()['noshear']
    assert np.abs(sky - pix).max() / np.abs(pix).max() > 0.05


def test_kspace_filters_finite_and_capped():
    """the hybrid filters (padded Np grid) are real, finite and capped at 1"""
    psf_im, gal_im, obs, wcs = _build(45.0)
    jmat = distortion_matrix(SCALE, theta_deg=45.0)
    pts, Np = delta_transfer_kspace(psf_im, wcs, DIM, STEP, TYPES)
    pts_rot, _ = delta_transfer_kspace(psf_im, wcs, DIM, STEP, TYPES, Np=Np,
                                       rotation=90 * galsim.degrees)
    G = make_hybrid_filters_kspace(pts, pts_rot, Np, SCALE, TYPES, jmat=jmat)
    for t in TYPES:
        assert G[t].shape == (Np, Np)
        assert np.all(np.isfinite(G[t]))
        assert G[t].min() >= 0.0 and G[t].max() <= 1.0 + 1e-12

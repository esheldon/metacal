"""
GPU engine parity vs the CPU path, with per-image psfs.
Skipped when cupy (or a device) is unavailable.

One engine processes observations with several DIFFERENT psfs
(varied Moffat fwhm and an elliptical one); each is compared
against the CPU production path built with that observation's
own psf.  Expected levels (measured, see the gpu-port document):
fp64 at ~1e-8 of the noise sigma, fp32 at ~1e-5; the psf
observations must be BITWISE identical (shared packaging code,
matched dither rng stream).
"""
import numpy as np
import pytest

cp = pytest.importorskip('cupy')
try:
    cp.cuda.runtime.getDeviceCount()
except Exception:
    pytest.skip('no CUDA device', allow_module_level=True)

import galsim  # noqa: E402
import ngmix  # noqa: E402
import metacal  # noqa: E402

SCALE = 0.2
N = 96

PSFS = [
    galsim.Moffat(beta=3.5, fwhm=0.8),
    galsim.Moffat(beta=3.5, fwhm=0.95),
    galsim.Moffat(beta=2.5, fwhm=0.85).shear(g1=0.03, g2=-0.02),
]


def _make_obs(seed, psf):
    rng = np.random.RandomState(seed)
    ps = psf.drawImage(nx=51, ny=51, scale=SCALE).array
    gal = galsim.Exponential(half_light_radius=0.5).shear(
        g1=0.05, g2=-0.03)
    image = galsim.Convolve(gal, psf).drawImage(
        nx=N, ny=N, scale=SCALE).array
    sig = image.max() / 30.0
    image = image + rng.normal(0, sig, image.shape)
    noise = rng.normal(0, sig, (N, N))
    jac = ngmix.DiagonalJacobian(
        row=(N - 1) / 2, col=(N - 1) / 2, scale=SCALE)
    pjac = ngmix.DiagonalJacobian(row=25, col=25, scale=SCALE)
    obs = ngmix.Observation(
        image, weight=np.ones((N, N)) / sig ** 2, jacobian=jac,
        noise=noise,
        psf=ngmix.Observation(ps.copy(), jacobian=pjac),
    )
    return obs, sig


@pytest.mark.parametrize('fp32,tol', [(False, 5.0e-8),
                                      (True, 5.0e-5)])
def test_per_psf_parity(fp32, tol):
    """one engine, three different psfs, each vs the CPU path"""
    from metacal.gpu import FusionEngine

    types = ('noshear', '1p', '1m')
    eng = FusionEngine(dim=N, types=types, fp32=fp32)
    for k, psf in enumerate(PSFS):
        obs, sig = _make_obs(11 + k, psf)
        oracle = metacal.metacal_obs(
            obs=obs, noise_filter=metacal.FusionFilter(),
            target_psf=metacal.AZGauss(),
            rng=np.random.RandomState(99), types=list(types),
        )
        mine = eng.metacal_obs(
            obs=obs, rng=np.random.RandomState(99))
        assert list(mine.keys()) == list(oracle.keys())
        for t in types:
            mo, oo = mine[t], oracle[t]
            d = np.abs(mo.image - oo.image).max() / sig
            assert d < tol, (k, t, d)
            assert np.array_equal(mo.weight, oo.weight)
            assert np.array_equal(mo.psf.image, oo.psf.image)
            assert np.array_equal(mo.psf.weight, oo.psf.weight)


def test_five_types():
    from metacal.gpu import metacal_obs

    types = ('noshear', '1p', '1m', '2p', '2m')
    obs, sig = _make_obs(12, PSFS[0])
    oracle = metacal.metacal_obs(
        obs=obs, noise_filter=metacal.FusionFilter(),
        target_psf=metacal.AZGauss(),
        rng=np.random.RandomState(3), types=list(types),
    )
    mine = metacal_obs(
        obs=obs, noise_filter=metacal.FusionFilter(),
        target_psf=metacal.AZGauss(),
        rng=np.random.RandomState(3), types=list(types),
    )
    for t in types:
        d = np.abs(mine[t].image - oracle[t].image).max() / sig
        assert d < 5.0e-5, (t, d)
        assert np.array_equal(mine[t].psf.image,
                              oracle[t].psf.image)


def test_bundle_reuse_and_shipping():
    """a worker-built bundle (cupy-free module) gives identical
    output to the engine's own derivation, and repeated psfs hit
    the bundle cache"""
    from metacal.gpu import FusionEngine
    from metacal.gpu.bundle import PsfBundle

    types = ('noshear', '1p', '1m')
    obs, sig = _make_obs(13, PSFS[1])
    eng = FusionEngine(dim=N, types=types, fp32=True)

    bundle = PsfBundle(
        psf_image=obs.psf.image,
        wcs=obs.jacobian.get_galsim_wcs(),
        types=types, dim=N,
    )
    r1 = eng.metacal_obs(obs=obs, rng=np.random.RandomState(5),
                         bundle=bundle)
    r2 = eng.metacal_obs(obs=obs, rng=np.random.RandomState(5))
    for t in types:
        assert np.array_equal(r1[t].image, r2[t].image)
        assert np.array_equal(r1[t].psf.image, r2[t].psf.image)

    b1 = eng.get_bundle(obs)
    b2 = eng.get_bundle(obs)
    assert b1 is b2


def test_rejects_unsupported():
    from metacal.gpu import metacal_obs
    from metacal.gpu.bundle import PsfBundle

    obs, _ = _make_obs(14, PSFS[0])
    with pytest.raises(NotImplementedError):
        metacal_obs(
            obs=obs,
            noise_filter=metacal.FusionFilter(),
            target_psf=metacal.Symmetrize(
                4, np.random.RandomState(1)),
            rng=np.random.RandomState(1),
            types=('noshear',),
        )
    with pytest.raises(NotImplementedError):
        PsfBundle(
            obs.psf.image,
            np.array([[0.2, 0.01], [0.0, 0.2]]),
            ('noshear',), N,
        )

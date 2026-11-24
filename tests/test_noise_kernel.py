"""
tests for the noise kernel (metacal.noise_kernel): the impulse pushed through
the metacal operator to measure the noise transfer, DeltaImpulse for white
noise and ImpulseResponse for correlated noise
"""
import numpy as np
import galsim
import pytest

import metacal
from metacal import DeltaImpulse, ImpulseResponse
from metacal.fusion_filter import (
    _impulse_transfer_kspace,
    _make_fusion_filters_kspace,
    _trapz_weights,
)
from metacal.wcs import distortion_matrix, galsim_wcs

DIM = 48
SCALE = 0.2
TYPES = ('noshear', '1p', '1m')


def _wcs():
    return galsim_wcs(
        distortion_matrix(SCALE, theta=0.0 * galsim.degrees, g1=0, g2=0)
    )


def _psf_image(wcs, psf_g1=0.0):
    psf = galsim.Moffat(beta=2.5, fwhm=0.9).shear(g1=psf_g1)
    return psf.drawImage(nx=DIM, ny=DIM, wcs=wcs).array


def _transfers(psf_image, wcs, noise_kernel, npix=None):
    """
    Get the un-rotated and sky-rotated transfers for this kernel.

    Returns
    -------
    pts, pts_rot, npix
    """
    impulse = noise_kernel(DIM, wcs)
    kw = dict(
        psf_image=psf_image, wcs=wcs, target_psf=metacal.AZGauss(),
        dim=DIM, types=TYPES, impulse=impulse,
    )
    pts, npix = _impulse_transfer_kspace(npix=npix, **kw)
    pts_rot, _ = _impulse_transfer_kspace(
        npix=npix, rotation=90 * galsim.degrees, **kw,
    )
    return pts, pts_rot, npix


def _m2(P, npix):
    """
    Project P onto annular harmonics with the trapz quadrature.

    Returns
    -------
    m0, c2, s2, nann
        the m=0 mean, the m=2 (cos, sin) coefficients and the mode count per
        annulus
    """
    k1 = 2 * np.pi * np.fft.fftfreq(npix, d=SCALE)
    kx, ky = np.meshgrid(k1, k1)
    kmag = np.hypot(kx, ky)
    th = np.arctan2(ky, kx)
    dk = 2 * np.pi / (npix * SCALE)
    ib = np.rint(kmag / dk).astype(int)
    nb = ib.max() + 1
    nann = np.bincount(ib.ravel(), minlength=nb)
    w = _trapz_weights(th, ib, nb)

    def az(x):
        return np.bincount(ib.ravel(), weights=(w * x).ravel(), minlength=nb)

    return az(P), az(P * np.cos(2 * th)), az(P * np.sin(2 * th)), nann


def _band(ref, npix):
    """
    Get the in-band annuli: power above 3% of peak and at least 24 modes.
    """
    m0, _, _, nann = _m2(ref, npix)
    return (m0 > 3e-2 * m0.max()) & (nann >= 24)


def _type_odd_m2(Pp, Pm, band, npix):
    """
    Get the band-mean type-odd m=2 anisotropy.

    The m=2 amplitude of the part of the noise power that differs between 1p
    and 1m, relative to the mean level.
    """
    m0p, cp, sp, _ = _m2(Pp, npix)
    m0m, cm, sm, _ = _m2(Pm, npix)
    m0 = 0.5 * (m0p + m0m)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.hypot(cp - cm, sp - sm) / m0
    return r[band].mean()


def test_delta_impulse_is_the_delta():
    im = DeltaImpulse()(DIM, _wcs())
    assert im.shape == (DIM, DIM)
    assert im.sum() == 1.0
    assert im[DIM // 2, DIM // 2] == 1.0


def test_fusion_filter_with_delta_matches_private_path():
    """
    FusionFilter with DeltaImpulse reproduces the filter built directly from
    the delta transfers
    """
    wcs = _wcs()
    psf_image = _psf_image(wcs, psf_g1=0.03)
    jmat = distortion_matrix(SCALE, theta=0.0 * galsim.degrees, g1=0, g2=0)

    hfilt, nvf = metacal.FusionFilter()(
        psf_image=psf_image, wcs=wcs, target_psf=metacal.AZGauss(),
        dim=DIM, types=TYPES, noise_kernel=DeltaImpulse(),
    )
    pts, pts_rot, npix = _transfers(psf_image, wcs, DeltaImpulse())
    ref = _make_fusion_filters_kspace(
        pts, pts_rot, npix, SCALE, TYPES, jmat=jmat,
    )
    for t in TYPES:
        assert np.array_equal(hfilt[t], ref[t])
    assert 1.0 < nvf < 2.0


def test_impulse_response_embeds_centered():
    kern = np.zeros((7, 5))
    kern[3, 2] = 1.0     # the center of a (7, 5) image
    kern[0, 0] = 0.5
    im = ImpulseResponse(kern)(DIM, _wcs())
    assert im.shape == (DIM, DIM)
    assert im[DIM // 2, DIM // 2] == 1.0
    assert im[DIM // 2 - 3, DIM // 2 - 2] == 0.5
    assert im.sum() == 1.5


def test_impulse_response_rejects_bad_input():
    with pytest.raises(ValueError):
        ImpulseResponse(np.zeros((5, 5)))
    with pytest.raises(ValueError):
        ImpulseResponse(np.ones(5))
    with pytest.raises(ValueError):
        ImpulseResponse(np.ones((DIM + 1, DIM + 1)))(DIM, _wcs())


def test_fusion_filter_rejects_bad_kernel():
    wcs = _wcs()
    psf_image = _psf_image(wcs)
    kw = dict(
        psf_image=psf_image, wcs=wcs, target_psf=metacal.AZGauss(),
        dim=DIM, types=TYPES,
    )
    with pytest.raises(ValueError):
        metacal.FusionFilter()(noise_kernel=lambda dim, wcs: np.ones(3), **kw)
    with pytest.raises(ValueError):
        metacal.FusionFilter()(
            noise_kernel=lambda dim, wcs: np.zeros((dim, dim)), **kw,
        )


def test_from_autocorr_has_the_right_power():
    """
    the kernel built from an autocorrelation xi has |FT|^2 = FT(xi)
    """
    n = 15
    y, x = np.mgrid[:n, :n] - n // 2
    G = np.exp(-0.5 * (x ** 2 + y ** 2) / 1.5 ** 2)

    # the autocorrelation of G on a padded grid, cropped to (2n-1)^2
    m = 4 * n
    Gf = np.zeros((m, m))
    Gf[m // 2 - n // 2:m // 2 - n // 2 + n,
       m // 2 - n // 2:m // 2 - n // 2 + n] = G
    Nf = np.abs(np.fft.fft2(np.fft.ifftshift(Gf))) ** 2
    xi = np.fft.fftshift(np.fft.ifft2(Nf).real)
    h = n - 1
    xi = xi[m // 2 - h:m // 2 + h + 1, m // 2 - h:m // 2 + h + 1]

    wcs = _wcs()
    im_ref = ImpulseResponse(G)(DIM, wcs)
    im_xi = ImpulseResponse.from_autocorr(xi)(DIM, wcs)
    N_ref = np.abs(np.fft.fft2(im_ref)) ** 2
    N_xi = np.abs(np.fft.fft2(im_xi)) ** 2
    inband = N_ref > 1e-6 * N_ref.max()
    assert np.allclose(N_xi[inband], N_ref[inband], rtol=1e-6)


def test_from_gsobject_draws_centered():
    wcs = _wcs()
    im = ImpulseResponse.from_gsobject(galsim.Gaussian(sigma=0.3))(DIM, wcs)
    assert im.shape == (DIM, DIM)
    assert np.all(np.isfinite(im))
    assert np.unravel_index(im.argmax(), im.shape) == (DIM // 2, DIM // 2)
    # symmetric about the center pixel
    c = DIM // 2
    assert np.allclose(im[c, c - 5:c], im[c, c + 1:c + 6][::-1])


def test_rotated_pass_shears_the_kernel_the_other_way():
    """
    an elongated kernel through the rotated pass (rotate, metacal, rotate
    back): in the final frame the kernel's own anisotropy is unchanged while
    the shear acts in the rotated frame, so for a round psf the rotated 1p
    transfer is the un-rotated 1m transfer, kernel included
    """
    wcs = _wcs()
    psf_image = _psf_image(wcs)
    kernel = ImpulseResponse.from_gsobject(
        galsim.Gaussian(sigma=0.25).shear(g1=0.4)
    )
    pts, pts_rot, npix = _transfers(psf_image, wcs, kernel)
    band = _band(pts['noshear'], npix)
    m0, c, s, _ = _m2(pts['noshear'], npix)
    with np.errstate(divide='ignore', invalid='ignore'):
        amp = (np.hypot(c, s) / m0)[band]
    assert amp.mean() > 0.05          # the kernel's spin-2 is in the transfer

    peak = pts['noshear'].max()
    # the shear makes a real difference between 1p and 1m ...
    assert np.abs(pts['1p'] - pts['1m']).max() > 1e-2 * peak
    # ... and the rotated pass swaps them, with noshear unchanged
    assert np.abs(pts_rot['noshear'] - pts['noshear']).max() < 1e-4 * peak
    assert np.abs(pts_rot['1p'] - pts['1m']).max() < 1e-4 * peak
    assert np.abs(pts_rot['1m'] - pts['1p']).max() < 1e-4 * peak


def test_correlated_kernel_filter_isotropizes():
    """
    with gaussian noise correlated over a pixel, the filter built from the
    white transfers leaves a large type-odd (1p vs 1m) spin-2 residual; the
    filter built from the kernel transfers brings it to the white-noise floor
    """
    wcs = _wcs()
    psf_image = _psf_image(wcs)
    jmat = distortion_matrix(SCALE, theta=0.0 * galsim.degrees, g1=0, g2=0)
    kernel = ImpulseResponse.from_gsobject(galsim.Gaussian(sigma=SCALE))

    pw, pw_rot, npix = _transfers(psf_image, wcs, DeltaImpulse())
    A, A_rot, _ = _transfers(psf_image, wcs, kernel, npix=npix)
    H_white = _make_fusion_filters_kspace(
        pw, pw_rot, npix, SCALE, TYPES, jmat=jmat,
    )
    H_kern = _make_fusion_filters_kspace(
        A, A_rot, npix, SCALE, TYPES, jmat=jmat,
    )

    band = _band(A['1p'], npix)

    def resid(H):
        return _type_odd_m2(
            A['1p'] + H['1p'] ** 2 * A_rot['1p'],
            A['1m'] + H['1m'] ** 2 * A_rot['1m'],
            band, npix,
        )

    before = _type_odd_m2(A['1p'], A['1m'], band, npix)
    floor = _type_odd_m2(
        pw['1p'] + H_white['1p'] ** 2 * pw_rot['1p'],
        pw['1m'] + H_white['1m'] ** 2 * pw_rot['1m'],
        _band(pw['1p'], npix), npix,
    )
    r_white, r_kern = resid(H_white), resid(H_kern)

    assert before > 0.02
    assert r_white > 0.2 * before          # the white filter is not adequate
    assert r_kern < 0.05 * before
    assert r_kern < 2.0 * floor            # at the white-noise floor


def test_noise_correct_runs_with_kernel():
    wcs = _wcs()
    psf_image = _psf_image(wcs, psf_g1=0.02)
    rng = np.random.RandomState(7)
    image = rng.normal(size=(DIM, DIM))
    noise = rng.normal(size=(DIM, DIM))
    kernel = ImpulseResponse.from_gsobject(galsim.Gaussian(sigma=0.5 * SCALE))

    res = metacal.metacal_noise_correct(
        image=image, psf_image=psf_image, noise_image=noise,
        noise_filter=metacal.FusionFilter(), noise_kernel=kernel,
        wcs=wcs, target_psf=metacal.AZGauss(), types=TYPES,
    )
    for t in TYPES:
        assert res[t].shape == (DIM, DIM)
        assert np.all(np.isfinite(res[t]))
    assert 1.0 < res.noise_var_factor < 2.0

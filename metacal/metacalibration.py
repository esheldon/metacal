"""
Metacalibration
"""

import numpy as np
import galsim

from .deficit import common_harmonic_deficits
from .wcs import galsim_wcs
from .metacal_result import MetacalResult
from .defaults import LANCZOS, DEFAULT_TYPES


def metacal_image(
    image,
    psf_image,
    noise_image,
    wcs,
    target_psf,
    step=0.01,
    types=DEFAULT_TYPES,
):
    """
    metacal one image

    Parameters
    ----------
    image: array
        the image to metacal
    psf_image: array
        the psf image
    noise_image: array
        A noise field for noise anisotropy correction.  Send None to not apply
        a correction.  The noise field should match the noise in image.
    wcs: galsim WCS
        May also be given as the 2x2 pixel->sky matrix (col, row order) instead
        of a galsim wcs
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

    if noise_image is None:
        mcal = Metacal(
            image=image,
            psf_image=psf_image,
            wcs=wcs,
            target_psf=target_psf,
            step=step,
            types=types,
        )
        return MetacalResult(
            images=mcal.get_images(),
            psf_image=mcal.target_psf_image(),
            noise_var_factor=1.0,
        )
    else:
        return _metacal_with_noise_correction(
            image,
            psf_image,
            wcs,
            target_psf=target_psf,
            noise_image=noise_image,
            step=step,
            types=types,
        )


class Metacal:
    """
    Metacalibrate a single image.

    You usually want to use one of the convenience functions metacal_image() or
    metacal_obs().  Only use this class if you know what you are doing.

    Parameters
    ----------
    image: (N, N) array
        the image to metacal (galaxy, noise field, or a unit delta)
    psf_image: (N, N) array
        the psf image (pixel-convolved, as drawn); deconvolved from the image
        and used to derive the round gaussian reconvolution target
    wcs: galsim local/Jacobian wcs, or a 2x2 array
        the pixel->sky wcs; a diagonal scale or a distorted (rotated/sheared)
        jacobian, handled exactly via profileToImage.  May be given as the 2x2
        pixel->sky matrix (col, row order) instead of a galsim wcs
    target_psf: A callable that returns a galsim object
        This should be callable with target_psf(psf=psf, flux=flux),
        with psf a galsim object such as galsim.InterpolatedImage.  For
        an example see metacal.AZGauss
    step: float
        metacal shear step
    types: sequence of str
        any of 'noshear', '1p', '1m', '2p', '2m'
    Np: int, optional
        the padded draw-grid size; default None computes galsim's own drawFFT
        size.  Pass an explicit Np only to force one shared grid across a set
        of Metacals (the filter and the noise it cancels must share a frame).
        This is not a tuning knob.
    rotation: galsim.Angle or None
        sky-frame rotation of the input field before metacal (the metacal'd
        field is rotated back by -rotation). Setting ``rotation=90 *
        galsim.degrees`` builds the noise correction field.
    x_interpolant: str
        the InterpolatedImage interpolant (default lanczos15, the metacal
        kernel)
    """

    def __init__(
        self,
        image,
        psf_image,
        wcs,
        target_psf,
        step=0.01,
        types=DEFAULT_TYPES,
        Np=None,
        rotation=None,
        x_interpolant=LANCZOS,
    ):
        if rotation is not None and not isinstance(rotation, galsim.Angle):
            raise TypeError(
                'rotation must be a galsim.Angle '
                '(e.g. 90 * galsim.degrees) or '
                f'None, got {type(rotation).__name__}'
            )
        self.N = image.shape[0]
        self.psf_image = psf_image
        self.wcs, self.wcs_matrix = _wcs_and_matrix(wcs)
        self.step = step
        self.types = list(types)
        self.rotation = rotation
        self._khats = None

        img = galsim.Image(np.asarray(image, dtype=float), wcs=self.wcs)
        self.image_int = galsim.InterpolatedImage(
            img, x_interpolant=x_interpolant
        )
        if self.rotation is not None:
            self.image_int = self.image_int.rotate(self.rotation)

        pimg = galsim.Image(np.asarray(psf_image, dtype=float), wcs=self.wcs)
        psf_int = galsim.InterpolatedImage(pimg, x_interpolant=LANCZOS)
        self.psf_flux = float(np.sum(psf_image))

        # deconvolve the psf (and the pixel it carries) from the image
        self.image_int_nopsf = galsim.Convolve(
            self.image_int, galsim.Deconvolve(psf_int)
        )

        # the round gaussian reconvolution target, dilated by 1 + 2*step; the
        # same target for every type (only the galaxy is sheared)
        self.target_psf = target_psf(psf_int, flux=self.psf_flux)
        self.target_psf = self.target_psf.dilate(1.0 + 2.0 * self.step)

        # the padded draw grid (galsim's own drawFFT size unless shared via Np)
        self.Np = self._galsim_kpad_size() if Np is None else int(Np)
        self._lo = (self.Np - self.N) // 2

        # numpy-fft-matched k grid on the Np draw grid: dk = 2*pi/Np, plus the
        # ifftshift reorder and the (Np-1)/2 centering phase ramp that match
        # galsim's drawKImage to fft2(drawImage(method='no_pixel'))
        self._dk = 2.0 * np.pi / self.Np
        k1 = 2.0 * np.pi * np.fft.fftfreq(self.Np)
        kxg, kyg = np.meshgrid(k1, k1)
        self._phase = np.exp(-1j * (kxg + kyg) * (self.Np - 1.0) / 2.0)

    def _galsim_kpad_size(self):
        """
        galsim's own FFT draw size for this metacal object, replicating
        ``GSObject.drawFFT_makeKImage``: the good image size from stepk, at
        least the stamp size, rounded to a good FFT size, floored at
        minimum_fft_size.  Set by the image size (stepk), so same-N images
        share it.
        """
        prof = self.wcs.profileToImage(
            galsim.Convolve([self.image_int_nopsf, self.target_psf])
        )
        n = max(prof.getGoodImageSize(1.0), self.N)
        n = galsim.Image.good_fft_size(n)
        return int(max(n, prof.gsparams.minimum_fft_size))

    def _khat(self, world_profile):
        """
        the matched-grid k-array of a world-frame profile: to image coords,
        drawKImage at dk, reorder to numpy fft layout, re-center
        """
        image_profile = self.wcs.profileToImage(world_profile)
        kim = image_profile.drawKImage(nx=self.Np, ny=self.Np, scale=self._dk)
        return np.fft.ifftshift(kim.array) * self._phase

    def _metacal_field(self, t):
        """
        the metacal'd (deconv-sheared-reconv) world profile for type t, with
        the sky rotate-back (-rotation) applied so the field is in the final
        image-metacal frame ready to add
        """
        sh = _shear_kwargs(t, self.step)
        nopsf = (
            self.image_int_nopsf
            if sh is None
            else self.image_int_nopsf.shear(**sh)
        )
        field = galsim.Convolve([nopsf, self.target_psf])
        if self.rotation is not None:
            field = field.rotate(-self.rotation)
        return field

    def _crop(self, arr):
        """
        crop the center N x N out of an Np x Np draw
        """
        lo = self._lo
        return arr[lo : lo + self.N, lo : lo + self.N]

    def get_khats(self):
        """
        dict type -> Np x Np metacal'd k-array (numpy fft layout); cached
        """
        if self._khats is None:
            self._khats = {
                t: self._khat(self._metacal_field(t)) for t in self.types
            }
        return self._khats

    def get_images(self):
        """
        dict type -> real metacal'd image, cropped to the center N x N (one
        ifft2 per type of the Np-grid get_khats)
        """
        return {
            t: self._crop(np.fft.ifft2(k).real)
            for t, k in self.get_khats().items()
        }

    def get_filtered_images(self, filters):
        """
        dict type -> real image of ifft2(khat_t * filters[t]), cropped to N;
        ``filters`` are Np x Np k-space filters (the hybrid H_t).  The filter
        is applied in k on the padded grid; no second fft; so the
        de-aliased correction field stays in the same frame as the transfer
        that built it.
        """
        kh = self.get_khats()
        return {
            t: self._crop(np.fft.ifft2(kh[t] * filters[t]).real)
            for t in self.types
        }

    def target_psf_image(self):
        """the round dilated-gaussian reconvolution psf image (for the fit);
        drawn the standard way (it is smooth, no aliasing concern)"""

        ny, nx = self.psf_image.shape
        return self.target_psf.drawImage(
            nx=nx, ny=ny, wcs=self.wcs, method='no_pixel'
        ).array


def _metacal_with_noise_correction(
    image,
    psf_image,
    wcs,
    target_psf,
    noise_image=None,
    step=0.01,
    types=DEFAULT_TYPES,

):
    if image.shape != noise_image.shape:
        raise ValueError(
            f'noise shape mistmatch, {noise_image.shape} != {image.shape}'
        )

    dim = image.shape[0]
    wcs, jmat = _wcs_and_matrix(wcs)

    # one shared padded grid for the transfer, the galaxy and the noise
    pts, npix = _delta_transfer_kspace(
        psf_image=psf_image,
        wcs=wcs,
        target_psf=target_psf,
        dim=dim,
        step=step,
        types=types,
    )
    pts_rot, _ = _delta_transfer_kspace(
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
        psf_image=mcal.target_psf_image(),
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


def _delta_transfer_kspace(
    psf_image, wcs, target_psf, dim, step, types, Np=None, rotation=None
):
    """
    the per-type metacal noise transfer P_t = |K_t|^2 on the padded Np x Np
    grid, computed k-natively: push a unit delta through ``Metacal`` and take
    |k-array|^2 directly (|fft2(delta)| = 1, so this is relative to a white
    input).  ``Np`` (default galsim's draw size) must match the Metacal whose
    noise this filter cancels.  ``rotation = 90 * galsim.degrees`` builds the
    rotated transfer P_t^rot = |rotateback(metacal(delta))|^2 (the corr-field
    transfer after the sky rotate-back).

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
        (_delta_transfer_kspace
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


def _shear_kwargs(t, step):
    """
    galsim .shear() kwargs for a metacal type (None for noshear).  1p/1m shear
    g1=+/-step, 2p/2m shear g2=+/-step ; 2p/2m give the g2 column of the full
    2x2 response (needed for the trace response Rbar; only 1p/1m for the R11
    path).
    """
    return {
        'noshear': None,
        '1p': {'g1': step, 'g2': 0.0},
        '1m': {'g1': -step, 'g2': 0.0},
        '2p': {'g1': 0.0, 'g2': step},
        '2m': {'g1': 0.0, 'g2': -step},
    }[t]


def _wcs_and_matrix(wcs):
    """
    resolve a wcs given as either a galsim local/Jacobian wcs or a 2x2
    pixel->sky jacobian matrix

    return both as (wcs, matrix)
    """
    if isinstance(wcs, galsim.BaseWCS):
        jac = wcs.jacobian()
        return wcs, np.array([[jac.dudx, jac.dudy], [jac.dvdx, jac.dvdy]])
    mat = np.asarray(wcs, dtype=float)
    return galsim_wcs(mat), mat

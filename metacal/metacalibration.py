"""
Metacalibration
"""

import numpy as np
import galsim

from .result import MetacalResult
from ._util import _wcs_and_matrix

# LANCZOS = 'lanczos15'
LANCZOS = 'lanczos3'
SHEAR_STEP = 0.01


def metacal_image(
    image,
    psf_image,
    wcs,
    target_psf,
    types,
):
    """
    metacal an image

    Parameters
    ----------
    image: array
        the image to metacal
    psf_image: array
        the psf image
    wcs: galsim WCS
        A galsim local wcs/Jacobian.  May also be given as the 2x2 pixel->sky
        matrix (col, row order) instead of a galsim wcs
    target_psf: A callable that returns a galsim object or a galsim.GSObject
        This should be a galsim.GSObject or a callable with target_psf(psf=psf,
        flux=flux), with psf a galsim object such as galsim.InterpolatedImage.
        For an example see metacal.AZGauss
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

    mcal = Metacal(
        image=image,
        psf_image=psf_image,
        wcs=wcs,
        target_psf=target_psf,
        types=types,
    )
    return MetacalResult(
        images=mcal.get_images(),
        psf_image=mcal.get_target_psf_image(),
        noise_var_factor=1.0,
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
        A galsim local wcs/Jacobian.  May also be given as the 2x2 pixel->sky
        matrix (col, row order) instead of a galsim wcs
    target_psf: A callable that returns a galsim object or a galsim.GSObject
        This should be a galsim.GSObject or a callable with target_psf(psf=psf,
        flux=flux), with psf a galsim object such as galsim.InterpolatedImage.
        For an example see metacal.AZGauss
    types: sequence of str
        any of 'noshear', '1p', '1m', '2p', '2m'
    npix: int, optional
        the padded draw-grid size; default None computes galsim's own drawFFT
        size.  Pass an explicit npix only to force one shared grid across a set
        of Metacals (the filter and the noise it cancels must share a frame).
        This is not a tuning knob.
    rotation: galsim.Angle or None
        sky-frame rotation of the input field before metacal (the metacal'd
        field is rotated back by -rotation). Setting ``rotation=90 *
        galsim.degrees`` builds the noise correction field.
    """

    def __init__(
        self,
        image,
        psf_image,
        wcs,
        target_psf,
        types,
        npix=None,
        rotation=None,
    ):
        if rotation is not None and not isinstance(rotation, galsim.Angle):
            raise TypeError(
                'rotation must be a galsim.Angle '
                '(e.g. 90 * galsim.degrees) or '
                f'None, got {type(rotation).__name__}'
            )
        self._N = image.shape[0]
        self._psf_image = psf_image
        self._wcs, self._wcs_matrix = _wcs_and_matrix(wcs)
        self._types = list(types)
        self._rotation = rotation
        self._khats = None

        img = galsim.Image(np.asarray(image, dtype=float), wcs=self._wcs)
        self._image_int = galsim.InterpolatedImage(img, x_interpolant=LANCZOS)
        if self._rotation is not None:
            self._image_int = self._image_int.rotate(self._rotation)

        pimg = galsim.Image(np.asarray(psf_image, dtype=float), wcs=self._wcs)
        psf_int = galsim.InterpolatedImage(pimg, x_interpolant=LANCZOS)
        self.psf_flux = float(np.sum(psf_image))

        # deconvolve the psf (and the pixel it carries) from the image
        self._image_int_nopsf = galsim.Convolve(
            self._image_int, galsim.Deconvolve(psf_int)
        )

        # the round gaussian reconvolution target, dilated by 1 + 2*step; the
        # same target for every type (only the galaxy is sheared)
        if isinstance(target_psf, galsim.GSObject):
            tpsf = target_psf.withFlux(self.psf_flux)
        else:
            tpsf = target_psf(psf_int, flux=self.psf_flux)

        self._target_psf = tpsf.dilate(1.0 + 2.0 * SHEAR_STEP)

        # the padded draw grid (galsim's own drawFFT size unless shared via npix)
        self._npix = self._galsim_kpad_size() if npix is None else int(npix)
        self._lo = (self._npix - self._N) // 2

        # numpy-fft-matched k grid on the npix draw grid: dk = 2*pi/npix, plus the
        # ifftshift reorder and the (npix-1)/2 centering phase ramp that match
        # galsim's drawKImage to fft2(drawImage(method='no_pixel'))
        self._dk = 2.0 * np.pi / self._npix
        k1 = 2.0 * np.pi * np.fft.fftfreq(self._npix)
        kxg, kyg = np.meshgrid(k1, k1)
        self._phase = np.exp(-1j * (kxg + kyg) * (self._npix - 1.0) / 2.0)

    def get_images(self):
        """
        Returns
        -------
        dict:
            The real metacal'd images, cropped to the center N x N (one
            ifft2 per type of the npix-grid get_khats)
        """
        return {
            t: self._crop(np.fft.ifft2(k).real)
            for t, k in self.get_khats().items()
        }

    def get_filtered_images(self, filters):
        """
        Returns
        -------
        dict:
            real image of ifft2(khat_t * filters[t]), cropped to N; ``filters``
            are npix x npix k-space filters (e.g. the fusion H_t).  The filter is
            applied in k on the padded grid; no second fft; so the de-aliased
            correction field stays in the same frame as the transfer that built
            it.
        """
        kh = self.get_khats()
        return {
            t: self._crop(np.fft.ifft2(kh[t] * filters[t]).real)
            for t in self._types
        }

    def get_target_psf_image(self):
        """
        Get an image of the round dilated-gaussian reconvolution psf.

        Returns
        -------
        array:
            The image array
        """

        ny, nx = self._psf_image.shape
        return self._target_psf.drawImage(
            nx=nx, ny=ny, wcs=self._wcs, method='no_pixel'
        ).array

    def get_khats(self):
        """
        dict type -> npix x npix metacal'd k-array (numpy fft layout); cached
        """
        if self._khats is None:
            self._khats = {
                t: self._khat(self._metacal_field(t)) for t in self._types
            }
        return self._khats

    @property
    def npix(self):
        return self._npix

    def _galsim_kpad_size(self):
        """
        galsim's own FFT draw size for this metacal object, replicating
        ``GSObject.drawFFT_makeKImage``: the good image size from stepk, at
        least the stamp size, rounded to a good FFT size, floored at
        minimum_fft_size.  Set by the image size (stepk), so same-N images
        share it.
        """
        prof = self._wcs.profileToImage(
            galsim.Convolve([self._image_int_nopsf, self._target_psf])
        )
        n = max(prof.getGoodImageSize(1.0), self._N)
        n = galsim.Image.good_fft_size(n)
        return int(max(n, prof.gsparams.minimum_fft_size))

    def _khat(self, world_profile):
        """
        the matched-grid k-array of a world-frame profile: to image coords,
        drawKImage at dk, reorder to numpy fft layout, re-center
        """
        image_profile = self._wcs.profileToImage(world_profile)
        kim = image_profile.drawKImage(
            nx=self._npix,
            ny=self._npix,
            scale=self._dk,
        )
        return np.fft.ifftshift(kim.array) * self._phase

    def _metacal_field(self, t):
        """
        the metacal'd (deconv-sheared-reconv) world profile for type t, with
        the sky rotate-back (-rotation) applied so the field is in the final
        image-metacal frame ready to add
        """
        sh = _shear_kwargs(t)
        nopsf = (
            self._image_int_nopsf
            if sh is None
            else self._image_int_nopsf.shear(**sh)
        )
        field = galsim.Convolve([nopsf, self._target_psf])
        if self._rotation is not None:
            field = field.rotate(-self._rotation)
        return field

    def _crop(self, arr):
        """
        crop the center N x N out of an npix x npix draw
        """
        lo = self._lo
        return arr[lo : lo + self._N, lo : lo + self._N]


def _shear_kwargs(t):
    """
    galsim .shear() kwargs for a metacal type (None for noshear).  1p/1m shear
    g1=+/-step, 2p/2m shear g2=+/-step ; 2p/2m give the g2 column of the full
    2x2 response (needed for the trace response Rbar; only 1p/1m for the R11
    path).
    """
    return {
        'noshear': None,
        '1p': {'g1': SHEAR_STEP, 'g2': 0.0},
        '1m': {'g1': -SHEAR_STEP, 'g2': 0.0},
        '2p': {'g1': 0.0, 'g2': SHEAR_STEP},
        '2m': {'g1': 0.0, 'g2': -SHEAR_STEP},
    }[t]

"""
k-space-native metacalibration with the rotated-hybrid noise correction.

The metacal deconvolve / shear / reconvolve is a chain of galsim k-space
GSObjects.  This module keeps it in k-space and samples it ONCE with
``drawKImage`` onto a grid matched to the numpy fft grid, so the per-type noise
transfer ``|K_t|^2`` and the filtered correction field are built WITHOUT a
second, aliasing real<->k round trip.  A single ``np.ifft2`` returns to real
space.

Grid matching
-------------
``drawImage(wcs=wcs, method='no_pixel')`` renders the profile in IMAGE
coordinates (galsim converts via ``wcs.profileToImage``).  So ``drawKImage`` of
``wcs.profileToImage(profile)`` at ``dk = 2*pi/Np`` samples exactly the
spectral grid a numpy ``fft2`` of that drawn image would -- on the axis-aligned
image-coordinate k-grid, so a rotated/sheared wcs needs no rotated k-grid.  The
two conventions to fix are galsim's centered (fftshift) k-image and its phase
reference: ``np.fft.ifftshift`` reorders to numpy fft layout and a phase ramp
``exp(-i (kx+ky) (Np-1)/2)`` re-centers to the drawImage centroid.

Padding
-------
The draw grid ``Np`` is galsim's own FFT draw size (``drawFFT_makeKImage``),
not the stamp size N: it gives the object's real-space tails room so the
periodic ifft does NOT wrap across the stamp.  ``Np`` is computed automatically
and is NOT a tuning knob; it is exposed only so a set of metacals (galaxy,
noise, transfer) can be forced onto ONE shared grid -- the filter and the noise
it cancels must live on a single frame.  The high-level ``metacal_hybrid`` does
that threading.

Sky-frame fixnoise rotation
---------------------------
The correction noise field is rotated 90 deg in the SKY (world) frame, not by a
pixel ``np.rot90``.  An ``InterpolatedImage`` built with the wcs is a world
profile, so ``.rotate(90*deg)`` is an exact coordinate rotation in sky angle (a
lazy Transform -- no resampling until the single final draw); rotate the noise
+90 before metacal and the metacal'd field -90 after, so ``FT(corr) = T_M(R90
k) N(k)`` -- the 90 lands on the analytic metacal transfer, the noise keeps its
single Lanczos wrap.  Under a non-conformal wcs this is the genuine sky 90 (a
pixel ``np.rot90`` is not), removing the spurious additive c it would leave.

Only the azgauss reconvolution path is implemented (the production kernel): a
round gaussian target from ``ngmix.metacal.azgauss_target_psf``, dilated by
1 + 2*step, the SAME target for every type (only the galaxy is sheared).  Needs
numpy, galsim and ngmix (on the ``mcal-gauss-stability`` branch).
"""

import numpy as np
import galsim

from ngmix.metacal.azgauss_target_psf import get_azgauss_target_psf

from .deficit import common_harmonic_deficits
from .wcs import galsim_wcs

LANCZOS = 'lanczos15'
DEFAULT_TYPES = ('noshear', '1p', '1m')


def _wcs_and_matrix(wcs):
    """
    resolve a wcs given as EITHER a galsim local/Jacobian wcs OR a 2x2
    pixel->sky jacobian matrix (col, row order, M = [[dudx, dudy], [dvdx,
    dvdy]]), and return the pair ``(galsim_wcs, matrix)`` -- the galsim wcs the
    metacal draws with and the matrix used for the pixel scale and the
    sky-angle deficit projection.  A galsim wcs is returned as-is with its
    jacobian extracted; a matrix is wrapped in a ``galsim.JacobianWCS``.
    """
    if isinstance(wcs, galsim.BaseWCS):
        jac = wcs.jacobian()
        return wcs, np.array([[jac.dudx, jac.dudy], [jac.dvdx, jac.dvdy]])
    mat = np.asarray(wcs, dtype=float)
    return galsim_wcs(mat), mat


def _shear_kwargs(t, step):
    """
    galsim .shear() kwargs for a metacal type (None for noshear).  1p/1m shear
    g1=+/-step, 2p/2m shear g2=+/-step -- 2p/2m give the g2 column of the full
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


class KMetacal:
    """
    k-space-native azgauss metacal of a single image (world shear frame).

    Builds the persistent galsim GSObjects once (interpolated image,
    deconvolved galaxy, dilated round-gaussian target psf) and exposes the
    per-type metacal'd field as the matched-grid k-array (``get_khats``) or the
    real image (``get_images``).

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
    step: float
        metacal shear step
    types: sequence of str
        any of 'noshear', '1p', '1m', '2p', '2m'
    Np: int, optional
        the padded draw-grid size; default None computes galsim's own drawFFT
        size.  Pass an explicit Np ONLY to force one shared grid across a set
        of KMetacals (the filter and the noise it cancels must share a frame)
        -- not a tuning knob.
    rotation: galsim.Angle or None
        sky-frame rotation of the input field before metacal (the metacal'd
        field is rotated back by -rotation); ``90 * galsim.degrees`` builds the
        fixnoise correction field, None (default) is the identity
        (galaxy/science path)
    x_interpolant: str
        the InterpolatedImage interpolant (default lanczos15, the metacal
        kernel)
    """

    def __init__(
        self,
        image,
        psf_image,
        wcs,
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
        # SAME target for every type (only the galaxy is sheared)
        gauss = get_azgauss_target_psf(psf_int, flux=self.psf_flux)
        self.target_psf = gauss.dilate(1.0 + 2.0 * self.step)

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
        """the matched-grid k-array of a WORLD-frame profile: to image coords,
        drawKImage at dk, reorder to numpy fft layout, re-center"""
        image_profile = self.wcs.profileToImage(world_profile)
        kim = image_profile.drawKImage(nx=self.Np, ny=self.Np, scale=self._dk)
        return np.fft.ifftshift(kim.array) * self._phase

    def _metacal_field(self, t):
        """the metacal'd (deconv-sheared-reconv) world profile for type t, with
        the sky rotate-back (-rotation) applied so the field is in the final
        image-metacal frame ready to add"""
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
        """crop the center N x N out of an Np x Np draw"""
        lo = self._lo
        return arr[lo : lo + self.N, lo : lo + self.N]

    def get_khats(self):
        """dict type -> Np x Np metacal'd k-array (numpy fft layout); cached"""
        if self._khats is None:
            self._khats = {
                t: self._khat(self._metacal_field(t)) for t in self.types
            }
        return self._khats

    def get_images(self):
        """dict type -> real metacal'd image, cropped to the center N x N (one
        ifft2 per type of the Np-grid get_khats)"""
        return {
            t: self._crop(np.fft.ifft2(k).real)
            for t, k in self.get_khats().items()
        }

    def get_filtered_images(self, filters):
        """
        dict type -> real image of ifft2(khat_t * filters[t]), cropped to N;
        ``filters`` are Np x Np k-space filters (the hybrid H_t).  The filter
        is applied in k on the padded grid -- no second fft -- so the
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
        return self.target_psf.drawImage(
            nx=self.N, ny=self.N, wcs=self.wcs, method='no_pixel'
        ).array


def delta_transfer_kspace(
    psf_image, wcs, dim, step, types, Np=None, rotation=None
):
    """
    the per-type metacal noise transfer P_t = |K_t|^2 on the padded Np x Np
    grid, computed k-natively: push a unit delta through ``KMetacal`` and take
    |k-array|^2 directly (|fft2(delta)| = 1, so this is relative to a white
    input).  ``Np`` (default galsim's draw size) MUST match the KMetacal whose
    noise this filter cancels.  ``rotation = 90 * galsim.degrees`` builds the
    ROTATED transfer P_t^rot = |rotateback(metacal(delta))|^2 (the corr-field
    transfer after the sky rotate-back).

    Returns the Np x Np power dict and the Np used.
    """
    delta = np.zeros((dim, dim))
    delta[dim // 2, dim // 2] = 1.0
    km = KMetacal(
        delta, psf_image, wcs, step=step, types=types, Np=Np, rotation=rotation
    )
    khats = km.get_khats()
    return {t: np.abs(khats[t]) ** 2 for t in types}, km.Np


def make_hybrid_filters_kspace(
    pts, pts_rot, npix, scale, types, extra_iso=0.0, ktol=1e-4, jmat=None
):
    """
    the per-type hybrid filters H_t (on the padded Np grid), in the FINAL
    (image-metacal) frame -- to apply to the sky-rotated, metacal'd correction
    noise (``KMetacal(rotation=90*galsim.degrees).get_filtered_images``).

    ``pts`` is the un-rotated image-metacal transfer (the noise to isotropize,
    in its own frame); ``pts_rot`` is the rotated-back corr-field transfer
    (|T_M(R90 k)|^2).  The common-level m=2/m=6 deficit D is built from ``pts``
    (sky-projected, trapz quadrature -- ``common_harmonic_deficits``) and
    divided by ``pts_rot``:

        H_t = min(sqrt(D_t / pts_rot_t), 1) * taper,

    so adding H_t * corr lands the total noise on the common isotropic level.
    The cap keeps the hybrid from adding more than the full rotated field
    (fixnoise) in any mode; the taper rolls H smoothly to zero out of band.

    Parameters
    ----------
    pts, pts_rot: dict type -> (Np, Np) array
        the un-rotated and sky-rotated metacal transfers (delta_transfer_kspace
        with rotation None and 90*galsim.degrees)
    npix: int
        the grid size Np
    scale: float
        pixel scale [arcsec/pixel]
    extra_iso: float
        extra isotropic deficit power (tunes toward fixnoise; 0 = minimal added
        variance)
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
        padd = deficits[t] + extra_iso
        eps = ktol * pt.max()
        hraw = np.sqrt(
            np.divide(padd, pt, out=np.zeros_like(padd), where=pt > 0)
        )
        out[t] = np.minimum(hraw, 1.0) * pt / (pt + eps)
    return out


def metacal(image, psf_image, wcs, step=0.01, types=DEFAULT_TYPES):
    """plain k-space metacal of one image (no noise correction): dict type ->
    metacal'd image."""
    return KMetacal(image, psf_image, wcs, step=step, types=types).get_images()


def metacal_hybrid(
    image,
    psf_image,
    wcs,
    noise_image,
    step=0.01,
    types=DEFAULT_TYPES,
    extra_iso=0.0,
):
    """
    k-space metacal with the rotated-HYBRID noise correction.

    The standard ``fixnoise`` correction restores the spin-2 symmetry of the
    metacal noise power by adding a full counter-rotated, metacal'd noise
    field.  It works but DOUBLES the variance.  Only the m=2/m=6 anisotropy
    needs cancelling, so the hybrid filters that same (sky-)rotated, metacal'd
    field down to just the anisotropic deficit before adding it: ~0.05x (round
    PSF) to ~0.5x (elliptical PSF) the added variance, vs 1.0x for fixnoise.
    Because the added field is the ACTUAL rotated metacal'd noise it carries
    the genuine (possibly non-stationary) covariance with the cancelling sign
    for free.

    Everything is on ONE shared padded k-grid (the galaxy, the sky-rotated
    noise, and the transfers), with the winning choices baked in: world shear
    frame, galsim's automatic padding, the trapz azimuthal quadrature, the
    sky-angle deficit projection, and the SKY-frame 90-deg rotation of the
    correction field.

    Parameters
    ----------
    image: (N, N) array
        the (noisy) science image to metacal
    psf_image: (N, N) array
        the psf image
    wcs: galsim local/Jacobian wcs, or a 2x2 array
        the pixel->sky wcs (diagonal or distorted); may be the 2x2 pixel->sky
        matrix (col, row order) instead of a galsim wcs
    noise_image: (N, N) array
        the correction noise field -- an independent realization matching the
        image's noise (its correlations / coverage); rotated, metacal'd,
        filtered and added to restore the noise symmetry
    step: float
        metacal shear step
    types: sequence of str
        metacal types (default noshear/1p/1m; add 2p/2m for the full 2x2
        response)
    extra_iso: float
        extra isotropic deficit power (tunes toward fixnoise; 0 = minimal added
        variance)

    Returns
    -------
    dict type -> (N, N) corrected metacal image (the metacal'd science image
    plus the filtered, sky-rotated correction field)
    """
    dim = image.shape[0]
    wcs, jmat = _wcs_and_matrix(wcs)
    scale = float(np.sqrt(abs(np.linalg.det(jmat))))

    # one shared padded grid for the transfer, the galaxy and the noise
    pts, npix = delta_transfer_kspace(psf_image, wcs, dim, step, types)
    pts_rot, _ = delta_transfer_kspace(
        psf_image, wcs, dim, step, types, Np=npix, rotation=90 * galsim.degrees
    )
    hfilt = make_hybrid_filters_kspace(
        pts, pts_rot, npix, scale, types, extra_iso=extra_iso, jmat=jmat
    )

    gal = KMetacal(
        image, psf_image, wcs, step=step, types=types, Np=npix
    ).get_images()
    # the sky-rotated correction field, metacal'd, filtered to the deficit, and
    # rotated back -- already in the final frame
    knoise = KMetacal(
        noise_image,
        psf_image,
        wcs,
        step=step,
        types=types,
        Np=npix,
        rotation=90 * galsim.degrees,
    )
    deficit = knoise.get_filtered_images(hfilt)
    return {t: gal[t] + deficit[t] for t in types}

"""
The impulse response of the process that generated the noise in the image.

The fusion filter measures the metacal noise transfer by pushing an impulse
through the full metacal operator.  For white noise that impulse is a unit
delta (``DeltaImpulse``).  For correlated noise it is an image G whose power
spectrum is the noise power spectrum, |FT(G)|^2 = N (``ImpulseResponse``); for
example the position-averaged resampling response of a coadd.  Pushing G
through the operator gives the transfer |K_t(k)|^2 N(S_t^-1 k), with the
metacal shear acting on N as it does on the data noise, and the deficit,
filter and noise variance factor then come out right for that noise.

A noise kernel is any callable ``noise_kernel(dim, wcs)`` returning a dim x
dim float array centered on pixel ``[dim // 2, dim // 2]``.  Only |FT|^2 is
used, so the normalization is immaterial.
"""
import numpy as np


class DeltaImpulse:
    """
    The unit delta impulse: the noise is white.

    Examples
    --------
    noise_kernel = metacal.DeltaImpulse()
    """
    def __call__(self, dim, wcs):
        """
        Make the delta impulse image.

        Parameters
        ----------
        dim: int
            the image dimension
        wcs: galsim WCS
            unused; part of the noise kernel protocol

        Returns
        -------
        array
            (dim, dim) array with a 1 at [dim // 2, dim // 2]
        """
        dim = int(dim)
        impulse = np.zeros((dim, dim))
        impulse[dim // 2, dim // 2] = 1.0
        return impulse

    def __repr__(self):
        return 'DeltaImpulse()'


class ImpulseResponse:
    """
    The impulse response of a correlated-noise process.

    An image G with |FT(G)|^2 = N, the noise power spectrum.  This is NOT the
    noise autocorrelation; use ``ImpulseResponse.from_autocorr`` for that.

    Parameters
    ----------
    image: 2d array
        the impulse response, centered on pixel [n0 // 2, n1 // 2].  Its
        support must fit within the image dimension it is called with.

    Examples
    --------
    # the position-averaged resampling response of a coadd
    noise_kernel = metacal.ImpulseResponse(response_image)

    # from the measured noise autocorrelation function xi
    noise_kernel = metacal.ImpulseResponse.from_autocorr(xi)

    # an analytic correlation, e.g. gaussian correlated noise
    noise_kernel = metacal.ImpulseResponse.from_gsobject(
        galsim.Gaussian(sigma=0.2),
    )
    """
    def __init__(self, image):
        image = np.asarray(image, dtype=float)
        if image.ndim != 2:
            raise ValueError(f'image must be 2d, got shape {image.shape}')
        if not np.all(np.isfinite(image)):
            raise ValueError('image must be finite')
        if not np.any(image):
            raise ValueError('image must not be all zero')
        self._image = image
        self._gsobject = None

    @classmethod
    def from_autocorr(cls, xi):
        """
        Build the impulse response from the noise autocorrelation function.

        G = FT^-1 sqrt(FT xi) is the zero-phase kernel with |FT(G)|^2 =
        FT(xi) = N.  Negative N from a noisy xi is clipped to zero.

        Parameters
        ----------
        xi: 2d array
            the autocorrelation xi(dx, dy), centered on [n0 // 2, n1 // 2];
            any normalization

        Returns
        -------
        ImpulseResponse
        """
        xi = np.asarray(xi, dtype=float)

        if xi.ndim != 2:
            raise ValueError(f'xi must be 2d, got shape {xi.shape}')

        # zero pad to twice the size so the circular transform does not wrap
        n0, n1 = xi.shape
        m = 2 * max(n0, n1)
        xi_full = np.zeros((m, m))
        r0, r1 = m // 2 - n0 // 2, m // 2 - n1 // 2
        xi_full[r0:r0 + n0, r1:r1 + n1] = xi

        N = np.fft.fft2(np.fft.ifftshift(xi_full)).real.clip(min=0.0)
        G = np.fft.fftshift(np.fft.ifft2(np.sqrt(N)).real)

        # the kernel's support is about half of xi's; crop to xi's size
        return cls(G[r0:r0 + n0, r1:r1 + n1])

    @classmethod
    def from_gsobject(cls, gsobject):
        """
        Build the impulse response from a galsim object.

        The object is drawn on the image grid when called, with
        method='no_pixel'.

        Parameters
        ----------
        gsobject: galsim.GSObject
            e.g. galsim.Gaussian(sigma=0.2) for gaussian correlated noise

        Returns
        -------
        ImpulseResponse
        """
        self = cls.__new__(cls)
        self._image = None
        self._gsobject = gsobject
        return self

    @property
    def image(self):
        """
        The impulse response image, None for a galsim object.
        """
        return self._image

    def __call__(self, dim, wcs):
        """
        Make the impulse response image, centered on [dim // 2, dim // 2].

        Parameters
        ----------
        dim: int
            the image dimension
        wcs: galsim WCS
            A galsim local wcs/Jacobian, used to draw a galsim object.  May
            also be given as the 2x2 pixel->sky matrix

        Returns
        -------
        array
            (dim, dim) array
        """
        dim = int(dim)

        if self._gsobject is not None:
            from ._util import _wcs_and_matrix

            wcs, _ = _wcs_and_matrix(wcs)
            # galsim centers on the true center; for even dim that is half
            # a pixel below [dim // 2, dim // 2]
            off = 0.5 if dim % 2 == 0 else 0.0
            im = self._gsobject.drawImage(
                nx=dim, ny=dim, wcs=wcs, method='no_pixel',
                offset=(off, off),
            )
            return np.asarray(im.array, dtype=float)

        image = self._image
        n0, n1 = image.shape
        r0, r1 = dim // 2 - n0 // 2, dim // 2 - n1 // 2
        if r0 < 0 or r1 < 0 or r0 + n0 > dim or r1 + n1 > dim:
            raise ValueError(
                f'impulse response support {image.shape} does not fit in '
                f'the image dimension {dim}'
            )
        impulse = np.zeros((dim, dim))
        impulse[r0:r0 + n0, r1:r1 + n1] = image
        return impulse

    def __repr__(self):
        if self._gsobject is not None:
            return f'ImpulseResponse.from_gsobject({self._gsobject!r})'
        return f'ImpulseResponse(<{self._image.shape} image>)'

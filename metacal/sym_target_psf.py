"""
Deshear/rotation-symmetrized target reconvolution PSF for metacal.

The target is built from the psf itself rather than replaced by a round
gaussian: the psf is recentered, desheared by its adaptive moments
shape, averaged over nrot rotations, and dilated by the minimal factor
that keeps the reconvolution filter |target/psf| from amplifying.
The result keeps the true psf profile (and hence the sharp look of the
dilate mode) while carrying no spin-2 to leak into galaxy shapes.

The rotation average is what carries the correctness: an nrot-fold
symmetric profile has identically zero spin-2 at every radius, by
symmetry, so the leakage null does not depend on the accuracy of the
measured shape (this includes radially varying psf ellipticity, which
moment-based rounding cannot remove).  The deshear step only reduces the
required dilation to (1 + e/2) with e the psf distortion, which is the
fundamental floor for any round non-amplifying target (it must cover the
psf major axis); without it the symmetrization would need (1 + e).  A
mis-measured shape moves the operating point along the
smoothing/amplification trade-off but cannot reintroduce leakage.

nrot=4 (rotations by multiples of 90 degrees) removes all spin-2, spin-6
and odd harmonics; nrot=8 also removes spin-4 (e.g. diffraction spikes)
at essentially no extra smoothing.
"""
import numpy as np
import galsim


class Symmetrize:
    """
    Get a symmetrized version of the input psf for use as the target
    reconvolution psf

    The psf is recentered, desheared, averaged over nrot rotations and
    dilated by (1 + e/2)

    Parameters
    ----------
    nrot: int
        Number of rotations to average over, 4 or 8.  4 removes all
        spin-2; 8 also removes spin-4
    rng: np.random.RandomState
        For the admom fit guesses
    """

    def __init__(self, nrot, rng):
        import ngmix

        if nrot not in (4, 8):
            raise ValueError(f'nrot must be 4 or 8, got {nrot}')
        self.nrot = nrot

        self.runner = ngmix.runners.Runner(
            fitter=ngmix.admom.AdmomFitter(rng=rng),
            guesser=ngmix.guessers.GMixPSFGuesser(
                rng=rng, ngauss=1, guess_from_moms=True,
            ),
            ntry=4,
        )

    def __call__(self, psf, flux):
        """
        Get the target psf.

        Parameters
        ----------
        psf: galsim object
            the psf, e.g. a galsim.InterpolatedImage
        flux: float
            flux of the output

        Returns
        -------
        galsim.GSObject
        """
        dx, dy, shape = _measure_moments(psf, self.runner)

        # rotations are about the origin, so recenter first; the target is
        # returned centered, consistent with the round gaussian targets
        p = psf.shift(-dx, -dy)

        e = np.hypot(shape.e1, shape.e2)
        p = p.shear(-shape)
        dilation = 1.0 + 0.5 * e

        step = 360.0 / self.nrot
        p = galsim.Sum(
            [p.rotate(i * step * galsim.degrees) for i in range(self.nrot)]
        ) * (1.0 / self.nrot)

        return p.dilate(dilation).withFlux(flux)


def _measure_moments(psf, runner):
    """
    adaptive moments (ngmix admom) centroid offset (in world units,
    relative to the profile origin) and shape of the psf, measured on an
    image drawn on a world-aligned grid at the profile's nyquist scale

    Parameters
    ----------
    psf: galsim object
        the psf, e.g. a galsim.InterpolatedImage
    runner: ngmix.runners.Runner
        the admom runner

    Returns
    -------
    dx, dy, shape
    """
    import ngmix
    from .wcs import ngmix_jacobian

    im = psf.drawImage(method='no_pixel')

    cen = im.true_center
    jac = ngmix_jacobian(
        row=cen.y - im.bounds.ymin,
        col=cen.x - im.bounds.xmin,
        jacmat=np.diag([im.scale, im.scale]),
    )
    obs = ngmix.Observation(image=im.array, jacobian=jac)

    res = runner.go(obs=obs)
    if res['flags'] != 0:
        raise RuntimeError(f'admom psf fit failed, flags={res["flags"]}')

    gm = res.get_gmix()
    dy, dx = gm.get_cen()
    e1, e2, _ = gm.get_e1e2T()

    return dx, dy, galsim.Shear(e1=e1, e2=e2)

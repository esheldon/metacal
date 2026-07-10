from .metacalibration import metacal_modecorr
from .defaults import DEFAULT_TYPES


def metacal_obs_modecorr(
    obs,
    target_psf,
    rng,
    step=0.01,
    types=DEFAULT_TYPES,
):
    """
    Metacal an ngmix.Observation.  If a noise image is present (obs.has_noise()
    is True), the noise corrections are applied.

    Parameters
    ----------
    obs: ngmix.Observation
        The ngmix Observation, with image, psf etc.
    target_psf: A callable that returns a galsim object
        This should be callable with target_psf(psf=psf, flux=flux),
        with psf a galsim object such as galsim.InterpolatedImage.  For
        an example see metacal.AZGauss
    rng: np.random.RandomState
        For adding a little noise to the final PSF image
    step: float
        metacal shear step
    types: sequence of str
        any of 'noshear', '1p', '1m', '2p', '2m'

    Notes
    -----
    ngmix is not actually required; the input "obs" can duck
    type the needed attributes and methods.
    """

    if not obs.has_psf():
        raise ValueError('observation must have a .psf')

    if not obs.has_noise():
        raise ValueError('observation must have a .noise set')

    wcs = obs.jacobian.get_galsim_wcs()

    res = metacal_modecorr(
        image=obs.image,
        psf_image=obs.psf.image,
        noise_image=obs.noise,
        wcs=wcs,
        target_psf=target_psf,
        step=step,
        types=types,
    )

    # pack each back into a ngmix.Observation, rescaling the weight map for the
    # noise-variance increase from the correction (a no-op when
    # noise_var_factor is 1, i.e. no correction was applied)
    odict = {}
    for key, image in res.items():

        oobs = obs.copy()
        oobs.image = image

        if res.noise_var_factor != 1.0:
            oobs.weight = oobs.weight / res.noise_var_factor

        _update_psf_in_obs(res=res, psf_obs=oobs.psf, rng=rng)

        odict[key] = oobs

    return odict


def _update_psf_in_obs(res, psf_obs, rng):
    psf_image = res.psf_image.copy()

    psf_noise = psf_image.max() / 50000.0
    psf_image += rng.normal(
        size=psf_image.shape,
        scale=psf_noise,
    )
    psf_weight = psf_image * 0 + 1.0 / psf_noise ** 2

    # pixels will be updated after exit from context
    with psf_obs.writeable():
        psf_obs.image[:, :] = psf_image
        psf_obs.weight[:, :] = psf_weight

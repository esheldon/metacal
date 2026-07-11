from .defaults import DEFAULT_TYPES


def metacal_noise_correct(
    image,
    psf_image,
    noise_image,
    noise_filter,
    wcs,
    target_psf,
    types=DEFAULT_TYPES,

):
    """
    metacal an image with spin-2 mode correction of the noise

    Parameters
    ----------
    image: array
        the image to metacal
    psf_image: array
        the psf image
    noise_image: array
        A noise field for noise anisotropy correction. The noise field should
        match the noise in image.
    noise_filter: callable
        A callable that generates the filters. See metacal.FusionFilter
        for an example
    wcs: galsim WCS
        A galsim local wcs/Jacobian.  May also be given as the 2x2 pixel->sky
        matrix (col, row order) instead of a galsim wcs
    target_psf: A callable that returns a galsim object
        This should be callable with target_psf(psf=psf, flux=flux),
        with psf a galsim object such as galsim.InterpolatedImage.  For
        an example see metacal.AZGauss
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
    import galsim
    from ._util import _wcs_and_matrix
    from .metacalibration import Metacal
    from .result import MetacalResult

    if image.shape != noise_image.shape:
        raise ValueError(
            f'noise shape mistmatch, {noise_image.shape} != {image.shape}'
        )

    dim = image.shape[0]
    wcs, jmat = _wcs_and_matrix(wcs)

    hfilt, noise_var_factor = noise_filter(
        psf_image=psf_image,
        wcs=wcs,
        target_psf=target_psf,
        dim=dim,
        types=types,
    )

    npix = hfilt[types[0]].shape[0]

    mcal = Metacal(
        image=image,
        psf_image=psf_image,
        wcs=wcs,
        target_psf=target_psf,
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
        psf_image=mcal.get_target_psf_image(),
        noise_var_factor=noise_var_factor,
    )

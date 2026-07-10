class MetacalResult:
    """
    The result of a :func:`metacal` run.

    Keyed by metacal type to the sheared images (``res['noshear']``); and is
    iterable / supports ``keys``/``values``/``items``/``in``/``len`` like a
    dict -- while also exposing the reconvolution psf image and the predicted
    noise-variance increase.

    Attributes
    ----------
    images: dict
        the sheared images keyed by metacal type
    psf_image: array
        the round reconvolution (target) psf image, common to all types
    noise_var_factor: float
        the factor by which the per-pixel noise variance increased due to the
        rotated-hybrid noise correction, relative to plain (uncorrected)
        metacal.  1.0 when no noise correction was applied; ~1.04 for a round
        psf, rising toward 2.0 (the full fixnoise level) as the psf becomes
        more elliptical.  The corrected noise lands at a common level across
        all types, so this is a single number.  Downstream, multiply the noise
        variance (or divide the weight map) by this factor.
    """

    def __init__(self, images, psf_image, noise_var_factor):
        self.images = images
        self.psf_image = psf_image
        self.noise_var_factor = noise_var_factor

    def __getitem__(self, key):
        return self.images[key]

    def __iter__(self):
        return iter(self.images)

    def __len__(self):
        return len(self.images)

    def __contains__(self, key):
        return key in self.images

    def keys(self):
        return self.images.keys()

    def values(self):
        return self.images.values()

    def items(self):
        return self.images.items()

    def __repr__(self):
        return (
            f'MetacalResult(types={list(self.images)}, '
            f'noise_var_factor={self.noise_var_factor:.4f})'
        )

# metacal

Implementation of Metacalibration weak lensing shear calibration.

This supercedes the metacal code that was part of ngmix.

It uses a new noise correction that increases the noise by a few percent for
typical PSFs, as compared to the old correction that increases it by sqrt(2) in
all cases.

Examples
---------

```python
import galsim
import metacal

# metacalibrate an image with noise correction.

res = metacal.metacal_modecorr(
    image=image,
    psf_image=psf_image,
    noise_image=noise_image,
    wcs=wcs,
    target_psf=metacal.AZGauss(),
)

# The correction uses the input noise image, matched to the noise of the data,
# along with the metacalibration transfer function to cancel spin-2 modes
# imprinted on the noise power spectrum.  A minimal amount of noise is added,
# increasing the total noise by a few percent for typical PSFs.
#
# The noise can be correlated and non stationary.
#
# The user can provice their own target_psf, either as a callable or a
# galsim.GSObject

# The returned res is a MetacalResult, which is a dict-like keyed by metacal
# type ('1p', 'noshear', etc).  Each item is the corresponding image.  The
# result also has attributes
#   .psf_image: the final target reconvolution PSF
#   .noise_var_vactor, the factor by which the noise^2 was increased by
#       the added noise

# The metacal_image function implements the basic metcalibration operations, It
# and the metacal.metacalibration.Metacal class can be used to build your own
# noise corrections.

res = metacal.metacal_image(
    image=image,
    psf_image=psf_image,
    wcs=wcs,
    target_psf=metacal.AZGauss(),
)
```

Metacalibration can also be run on an ngmix Observation (although ngmix is not
a requirement).
```python
import galsim

res = metacal.metacal_modecorr_obs(obs)

# Here res is a dict holding observations for each requested metacal
# type.  The Observation must have a the .noise filled
# for use in corrections
```

Why a new package?
------------------

The new noise method and reconvolution kernel AZGauss included in this package
are significantly better than the old defaults from ngmix. But, due to poor
design choices in ngmix (using a "default keyword" structure for these
features), there was no easy way to slot them in as the new defaults without
breaking backwards compatibility.  We decided to make a clean break.

This package is designed so that the user must explicity send the object that
creates the target psf (we provide AZGauss) or their own galsim.GSObject.  No
default is provided.

Simularly, we provide an explicit function that implements metacal with the
"modecorr" noise correction.  If another method is developed, we will provide
new function rather than change the behavior of the existing one.  And the user
can create of course create their own.

## Dependencies

numpy, galsim

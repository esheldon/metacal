# metacal

Implementation of Metacalibration weak lensing shear calibration.

This supercedes the metacal code that was part of ngmix.

It uses a new "hybrid" noise correction that increases the noise by about 2%
for typical PSFs, as compared to the old correction that increases it by
sqrt(2) in all cases.  It uses the reconvolution kernel "azgauss" that is
robust to noise and works well for even complex optical PSFs.

## What it does

Runs the metacalibration algorithm on images.

`metacal(image, psf_image, wcs, noise_image=noise_image)` metacalibrates an
image and applies the cheap hybrid noise correction in one shared k-space
frame:

Why a new package?
------------------

The new methods a clearly better than the old ones, but due to poor design
choices in ngmix (using a "default keyword" structure for these features),
there was no easy way to slot them in as the new defaults.  Easier to make a
clean break.

Examples
---------

```python
import galsim
from metacal import metacal

# send noise_image= to apply the noise correction.  Noise can
# be correlated and non stationary
res = metacal(image, psf_image, wcs, noise_image=noise_image)
```

It can also work on an ngmix Observation (although ngmix is not a requirement).
```python
import galsim
from metacal import metacal_obs

# if the obs has a .noise attribute, the noise correction is applied
res = metacal_obs(obs)
```

## Dependencies

numpy, galsim

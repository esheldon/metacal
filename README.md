# metacal

Implementation of Metacalibration weak lensing shear calibration.

## What it does

`metacal(image, psf_image, wcs, noise_image=noise_image)` metacalibrates an
image and applies the cheap hybrid noise correction in one shared k-space
frame:

- the deconvolve / shear / reconvolve is kept in galsim k-space and sampled once
  with `drawKImage` onto a numpy-fft-matched grid — no second, aliasing
  real↔k round trip;
- Uses the hybrid noise correction, rather than old "fixnoise".  This applies
  a minimal amount of extra noise to reduce noise biases.
  Typically increases noise by about 2.5%.  Works for non stationary
  noise fields.

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

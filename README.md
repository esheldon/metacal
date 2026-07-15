# metacal

Metacalibration weak lensing shear calibration.

This package supercedes the metacal code that was part of `ngmix`.

We provide the "fusion" noise correction that increases the noise by only a few
percent for typical PSFs, as compared to the old correction that increases it
by `sqrt(2)` in all cases.  We also provide the "azgauss" psf target.

Examples
---------

### Process an image without any noise correction applied

```python
import metacal

#
# metacalibrate an image.
#

res = metacal.metacal_image(
    image=image,
    psf_image=psf_image,
    wcs=wcs,
    target_psf=metacal.AZGauss(),
    types=('noshear', '1p', '1m', '2p', '2m'),
)
```

`metacal_image` implements the basic metcalibration operations.  No noise
correction is applied.  The `target_psf` is specified by the user, either as a
callable or a `galsim.GSObject`.  We provide `AZGauss`.  The `wcs` is a
`local/Jacobian galsim WCS`.

The returned `res` is a `MetacalResult`, which is a `dict`-like keyed by
metacal type `('1p', 'noshear', etc)` with each item the corresponding image.
The result also has attributes
- `.psf_image` the final target reconvolution PSF
- `.noise_var_vactor` the factor by which the `noise^2` was increased by the added noise

### Noise Correction

```python
# correct the noise, using the Fusion noise filter
res = metacal.metacal_noise_correct(
    image=image,
    psf_image=psf_image,
    noise_image=noise_image,
    noise_filter=metacal.FusionFilter(),
    wcs=wcs,
    target_psf=metacal.AZGauss(),
    types=('noshear', '1p', '1m', '2p', '2m'),
)
```

The fusion correction uses the input noise image along with the metacalibration
transfer function to cancel spin-2 modes imprinted on the noise power spectrum
by the metacalibration process.  A minimal amount of noise is added, increasing
the total noise by a few percent for typical PSFs.

The noise can be correlated and non stationary.

# Processing an `ngmix.Observation`

Metacalibration can also be run on an `ngmix.Observation` (although ngmix is not
a requirement, you can duck type it).

```python
import metacal

res = metacal.metacal_obs(
    obs=obs,
    noise_filter=metacal.FusionFilter(),
    target_psf=metacal.AZGauss(),
    types=('noshear', '1p', '1m', '2p', '2m'),
    rng=rng,
)
```
The input `Observation` must have a the `.noise` filled for use in corrections.
Here `res` is a dict holding `Observations` for each requested metacal type.

Why a new package?
------------------

The new fusion noise method and azgauss reconvolution kernel included in this
package are significantly better than the old defaults from `ngmix`. But, due
to poor design choices in `ngmix` (using a "default keyword" structure for
these features), there is no easy way to slot them in as the new defaults
without breaking backwards compatibility.  We decided to make a clean break.

This package is designed so that the user must explicity send the callable or
`galsim.GSObject` that creates the target psf.  No default is provided.

Similarly, the user must explicitly send the callable that provides the filter
for the noise used in the noise correction.

## Dependencies

numpy, galsim

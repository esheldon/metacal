# metacal

k-space-native metacalibration with the **rotated-hybrid** noise correction.

This is the validated, production-oriented subset of the `mcal_hybrid` research
package, with the winning choices baked in and the dead-end experiments removed.

## What it does

`metacal_hybrid(image, psf_image, wcs, noise_image)` metacalibrates an image and
applies the cheap hybrid noise correction in one shared k-space frame:

- the deconvolve / shear / reconvolve is kept in galsim k-space and sampled once
  with `drawKImage` onto a numpy-fft-matched grid — no second, aliasing
  real↔k round trip;
- the standard `fixnoise` correction restores the spin-2 symmetry of the metacal
  noise power by adding a full counter-rotated metacal'd noise field, which
  **doubles the variance**; only the m=2/m=6 anisotropy needs cancelling, so the
  hybrid filters that same rotated, metacal'd field down to just the anisotropic
  **deficit** (~0.05× round PSF to ~0.5× elliptical the added variance);
- because the added field is the **actual** rotated metacal'd noise (just
  attenuated), it carries the genuine, possibly non-stationary covariance with
  the cancelling sign for free.

```python
import galsim
from metacal import metacal_hybrid, distortion_matrix, galsim_wcs

wcs = galsim_wcs(distortion_matrix(0.2, theta_deg=10, g1=0.01, g2=-0.005))
mcal = metacal_hybrid(image, psf_image, wcs, noise_image,
                      types=('noshear', '1p', '1m', '2p', '2m'))
# mcal['noshear'], mcal['1p'], ...  -> corrected metacal images
```

Use the 5 types `noshear/1p/1m/2p/2m` for the full 2×2 response (the trace
**Rbar** = ½(R₁₁+R₂₂) is the recommended estimator — see "Choices" below).

Lower-level pieces are public too: `KMetacal` (the single-image k-space metacal),
`delta_transfer_kspace` (the per-type noise transfer), and
`make_hybrid_filters_kspace` (the deficit filter).

## The choices baked in (and why)

These were established empirically in `mcal_hybrid` (full derivations in its
`docs/`); here they are simply the defaults, not options.

| choice | what | why |
|---|---|---|
| **k-space metacal** | one `drawKImage` onto the fft-matched grid | removes the redundant real↔k round trip that aliases the near-Nyquist m=2 and leaks a rotation-dependent bias |
| **sky-frame rotation** | the correction noise is rotated 90° in the world frame (not pixel `np.rot90`) | under a non-conformal wcs a pixel 90° is not a sky 90°, leaving a spurious additive c; the sky rotation removes it |
| **trapz quadrature** | the azimuthal m-mode projection weights each mode by its angular Voronoi cell | removes the square-grid cos4φ projection gain error (the conformal-rotation leak) |
| **sky-angle deficit** | the m=2/m=6 deficit is projected in sky angle (via the wcs jacobian) | the shear bias is a sky-frame contraction, so it is the sky-frame spin-2 that must be isotropized; a diagonal wcs is an exact no-op |
| **automatic padding** | the padded draw grid is galsim's own `drawFFT` size | enough to keep the periodic ifft from wrapping; a manual override is not needed |
| **world shear frame** | the metacal shears along the world axes | the grid-frame shear was a tested dead end (did not fix the leak) |

**Recommended estimator: the trace response Rbar = ½(R₁₁+R₂₂).** Metacal
reconvolves to a round target, so the true response is isotropic; the trace is
blind to the small residual spin-2 contaminations (the cos4φ projection leak and
the intrinsic finite-shear response anisotropy), removing them at the estimator
for free. Build the 2×2 response from the 5 types.

## Dependencies

numpy, galsim, and **ngmix on the `mcal-gauss-stability` branch** (for the
`azgauss` reconvolution target psf — the noise-robust azimuthal-average kernel).

## Tests

`pytest` (needs the ngmix branch above). The key test asserts the k-space
metacal field equals ngmix's real-space drawn metacal to ~machine precision.

## Status / not yet here

This is the metacal + correction core. Not ported (lives in `mcal_hybrid`): the
shear-bias validation sim, the condor/config harness, and the edgy-coadd
non-stationary noise models. An ngmix-`Observation` convenience wrapper (building
the wcs from the obs jacobian and folding the added correction variance into the
weight map) is the natural next addition.

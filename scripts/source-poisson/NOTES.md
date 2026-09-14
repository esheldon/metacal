# Fusion noise correction with source Poisson noise

Question: does the fusion noise correction still give unbiased shear when
the noise includes a source Poisson term, so that the noise is
non-stationary and correlated with the galaxy?  All earlier tests used sky
noise only.

## Design (`poisson_noise_test.py`)

- Round Exponential (hlr 0.5"), Moffat psf (beta 2.5, fwhm 0.9", g1 = 0.05),
  48x48 stamps at 0.2"/pixel, perfectly centered.  AZGauss target psf, all
  five metacal types, ngmix GaussMom (fwhm 1.2") for the shapes.
- True shear +/-0.02 in g1, analyzed as pairs:
  m = <e1+ - e1-> / (2 R11 g) - 1 and c1 = <e1+ + e1-> / (2 R11).
- The object is fixed by its flux; the sky variance is set separately.
  `--calibrate SNR` finds the sky variance giving that typical GaussMom
  s2n with sky noise alone.  Never derive the flux from a target S/N: that
  couples the object to the noise level and its own profile.
- Variance reduction, both essential on CPU:
  1. the +g and -g images share the standard-normal draws, scaled by each
     image's own variance map;
  2. every measurement is averaged over the noise field and its negative
     (image noise and correction noise together).  Metacal is linear in
     the image, so this cancels all odd orders in the noise and keeps the
     quadratic noise bias.  It cuts the error on m by ~10x at fixed
     realization count (the per-object R11 scatter is ~0.35 at S/N 15,
     which would otherwise need ~1e6 realizations for 1e-3 on m).
- Errors are bootstrap over realizations within a seed; `combine.py`
  merges independent seeds by inverse-variance weighting and reports a
  chi2 across seeds.

Cases (image noise variance / noise-image variance):

| case     | image          | noise image    | purpose                       |
|----------|----------------|----------------|-------------------------------|
| sky      | sky            | sky            | the usual matched test        |
| poisson  | sky + gal_true | sky + gal_true | matched, source term included |
| mismatch | sky + gal_true | sky            | weight map lacks source term  |
| nocorr   | sky + gal_true | none           | plain metacal, sensitivity    |

The Poisson term is Gaussian with the model variance, not a true Poisson
draw, so the test isolates the non-stationarity from the skewness.

## Running

```
cd scripts/source-poisson
python poisson_noise_test.py --flux 1000 --seed 3 --calibrate 15
#   -> flux 1000.0 snr 15.0 sky_var 23.0349
for seed in 21 22 23 24 25 26; do
  python poisson_noise_test.py --case poisson --nreal 3000 --seed $seed \
      --flux 1000 --skyvar 23.0349 --out quick/poisson_s$seed.json &
done
python combine.py "quick/poisson_s*.json"
```

About 0.25 s per realization for the corrected cases (four metacal calls
each), half that for nocorr.  3000 realizations x 6 seeds takes ~13 min on
6 cores and gives sigma_m ~ 1e-3.

## Results (2026-09-14)

Sky variance 23.03 throughout (typical s2n 15 at flux 1000).  At a fixed
sky level both the S/N and the source-Poisson-to-sky variance ratio scale
with flux, so the Poisson-dominated regime is a brighter object at the
same sky, not a fainter object with a rescaled sky.

| flux | s2n  | peak source var / sky var | case    | n     | m                   | c1                   |
|------|------|---------------------------|---------|-------|---------------------|----------------------|
| 1000 | 15   | 0.58                      | poisson | 18000 | -0.0008 +/- 0.0011  | -0.00017 +/- 0.00014 |
| 3000 | ~45  | 1.73                      | poisson | 18000 | +0.00045 +/- 0.00016 | -0.00002 +/- 0.00002 |

The pair estimator has a noiseless offset of m = +0.00035 at both fluxes
(finite shear step plus the +/-0.02 nonlinearity; c1 is ~1e-7).  That is
not noise bias; the script now records it as `m_noiseless` and
`combine.py` prints m - m_noiseless.  Relative to it:

| flux | m - m_noiseless        |
|------|------------------------|
| 1000 | -0.0012 +/- 0.0011     |
| 3000 | +0.0001 +/- 0.00016    |

For scale, the uncorrected (nocorr) bias at flux 1000 is roughly +0.6 to
+1 percent from short sizing runs, so the correction removes it to within
about a tenth of its size with the source term present.  At flux 3000
the expected uncorrected bias is ~1e-3 (it scales as 1/s2n^2), so the
0.00016 error there still resolves it.  The sky-only baseline at flux
3000 is running (seeds 41-46) to confirm the corrected sky case sits at
the same place.

## Not yet done

- sky and nocorr baselines at the same precision with the fixed-flux setup
- the mismatch case
- flux 10000 at sky variance 23 (peak ratio ~5.8)
- a detection phase: detected centers instead of known ones
- no GPU runs for these tests (CPU only, by decision)

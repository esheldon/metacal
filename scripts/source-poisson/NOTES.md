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
| 3000 | ~45  | (sky noise only)          | sky     | 18000 | +0.00034 +/- 0.00010 | +0.00001 +/- 0.00001 |

The pair estimator has a noiseless offset of m = +0.00035 at both fluxes
(finite shear step plus the +/-0.02 nonlinearity; c1 is ~1e-7).  That is
not noise bias; the script now records it as `m_noiseless` and
`combine.py` prints m - m_noiseless.  Relative to it:

| flux | case    | m - m_noiseless        |
|------|---------|------------------------|
| 1000 | poisson | -0.0012 +/- 0.0011     |
| 3000 | poisson | +0.0001 +/- 0.00016    |
| 3000 | sky     | -0.0000 +/- 0.00010    |

Conclusion so far: with the source Poisson term in both the image and the
noise realization, the corrected shear is unbiased to 1e-3 at s2n 15 and
to 2e-4 at s2n 45, and at s2n 45 it agrees with the sky-only case
(difference +0.0001 +/- 0.0002).  c1 and c2 are zero at 2e-5.

For scale, the uncorrected (nocorr) bias at flux 1000 is roughly +0.6 to
+1 percent from short sizing runs, so the correction removes it to within
about a tenth of its size with the source term present.  At flux 3000
the expected uncorrected bias is ~1e-3 (it scales as 1/s2n^2), so the
0.00016 error there still resolves it, but nocorr has not been run at
that flux.

## Detection phase (`mdet/`, started 2026-09-15)

Objects on a grid at random sub-pixel positions, detected with sep and
measured with weighted moments through simcoadd-mdet, so the centers
come from detection.

- simcoadd: `SourcePoissonNoiseField(rng, base, gain)` in
  `simcoadd/noise/poisson.py` wraps any sky noise field (white or
  coadd) and adds Gaussian noise with per-pixel variance model / gain,
  folding it into the weight map.  `make_obs` keeps the noiseless image
  after drawing and passes it to both the image-noise call and the
  independent noise realization, so `obs.noise` and `obs.weight` carry
  the source term matched to the image.  `Survey.gain` (electrons per
  flux unit, 711 for LSST i) is 1 / flux_factor; flux * gain reproduces
  the descwl electron count for both the fixed and wldb catalogs, and
  the sky closes the same way.  Tests in `tests/test_source_poisson.py`.
- simcoadd-mdet: required `source_poisson: bool` in the noise config,
  wired through `_get_noise_field`; every config in `configs/` got
  `source_poisson: false`.

Units: the source term is Gaussian with the Poisson variance from the
true model, not a Poisson draw.

Regime: the peak-source-variance-to-sky-variance ratio is the ratio of
the object's peak surface brightness to the sky surface brightness,
and it is invariant under exposure time and the number of coadded
exposures (sky and source counts both scale the same way; only S/N
grows).  descwl's LSST i sky is 20.5 mag/arcsec^2 (32.5 e-/s/pixel,
975 e- per 30 s visit, 179476 e- over the 184-visit year-10 coadd).
For the exp hlr 0.5" galaxy the psf-convolved peak is 0.43 mag/arcsec^2
brighter than the total magnitude, so at the real LSST sky:

| mag  | matched S/N | peak src var / sky var |
|------|-------------|------------------------|
| 20.0 | 1450        | 0.67                   |
| 21.0 | 580         | 0.27                   |
| 22.0 | 230         | 0.11                   |
| 24.5 | 23          | 0.011                  |

So in LSST the source term matters only for objects at S/N in the
hundreds, where noise bias is negligible.  A regime with both effects
at once is a dark (space-like) sky; `noise_factor` scales the sky
alone, so mag 28.8 with noise_factor 0.0143 gives S/N 31 with ratio
0.99 (2100 e- on 37 e-/pixel of sky).  That is the same regime as the
perfect-center test (sky 23 e-/pixel), a stress test, not LSST.

Configs: `lsst-bright-*` (mag 21, real sky: the LSST-realistic case,
run locally) and `stress-*` (mag 28.8, noise_factor 0.0143: the
stress test, for Erin's remote condor system).

`mdet/run_pairs.py` builds the +g/-g configs, runs matched seeds in a
process pool and calls `simcoadd-mdet-doshear-cancel` with
`mdet/cuts-wmom.yaml` (Trat_min 1.2: wmom sizes are post-psf).  One
trial (350x350, 25 objects, 5 types) takes ~3 s.

Calibration run 2026-09-15 (mag 28.8, nf 0.0143, source poisson on,
psf g_sigma 0.03, 60 seeds x 2 trials per sign, 6 cores, 3.5 min):
3000 objects, wmom S/N 22.7, T/Tpsf 1.45, R 0.38, m1 = -0.07 +/- 0.05.
So sigma_m ~ 2.8 / sqrt(N_obj); 1e-3 needs ~8e6 objects.  Throughput
is ~90k objects per hour per sign on 6 cores at the 9.5" grid spacing,
i.e. sigma_m ~ 9e-3 / sqrt(hours) per case.  A 6" spacing (81 objects
per 350x350 trial, same metacal cost) would give ~5e-3 / sqrt(hours).
The +/-noise antithetic trick that made the perfect-center test cheap
is not available in the sim without adding a noise-sign option.

The 9.5" spacing is the simcoadd-mdet example value (47.5 pixels,
matching the 49-pixel stamps of the single-object fitters); for wmom a
6" spacing would be safe (neighbors ~12 weight sigmas out) but was not
used.

### Running on condor

The explicit configs `mdet/stress-{poisson,sky}-{plus,minus}.yaml` are
generated by `run_pairs.make_config` and are what a remote run needs,
plus `mdet/cuts-wmom.yaml`.  The remote must have simcoadd and
simcoadd-mdet with the source-poisson changes installed.
`simcoadd-mdet-make-condor` draws the per-job seeds from `--seed`, so
the same `--seed` for the plus and minus sets gives matched pairs:

```
simcoadd-mdet-make-condor --run stress-poisson-plus  --seed 3001 \
    --njobs 1000 --ntrial 320 --config stress-poisson-plus.yaml
simcoadd-mdet-make-condor --run stress-poisson-minus --seed 3001 \
    --njobs 1000 --ntrial 320 --config stress-poisson-minus.yaml
# submit stress-poisson-plus-3001/stress-poisson-plus-3001.condor etc.
# then, with one output path per line in each list:
simcoadd-mdet-doshear-cancel --config cuts-wmom.yaml \
    --flist-plus plus.txt --flist-minus minus.txt \
    --outfile shear-stress-poisson.fits --seed 1 --nproc 8
```

1000 jobs x 320 trials x 25 objects = 8e6 objects per sign, ~1e-3 on
m; each job is ~16 min.  The same for the sky case with its configs.

```
python mdet/run_pairs.py --outdir out --mag 28.8 --noise-factor 0.0143 \
    --source-poisson true --g 0.02 --psf-g-sigma 0.03 \
    --nseeds 60 --ntrial 2 --seed0 1000 --nproc 6
```

## Not yet done

- sky and nocorr baselines at the same precision with the fixed-flux setup
- the mismatch case
- flux 10000 at sky variance 23 (peak ratio ~5.8)
- detection phase: calibration run for the error scaling, then the
  source-poisson and sky-only pairs at the chosen regime
- no GPU runs for these tests (CPU only, by decision)

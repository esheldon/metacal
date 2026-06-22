# metacal — k-space metacalibration + rotated-hybrid noise correction

The clean, production-oriented extraction of the **winning** method from the
`mcal_hybrid` research package (`~/git/mcal_hybrid`).  That repo explored many
ideas and kept everything as switchable options; this repo keeps ONLY what works
and bakes it in.  When in doubt about *why* a choice is the way it is, the
derivations and the condor validation live in `mcal_hybrid` (its `CLAUDE.md` and
`docs/`, especially `docs/cos4phi-gain-error.tex`).

## Package layout

- `metacal/kmetacal.py` — the core.
  - `KMetacal(image, psf_image, wcs, step, types, Np=None, rotation=None)` —
    single-image k-space azgauss metacal (world shear frame).  deconv/shear/
    reconv in galsim k-space, one `drawKImage` onto the numpy-fft-matched grid,
    `ifft2`, crop.  `rotation=90*galsim.degrees` (a `galsim.Angle`, not a bare
    number; None default) is the SKY-frame fixnoise rotation (rotate +90 world
    before metacal, −90 after).  `Np` is galsim's automatic drawFFT size; it is
    exposed ONLY so a set of metacals can share one grid (the filter and the
    noise it cancels must share a frame) — NOT a tuning knob.
  - `delta_transfer_kspace(...)` — the per-type transfer `P_t=|K_t|²` (a delta
    impulse through KMetacal); `rotation=90*galsim.degrees` gives the rotated
    transfer.
  - `make_hybrid_filters_kspace(pts, pts_rot, npix, scale, types, jmat)` — the
    final-frame deficit filter `H=min(√(D/pts_rot),1)·taper`, `D` from `pts`.
  - `metacal(image, psf_image, wcs, ...)` — plain metacal (no correction).
  - `metacal_hybrid(image, psf_image, wcs, noise_image, ...)` — THE deliverable:
    metacal + the rotated-hybrid correction on one shared grid, all winning
    choices baked in.
- `metacal/deficit.py` — `common_harmonic_deficits` (trapz quadrature, sky-angle
  projection via `jac`), `_trapz_weights`, `jacobian_matrix` (ngmix → 2×2).
  numpy only.
- `metacal/wcs.py` — `distortion_matrix`, `galsim_wcs`, `ngmix_jacobian`.
- `tests/` — `test_kmetacal.py` (k-space metacal == ngmix draw to ~machine
  precision; transfer; sky-rotation conformal-match / distortion-diverge),
  `test_hybrid.py` (`metacal_hybrid` runs, filters capped, correction is the
  spin-2 deficit).

## The baked-in choices (were options in mcal_hybrid)

- **k-space metacal** (not the real-space ngmix path + a 2nd fft).
- **sky-frame 90° rotation** of the correction field
  (`rotation=90*galsim.degrees`), not the pixel `np.rot90` — fixes the
  non-conformal additive c.
- **trapz** azimuthal quadrature — fixes the square-grid cos4φ rotation leak.
- **sky-angle deficit** projection — always pass the wcs jacobian; a diagonal
  wcs is an exact identity no-op, so there is no separate flag.
- **automatic padding** — galsim's drawFFT size; no manual `Np` override.
- **world shear frame** — the grid frame was a dead end.

## Recommended estimator

**Rbar = ½(R₁₁+R₂₂)**, the trace of the 2×2 response (build from the 5 types
`noshear/1p/1m/2p/2m`).  Metacal's reconvolution target is round, so the true
response is isotropic and the trace is blind to the small residual spin-2
contaminations (the cos4φ leak and the intrinsic finite-shear g² response
anisotropy).  The metacal package produces the sheared images; the response /
Rbar is formed downstream by the shape estimator.

## What was deliberately LEFT OUT (dead ends / out of scope)

- the real-space correction (`mcal_hybrid/correction.py`: `_delta_transfer` rfft,
  `_kfilter`, `make_hybrid_filters`, the ngmix `get_all_metacal_fixnoise_hybrid`,
  `rot_power_90/270`, `rfft_power_to_full`) — superseded by the k-space path;
- the grid-frame metacal shear (`shear_frame='grid'`);
- the user `Np`/k-space padding override (the padding sweep);
- the uniform azimuthal quadrature (`quad='uniform'`);
- the pixel `np.rot90` rotate-back path;
- the shear-bias validation sim, the YAML/condor harness, and the edgy-coadd
  non-stationary noise models — those stay in `mcal_hybrid`.

## Dependencies

numpy, galsim, and **ngmix on the `mcal-gauss-stability` branch** (the `azgauss`
target psf).  `pip install -e .` or `PYTHONPATH=.`.

## Workflow

- Commit/push only when asked (the user handles commits/merges).
- Match the existing code's style (comment density, naming, idiom).

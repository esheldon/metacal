"""
Shear recovery with the fusion noise correction when the noise includes
source Poisson noise.

Round Exponential galaxy, elliptical Moffat psf, sheared by +g and -g.
Each realization is an antithetic pair: the +g and -g images share the
same standard-normal draws (scaled by each image's own variance map),
so the noise largely cancels in e+ - e-.  Each measurement is also
averaged over the noise field and its negative (both the image noise
and the correction noise); metacal is linear in the image, so this
cancels every odd order in the noise and leaves the quadratic noise
bias with much smaller scatter.

    m  = <e1+ - e1-> / (2 R11 g) - 1
    c1 = <e1+ + e1-> / (2 R11)

Cases (image noise variance / noise_image variance):
    sky       sky            / sky              the usual matched test
    poisson   sky + gal_true / sky + gal_true   matched, source term included
    mismatch  sky + gal_true / sky              noise image lacks source term
    nocorr    sky + gal_true / (none)           plain metacal, no correction
"""
import argparse
import json
import time
import numpy as np
import galsim
import ngmix

import metacal
from metacal.metacalibration import SHEAR_STEP

DIM = 48
SCALE = 0.2
TYPES = ('noshear', '1p', '1m', '2p', '2m')
PSF_G1 = 0.05
GTRUE = 0.02
WMOM_FWHM = 1.2
CEN = (DIM - 1) / 2


def make_scene(flux):
    """
    the object is fixed by its flux and profile alone; the noise level is
    set separately (see calibrate_sky_var)
    """
    psf = galsim.Moffat(beta=2.5, fwhm=0.9).shear(g1=PSF_G1)
    psf_im = psf.drawImage(nx=DIM, ny=DIM, scale=SCALE).array
    gal0 = galsim.Exponential(half_light_radius=0.5).withFlux(flux)

    gals = {}
    for sign in (+1, -1):
        gal = gal0.shear(g1=sign * GTRUE)
        gals[sign] = galsim.Convolve(gal, psf).drawImage(
            nx=DIM, ny=DIM, scale=SCALE,
        ).array
    return psf_im, gals


def calibrate_sky_var(flux, snr, seed, nreal=500):
    """
    the sky variance at which the typical (mean) GaussMom s2n of the
    fixed object, with sky noise only, is ``snr``.  Measured at a trial
    variance and scaled: the moments flux is linear in the image and its
    error is proportional to sigma, so the mean s2n goes as 1/sigma.
    """
    psf_im, gals = make_scene(flux)
    rng = np.random.RandomState(seed)
    jac = ngmix.DiagonalJacobian(row=CEN, col=CEN, scale=SCALE)
    trial = 1.0
    s2n = []
    for i in range(nreal):
        for s in gals:
            image = gals[s] + rng.normal(size=(DIM, DIM)) * np.sqrt(trial)
            obs = ngmix.Observation(
                image=image, weight=np.ones_like(image) / trial,
                jacobian=jac,
            )
            s2n.append(ngmix.gaussmom.GaussMom(fwhm=WMOM_FWHM).go(obs)['s2n'])
    s2n_trial = float(np.mean(s2n))
    return trial * (s2n_trial / snr) ** 2


def measure(image):
    jac = ngmix.DiagonalJacobian(row=CEN, col=CEN, scale=SCALE)
    obs = ngmix.Observation(
        image=image, weight=np.ones_like(image), jacobian=jac,
    )
    res = ngmix.gaussmom.GaussMom(fwhm=WMOM_FWHM).go(obs)
    if res['flags'] != 0:
        return None
    return res['e']


def run_case(case, nreal, seed, flux, sky_var):
    psf_im, gals = make_scene(flux)
    wcs = np.diag([SCALE, SCALE])
    target_psf = metacal.AZGauss()

    # the psf is fixed, so build the fusion filter once
    hfilt, fac = metacal.FusionFilter()(
        psf_image=psf_im, wcs=wcs, target_psf=target_psf,
        dim=DIM, types=TYPES,
    )

    def noise_filter(**kw):
        return hfilt, fac

    if case == 'sky':
        imvar = {s: sky_var + 0 * gals[s] for s in gals}
        nzvar = imvar
    elif case == 'poisson':
        imvar = {s: sky_var + gals[s] for s in gals}
        nzvar = imvar
    elif case == 'mismatch':
        imvar = {s: sky_var + gals[s] for s in gals}
        nzvar = {s: sky_var + 0 * gals[s] for s in gals}
    elif case == 'nocorr':
        imvar = {s: sky_var + gals[s] for s in gals}
        nzvar = None
    else:
        raise ValueError(case)

    rng = np.random.RandomState(seed)
    e = {s: {t: [] for t in TYPES} for s in gals}
    nfail = 0
    t0 = time.time()
    for i in range(nreal):
        u1 = rng.normal(size=(DIM, DIM))
        u2 = rng.normal(size=(DIM, DIM))
        pair = {}
        for s in gals:
            # +noise / -noise antithetic pair: metacal is linear in the
            # image, so the mean cancels all odd orders in the noise and
            # keeps the (even-order) noise bias with far less scatter
            vals = []
            for ns in (+1, -1):
                image = gals[s] + ns * u1 * np.sqrt(imvar[s])
                if nzvar is None:
                    res = metacal.metacal_image(
                        image=image, psf_image=psf_im, wcs=wcs,
                        target_psf=target_psf, types=TYPES,
                    )
                else:
                    noise = ns * u2 * np.sqrt(nzvar[s])
                    res = metacal.metacal_noise_correct(
                        image=image, psf_image=psf_im, noise_image=noise,
                        noise_filter=noise_filter, wcs=wcs,
                        target_psf=target_psf, types=TYPES,
                    )
                vals.append({t: measure(res[t]) for t in TYPES})
            pair[s] = {
                t: (
                    None
                    if vals[0][t] is None or vals[1][t] is None
                    else 0.5 * (vals[0][t] + vals[1][t])
                )
                for t in TYPES
            }

        if any(v is None for s in pair for v in pair[s].values()):
            nfail += 1
            continue
        for s in gals:
            for t in TYPES:
                e[s][t].append(pair[s][t])

    dt = time.time() - t0
    e = {s: {t: np.array(v) for t, v in e[s].items()} for s in e}
    return summarize(e, nfail, dt, fac, gals, flux, sky_var)


def summarize(e, nfail, dt, fac, gals, flux, sky_var):
    n = e[1]['noshear'].shape[0]
    # per-pair quantities, e1 index 0, e2 index 1
    ep = e[1]['noshear']
    em = e[-1]['noshear']
    r11 = 0.5 * (
        (e[1]['1p'][:, 0] - e[1]['1m'][:, 0])
        + (e[-1]['1p'][:, 0] - e[-1]['1m'][:, 0])
    ) / (2 * SHEAR_STEP)
    r22 = 0.5 * (
        (e[1]['2p'][:, 1] - e[1]['2m'][:, 1])
        + (e[-1]['2p'][:, 1] - e[-1]['2m'][:, 1])
    ) / (2 * SHEAR_STEP)

    def stats(sel):
        R11 = r11[sel].mean()
        R22 = r22[sel].mean()
        d = (ep[sel, 0] - em[sel, 0]).mean()
        m = d / (2 * R11 * GTRUE) - 1
        c1 = (ep[sel, 0] + em[sel, 0]).mean() / (2 * R11)
        c2 = (ep[sel, 1] + em[sel, 1]).mean() / (2 * R22)
        return np.array([m, c1, c2, R11])

    full = stats(np.arange(n))
    brng = np.random.RandomState(1234)
    boots = np.array([
        stats(brng.randint(0, n, n)) for _ in range(300)
    ])
    err = boots.std(axis=0)

    peak = max(g.max() for g in gals.values())
    return dict(
        n=int(n), nfail=int(nfail), seconds=round(dt, 1),
        noise_var_factor=float(fac),
        flux=float(flux), peak_over_skyvar=float(peak / sky_var),
        m=float(full[0]), m_err=float(err[0]),
        c1=float(full[1]), c1_err=float(err[1]),
        c2=float(full[2]), c2_err=float(err[2]),
        R11=float(full[3]),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--flux', type=float, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument(
        '--calibrate', type=float, metavar='SNR',
        help='print the sky variance giving this typical s2n, then exit',
    )
    p.add_argument('--case')
    p.add_argument('--nreal', type=int)
    p.add_argument('--skyvar', type=float)
    p.add_argument('--out')
    args = p.parse_args()

    if args.calibrate is not None:
        sky_var = calibrate_sky_var(args.flux, args.calibrate, args.seed)
        print(f'flux {args.flux} snr {args.calibrate} sky_var {sky_var:.4f}')
        return

    for name in ('case', 'nreal', 'skyvar', 'out'):
        if getattr(args, name) is None:
            p.error(f'--{name} is required for a run')

    res = run_case(args.case, args.nreal, args.seed, args.flux, args.skyvar)
    res['case'] = args.case
    res['flux'] = args.flux
    res['sky_var'] = args.skyvar
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()

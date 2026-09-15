"""
Run simcoadd-mdet on a grid of fixed galaxies with detection, for +g and
-g with matched seeds, and measure the shear with noise cancellation.

The config is built here so every choice is explicit: grid layout,
fixed Exponential galaxies, white sky noise with or without the source
Poisson term, Moffat psf with random small ellipticity, fusion metacal
with the azgauss target, sep detection and weighted moments.

At a fixed sky level both the S/N and the source-Poisson-to-sky
variance ratio scale with the flux, so a Poisson-dominated regime at
moderate S/N needs a faint object on a dark sky: noise_factor scales
the sky noise alone, so (mag, noise_factor) set (S/N, ratio)
independently.

Outputs go under outdir: the two configs, plus/ and minus/ with one
fits file per seed, the file lists, and shear.fits from
simcoadd-mdet-doshear-cancel
"""
import argparse
import os
import subprocess
from multiprocessing import Pool
import yaml

CUTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'cuts-wmom.yaml',
)


def make_config(mag, noise_factor, source_poisson, g1, psf_g_sigma):
    return {
        'metacal': {
            'noise_correction': 'fusion',
            'target_psf': 'azgauss',
            'types': ['noshear', '1p', '1m', '2p', '2m'],
        },
        'survey': {'name': 'LSST', 'bands': ['i']},
        'noise': {
            'type': 'white',
            'noise_factor': noise_factor,
            'source_poisson': source_poisson,
        },
        'layout': {'type': 'grid', 'dim': 350, 'buff': 50, 'spacing': 9.5},
        'shear': {'g1': g1, 'g2': 0.0},
        'psf': {
            'type': 'moffat',
            'moffat_beta': 3.5,
            'fwhm_mean': 0.8,
            'fwhm_sigma': 0.0,
            'fwhm_min': 0.6,
            'fwhm_max': 2.0,
            'g_sigma': psf_g_sigma,
            'noise': 1.0e-6,
            'dim': 51,
        },
        'galaxies': {'type': 'fixed', 'morph': 'exp', 'hlr': 0.5, 'mag': mag},
        'detection': {'name': 'sep'},
        'fitter': {'name': 'wmom'},
    }


def run_one(job):
    config_file, seed, ntrial, outfile = job
    if os.path.exists(outfile):
        return outfile
    cmd = [
        'simcoadd-mdet-run',
        '--config', config_file,
        '--seed', str(seed),
        '--ntrial', str(ntrial),
        '--outfile', outfile,
    ]
    with open(outfile + '.log', 'w') as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    return outfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--outdir', required=True)
    p.add_argument('--mag', type=float, required=True)
    p.add_argument('--noise-factor', type=float, required=True)
    p.add_argument(
        '--source-poisson', required=True, choices=['true', 'false'],
    )
    p.add_argument('--g', type=float, required=True, help='true |g1|')
    p.add_argument('--psf-g-sigma', type=float, required=True)
    p.add_argument('--nseeds', type=int, required=True)
    p.add_argument('--ntrial', type=int, required=True)
    p.add_argument('--seed0', type=int, required=True)
    p.add_argument('--nproc', type=int, required=True)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    source_poisson = args.source_poisson == 'true'

    jobs = []
    flists = {}
    for sign, name in ((+1, 'plus'), (-1, 'minus')):
        config = make_config(
            mag=args.mag,
            noise_factor=args.noise_factor,
            source_poisson=source_poisson,
            g1=sign * args.g,
            psf_g_sigma=args.psf_g_sigma,
        )
        config_file = os.path.join(args.outdir, f'config-{name}.yaml')
        with open(config_file, 'w') as f:
            yaml.safe_dump(config, f)

        rundir = os.path.join(args.outdir, name)
        os.makedirs(rundir, exist_ok=True)
        outfiles = []
        for i in range(args.nseeds):
            seed = args.seed0 + i
            outfile = os.path.join(rundir, f'run-{seed:06d}.fits')
            outfiles.append(outfile)
            jobs.append((config_file, seed, args.ntrial, outfile))

        flist = os.path.join(args.outdir, f'flist-{name}.txt')
        with open(flist, 'w') as f:
            f.write('\n'.join(outfiles) + '\n')
        flists[name] = flist

    # interleave plus and minus so partial progress stays paired
    jobs.sort(key=lambda j: (j[1], j[0]))
    with Pool(args.nproc) as pool:
        for i, outfile in enumerate(pool.imap_unordered(run_one, jobs)):
            print(f'[{i + 1}/{len(jobs)}] {outfile}', flush=True)

    shear_file = os.path.join(args.outdir, 'shear.fits')
    subprocess.run(
        [
            'simcoadd-mdet-doshear-cancel',
            '--config', CUTS,
            '--flist-plus', flists['plus'],
            '--flist-minus', flists['minus'],
            '--outfile', shear_file,
            '--seed', '1',
            '--nproc', str(args.nproc),
        ],
        check=True,
    )


if __name__ == '__main__':
    main()

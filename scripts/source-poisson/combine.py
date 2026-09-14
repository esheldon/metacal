"""
combine the per-seed json outputs of poisson_noise_test.py: seeds are
independent samples of equal size, so the combined value is the
inverse-variance weighted mean and its error
"""
import sys
import glob
import json
import numpy as np


def combine(files):
    rows = [json.load(open(f)) for f in files]
    out = {}
    for key in ('m', 'c1', 'c2'):
        v = np.array([r[key] for r in rows])
        e = np.array([r[key + '_err'] for r in rows])
        w = 1.0 / e**2
        mean = (w * v).sum() / w.sum()
        err = 1.0 / np.sqrt(w.sum())
        chi2 = ((v - mean) ** 2 * w).sum()
        out[key] = (mean, err, chi2, len(v) - 1)
    out['n'] = sum(r['n'] for r in rows)
    out['nfail'] = sum(r['nfail'] for r in rows)
    out['R11'] = float(np.mean([r['R11'] for r in rows]))
    out['case'] = rows[0]['case']
    out['flux'] = rows[0].get('flux')
    out['sky_var'] = rows[0].get('sky_var')
    out['peak_over_skyvar'] = rows[0]['peak_over_skyvar']
    out['noise_var_factor'] = rows[0]['noise_var_factor']
    out['m_noiseless'] = rows[0].get('m_noiseless')
    return out


def main():
    for pattern in sys.argv[1:]:
        files = sorted(glob.glob(pattern))
        if not files:
            continue
        o = combine(files)
        print(
            f"{o['case']:9s} flux={o['flux']} sky_var={o['sky_var']} "
            f"peak/skyvar={o['peak_over_skyvar']:.2f} "
            f"nvf={o['noise_var_factor']:.3f} n={o['n']} nfail={o['nfail']} "
            f"R11={o['R11']:.4f}"
        )
        for key in ('m', 'c1', 'c2'):
            mean, err, chi2, dof = o[key]
            print(
                f"    {key:3s} = {mean:+.5f} +/- {err:.5f}   "
                f"({mean / err:+.1f} sigma; seed chi2/dof = {chi2:.1f}/{dof})"
            )
        if o['m_noiseless'] is not None:
            mean, err, _, _ = o['m']
            m0 = o['m_noiseless']
            d = mean - m0
            print(
                f"    m - m_noiseless = {d:+.5f} +/- {err:.5f}   "
                f"({d / err:+.1f} sigma; m_noiseless = {m0:+.5f})"
            )


if __name__ == '__main__':
    main()

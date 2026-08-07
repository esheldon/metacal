# flake8: noqa
"""
GPU backend for the fusion/AZGauss metacal pipeline (requires
cupy and an NVIDIA device).  Per-image psfs are the fundamental
case: the engine holds only psf-independent device state, and
everything derived from a psf lives in a PsfBundle (pure CPU;
see bundle.py) assembled into device factors in milliseconds,
with an LRU so repeated psfs cost nothing.

Explicit engine (production pipelines):

    from metacal.gpu import FusionEngine
    eng = FusionEngine(dim=dim, types=types, fp32=True)
    for obs in observations:            # any psfs
        res = eng.metacal_obs(obs=obs, rng=rng)

Workers without device access build bundles themselves
(metacal.gpu.bundle imports no cupy) and ship them to a
device-owning process:

    from metacal.gpu.bundle import PsfBundle
    bundle = PsfBundle(psf_image, wcs, types, dim)

Drop-in (convenience): same signature as metacal.metacal_obs,
backed by an engine cache on (dim, types, precision):

    from metacal.gpu import metacal_obs
    res = metacal_obs(obs=obs, noise_filter=FusionFilter(),
                      target_psf=AZGauss(), rng=rng, types=types)
"""

from .bundle import PsfBundle
from .engine import FusionEngine

_ENGINE_CACHE = {}


def clear_cache():
    """drop all cached engines (frees device memory)"""
    _ENGINE_CACHE.clear()


def get_engine(dim, types, fp32=True):
    """a cached FusionEngine for this configuration"""
    key = (int(dim), tuple(types), bool(fp32))
    if key not in _ENGINE_CACHE:
        _ENGINE_CACHE[key] = FusionEngine(
            dim=dim, types=types, fp32=fp32,
        )
    return _ENGINE_CACHE[key]


def metacal_obs(obs, noise_filter, target_psf, rng, types,
                fp32=True):
    """
    drop-in for metacal.metacal_obs running on the gpu; see
    metacal.obs.metacal_obs for the contract.  Handles any psf
    (the per-psf derivation is cached on psf content).
    """
    from ..azgauss_target_psf import AZGauss
    from ..fusion_filter import FusionFilter

    if target_psf is not None and \
            not isinstance(target_psf, AZGauss):
        raise NotImplementedError(
            f'the gpu engine implements the AZGauss target '
            f'only, got {type(target_psf).__name__}'
        )
    if noise_filter is not None and \
            not isinstance(noise_filter, FusionFilter):
        raise NotImplementedError(
            f'the gpu engine implements the FusionFilter noise '
            f'correction only, got '
            f'{type(noise_filter).__name__}'
        )
    eng = get_engine(dim=obs.image.shape[0], types=types,
                     fp32=fp32)
    return eng.metacal_obs(obs=obs, rng=rng)

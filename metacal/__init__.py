# flake8: noqa
"""
metacal: the k-space-native metacalibration with the rotated-hybrid noise
correction;the validated subset of the mcal_hybrid research package, with the
winning choices baked in (see README).
"""

from .kmetacal import (
    KMetacal,
    metacal,
    metacal_hybrid,
    delta_transfer_kspace,
    make_hybrid_filters_kspace,
    DEFAULT_TYPES,
    LANCZOS,
)
from .deficit import common_harmonic_deficits, jacobian_matrix
from .wcs import distortion_matrix, galsim_wcs, ngmix_jacobian

__all__ = [
    'KMetacal',
    'metacal',
    'metacal_hybrid',
    'delta_transfer_kspace',
    'make_hybrid_filters_kspace',
    'common_harmonic_deficits',
    'jacobian_matrix',
    'distortion_matrix',
    'galsim_wcs',
    'ngmix_jacobian',
    'DEFAULT_TYPES',
    'LANCZOS',
]

__version__ = '0.1.0'

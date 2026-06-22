"""
local WCS helpers.

A uniform (linear) WCS is the 2x2 jacobian M mapping a pixel offset
(col, row) = (x, y) to a world offset (u, v) in arcsec, stored in (x, y) order
M = [[dudx, dudy], [dvdx, dvdy]].  A plain diagonal scale is M = scale * I; a
distorted system adds a rotation R(theta) and an area-preserving shear S(g1, g2):

    M = scale * R(theta) @ S(g1, g2),

so |det M| = scale^2 (the pixel area, flux and matched-filter S/N are unchanged).

galsim and ngmix both express the jacobian as u = dudx*x + dudy*y with x = column,
y = row, so galsim ``dudx`` maps to ngmix ``dudcol`` and ``dudy`` to ``dudrow``.
"""
import numpy as np
import galsim
import ngmix


def distortion_matrix(scale, theta_deg=0.0, g1=0.0, g2=0.0):
    """the 2x2 pixel->sky jacobian M = scale * R(theta_deg) @ S(g1, g2), in
    (x, y) = (col, row) order.  theta_deg = g1 = g2 = 0 gives scale * I."""
    th = np.deg2rad(theta_deg)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    norm = 1.0 / np.sqrt(1.0 - g1**2 - g2**2)
    shear = norm * np.array([[1.0 + g1, g2], [g2, 1.0 - g1]])
    return scale * (rot @ shear)


def galsim_wcs(jacmat):
    """a galsim.JacobianWCS for the 2x2 matrix jacmat (x, y order)"""
    return galsim.JacobianWCS(
        dudx=jacmat[0, 0], dudy=jacmat[0, 1],
        dvdx=jacmat[1, 0], dvdy=jacmat[1, 1],
    )


def ngmix_jacobian(row, col, jacmat):
    """an ngmix.Jacobian centered at (row, col) for the 2x2 matrix jacmat"""
    return ngmix.Jacobian(
        row=row, col=col,
        dudcol=jacmat[0, 0], dudrow=jacmat[0, 1],
        dvdcol=jacmat[1, 0], dvdrow=jacmat[1, 1],
    )

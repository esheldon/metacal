"""
local WCS helpers.
"""


def distortion_matrix(scale, theta=None, g1=0.0, g2=0.0):
    """
    Create a distortion matrix from scale, angle, shear

    Parameters
    ----------
    scale: float
        The pixel scale, e.g. 0.2 arcsec/pixel
    theta: galsim.Angle or None
        The rotation as a galsim.Angle, e.g. 30 * galsim.degrees; None
        (default) is no rotation
    g1: float
        Shear g1
    g2: float
        Shear g2

    Returns
    -------
    matrix: array
        Matrix of shape [2, 3]
    """
    import numpy as np
    import galsim

    if theta is not None and not isinstance(theta, galsim.Angle):
        raise TypeError(
            'theta must be a galsim.Angle (e.g. 30 * galsim.degrees) or '
            f'None, got {type(theta).__name__}'
        )
    th = 0.0 if theta is None else theta / galsim.radians
    rot = np.array(
        [[np.cos(th), -np.sin(th)],
         [np.sin(th), np.cos(th)]]
    )
    norm = 1.0 / np.sqrt(1.0 - g1**2 - g2**2)
    shear = norm * np.array([[1.0 + g1, g2], [g2, 1.0 - g1]])
    return scale * (rot @ shear)


def galsim_wcs(jacmat):
    """
    Convert a jacobian matrix to a galsim.JacobianWCS

    Parameters
    ----------
    jacmat: array
        [2, 2] array

    Returns
    -------
    galsim.JacobianWCS
    """
    import galsim

    return galsim.JacobianWCS(
        dudx=jacmat[0, 0],
        dudy=jacmat[0, 1],
        dvdx=jacmat[1, 0],
        dvdy=jacmat[1, 1],
    )


def ngmix_jacobian(row, col, jacmat):
    """
    Get an ngmix.Jacobian centered at (row, col) for the 2x2 matrix jacmat

    Parameters
    ----------
    row: float
        Row center
    col: float
        Col center
    jacmat: array
        [2, 2] array

    Returns
    -------
    ngmix.Jacobian
    """
    import ngmix

    return ngmix.Jacobian(
        row=row,
        col=col,
        dudcol=jacmat[0, 0],
        dudrow=jacmat[0, 1],
        dvdcol=jacmat[1, 0],
        dvdrow=jacmat[1, 1],
    )

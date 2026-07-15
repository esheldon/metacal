def _wcs_and_matrix(wcs):
    """
    resolve a wcs given as either a galsim local/Jacobian wcs or a 2x2
    pixel->sky jacobian matrix

    return both as (wcs, matrix)
    """
    import numpy as np
    import galsim
    from .wcs import galsim_wcs

    if isinstance(wcs, galsim.BaseWCS):
        jac = wcs.jacobian()
        return wcs, np.array([[jac.dudx, jac.dudy], [jac.dvdx, jac.dvdy]])

    mat = np.asarray(wcs, dtype=float)

    return galsim_wcs(mat), mat

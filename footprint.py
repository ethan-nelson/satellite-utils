import numpy as np


def footprint(rx, ry, gx, gy, weight_flag):
    """
    Create a footprint mask on a 2D grid centered at [0,0]. Numpy's
    roll can be used to center the mask about a grid point.

    Parameters
    ----------
    rx : int
        Radius in x direction.
    ry : int
        Radius in y direction.
    gx : int
        Grid size in x direction.
    gy : int
        Grid size in y direction
    weight_flag : boolean
        Flag whether to use Gaussian average (1=yes, 0=no)

    Returns
    -------
    out_grid : array_like
        Grid with areas outside of footprint defined as 0 and areas
        in the footprint defined as either 1 or a Gaussian weight.

    Notes
    -----
    Radius sizes should include the center gridpoint; in example, rx=2 
    means the original grid point plus one more in the x direction.

    Examples
    --------
    >>> grid = footprint(2,2,4,4,0)
    >>> grid
    array([[1., 1., 0., 1.,],
           [1., 0., 0., 0.,],
           [0., 0., 0., 0.,],
           [1., 0., 0., 1.,]])

    Changelog
    ---------
    08/12/15 ELN - Added Gaussian calculation and documentation.
    """
    msg = 'footprint: %s (%d) must be >= %d'
    if not (1 < rx):
        raise ValueError(msg % ('radius in x', rx, 2))
    if not (1 < ry):
        raise ValueError(msg % ('radius in y', ry, 2))
    msg = 'footprint: %s must be < %s'
    if not (rx < gx):
        raise ValueError(msg % ('radius in x', 'grid in x'))
    if not (ry < gy):
        raise ValueError(msg % ('radius in y', 'grid in y'))

    L, R, B, T = int(-np.ceil(gy/2.0)+1), int(np.floor(gy/2.0)+1), int(-np.ceil(gx/2.0)+1), int(np.floor(gx/2.0)+1)

    rxr = rx - 1
    ryr = ry - 1

    grid = np.zeros([gx,gy])

    xv, yv = np.meshgrid(np.arange(B,T), np.arange(L,R))

    if weight_flag == 0:
        grid = 1.0
    else:
        grid = np.exp(-(1.0/2.0) * (np.sqrt(xv**2.0 + yv**2.0) / ((rxr + ryr)/2.0 ))**2.0)

    TorF = ((xv**2.0 / rxr**2.0) + (yv**2.0 / ryr**2.0)) <= 1.0
    out_grid = np.roll(np.roll(grid * TorF,L,axis=0),B,axis=1)
    return out_grid

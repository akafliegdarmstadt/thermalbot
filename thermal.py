from scipy.interpolate import interp1d
from math import sqrt, sin, cos
import numpy as np
import numba


# @numba.njit
def thermal(x, y, z, thermal_pos=[0.0, 0.0], z_i=1213.0, w_star=1.97):
    """Calculate thermals following Allen 2006"""

    if z > 0.9 * z_i:
        return 0.0
    else:
        r = sqrt((x - thermal_pos[0]) ** 2 + (y - thermal_pos[1]) ** 2)
        w_mean = w_star * (z / z_i) ** 3 * (1.0 - 1.1 * (z / z_i))
        r2 = max((10.0, 0.102 * (z / z_i) ** (1 / 3) * (1 - 0.25 * (z / z_i)) * z_i))
        r1 = (0.8 if r2 > 600 else 0.0011 * r2 + 0.14) * r2
        w_peak = 3 * w_mean * (r2 ** 3 - r2 ** 2 * r1) / (r2 ** 3 - r1 ** 3)

        w_D = 0.0  # TODO: calculate

        k1, k2, k3, k4 = _get_ks_faster(r1 / r2)
        w = w_peak * (1 / (1 + abs(k1 * r / r2 + k3) ** k2) + k4 * r / r2 + w_D)

        return w


rrs = np.array([0.14, 0.25, 0.36, 0.47, 0.58, 0.69, 0.80])
ks = np.array(
    [
        [1.5352, 2.5826, -0.0113, 0.0008],
        [1.5265, 3.6054, -0.0176, 0.0005],
        [1.4866, 4.8354, -0.0320, 0.0001],
        [1.2042, 7.7904, 0.0848, 0.0001],
        [0.8816, 13.972, 0.3404, 0.0001],
        [0.07067, 23.994, 0.5689, 0.0002],
        [0.6189, 42.797, 0.7157, 0.0001],
    ]
)


def _get_ks(rr):
    return interp1d(rrs, ks, kind="nearest", axis=0)(rr)


@numba.jit
def _get_ks_faster(rr):

    for i, ar in enumerate(rrs):
        if ar < rr:
            break

    if i == 0:
        return ks[0, :]
    elif i == ks.shape[1]:
        return ks[-1, :]
    else:
        if (rr - ar) / (rrs[i + 1] - rrs[i]) >= 0.5:
            return ks[i + 1, :]
        else:
            return ks[i, :]

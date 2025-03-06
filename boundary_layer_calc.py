import dataclasses

import numpy as np

@dataclasses.dataclass
class BoundaryLayer:
    y0: float
    n: int
    GR: float

def boundary_layer_mesh_stats(rho, V, mu, L, x, yplus, GR):
    # Reynolds number for leading edge y+ calculation (critical)
    # L should be LE radius for an aerofoil
    Re_L = rho * V * L / mu
    # Schlichting 1979 turbulent BL thickness approx for a flat plate using LE Re approx.
    C_f = 0.25 * 0.0576 * Re_L ** (-1 / 5)
    # compute first layer height
    tau_w = 0.5 * C_f * rho * V ** 2
    u_tau = np.sqrt(tau_w / rho)
    y0 = yplus * mu / rho / u_tau

    # Reynolds number for calculating final boundary layer height at TE
    Re_x = rho * V * x / mu
    delta_x = (0.37 * x * Re_x ** (-1 / 5))
    n = np.log(1+(delta_x / y0)*(GR-1)) / np.log(GR)

    final_layer_height = y0 * GR ** (n-1)

    print(f"""y0 \t\t\t = {y0.to('mm'): .5f~P}
BL height    = {delta_x.to('mm'):0= 3g~P}
prism layers = {int(n): d}
last layer   = {final_layer_height.to('mm'): 3g~P}""")

    return BoundaryLayer(y0, int(n), GR)
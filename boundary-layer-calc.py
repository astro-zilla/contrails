import numpy as np
from flightcondition import FlightCondition, unit


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

    final_layer_height = y0 * GR ** n

    print(f"""y0 \t\t\t = {y0.to('mm'):0= 5g~P}
BL height    = {delta_x.to('mm'):0= 3g~P}
prism layers = {int(n): d}
last layer   = {final_layer_height.to('mm'): 3g~P}""")


if __name__ == '__main__':
    condition = FlightCondition(M=0.78, L=3.8 * unit('m'), h=37000 * unit('ft'), units='SI')
    print('\nEXTERNAL')
    boundary_layer_mesh_stats(rho=condition.rho, V=condition.TAS, mu=condition.mu,
                              L=3.8 * unit('m') * 0.0025, x=3.8 * unit('m'),
                              yplus=1.0, GR=1.2)
    print('\nBYPASS')
    boundary_layer_mesh_stats(rho=condition.rho, V=294.24 * unit('m/s'), mu=condition.mu,
                              L=0.8*unit('m'), x=2.1 * unit('m'),
                              yplus=1.0, GR=1.175)
    print('\nCORE')
    boundary_layer_mesh_stats(rho=condition.rho, V=367.8 * unit('m/s'), mu=condition.mu,
                              L=3.8 * unit('m'), x=1.0 * unit('m'),
                              yplus=1.0, GR=1.15)
    print('\nWAKE')
    boundary_layer_mesh_stats(rho=condition.rho, V=(367.8 - 294.24) * unit('m/s'), mu=condition.mu,
                              L=0.4 * unit('m'), x=4.9 * unit('m'),
                              yplus=1.0, GR=1.165)
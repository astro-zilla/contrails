import numpy as np
from flightcondition import FlightCondition, unit


def calc_boundary_layer(condition: FlightCondition, L: float, yplus: float, GR: float):
    Re_x = condition.rho * condition.TAS * L / condition.mu
    C_f = 0.25 * 0.0576 * Re_x ** (-1 / 5)
    tau_w = 0.5 * C_f * condition.rho * condition.TAS ** 2
    u_tau = np.sqrt(tau_w / condition.rho)
    y0 = yplus * condition.mu / condition.rho / u_tau

    delta_L = 0.37 * L * Re_x ** (-1 / 5)
    n = np.log(delta_L / y0) / np.log(GR)

    print(f"""y = {y0:.5g~P}
delta_L = {delta_L:.5g~P}
n = {n:.5g~P}""")


if __name__ == '__main__':
    condition = FlightCondition(M=0.78, L=3.8 * unit('m'), h=37000 * unit('ft'), units='SI')

    calc_boundary_layer(condition=condition,
                        L=2 * unit('m'),
                        yplus=1.0,
                        GR=1.2)

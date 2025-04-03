import numpy as np


def psat_water(p, T):
    ew = np.exp(-6096.9385 / T + 21.2409642 - 2.711193e-2 * T + 1.673952e-5 * T ** 2 + 2.433502 * np.log(T))
    fw = 1.0016 + 3.15e-8 * p - 7.4e-4 / p
    return ew * fw


def psat_ice(p, T):
    ei = np.exp(-6024.528211 / T + 29.32707 + 1.0613868e-2 * T + -1.3198825e-5 * T ** 2 - 0.49382577 * np.log(T))
    fi = 1.0016 + 3.15e-8 * p - 7.4e-4 / p
    return ei * fi

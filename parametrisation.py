import numpy as np
from flightcondition import FlightCondition
from matplotlib import pyplot as plt
from pint import UnitRegistry
ureg = UnitRegistry()

rho_b=0.81e3


def D_v(p: float,T: float) -> float:
    return 2.11e-5 * (T/273.15)**1.94 * (101325/p)

def r_a(m):
    m_t=2.146e-13
    alpha = np.where(m<m_t, 526.1, 0.04142)
    beta = np.where(m<m_t, 3.0, 2.2)
    return np.where(m<m_t, 1.0,np.sqrt(np.sqrt(27)*rho_b/(8*alpha**(3/beta)))*m**((3-beta)/(2*beta)))

def L(m,r_a):
    return np.pow((m/rho_b*r_a**2*8/np.sqrt(27)),1/3)

def A(L,r_a):
    D=L/r_a
    b=D/2
    return 6*b*L+3*np.sqrt(3)*b**2

def C(L,r_a):
    a=L/2
    b=a/r_a
    A_dash = np.sqrt(a**2-b**2)
    return A_dash/np.log((a+A_dash)/b)

def f1(p,T):
    rstar=1
    lstar_M=1
    return rstar / (rstar(+lstar_M))

def mu_lognormal(k, N_c, mean, sigma):
    r_0 = np.exp(np.log(np.sqrt(sigma)) ** 2)
    return N_c*mean**k*r_0**(k*(k-1)/2)

ms = np.logspace(-17,-7,1000)
r_as = r_a(ms)
Ls = L(ms, r_as)
Ds = Ls/r_as
plt.semilogx(Ls*1e6, r_as, label='r_a(m)')
plt.grid()
plt.show()



# fc = FlightCondition(M=0.78, L=3.8 * ureg.meter, h=37000 * ureg.foot, units='SI')
# p= fc.p.magnitude
# T= fc.T.magnitude
# dm_dt=4*np.pi*D_v(p,T)*f1(p,T)*f2(p,T)*(rho_v(T)-rho_si(T_s))
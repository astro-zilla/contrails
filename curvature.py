import time

import numpy as np
import matplotlib.pyplot as plt
import scipy


def H(a, c, beta):
    return c * (2 * a ** 2 + (c ** 2 - a ** 2) * np.cos(beta) ** 2) / (
                2 * a * (a ** 2 + (c ** 2 - a ** 2) * np.cos(beta) ** 2) ** (3/2))


def K(a, c, beta):
    return c ** 2 / ((a ** 2 + (c ** 2 - a ** 2) * np.cos(beta) ** 2) ** 2)



def dS(a, c, beta):
    return 2 * np.pi * a*np.cos(beta)*np.sqrt(a**2*np.sin(beta)**2+c**2*np.cos(beta)**2)


def curv(a, c):
    beta = np.linspace(-np.pi / 2, np.pi / 2, 1000)
    return np.sum(2 * H(a, c, beta) * dS(a, c, beta)) / np.sum(dS(a, c, beta))



x=[0.45,0.9]
def curv_est(a,c,x,beta):
    r_a = a / c
    r_a_x2 = beta*r_a
    alpha=1.2732384954398976
    return alpha/a*(np.exp(x[0]+x[0]*np.tanh(x[1]*np.log(r_a))))#*(r_a_x2**3+r_a_x2**2+r_a_x2+1)/(r_a_x2**4+r_a_x2**3+r_a_x2**2+r_a_x2+1)


def fit(x):
    logcoords = np.logspace(0,1)


    beta = curv_est(1e10,1e-10,x,1)/curv(1e10,1e-10)
    fitfunc=np.array([curv_est(1,lc,x,beta) for lc in logcoords])
    actfunc = np.array([curv(1,lc) for lc in logcoords])
    e = np.sum((fitfunc - actfunc) ** 2)
    print(e,beta)
    return e

res = scipy.optimize.minimize(fit, x)
print(res.x)
x=res.x
beta = curv_est(1e10,1e-10,x,1)/curv(1e10,1e-10)

c_vals = np.logspace(-4,4,1000)
a_vals = np.logspace(-0.1,0.1,10)
for i,a in enumerate(a_vals):
    spheroid = np.array([curv(a,c) for c in c_vals])
    spheroid_est = np.array([curv_est(a,c,x,beta) for c in c_vals])






    plt.loglog(a/c_vals,spheroid*a,color=f'red')
    plt.loglog(a/c_vals, spheroid_est*a,color=f'blue',linestyle='-')


plt.legend()
plt.show()


m=np.linspace(1e-13,5e-8,1000)
V= m/(0.81e3)
m_t=2.146e-13
alpha_c = np.where(m<m_t, 526.1, 0.04142)
beta_c = np.where(m<m_t, 3.0, 2.2)
r_a = np.where(m<m_t,1,np.sqrt(np.sqrt(27)*0.81e3/(8*alpha_c**(3/beta_c)))*m**((3-beta_c)/(2*beta_c)))


# radius of sphere of volume V
r = (3 * V / (4 * np.pi)) ** (1 / 3)
c_vals = r/r_a**(2/3)
a_vals = r_a*c_vals

curv_sphere = 2/r * np.ones_like(r_a)
curv_spheroid = [curv(a,c) for a, c in zip(a_vals, c_vals)]
curv_spheroid_est = [curv_est(a,c,x,beta) for a,c in zip(a_vals,c_vals)]

plt.semilogx(m, curv_sphere, 'r', label='Sphere')
plt.semilogx(m, curv_spheroid, 'b', label='Spheroid')
# plt.semilogx(m, curv_spheroid_est, 'g', label='Spheroid est.')

# plt.ylim(1,3)

plt.legend()
plt.show()

# Curvature of a spheroid
# need $\frac{1}{\kappa_1}+\frac{1}{\kappa_2}$
# \begin{equation}
#     \frac{1}{\kappa_1}+\frac{1}{\kappa_2}\approx a\left(\alpha r_a+\beta r_a^{-2}\right)\left(1 + \frac{c_0 + c_1  \tanh(c_2\log{r_a} - c_3) }{ (\cosh(c_4\log{r_a} - c_5)^{c_6}}\right)
# \end{equation}
# where the constants are given in Table \ref{tab:curvature-model-constants}.
# \begin{table}[]
#     \centering
#     \begin{tabular}{ c | c }
#         \text{constant} & \text{value} \\
#         \hline
#         $\alpha$ & 1.06666425\\
#         $\beta$ & 0.67906109\\
#         $c_0$ & -0.51712312\\
#         $c_1$ & 0.84778982\\
#         $c_2$ & -0.46601708\\
#         $c_3$ & -1.05701554\\
#         $c_4$ & 1.32793174\\
#         $c_5$ & -0.14066292\\
#         $c_6$ & 1.48094421
#     \end{tabular}
#     \caption{Model Constants}
#     \label{tab:curvature-model-constants}
# \end{table}

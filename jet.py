import matplotlib
import numpy as np
from flightcondition import FlightCondition, unit
from thermo import ChemicalConstantsPackage, CEOSGas, PRMIX, CEOSLiquid, FlashVL
from thermo.interaction_parameters import IPDB
import matplotlib.pyplot as plt
from svgpathtools import parse_path, svg2paths

# Flight Condition

condition = FlightCondition(M=0.85, L=3.8 * unit('m'), h=37000 * unit('ft'), units='SI')
print(condition)

# Chemistry
constants, properties = ChemicalConstantsPackage.from_IDs(['water', 'carbon dioxide', 'oxygen', 'nitrogen'])
kijs = IPDB.get_ip_asymmetric_matrix('ChemSep PR', constants.CASs, 'kij')
eos_kwargs = {'Pcs': constants.Pcs, 'Tcs': constants.Tcs, 'omegas': constants.omegas, 'kijs': kijs}
gas = CEOSGas(PRMIX, eos_kwargs=eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
liquid = CEOSLiquid(PRMIX, eos_kwargs=eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
flasher = FlashVL(constants, properties, liquid=liquid, gas=gas)

R = 287 * unit('m^2/s^2/K')
gam_a = 1.4
cP_a = 1005 * unit('kg.m^2/s^2/kg/K')
gam_e = 1.31
cP_e = 830 * unit('kg.m^2/s^2/kg/K')

eta_c = 0.85
eta_f = 0.92345
eta_t = 0.9

r_pf = 1.4398
r_pc = 40 / r_pf
r_pt = np.nan
Vjb_Vjc = 0.8

BPR = 11
D = (78 * unit('in')).to('m')
sfc_cruise_des = 0.0144 * unit('kg/kN/s')
LD = 18
L = 55000 * 0.00981 * unit('kN')
F_cruise_des = L / LD / 2
mdotf = sfc_cruise_des * F_cruise_des
LCV_fuel = 43.2 * unit('MJ/kg')
C_f = 10.8
H_f = 21.6
M_f = 151.9
M_H2O = 18.02
M_CO2 = 44.01
M_N2 = 28.01
M_O2 = 32.00

# TET can be 1550K
# EGT can be 15-25% for cooling - taken after HPC
# epsilon Tg-Tm / Tg-Tc for cooling is around 0.6-0.7

pa = condition.p

h01 = 249195 * unit('J/kg')
p01 = condition.p0
T01 = condition.T0
ro01 = p01 / R / T01

p03b = p02 = p01 * r_pf
T03b = T02 = T01 * r_pf ** ((gam_a - 1) / (gam_a * eta_f))
ro03b = ro02 = ro01 * r_pf ** (1 / gam_a / eta_f)

p3b = p03b * (1 + (gam_a - 1) / 2) ** -(gam_a / (gam_a - 1))
T3b = T03b * (1 + (gam_a - 1) / 2) ** -1
ro3b = ro03b * (1 + (gam_a - 1) / 2) ** -(1 / (gam_a - 1))

p03 = p02 * r_pc
T03 = T02 * r_pc ** ((gam_a - 1) / (gam_a * eta_c))
wc = cP_a * (T03 - T02)
h03 = h01 + wc

# assume bypass is choked. 1/2Vjb^2 = cP(T03b-T3b) OR Vjb = sqrt(gam_a * R * T3b)
Vjb = np.sqrt(2 * cP_a * (T03b - T3b))
Vf = condition.TAS

tb = (Vjb - Vf + (p3b - pa) / (ro3b * Vjb))

wfan = cP_a * (T02 - T01)
wcore = BPR * wfan

Vjc = Vjb / Vjb_Vjc

Q = mdotf * LCV_fuel

tc = (Vjc - Vf)

mdotc = F_cruise_des / (tc + BPR * tb)
mdotb = mdotc * BPR

p04 = 0.99 * p03
h04 = h03 + Q / mdotc

#  C 10.8 H 21.6 + phi*(C_f+H_f/4)(O2 + 79/21N2) --> C_f CO2 + H_f/2 H2O + phi*(C_f+H_f/4) - H_f/4 O2 + phi*(C_f+H_f/4) * 79/21 N2
AFR_s = (C_f + H_f / 4) * (M_O2 + 79 / 21 * M_N2) / M_f
AFR = mdotc / mdotf
phi = (AFR / AFR_s).to_base_units()

nCO2 = C_f
nH2O = H_f / 2
nO2 = phi * (C_f + H_f / 4) - nH2O / 2 - nCO2
nN2 = phi * (C_f + H_f / 4) * 79 / 21
nTOTAL = nCO2 + nH2O + nO2 + nN2
MTOTAL = nCO2 * M_H2O + nH2O * M_H2O + nO2 * M_O2 + nN2 * M_H2O

wtCO2 = (nCO2 * M_CO2) / MTOTAL
wtH2O = (nH2O * M_H2O) / MTOTAL
wtO2 = (nO2 * M_O2) / MTOTAL
wtN2 = (nN2 * M_N2) / MTOTAL

zs = [wtH2O, wtCO2, wtO2, wtN2]

T04 = flasher.flash(H_mass=h04.magnitude, P=p04.magnitude, zs=zs).T * unit('K')

h05 = h04 - wc - wcore
T05 = flasher.flash(H_mass=h05.magnitude, P=p04.magnitude, zs=zs).T * unit('K')
r_pt = (T04 / T05) ** (gam_e / (gam_e - 1) / eta_t)
p05 = p04 / r_pt
T05 = flasher.flash(H_mass=h05.magnitude, P=p05.magnitude, zs=zs).T * unit('K')
T5 = T05 - 0.5 / cP_e * Vjb ** 2
r_pt = (T04 / T05) ** (gam_e / (gam_e - 1) / eta_t)
p05 = p04 / r_pt

print(T04, T05, T5, T03 ** 2 / T02)
print(f"{r_pt=}\n")

Mjc = Vjc / np.sqrt(gam_e * R * T5)
print(Mjc)

Ab = mdotb * np.sqrt(cP_a * T03b) / 1.281 / p03b
Ac = mdotc * np.sqrt(cP_e * T05) / (
            gam_e / np.sqrt(gam_e - 1) * Mjc * (1 + (gam_e - 1) / 2 * Mjc ** 2) ** 0.5) / condition.p

r = np.linspace(0.00000001*unit('m'), D / 2)
b_annulus_t = Ab / 2 / np.pi / r
c_annulus_t = Ac / 2 / np.pi / r

# r = [0.4,0.74]

print(f"BYPASS:\n"
      f"p03b = {p03b.to('kPa'):.3f}\n"
      f"h03b = {(cP_a * (T02 - T01) + h01).to('kJ/kg')}\n"
      f"Ab = {Ab.to_base_units():.3f}\n"
      f"\nCORE:\n"
      f"wH2O = {wtH2O:.5f}\n"
      f"p05 = {p05.to('kPa'):.3f}\n"
      f"h05 = {h05.to('kJ/kg')}\n"
      f"Ac = {Ac.to_base_units():.3f}\n")

print((mdotc*p05+mdotb*p03b)/(mdotc+mdotb))
print((mdotc*h05+mdotb*(cP_a * (T02 - T01) + h01))/(mdotc+mdotb))


# plt.plot(r, np.clip((r + b_annulus_t / 2).magnitude,0,(D/2).magnitude), color='blue', label='BYPASS')
# plt.plot(r, np.clip((r - b_annulus_t / 2).magnitude, 0, (D/2).magnitude), color='blue')
# plt.plot(r, np.clip((r + c_annulus_t / 2).magnitude,0,(D/2).magnitude), color='red', label='CORE')
# plt.plot(r, np.clip((r - c_annulus_t / 2).magnitude, 0, (D/2).magnitude), color='red')
# plt.legend()
# plt.xlabel('$r$ [m]')
# plt.ylabel('annulus extent [m]')
# plt.show()
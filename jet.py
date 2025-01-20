import matplotlib
import numpy as np
from scipy.optimize import root_scalar
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

# r_pf = 1.4398
r_pf = 1.27
r_pc = 42 / r_pf  # PW1100G
r_pt = np.nan
Vjb_Vjc = 0.8

BPR = 12.5  # PW1100G
D = (81 * unit('in')).to('m')  # PW1100G
Af = 3.013 * unit('m^2')  # PW1100G from nacelle.py
sfc_cruise_des = 0.0144 * unit('kg/kN/s')  # PW1100G
LD = 18  # A320
L = 55000 * 0.00981 * unit('kN')  # A320
F_cruise_des = L / LD / 2
mdotf = sfc_cruise_des * F_cruise_des
LCV_fuel = 43.2 * unit('MJ/kg')  # Jet A-1
C_f = 10.8  # Jet A-1
H_f = 21.6  # Jet A-1
M_f = 151.9  # Jet A-1
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
r_pt = (T04 / T05) ** (gam_e / (gam_e - 1) / eta_t)
p05 = p04 / r_pt

T5 = T05 - 0.5 / cP_e * Vjb ** 2

print(T04, T05, T5, T03 ** 2 / T02)
print(f"{r_pt=}\n")

Mjc = Vjc / np.sqrt(gam_e * R * T5)

Ab = mdotb * np.sqrt(cP_a * T03b) / 1.281 / p03b
p_ce = condition.q_c*0.7+condition.p
Ac = mdotc * np.sqrt(cP_e * T05) / (
        gam_e / np.sqrt(gam_e - 1) * Mjc * (1 + (gam_e - 1) / 2 * Mjc ** 2) ** 0.5) / p_ce

# get choke area for fan and check it is smaller than actual area
Afstar = (mdotb + mdotc) * np.sqrt(cP_a * T01) / 1.281 / p01
if (Afstar > Af):
      raise(Exception(f"Inlet choked!: Af = {Af.to_base_units():.3f}, Afstar = {Afstar.to_base_units():.3f}"))
fm = (mdotb + mdotc) * np.sqrt(cP_a * T01) / Af / p01
fm = fm.to_base_units()
print(fm)
def f(M):
    return fm - gam_a / np.sqrt(gam_a - 1) * M * (1 + (gam_a - 1) / 2 * M ** 2) ** (-0.5 * (gam_a + 1) / (gam_a - 1))


M2 = root_scalar(f, bracket=[0, 1]).root
p2 = p01 * (1 + (gam_a - 1) / 2 * M2 ** 2) ** -(gam_a / (gam_a - 1))
V2 = M2 * np.sqrt(gam_a*R*condition.T)
N1 = 0.8
RPMFAN = 3281 * unit('2*pi/minute')
U2 = (N1 * RPMFAN * D/2).to_base_units()

r = np.linspace(0.00000001 * unit('m'), D / 2)
b_annulus_t = Ab / 2 / np.pi / r
c_annulus_t = Ac / 2 / np.pi / r

# r = [0.4,0.74]
# phi is 0.6

print(f"""
BYPASS:
  p03b = {p03b.to("kPa"):.5g~P}
  h03b = {(cP_a * (T02 - T01) + h01).to("kJ/kg"):.5g~P}
  Ab = {Ab.to_base_units():.5g~P}
  
CORE:
  wH2O = {wtH2O:.5g}
  p05 = {p05.to("kPa"):.5g~P}
  h05 = {h05.to("kJ/kg"):.5g~P}
  Ac = {Ac.to_base_units():.5g~P}

INTAKE:
  M2 = {M2:.5g}
  p2 = {p2.to("kPa"):.5g~P}
  V2 = {V2:.5g~P}
  phi2 = {(V2 / U2).magnitude:.5g}

TOTAL:
  mdot = {mdotc.to_base_units():.5g~P} + {mdotb.to_base_units():.5g~P} = {(mdotb + mdotc).to_base_units():.5g~P}
  F = {F_cruise_des.to("kN"):.5g~P}
""")

print((mdotc * p05 + mdotb * p03b) / (mdotc + mdotb))
print((mdotc * h05 + mdotb * (cP_a * (T02 - T01) + h01)) / (mdotc + mdotb))

Cdf =  219.3/220.1
Cdb = 207.9/205.6
Cdc = 25/16.4

# give bypass ratio and fan area in filename so paraview script can run jet with correct params
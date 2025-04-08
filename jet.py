from dataclasses import dataclass

import numpy as np
from flightcondition import FlightCondition, unit
from thermo import ChemicalConstantsPackage, CEOSGas, PRMIX, CEOSLiquid, FlashVL
from thermo.interaction_parameters import IPDB
from scipy.optimize import root_scalar


@dataclass
class Hydrocarbon:
    n_C: float
    n_H: float
    LCV: float

    @property
    def M(self):
        return 12.011 * self.n_C + 1.00784 * self.n_H


def get_exhaust_comp(AFR: float, fuel: Hydrocarbon, Y_h2o: float):
    M_H2O = 18.02
    M_CO2 = 44.01
    M_N2 = 28.01
    M_O2 = 32.00

    AFR_s = (fuel.n_C + fuel.n_H / 4) * (M_O2 + 79 / 21 * M_N2) / fuel.M

    phi = (AFR / AFR_s)

    nCO2 = fuel.n_C
    nH2O = fuel.n_H / 2
    nO2 = phi * (fuel.n_C + fuel.n_H / 4) - nH2O / 2 - nCO2
    nN2 = phi * (fuel.n_C + fuel.n_H / 4) * 79 / 21
    MTOTAL = nCO2 * M_H2O + nH2O * M_H2O + nO2 * M_O2 + nN2 * M_H2O

    wtCO2 = (nCO2 * M_CO2) / MTOTAL
    wtH2O = (nH2O * M_H2O) / MTOTAL + Y_h2o
    wtO2 = (nO2 * M_O2) / MTOTAL
    wtN2 = (nN2 * M_N2) / MTOTAL

    return [wtH2O, wtCO2, wtO2, wtN2]


@dataclass
class Engine:
    D: float
    BPR: float
    sfc_cruise: float
    fuel: Hydrocarbon

    eta_c: float
    eta_f: float
    eta_t: float

    r_pf: float
    r_po: float
    N1: float
    Vjb_Vjc: float

    @property
    def r_pc(self):
        return self.r_po / self.r_pf


class JetCondition:
    def __init__(self, flight_condition: FlightCondition, engine: Engine, Af: float, Y_h2o: float):
        self.fc = flight_condition
        self.engine = engine
        self.Af = Af
        self.h_ref = 297931 * unit('J/kg')

        # chemistry
        constants, properties = ChemicalConstantsPackage.from_IDs(['water', 'carbon dioxide', 'oxygen', 'nitrogen'])
        kijs = IPDB.get_ip_asymmetric_matrix('ChemSep PR', constants.CASs, 'kij')
        eos_kwargs = {'Pcs': constants.Pcs, 'Tcs': constants.Tcs, 'omegas': constants.omegas, 'kijs': kijs}
        gas = CEOSGas(PRMIX, eos_kwargs=eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
        liquid = CEOSLiquid(PRMIX, eos_kwargs=eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
        self.flasher = FlashVL(constants, properties, liquid=liquid, gas=gas)
        zs_air = [0, 0, 0.21, 0.79]

        # gas properties
        R = 287 * unit('m^2/s^2/K')
        # this could be done with the EOS
        gam_a = 1.4
        cP_a = 1005 * unit('kg.m^2/s^2/kg/K')
        gam_e = 1.31
        cP_e = 830 * unit('kg.m^2/s^2/kg/K')

        LD = 18  # A320
        L = 55000 * 0.00981 * unit('kN')  # A320
        self.Fnet = L / LD / 2
        print(f"Thrust: {self.Fnet.to('kN'):.5g~P}")
        mdotf = self.engine.sfc_cruise * self.Fnet
        # TET can be 1550K
        # EGT can be 15-25% for cooling - taken after HPC
        # epsilon Tg-Tm / Tg-Tc for cooling is around 0.6-0.7

        # station 1 - inlet
        p01 = self.fc.p0
        T01 = self.fc.T0
        self.station_01 = self.flasher.flash(T=T01.magnitude, P=p01.magnitude, zs=zs_air)
        h01 = self.station_01.H_mass() * unit('J/kg')

        # station 2 - fan exit
        p02 = p01 * self.engine.r_pf
        T02 = T01 * self.engine.r_pf ** ((gam_a - 1) / (gam_a * self.engine.eta_f))
        self.station_02 = self.flasher.flash(T=T02.magnitude, P=p02.magnitude, zs=zs_air)

        # station 3b - bypass exit
        p03b = p02
        T03b = T02
        self.station_03b = self.flasher.flash(T=T03b.magnitude, P=p03b.magnitude, zs=zs_air)
        h03b = self.station_03b.H_mass() * unit('J/kg')

        p3b = p03b * (1 + (gam_a - 1) / 2) ** -(gam_a / (gam_a - 1))
        T3b = T03b * (1 + (gam_a - 1) / 2) ** -1
        self.station_3b = self.flasher.flash(T=T3b.magnitude, P=p3b.magnitude, zs=zs_air)
        ro3b = self.station_3b.rho_mass() * unit('kg/m^3')

        # station 3 - compressor exit
        p03 = p02 * self.engine.r_pc
        T03 = T02 * self.engine.r_pc ** ((gam_a - 1) / (gam_a * self.engine.eta_c))
        self.station_03 = self.flasher.flash(T=T03.magnitude, P=p03.magnitude, zs=zs_air)
        wc = (self.station_03.H_mass() - self.station_02.H_mass()) * unit('J/kg')
        h03 = h01 + wc

        # jet velocities
        # assume bypass is choked. 1/2Vjb^2 = cP(T03b-T3b) OR Vjb = sqrt(gam_a * R * T3b)
        self.Vjb = np.sqrt(2 * cP_a * (T03b - T3b))
        self.Vjc = self.Vjb / engine.Vjb_Vjc

        # specific thrusts
        tb = (self.Vjb - self.fc.TAS + (p3b - self.fc.p) / (ro3b * self.Vjb))
        tc = (self.Vjc - self.fc.TAS)

        # work done by fan specific to bypass flow
        wfan = (self.station_02.H_mass() - self.station_01.H_mass()) * unit('J/kg')
        # specific to core flow
        wcore = self.engine.BPR * wfan

        # required mass flows
        mdotc = (self.Fnet / (tc + self.engine.BPR * tb)).to_base_units()
        mdotb = mdotc * self.engine.BPR
        self.mdot = mdotb + mdotc

        # station 4 - combustor exit
        self.zs_exhaust = get_exhaust_comp(AFR=(mdotc / mdotf).to_base_units(), fuel=engine.fuel, Y_h2o=Y_h2o)
        Q = mdotf * engine.fuel.LCV
        p04 = 0.99 * p03
        h04 = h03 + Q / mdotc
        self.station_04 = self.flasher.flash(H_mass=h04.magnitude, P=p04.magnitude, zs=self.zs_exhaust)
        T04 = self.station_04.T * unit('K')

        # station 5 - turbine exit
        # this is all we need for the boundary condition - rest is for interest and boundary layer calc
        h05 = h04 - wc - wcore
        # iterate as we don't know p05
        # guess...
        p05 = p04
        T05 = None
        for i in range(2):
            self.station_05 = self.flasher.flash(H_mass=h05.magnitude, P=p05.magnitude, zs=self.zs_exhaust)
            T05 = self.station_05.T * unit('K')
            r_pt = (T04 / T05) ** (gam_e / (gam_e - 1) / engine.eta_t)
            p05 = p04 / r_pt
        T5 = T05 - 0.5 / cP_e * self.Vjb ** 2
        Mjc = self.Vjc / np.sqrt(gam_e * R * T5)

        # exit areas from assumption that bypass is choked and core exits at ambient pressure

        # @incollection{HOUGHTON2013427, title = {Chapter 7 - Airfoils and Wings in Compressible Flow}, editor = {E.L. Houghton and P.W. Carpenter and Steven H. Collicott and Daniel T. Valentine}, booktitle = {Aerodynamics for Engineering Students (Sixth Edition)}, publisher = {Butterworth-Heinemann}, edition = {Sixth Edition}, address = {Boston}, pages = {427-477}, year = {2013}, isbn = {978-0-08-096632-8}, doi = {https://doi.org/10.1016/B978-0-08-096632-8.00007-2}, url = {https://www.sciencedirect.com/science/article/pii/B9780080966328000072}, author = {E.L. Houghton and P.W. Carpenter and Steven H. Collicott and Daniel T. Valentine}, keywords = {critical Mach number, linearized subsonic flow, linearized supersonic flow, Prandtl-Glauert rule, compressibility correction, small disturbance theory, supersonic wings, wing sweep, Ackert's rule}, abstract = {Publisher Summary: The chapter discusses the subsonic linearized compressible flow theory for extending the trusted results from incompressible flow into high subsonic flight. The most sweeping approximations, producing the simplest solutions, are made here and result in soluble linear differential equations. This leads to the expression linearized theory associated with airfoils. The chapter summarizes the supersonic linearized theory such as symmetrical double wedge airfoil in supersonic flow, supersonic biconvex circular arc airfoil in supersonic flow, general airfoil section, airfoil section made up of unequal circular arcs, double-wedge airfoil section. Several other aspects of supersonic wings such as the shock-expansion approximation, wings of finite span, computational methods are also presented in this chapter. The compressible-flow equations in various forms are considered in order to predict the behavior of airfoil sections in high sub- and supersonic flows. The wings in compressible flow, such as transonic flow, subcritical flow, supersonic linearized theory, and other aspects of supersonic wings are discussed in this chapter. The analysis of this regime involves solving a set of nonlinear differential equations, a task that demands either advanced computational techniques or some form of approximation. The approximations come about mainly from assuming that all disturbances are small disturbances or small perturbations to the free-stream flow conditions. The chapter also explores the phenomenon of wave drag in supersonic flight and how it is predicted by both the shock-expansion method and linearized supersonic flow.}}
        cp = 0.0
        p_core_exit = self.fc.q_inf * cp/np.sqrt(1-self.fc.M**2) + self.fc.p
        self.Ab = mdotb * np.sqrt(cP_a * T03b) / 1.281 / p03b
        self.Ac = mdotc * np.sqrt(cP_e * T05) / (
                gam_e / np.sqrt(gam_e - 1) * Mjc * (1 + (gam_e - 1) / 2 * Mjc ** 2) ** 0.5) / p_core_exit

        # station 1 - intake
        f = (mdotb + mdotc) * np.sqrt(cP_a * T01) / self.Af / p01

        def f0(M, ga):
            return f - ga / np.sqrt(ga - 1) * M * (1 + (ga - 1) / 2 * M ** 2) ** (-0.5 * (ga + 1) / (ga - 1))

        M1 = root_scalar(f0, (gam_a,), bracket=[0, 1]).root
        self.p1 = p01 * (1 + (gam_a - 1) / 2 * M1 ** 2) ** -(gam_a / (gam_a - 1))
        V1 = np.sqrt(cP_a * T01) * (gam_a - 1) * M1 * (1 + (gam_a - 1) / 2 * M1 ** 2) ** -0.5
        U1 = self.engine.N1.to_base_units() * self.engine.D / 2

        # jet dimensions for wake estimation
        Aj = self.Ab + self.Ac
        rj = np.sqrt(Aj / (2 * np.pi))
        rc = np.sqrt(self.Ac / (2 * np.pi))

        print(f"""
        BYPASS:
          p03b = {p03b.to("kPa"):.5g~P}
          h03b = {(h03b + self.h_ref).to("kJ/kg"):.5g~P}
          Ab = {self.Ab.to_base_units():.5g~P}
          
        CORE:
          wH2O = {self.zs_exhaust[0]:.5g}
          p05 = {p05.to("kPa"):.5g~P}
          h05 = {(h05 + self.h_ref).to("kJ/kg"):.5g~P}
          Ac = {self.Ac.to_base_units():.5g~P}
        
        INTAKE:
          M2 = {M1:.5g}
          p2 = {self.p1.to("kPa"):.5g~P}
          V2 = {V1: .5g~P}
          U2 = {U1:.5g~P}
          phi2 = {(V1 / U1).magnitude: .5g}
        
        TOTAL:
          rc = {rc:.5g~P}
          rj = {rj:.5g~P}
          mdot = {mdotc.to_base_units():.5g~P} + {mdotb.to_base_units():.5g~P} = {(mdotb + mdotc).to_base_units():.5g~P}
          F = {self.Fnet.to("kN"):.5g~P}
        """)

# give bypass ratio and fan area in filename so paraview script can run jet with correct params

import copy
from dataclasses import dataclass

import numpy as np
from flightcondition import FlightCondition, unit
from fluids.numerics import minimize
from thermo import ChemicalConstantsPackage, CEOSGas, PRMIX, CEOSLiquid, FlashVL
from thermo.interaction_parameters import IPDB
from scipy.optimize import minimize_scalar, root_scalar


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
    sfc_sl: float
    fuel: Hydrocarbon

    eta_c: float
    eta_f: float
    eta_t: float
    Fmax: float

    r_po: float
    N1: float
    Vjb_Vjc: float

    cp_bypass: float = 0.05
    cp_core: float = 0.35

    r_pf: float = None
    r_p_mult: float = 1.0

    @property
    def r_pc(self):
        return self.r_po / self.r_pf


jetA1 = Hydrocarbon(n_C=10.8, n_H=21.6, LCV=43.15 * unit('MJ/kg'))

PW1100G = Engine(D=(81 * unit('in')).to('m'), BPR=12.5,
                 sfc_cruise=0.0144 * unit('kg/kN/s'), sfc_sl=0.0079, fuel=jetA1,
                 eta_c=0.85, eta_f=0.92345, eta_t=0.9, Fmax=120 * unit('kN'),
                 r_po=42, N1=0.85 * 3281 * unit('2*pi/min'),  # 42
                 Vjb_Vjc=0.8)

LEAP1A = Engine(D=(78 * unit('in')).to('m'), BPR=11,
                sfc_cruise=0.0144 * unit('kg/kN/s'), sfc_sl=0.0079, fuel=jetA1,
                eta_c=0.85, eta_f=0.92345, eta_t=0.9, Fmax=120 * unit('kN'),
                r_po=40, N1=0.85 * 3894 * unit('2*pi/min'),  # 40
                Vjb_Vjc=0.8)


class JetCondition:
    def calibrate_PR_cruise(self):

        def f(r_p_mult):
            station_01, station_02, station_03, station_03b, station_3b = self.bypass_conditions(r_p_mult=r_p_mult,
                                                                                                 cruise=True)
            mach = (((self.fc.p.magnitude / station_03b.P) ** (-(self.gam_a - 1) / self.gam_a) - 1) * 2 / (
                        self.gam_a - 1)) ** 0.5
            return 1 - mach

        self.engine.r_p_mult = minimize_scalar(f, bounds=[0.6, 1.0]).x



    def calibrate_fan_sl(self):

        def f(r_pf):
            station_01, station_02, station_03, station_03b, station_3b = self.bypass_conditions(cruise=False, r_pf=r_pf)

            mach_bypass = (((station_3b.P / station_03b.P) ** (-(self.gam_a - 1) / self.gam_a) - 1) * 2 / (
                    self.gam_a - 1)) ** 0.5
            Vjb = mach_bypass * np.sqrt(self.gam_a * self.R * station_3b.T)
            mdotb = (self.Ab * station_3b.rho_mass() * Vjb).magnitude

            mdotb_req, mdotc_req, mdotf_req, Vjb, Vjc = self.mdots(station_01, station_02, station_03, station_03b, station_3b, False)
            return mdotb - mdotb_req


        self.engine.r_pf = root_scalar(f, x0=1.4, bracket=[1.01, 1.5], rtol=1e-4).root

    def bypass_conditions(self, cruise: bool = True, r_pf: float = None, r_p_mult: float = None):
        if not r_p_mult:
            r_p_mult = self.engine.r_p_mult
        if not r_pf:
            r_pf = self.engine.r_pf
        if cruise:
            fc = self.fc
        else:
            fc = FlightCondition(M=0, h=0 * unit('m'))
            r_p_mult = 1

        r_pc = self.engine.r_po / r_pf

        # TET can be 1550K
        # EGT can be 15-25% for cooling - taken after HPC
        # epsilon Tg-Tm / Tg-Tc for cooling is around 0.6-0.7
        # station 1 - inlet
        station_0_0 = self.flasher.flash(
            T=fc.T0.magnitude,
            P=fc.p0.to('Pa').magnitude,
            zs=self.zs_air
        )

        # station 2 - fan exit
        station_13_0 = self.flasher.flash(
            T=station_0_0.T * (r_pf * r_p_mult) ** ((self.gam_a - 1) / (self.gam_a * self.engine.eta_f)),
            P=station_0_0.P * (r_pf * r_p_mult),
            zs=self.zs_air
        )

        # station 3b - bypass exit total
        station_19_0 = self.flasher.flash(
            T=station_13_0.T,
            P=station_13_0.P,
            zs=self.zs_air
        )

        # station 3b - bypass exit static
        p3star = station_19_0.P * (1 + (self.gam_a - 1) / 2) ** -(self.gam_a / (self.gam_a - 1))
        p3ext = (self.engine.cp_bypass * fc.q_inf / np.sqrt(1 - self.fc.M ** 2) + fc.p).to('Pa').magnitude
        p3b = max(p3star, p3ext)
        # isentropic expansion to p3b
        station_19 = self.flasher.flash(
            S_mass = station_19_0.S_mass(),
            P=p3b,
            zs=self.zs_air
        )


        # station 3 - compressor exit
        station_3_0 = self.flasher.flash(
            T=station_13_0.T * (r_pc * r_p_mult) ** ((self.gam_a - 1) / (self.gam_a * self.engine.eta_c)),
            P=station_13_0.P * (r_pc * r_p_mult),
            zs=self.zs_air
        )

        return station_0_0, station_13_0, station_3_0, station_19_0, station_19

    def mdots(self, station_01, station_02, station_03, station_03b, station_3b, cruise=True):
        if cruise:
            F = self.Fnet
            sfc = self.engine.sfc_cruise
            fc = self.fc
        else:
            F = self.engine.Fmax
            sfc = self.engine.sfc_sl
            fc = FlightCondition(M=0, h=0 * unit('m'))

        Vjb = np.sqrt(2 * self.cP_a * (station_03b.T - station_3b.T)).magnitude
        Vjc = Vjb / self.engine.Vjb_Vjc

        # specific thrusts
        tb = (Vjb - fc.TAS.magnitude + (station_3b.P - fc.p.magnitude) / (station_3b.rho_mass() * Vjb)) * unit('m/s')
        tc = (Vjc - fc.TAS.magnitude) * unit('m/s')

        # required mass flows
        mdotc = (F / (tc + self.engine.BPR * tb)).to_base_units()
        mdotb = mdotc * self.engine.BPR

        return mdotb.magnitude, mdotc.magnitude, (F * sfc).magnitude, Vjb, Vjc

    def __init__(self, flight_condition: FlightCondition, engine: Engine, Af: float, Ajb: float, Ajc: float, Y_h2o: float):
        self.fc = flight_condition
        self.engine = engine
        self.Af = Af
        self.Ab = Ajb
        self.Ac = Ajc
        self.h_ref = 297931 * unit('J/kg')

        # chemistry
        constants, properties = ChemicalConstantsPackage.from_IDs(['water', 'carbon dioxide', 'oxygen', 'nitrogen'])
        kijs = IPDB.get_ip_asymmetric_matrix('ChemSep PR', constants.CASs, 'kij')
        eos_kwargs = {'Pcs': constants.Pcs, 'Tcs': constants.Tcs, 'omegas': constants.omegas, 'kijs': kijs}
        gas = CEOSGas(PRMIX, eos_kwargs=eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
        liquid = CEOSLiquid(PRMIX, eos_kwargs=eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases)
        self.flasher = FlashVL(constants, properties, liquid=liquid, gas=gas)
        self.zs_air = [0, 0, 0.21, 0.79]

        LD = 18  # A320
        L = 55000 * 0.00981 * unit('kN')  # A320 70000
        self.Fnet = L / LD / 2

        self.gam_e = 1.31
        self.cP_e = 830 * unit('kg.m^2/s^2/kg/K')
        self.R = 287 * unit('m^2/s^2/K')
        self.gam_a = 1.4
        self.cP_a = 1005 * unit('kg.m^2/s^2/kg/K')

        self.calibrate_fan_sl()
        print(f"r_pf = {self.engine.r_pf}")
        self.calibrate_PR_cruise()
        # bypass conditions station_0_0, station_13_0, station_3_0, station_19_0, station_19
        self.station_0_0, self.station_13_0, self.station_3_0, self.station_19_0, self.station_19 = self.bypass_conditions()
        # mass flows
        mdotb, mdotc, mdotf, self.Vjb, self.Vjc = self.mdots(self.station_0_0, self.station_13_0, self.station_3_0, self.station_19_0,
                                                             self.station_19)
        self.mdot = mdotc+mdotb

        # station 4 - combustor exit
        self.zs_exhaust = get_exhaust_comp(AFR=(mdotc / mdotf), fuel=engine.fuel, Y_h2o=Y_h2o)
        Q = mdotf * engine.fuel.LCV
        wfan = (self.station_13_0.H_mass() - self.station_0_0.H_mass())
        wcomp = (self.station_3_0.H_mass() - self.station_13_0.H_mass())


        self.station_4_0 = self.flasher.flash(
            H_mass=self.station_3_0.H_mass() + Q.to_base_units().magnitude / mdotc,
            P=0.99 * self.station_3_0.P,
            zs=self.zs_exhaust
        )

        # station 5 - turbine exit
        # this is all we need for the boundary condition - rest is for interest and boundary layer calc
        # iterate as we don't know p05
        # guess...
        p05 = self.station_4_0.P
        T05 = None
        for i in range(4):

            self.station_5_0 = self.flasher.flash(
                H_mass=self.station_4_0.H_mass() - wcomp - self.engine.BPR * wfan,
                P=p05,
                zs=self.zs_exhaust
            )
            T05 = self.station_5_0.T
            p05 = self.station_4_0.P / (self.station_4_0.T / self.station_5_0.T) ** (
                    self.gam_e / (self.gam_e - 1) / engine.eta_t)

        T5 = T05 - 0.5 / self.cP_e.magnitude * self.Vjc ** 2
        Mjc = self.Vjc / np.sqrt(self.gam_e * self.R.magnitude * T5)

        # exit areas from assumption that bypass is choked and core exits at ambient pressure

        # @incollection{HOUGHTON2013427, title = {Chapter 7 - Airfoils and Wings in Compressible Flow}, editor = {E.L. Houghton and P.W. Carpenter and Steven H. Collicott and Daniel T. Valentine}, booktitle = {Aerodynamics for Engineering Students (Sixth Edition)}, publisher = {Butterworth-Heinemann}, edition = {Sixth Edition}, address = {Boston}, pages = {427-477}, year = {2013}, isbn = {978-0-08-096632-8}, doi = {https://doi.org/10.1016/B978-0-08-096632-8.00007-2}, url = {https://www.sciencedirect.com/science/article/pii/B9780080966328000072}, author = {E.L. Houghton and P.W. Carpenter and Steven H. Collicott and Daniel T. Valentine}, keywords = {critical Mach number, linearized subsonic flow, linearized supersonic flow, Prandtl-Glauert rule, compressibility correction, small disturbance theory, supersonic wings, wing sweep, Ackert's rule}, abstract = {Publisher Summary: The chapter discusses the subsonic linearized compressible flow theory for extending the trusted results from incompressible flow into high subsonic flight. The most sweeping approximations, producing the simplest solutions, are made here and result in soluble linear differential equations. This leads to the expression linearized theory associated with airfoils. The chapter summarizes the supersonic linearized theory such as symmetrical double wedge airfoil in supersonic flow, supersonic biconvex circular arc airfoil in supersonic flow, general airfoil section, airfoil section made up of unequal circular arcs, double-wedge airfoil section. Several other aspects of supersonic wings such as the shock-expansion approximation, wings of finite span, computational methods are also presented in this chapter. The compressible-flow equations in various forms are considered in order to predict the behavior of airfoil sections in high sub- and supersonic flows. The wings in compressible flow, such as transonic flow, subcritical flow, supersonic linearized theory, and other aspects of supersonic wings are discussed in this chapter. The analysis of this regime involves solving a set of nonlinear differential equations, a task that demands either advanced computational techniques or some form of approximation. The approximations come about mainly from assuming that all disturbances are small disturbances or small perturbations to the free-stream flow conditions. The chapter also explores the phenomenon of wave drag in supersonic flight and how it is predicted by both the shock-expansion method and linearized supersonic flow.}}


        def f0(M, f, ga):
            return f - ga / np.sqrt(ga - 1) * M * (1 + (ga - 1) / 2 * M ** 2) ** (-0.5 * (ga + 1) / (ga - 1))

        # approximate isentropic expansion to ambient for bypass:
        station_jb = self.flasher.flash(
            S_mass=self.station_19_0.S_mass(),
            P=self.fc.p.magnitude,
            zs=self.zs_air
        )
        Mjb = (((station_jb.P / self.station_19_0.P) ** (-(self.gam_a - 1) / self.gam_a) - 1) * 2 / (
                self.gam_a - 1)) ** 0.5
        Ajb = mdotb * np.sqrt(self.cP_a * self.station_19_0.T) / (
                self.gam_a / np.sqrt(self.gam_a - 1) * Mjb * (1 + (self.gam_a - 1) / 2 * Mjb ** 2) ** 0.5) / self.fc.p

        # approximate isentropic expansion to ambient for core:
        station_jc = self.flasher.flash(
            S_mass=self.station_5_0.S_mass(),
            P=self.fc.p.magnitude,
            zs=self.zs_exhaust
        )
        Mjc = (((station_jc.P / self.station_5_0.P) ** (-(self.gam_e - 1) / self.gam_e) - 1) * 2 / (
                self.gam_e - 1)) ** 0.5
        Ajc = mdotc * np.sqrt(self.cP_e * self.station_5_0.T) / (
                self.gam_e / np.sqrt(self.gam_e - 1) * Mjc * (1 + (self.gam_e - 1) / 2 * Mjc ** 2) ** 0.5) / self.fc.p

        # station 1 - intake
        f2 = (mdotb + mdotc) * np.sqrt(self.cP_a.magnitude * self.fc.T0.magnitude) / self.Af.magnitude / self.fc.p0.magnitude

        M2 = root_scalar(f0, (f2, self.gam_a,), bracket=[0, 1]).root
        self.p2 = self.station_0_0.P * (1 + (self.gam_a - 1) / 2 * M2 ** 2) ** -(self.gam_a / (self.gam_a - 1))
        V2 = np.sqrt(self.cP_a * self.station_0_0.T) * (self.gam_a - 1) * M2 * (1 + (self.gam_a - 1) / 2 * M2 ** 2) ** -0.5
        U2 = self.engine.N1.to_base_units() * self.engine.D / 2

        # jet dimensions for wake estimation
        Aj = Ajb.magnitude + Ajc.magnitude
        rj = np.sqrt(Aj / (2 * np.pi))
        rc = np.sqrt(self.Ac / (2 * np.pi))

        print(self.Af.magnitude * V2 * self.fc.rho.magnitude, self.Ab * self.Vjb * self.station_19.rho_mass(), self.Ac * self.Vjc * self.station_5_0.rho_mass())
        print(f"""
        BYPASS:
          p19_0 = {self.station_19_0.P}
          h19_0 = {(self.station_19_0.H_mass() + self.h_ref.magnitude)}
          Ab = {self.Ab.to_base_units():.5g~P}
          
        CORE:
          wH2O = {self.zs_exhaust[0]:.5g}
          p09 = {p05}
          h09 = {(self.station_5_0.H_mass() + self.h_ref.magnitude)}
          Ac = {self.Ac.to_base_units():.5g~P}
        
        INTAKE:
          M2 = {M2}
          p2 = {self.p2}
          V2 = {V2},{self.fc.TAS}
          U2 = {U2}
          phi2 = {(V2 / U2).magnitude: .5g}
        
        TOTAL:
          rc = {rc}
          rj = {rj}
          Vjb = {self.Vjb}
          Vjc = {self.Vjc}
          mdot = {mdotc} + {mdotb} = {(mdotb + mdotc)}
          F = {self.Fnet},{(mdotb*self.Vjb+mdotc*self.Vjc-(mdotb+mdotc)*V2.magnitude+self.Ab.magnitude*self.station_19.P-self.Af.magnitude*self.p2)/1000}
        """)

# give bypass ratio and fan area in filename so paraview script can run jet with correct params

import copy
import json

import numpy as np
from ansys.geometry.core.math.point import Point2D
from flightcondition import FlightCondition, unit
from matplotlib import pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from svgpathtools import Arc, svg2paths

from geometry import fillet
from hygrometry import psat_ice, psat_water
from jet import JetCondition, PW1100G
from setup import AdvancedJSONEncoder, BoundaryCondition, boundary_layer_mesh_stats

M_h2o = 18.01528  # g/mol
M_air = 28.9647  # g/mol

engine = PW1100G

# get bsplines traced from my very specific SVG. not portable, not generalised, terrible coding practice
paths, attributes = svg2paths("images/exhaust_traced.svg")
# order of this dict matters as it reflects the numbering of lines in the svg.
# dict ordering is standard in modern python and in svg, but not guaranteed in early python so not pythonic?
lines = {"nacelle": [],
         "bypass_tail_outer": [],
         "bypass_tail_inner": [],
         "core_tail_outer": [],
         "core_iface": [],
         "core_tail_inner": [],
         "centreline": [],
         "fan_iface": [],
         "bypass_iface": [],
         "tail_zero_rad": [],
         "nosecone": [],
         "wake1_i": [],
         "wake1": [],
         "wake2_i": [],
         "wake2": []}

origin = paths[9][0].end
scale = abs((engine.D / 2 / (paths[7][0].start.imag - paths[6][0].start.imag)).magnitude)
for i, path in enumerate(paths):
    paths[i] = path.translated(-origin).scaled(scale)

paths[3] = paths[3].reversed()

p1 = paths[1][0].end
p2 = paths[2][1].start
L = abs(p1 - p2)
Ab = abs(np.pi * L * (p1.imag + p2.imag)) * unit('m^2')
Ac = abs(np.pi * (paths[4][0].start.imag ** 2 - paths[4][0].end.imag ** 2)) * unit('m^2')
plt.plot([p1.real, p2.real], [-p1.imag, -p2.imag])

paths_orig = copy.deepcopy(paths)
lines_orig = copy.deepcopy(lines)

for lwake in [2,5,10,20,50]:
    paths = copy.deepcopy(paths_orig)
    lines = copy.deepcopy(lines_orig)


    f11, f12, wake1_i, wake1 = fillet(paths[0][-1], paths[1][-1], 0.006, 0.135, 0.6, lwake)
    f21, f22, wake2_i, wake2 = fillet(paths[2][-1], paths[3][-1], 0.006, 0.135, 0.25, lwake)
    paths[0].append(f11)
    paths[1].append(f12)
    paths[2].append(f21)
    paths[3].append(f22)
    paths.append(wake1_i)
    paths.append(wake1)
    paths.append(wake2_i)
    paths.append(wake2)

    Af = np.pi * (paths[0][0].start.imag ** 2 - paths[10][0].start.imag ** 2) * unit('m^2')
    te_thickness = 15 * unit('mm')

    # interpolate through splines, 10 samples per spline segment
    t = np.linspace(0, 1, 100)
    colors = [c for c in TABLEAU_COLORS.values()] + ['black']
    for i, key in enumerate(lines.keys()):
        for j, segment in enumerate(paths[i]):
            if isinstance(segment, Arc):
                pts = segment.point(t[j > 0:])
            else:
                pts = segment.points(t[j > 0:])
            [lines[key].append(Point2D(pt * unit('m'))) for pt in zip(pts.real, pts.imag)]

    # core tail outer needs reversing to match syntax: x increasing
    lines["core_tail_outer"] = lines["core_tail_outer"][::-1]
    tedges = []
    # ensure closed profile by matching points
    lines["nosecone"][0].x = lines["fan_iface"][-1].x
    lines["fan_iface"][-1] = lines["nosecone"][0]

    lines["nacelle"][0].x = lines["fan_iface"][0].x
    lines["fan_iface"][0] = lines["nacelle"][0]
    # te
    # tedges.append(["nacelle", "bypass_tail_outer"][cut_te(lines["nacelle"], lines["bypass_tail_outer"],te_thickness)])
    # lines["nacelle"][-1] = lines["bypass_tail_outer"][-1]
    lines["bypass_tail_outer"][0].x = lines["bypass_iface"][0].x
    lines["bypass_iface"][0] = lines["bypass_tail_outer"][0]

    lines["bypass_tail_inner"][0].x = lines["bypass_iface"][-1].x
    lines["bypass_iface"][-1] = lines["bypass_tail_inner"][0]
    # te
    # tedges.append(["bypass_tail_inner", "core_tail_outer"][cut_te(lines["bypass_tail_inner"], lines["core_tail_outer"],te_thickness)])
    # lines["bypass_tail_inner"][-1] = lines["core_tail_outer"][-1]
    lines["core_tail_outer"][-1].x = lines["core_iface"][-1].x
    lines["core_iface"][-1] = lines["core_tail_outer"][-1]

    lines["core_tail_inner"][0].x = lines["core_iface"][0].x
    lines["core_iface"][0] = lines["core_tail_inner"][0]

    lines["core_tail_inner"][-1].x = lines["tail_zero_rad"][0].x
    lines["tail_zero_rad"][0] = lines["core_tail_inner"][-1]

    with open(f"geom/lines-{lwake}m.txt", "w") as f:
        f.write(f'3d=true\nfit=true\n')
        for i, key in enumerate(lines.keys()):
            plt.plot([point.x.magnitude for point in lines[key]], [- point.y.magnitude for point in lines[key]], '-',
                     label=key,
                     color='black')
            for point in lines[key][:-1]:
                f.write(f'0 {1000 * point.x.magnitude} {abs(1000 * point.y.magnitude)}\n')
            if key in tedges:
                f.write(f'\n0 {1000 * lines[key][-2].x.magnitude} {abs(1000 * lines[key][-2].y.magnitude)}\n')
            f.write(f'0 {1000 * lines[key][-1].x.magnitude} {abs(1000 * lines[key][-1].y.magnitude)}\n')

            f.write('\n')
            # color=colors[i])



condition = FlightCondition(M=0.78, L=3.8 * unit('m'), h=37000 * unit('ft'), units='SI')
print(condition)

pv_w = psat_water(condition.p.magnitude, condition.T.magnitude) * unit('Pa')
pv_i = psat_ice(condition.p.magnitude, condition.T.magnitude) * unit('Pa')

p_h2o = (
                pv_w + pv_i) / 2  # between limits. can also be set to 120% humidity [Petzold, A. et al.](https://doi.org/10.5194/acp-20-8157-2020)
Y_h2o = (M_h2o * p_h2o) / (M_h2o * p_h2o + M_air * (condition.p - p_h2o))

jet = JetCondition(condition, engine, Af, Ab, Ac, Y_h2o)
Ab_des = jet.Ab.to_base_units()
Ac_des = jet.Ac.to_base_units()

print('\nEXTERNAL')
BL1 = boundary_layer_mesh_stats(rho=condition.rho, V=condition.TAS, mu=condition.mu,
                                L=3.8 * unit('m') * 0.0025, x=3.8 * unit('m'),
                                yplus=1.0, GR=1.2)
# annulus width
print('\nBYPASS')
BL2 = boundary_layer_mesh_stats(rho=jet.station_19_0.rho_mass() * unit('kg/m^3'), V=jet.Vjb * unit('m/s'),
                                mu=jet.station_19_0.mu() * unit('Pa.s').to_base_units(),
                                L=0.6 * unit('m'), x=2.1 * unit('m'),
                                yplus=1.0, GR=1.2)
# annulus width
print('\nCORE')
BL3 = boundary_layer_mesh_stats(rho=jet.station_5_0.rho_mass() * unit('kg/m^3'), V=jet.Vjc * unit('m/s'),
                                mu=jet.station_5_0.mu() * unit('Pa.s').to_base_units(),
                                L=0.4 * unit('m'), x=1.0 * unit('m'),
                                yplus=1.0, GR=1.2)
# use bypass annulus as reference dimension
print('\nWAKE')
BL4 = boundary_layer_mesh_stats(rho=jet.station_19_0.rho_mass() * unit('kg/m^3'), V=0.5 * (jet.Vjc - jet.Vjb) * unit('m/s'),
                                mu=jet.station_19_0.mu() * unit('Pa.s').to_base_units(),
                                L=0.4 * unit('m'), x=50 * unit('m'),
                                yplus=500.0, GR=1)

C_mu = 0.09  # https://doi.org/10.1016/j.jcp.2004.08.009

I_bypass = 0.05  # Russo, F. and Basse, N.T. (2016)
I_core = 0.05  # https://www.cfd-online.com/Wiki/Turbulence_intensity

l_bypass = 0.05 * np.pi * 2 * abs(
    (lines["bypass_iface"][0].y.magnitude + lines["bypass_iface"][-1].y.magnitude) / 2) * unit(
    'm') / 20  # https://www.cfd-online.com/Wiki/Turbulent_length_scale
l_core = 0.05 * np.pi * 2 * abs(
    (lines["core_iface"][0].y.magnitude + lines["core_iface"][-1].y.magnitude) / 2) * unit(
    'm') / 20  # https://www.cfd-online.com/Wiki/Turbulent_length_scale

nu_bypass = np.sqrt(3 / 2) * condition.TAS * I_bypass * l_bypass
nu_core = np.sqrt(3 / 2) * condition.TAS * I_core * l_core

k_bypass = 3 / 2 * (condition.TAS * I_bypass) ** 2
k_core = 3 / 2 * (condition.TAS * I_core) ** 2

w_bypass = C_mu ** 0.75 * k_bypass ** 0.5 / l_bypass
w_core = C_mu ** 0.75 * k_core ** 0.5 / l_core

print(nu_bypass, nu_core, condition.mu / condition.rho)
print(k_bypass, k_core)
print(w_bypass, w_core)

bc = BoundaryCondition(M=condition.M,
                       alpha=0,
                       T=condition.T,
                       h0=jet.station_0_0.H_mass() + jet.h_ref.magnitude,
                       p=condition.p,
                       p0=condition.p0,
                       nu=condition.nu,
                       k=10e-6 * condition.TAS.magnitude ** 2,
                       w=5*condition.TAS.magnitude/condition.L.magnitude,
                       Y_h2o=Y_h2o,
                       vx=condition.TAS,
                       mdot=jet.mdot,
                       BPR=engine.BPR,
                       Fnet=jet.Fnet,
                       h0_bypass=jet.station_19_0.H_mass() + jet.h_ref.magnitude,
                       nu_bypass=nu_bypass,
                       k_bypass=k_bypass,
                       w_bypass=w_bypass,
                       A_bypass=Ab,
                       h0_core=jet.station_5_0.H_mass() + jet.h_ref.magnitude,
                       nu_core=nu_core,
                       k_core=k_core,
                       w_core=w_core,
                       A_core=Ac,
                       Y_h2o_core=jet.zs_exhaust[0],
                       p_fan=jet.p2,
                       A_fan=abs(np.pi * (lines["fan_iface"][0].y.magnitude ** 2 - lines["fan_iface"][-1].y.magnitude ** 2)),
                       bl_nacelle_bypass=BL1,
                       bl_bypass_core=BL2,
                       bl_core_tail=BL3,
                       bl_wake=BL4)
with open("geom/nacelle.json", "w", encoding='utf-8') as f:
    json.dump(bc, f, ensure_ascii=False, cls=AdvancedJSONEncoder, indent=4)

for tedge in tedges:
    print(
        f"trailing edge thickness on {tedge} = {1000 * abs(lines[tedge][-2].y.magnitude - lines[tedge][-1].y.magnitude):.2f}mm")

plt.gca().set_aspect('equal', adjustable='box')
plt.show()

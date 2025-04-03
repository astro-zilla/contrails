import json

import numpy as np
from flightcondition import FlightCondition, unit
from matplotlib import pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from svgpathtools import svg2paths
from ansys.geometry.core import __version__, launch_modeler
from ansys.geometry.core.sketch import Sketch
from ansys.geometry.core.math import Point2D, Vector3D, Point3D, UNITVECTOR3D_X, ZERO_POINT3D

from jet import Hydrocarbon, JetCondition, Engine
from setup import AdvancedJSONEncoder, BoundaryCondition, boundary_layer_mesh_stats
from geometry import move_path, move_pt
from hygrometry import psat_ice, psat_water

print(f"PyAnsys Geometry version: {__version__}")

M_h2o = 18.01528  # g/mol
M_air = 28.9647  # g/mol

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
         "nosecone": []}

y0 = paths[6][0].start.imag
# turbomachinery performance

jetA1 = Hydrocarbon(n_C=10.8, n_H=21.6, LCV=43.15 * unit('MJ/kg'))

PW1100G = Engine(D=(81 * unit('in')).to('m'), BPR=12.5,
                 sfc_cruise=0.0144 * unit('kg/kN/s'), fuel=jetA1,
                 eta_c=0.85, eta_f=0.92345, eta_t=0.9,
                 r_pf=1.23, r_po=37, N1=0.85 * 3281 * unit('2*pi/min'),  # 42
                 Vjb_Vjc=0.9)

LEAP1A = Engine(D=(78 * unit('in')).to('m'), BPR=11,
                sfc_cruise=0.0144 * unit('kg/kN/s'), fuel=jetA1,
                eta_c=0.85, eta_f=0.92345, eta_t=0.9,
                r_pf=1.24, r_po=35, N1=0.85 * 3894 * unit('2*pi/min'),  # 40
                Vjb_Vjc=0.8)

engine = PW1100G

scale = abs((engine.D / 2 / (paths[7][0].start.imag - paths[6][0].start.imag)).magnitude) * unit('m')
Af = np.pi * (((paths[0][0].start.imag - y0) * scale) ** 2 - ((paths[10][0].start.imag - y0) * scale) ** 2)
te_thickness = 15 * unit('mm')

print(f"{engine.D = :f}\n"
      f"{scale = :f}\n"
      f"{Af = :f}")

condition = FlightCondition(M=0.78, L=3.8 * unit('m'), h=37000 * unit('ft'), units='SI')
print(condition)


pv_w = psat_water(condition.p.magnitude,condition.T.magnitude)*unit('Pa')
pv_i = psat_ice(condition.p.magnitude,condition.T.magnitude)*unit('Pa')

p_h2o = (pv_w+pv_i)/2  # between limits. can also be set to 120% humidity [Petzold, A. et al.](https://doi.org/10.5194/acp-20-8157-2020)
Y_h2o = (M_h2o * p_h2o) / (M_h2o * p_h2o + M_air * (condition.p - p_h2o))

print(pv_w,pv_i,p_h2o,Y_h2o)



jet = JetCondition(condition, engine, Af, Y_h2o)
Ab_des = jet.Ab.to_base_units()
Ac_des = jet.Ac.to_base_units()



# outer radius stays same
outer = scale * paths[1][0].end
inner = scale * paths[2][1].start
l0 = abs(outer - inner)
# inner control point follows straight line from inital point to outside te
rb_outer = abs(outer.imag - y0 * scale)
delta_rb_l = abs(outer.imag - inner.imag) / l0

# A = np.pi*(rb_outer+l0*rb_inner_l)*l0
# by quadratic formula
# l = (-np.pi * rb_outer + np.sqrt(np.pi ** 2 * rb_outer ** 2 + 4 * rb_outer * Ab_des)) / 2 / rb_inner_l
l = (rb_outer - np.sqrt(rb_outer ** 2 - delta_rb_l * Ab_des / np.pi)) / delta_rb_l
t = l / l0

move_pt(paths[2], 1, t * inner / scale + (1 - t) * outer / scale)

r_core_inner = np.sqrt(abs((paths[4][-1].end.imag - y0) * scale) ** 2 - Ac_des / np.pi)
delta = (r_core_inner - abs((paths[5][0].start.imag - y0) * scale)) / scale * complex(0, -1)
move_path(paths[5], delta)
paths[5][-1].end -= delta
paths[5][-1].control2 -= delta
paths[4][0].start += delta

# interpolate through splines, 10 samples per spline segment
t = np.linspace(0, 1, 100)
colors = [c for c in TABLEAU_COLORS.values()] + ['black']
for i, key in enumerate(lines.keys()):
    for j, segment in enumerate(paths[i]):
        pts = scale * segment.poly()(t[j > 0:])
        [lines[key].append(Point2D(pt)) for pt in zip(pts.real.magnitude, pts.imag.magnitude)]
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
lines["nacelle"][-1] = lines["bypass_tail_outer"][-1]
lines["bypass_tail_outer"][0].x = lines["bypass_iface"][0].x
lines["bypass_iface"][0] = lines["bypass_tail_outer"][0]

lines["bypass_tail_inner"][0].x = lines["bypass_iface"][-1].x
lines["bypass_iface"][-1] = lines["bypass_tail_inner"][0]
# te
# tedges.append(["bypass_tail_inner", "core_tail_outer"][cut_te(lines["bypass_tail_inner"], lines["core_tail_outer"],te_thickness)])
lines["bypass_tail_inner"][-1] = lines["core_tail_outer"][-1]
lines["core_tail_outer"][0].x = lines["core_iface"][-1].x
lines["core_iface"][-1] = lines["core_tail_outer"][0]

lines["core_tail_inner"][0].x = lines["core_iface"][0].x
lines["core_iface"][0] = lines["core_tail_inner"][0]

lines["core_tail_inner"][-1].x = lines["tail_zero_rad"][0].x
lines["tail_zero_rad"][0] = lines["core_tail_inner"][-1]

y0 = lines["centreline"][0].y.magnitude
# Ab = np.pi * ((lines["bypass_iface"][0].y.magnitude - y0) ** 2 - (lines["bypass_iface"][-1].y.magnitude - y0) ** 2)
# Ac = np.pi * ((lines["core_iface"][-1].y.magnitude - y0) ** 2 - (lines["core_iface"][0].y.magnitude - y0) ** 2)
# print(f"{Ab = :f}\n{Ac = :f}")
with open("CAD/lines.txt", "w") as f:
    f.write(f'3d=true\nfit=true\n')
    for i, key in enumerate(lines.keys()):
        plt.plot([point.x.magnitude for point in lines[key]], [y0 - point.y.magnitude for point in lines[key]], '-',
                 label=key,
                 color='black')
        for point in lines[key][:-1]:
            f.write(f'0 {1000 * point.x.magnitude} {abs(1000 * point.y.magnitude - 1000 * y0)}\n')
        if key in tedges:
            f.write(f'\n0 {1000 * lines[key][-2].x.magnitude} {abs(1000 * lines[key][-2].y.magnitude - 1000 * y0)}\n')
        f.write(f'0 {1000 * lines[key][-1].x.magnitude} {abs(1000 * lines[key][-1].y.magnitude - 1000 * y0)}\n')

        f.write('\n')
        # color=colors[i])
# plt.legend(loc="upper right")


print('\nEXTERNAL')
BL1 = boundary_layer_mesh_stats(rho=condition.rho, V=condition.TAS, mu=condition.mu,
                          L=3.8 * unit('m') * 0.0025, x=3.8 * unit('m'),
                          yplus=1.0, GR=1.2)
# annulus width
print('\nBYPASS')
BL2 = boundary_layer_mesh_stats(rho=jet.station_03b.rho_mass() * unit('kg/m^3'), V=jet.Vjb,
                          mu=jet.station_03b.mu() * unit('Pa.s').to_base_units(),
                          L=0.6 * unit('m'), x=2.1 * unit('m'),
                          yplus=1.0, GR=1.2)
# annulus width
print('\nCORE')
BL3 = boundary_layer_mesh_stats(rho=jet.station_05.rho_mass() * unit('kg/m^3'), V=jet.Vjc,
                          mu=jet.station_05.mu() * unit('Pa.s').to_base_units(),
                          L=0.4 * unit('m'), x=1.0 * unit('m'),
                          yplus=1.0, GR=1.2)
# use bypass annulus as reference dimension
print('\nWAKE')
BL4 = boundary_layer_mesh_stats(rho=jet.station_03b.rho_mass() * unit('kg/m^3'), V=0.5 * (jet.Vjc - jet.Vjb),
                          mu=jet.station_03b.mu() * unit('Pa.s').to_base_units(),
                          L=0.6 * unit('m'), x=5.291 * unit('m'),
                          yplus=30.0, GR=1.2)

bc = BoundaryCondition(M=condition.M,
                       alpha=0,
                       T=condition.T,
                       h0=jet.station_01.H_mass()+jet.h_ref.magnitude,
                       p=condition.p,
                       p0=condition.p0,
                       mu=condition.mu,
                       Y_h2o=Y_h2o,
                       vx=condition.TAS,
                       mdot=jet.mdot,
                       BPR=engine.BPR,
                       h0_bypass=jet.station_03b.H_mass()+jet.h_ref.magnitude,
                       mu_bypass=jet.station_03b.mu(),
                       A_bypass=abs(np.pi*((lines["bypass_iface"][0].y.magnitude-y0)**2 - (lines["bypass_iface"][-1].y.magnitude-y0)**2)),
                       h0_core=jet.station_05.H_mass()+jet.h_ref.magnitude,
                       mu_core=jet.station_05.mu(),
                       A_core=abs(np.pi*((lines["core_iface"][0].y.magnitude-y0)**2 - (lines["core_iface"][-1].y.magnitude-y0)**2)),
                       Y_h2o_core=jet.zs_exhaust[0],
                       p_fan=jet.p1,
                       A_fan=abs(np.pi*((lines["fan_iface"][0].y.magnitude-y0)**2 - (lines["fan_iface"][-1].y.magnitude-y0)**2)),
                       bl_nacelle_bypass=BL1,
                       bl_bypass_core=BL2,
                       bl_core_tail=BL3,
                       bl_wake=BL4)
with open("CAD/boundary_conditions.json", "w", encoding='utf-8') as f:
    json.dump(bc, f, ensure_ascii=False, cls=AdvancedJSONEncoder, indent=4)

for tedge in tedges:
    print(
        f"trailing edge thickness on {tedge} = {1000 * abs(lines[tedge][-2].y.magnitude - lines[tedge][-1].y.magnitude):.2f}mm")

plt.ylim(-1.5, 2.8)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
exit()
# draw lines in sketch segment by segment
sketch = Sketch()
for key in lines.keys():
    if key != "centreline":
        points = lines[key]
        sketch.segment(points[0], points[1])
        for i in range(len(points) - 2):
            sketch.segment_to_point(points[i + 2])

sketch.segment(lines["nosecone"][-1], lines["tail_zero_rad"][-1])

# translate so that tip of nosecone is origin
nose = Point3D((lines["nosecone"][-1].x.magnitude, lines["nosecone"][-1].y.magnitude, 0))
sketch.translate_sketch_plane(-Vector3D(nose))

# revolve to 3d in SpaceClaim
# has to be run on windows
modeler = launch_modeler()
design = modeler.create_design(f"nacelle")
design.revolve_sketch("Nacelle", sketch, UNITVECTOR3D_X, 360 * unit("degrees"), ZERO_POINT3D)
design.plot()

scdocx_location = design.export_to_scdocx()
print(f"design saved to {scdocx_location}")

modeler.close()

# handles, labels = plt.gca().get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# plt.legend(by_label.values(), by_label.keys())
# plt.axis("equal")
# plt.show()

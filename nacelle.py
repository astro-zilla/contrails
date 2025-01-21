import numpy as np
from flightcondition import FlightCondition, unit
from matplotlib import pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from svgpathtools import svg2paths
from ansys.geometry.core import __version__, launch_modeler
from ansys.geometry.core.sketch import Sketch
from ansys.geometry.core.math import Point2D, Vector3D, Point3D, UNITVECTOR3D_X, UNITVECTOR3D_Y, ZERO_POINT3D
from pathlib import Path
from jet import JetCondition, Engine

print(f"PyAnsys Geometry version: {__version__}")


def cut_te(upper, lower):
    u2, u1 = upper[-2:]
    l2, l1 = lower[-2:]
    if u2.x > l2.x:
        interp_var = (l1.x - u2.x) / (l1.x - l2.x)
        lower[-1].x = upper[-2].x
        lower[-1].y = interp_var * l2.y + (1 - interp_var) * l1.y
        upper[-1] = lower[-1]
    else:
        interp_var = (u1.x - l2.x) / (u1.x - u2.x)
        upper[-1].x = lower[-2].x
        upper[-1].y = interp_var * u2.y + (1 - interp_var) * u1.y
        lower[-1] = upper[-1]


def move_pt(line, seg_idx, pos):
    delta = pos - line[seg_idx].start
    line[seg_idx - 1].end += delta
    line[seg_idx - 1].control2 += delta
    line[seg_idx].start += delta
    line[seg_idx].control1 += delta


# get bsplines traced from my very specific SVG. not portable, not generalised, terrible coding practice
paths, attributes = svg2paths("exhaust_traced.svg")
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

PW1100G = Engine(D=(81 * unit('in')).to('m'), BPR=12.5,
                 sfc_cruise=0.0144 * unit('kg/kN/s'),
                 eta_c=0.85, eta_f=0.92345, eta_t=0.9,
                 r_pf=1.2, r_po=37, # 42
                 Vjb_Vjc=0.8)

LEAP1A = Engine(D=(78 * unit('in')).to('m'), BPR=11,
                sfc_cruise=0.0144 * unit('kg/kN/s'),
                eta_c=0.85, eta_f=0.92345, eta_t=0.9,
                r_pf=1.22, r_po=35, # 40
                Vjb_Vjc=0.8)

condition = FlightCondition(M=0.78, L=3.8 * unit('m'), h=37000 * unit('ft'), units='SI')
jet = JetCondition(condition, LEAP1A)
# Ab_des = 2.449
# Ac_des = 0.446
Ab_des = jet.Ab.to_base_units()
Ac_des = jet.Ac.to_base_units()

scale = abs((jet.engine.D / 2 / (paths[7][0].start.imag - paths[6][0].start.imag)).magnitude)
print(f"{jet.engine.D = :f}\n"
      f"{scale = :f}")

# outer radius stays same
outer = scale * paths[1][0].end
inner = scale * paths[2][1].start
l0 = abs(outer - inner) * unit('m')
# inner control point follows straight line from inital point to outside te
rb_outer = abs(outer.imag - y0 * scale) * unit('m')
delta_rb_l = abs(outer.imag - inner.imag) / l0 * unit('m')

# A = np.pi*(rb_outer+l0*rb_inner_l)*l0
# by quadratic formula
# l = (-np.pi * rb_outer + np.sqrt(np.pi ** 2 * rb_outer ** 2 + 4 * rb_outer * Ab_des)) / 2 / rb_inner_l
l = (rb_outer - np.sqrt(rb_outer ** 2 - delta_rb_l * Ab_des / np.pi)) / delta_rb_l
t = l / l0
print(f"{t = :f}")
move_pt(paths[2], 1, t * inner / scale + (1 - t) * outer / scale)

# interpolate through splines, 10 samples per spline segment
t = np.linspace(0, 1, 1000)
colors = [c for c in TABLEAU_COLORS.values()] + ['black']
for i, key in enumerate(lines.keys()):
    for j, segment in enumerate(paths[i]):
        pts = scale * segment.poly()(t[j > 0:])
        [lines[key].append(Point2D(pt)) for pt in zip(pts.real, pts.imag)]
# core tail outer needs reversing to match syntax: x increasing
lines["core_tail_outer"] = lines["core_tail_outer"][::-1]

# ensure closed profile by matching points
lines["nosecone"][0] = lines["fan_iface"][-1]

lines["nacelle"][0] = lines["fan_iface"][0]
# te
cut_te(lines["nacelle"], lines["bypass_tail_outer"])
# lines["nacelle"][-1] = lines["bypass_tail_outer"][-1]
lines["bypass_tail_outer"][0] = lines["bypass_iface"][0]

lines["bypass_tail_inner"][0] = lines["bypass_iface"][-1]
# te
cut_te(lines["bypass_tail_inner"], lines["core_tail_outer"])
# lines["bypass_tail_inner"][-1] = lines["core_tail_outer"][-1]
lines["core_tail_outer"][0] = lines["core_iface"][-1]

lines["core_tail_inner"][0] = lines["core_iface"][0]
lines["core_tail_inner"][-1] = lines["tail_zero_rad"][0]

y0 = lines["centreline"][0].y.magnitude
Ab = np.pi * ((lines["bypass_iface"][0].y.magnitude - y0) ** 2 - (lines["bypass_iface"][-1].y.magnitude - y0) ** 2)
Ac = np.pi * ((lines["core_iface"][-1].y.magnitude - y0) ** 2 - (lines["core_iface"][0].y.magnitude - y0) ** 2)
print(f"{Ab = :f}\n{Ac = :f}")

for i, key in enumerate(lines.keys()):
    plt.plot([point.x.magnitude for point in lines[key]], [y0 - point.y.magnitude for point in lines[key]], '-', label=key,
             color='black')
    # color=colors[i])
# plt.legend(loc="upper right")

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

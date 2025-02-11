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
    del upper[-5:-1], lower[-5:-1]
    u2, u1 = upper[-2:]
    l2, l1 = lower[-2:]
    if u2.x > l2.x:
        interp_var = (l1.x - u2.x) / (l1.x - l2.x)
        lower[-1].x = upper[-2].x
        lower[-1].y = interp_var * l2.y + (1 - interp_var) * l1.y
        upper[-1] = lower[-1]
        return 0
    else:
        interp_var = (u1.x - l2.x) / (u1.x - u2.x)
        upper[-1].x = lower[-2].x
        upper[-1].y = interp_var * u2.y + (1 - interp_var) * u1.y
        lower[-1] = upper[-1]
        return 1


def move_pt(line, seg_idx, pos):
    delta = pos - line[seg_idx].start
    line[seg_idx - 1].end += delta
    line[seg_idx - 1].control2 += delta
    line[seg_idx].start += delta
    line[seg_idx].control1 += delta


def move_path(path, delta):
    for seg in path:
        seg.start += delta
        seg.control1 += delta
        seg.control2 += delta
        seg.end += delta


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
                 r_pf=1.22, r_po=37, N1=0.85 * 3281 * unit('2*pi/min'),  # 42
                 Vjb_Vjc=0.8)

LEAP1A = Engine(D=(78 * unit('in')).to('m'), BPR=11,
                sfc_cruise=0.0144 * unit('kg/kN/s'),
                eta_c=0.85, eta_f=0.92345, eta_t=0.9,
                r_pf=1.24, r_po=35, N1=0.85 * 3894 * unit('2*pi/min'),  # 40
                Vjb_Vjc=0.8)

engine = PW1100G

scale = abs((engine.D / 2 / (paths[7][0].start.imag - paths[6][0].start.imag)).magnitude) * unit('m')
Af = np.pi * (((paths[0][0].start.imag - y0) * scale) ** 2 - ((paths[10][0].start.imag - y0) * scale) ** 2)

print(f"{engine.D = :f}\n"
      f"{scale = :f}\n"
      f"{Af = :f}")

condition = FlightCondition(M=0.78, L=3.8 * unit('m'), h=37000 * unit('ft'), units='SI')
print(condition)
jet = JetCondition(condition, engine, Af)
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
print(f"{t = :f}")
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
tedges.append(["nacelle", "bypass_tail_outer"][cut_te(lines["nacelle"], lines["bypass_tail_outer"])])
# lines["nacelle"][-1] = lines["bypass_tail_outer"][-1]
lines["bypass_tail_outer"][0].x = lines["bypass_iface"][0].x
lines["bypass_iface"][0] = lines["bypass_tail_outer"][0]

lines["bypass_tail_inner"][0].x = lines["bypass_iface"][-1].x
lines["bypass_iface"][-1] = lines["bypass_tail_inner"][0]
# te
tedges.append(["bypass_tail_inner", "core_tail_outer"][cut_te(lines["bypass_tail_inner"], lines["core_tail_outer"])])
# lines["bypass_tail_inner"][-1] = lines["core_tail_outer"][-1]
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
with open("lines.txt", "w") as f:
    f.write(f'3d=true\nfit=true\n')
    for i, key in enumerate(lines.keys()):
        plt.plot([point.x.magnitude for point in lines[key]], [y0 - point.y.magnitude for point in lines[key]], '-',
                 label=key,
                 color='black')
        for point in lines[key][:-1]:
            f.write(f'0 {1000*point.x.magnitude} {1000*point.y.magnitude - 1000*y0}\n')
        if key in tedges:
            f.write(f'\n0 {1000*lines[key][-2].x.magnitude} {1000*lines[key][-2].y.magnitude - 1000*y0}\n')
        f.write(f'0 {1000*lines[key][-1].x.magnitude} {1000*lines[key][-1].y.magnitude - 1000*y0}\n')

        f.write('\n')
        # color=colors[i])
# plt.legend(loc="upper right")

for tedge in tedges:
    print(f"trailing edge thickness on {tedge} = {1000*abs(lines[tedge][-2].y.magnitude - lines[tedge][-1].y.magnitude):.2f}mm")

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

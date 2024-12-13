import numpy as np
from flightcondition import unit
from matplotlib import pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from svgpathtools import svg2paths
from ansys.geometry.core import __version__, launch_modeler
from ansys.geometry.core.sketch import Sketch
from ansys.geometry.core.math import Point2D, Vector3D, Point3D, UNITVECTOR3D_X, UNITVECTOR3D_Y, ZERO_POINT3D
from pathlib import Path

print(f"PyAnsys Geometry version: {__version__}")


def cut_te(upper, lower):
    u2, u1 = upper[-2:]
    l2, l1 = lower[-2:]
    if u2.x > l2.x:
        interp_var = (l1.x - u2.x) / (l1.x - l2.x)
        lower[-1].x = upper[-2].x
        lower[-1].y = interp_var * l2.y + (1-interp_var) * l1.y
        upper[-1] = lower[-1]
    else:
        interp_var = (u1.x - l2.x) / (u1.x - u2.x)
        upper[-1].x = lower[-2].x
        upper[-1].y = interp_var * u2.y + (1 - interp_var) * u1.y
        lower[-1] = upper[-1]



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

Dfan = 68.3*unit('in').to_base_units()
scale = abs((Dfan/2/(paths[7][0].poly()(0).imag-paths[6][0].poly()(0).imag)).magnitude)
print(f"{scale = :f}")

# interpolate through splines, 10 samples per spline segment
t = np.linspace(0, 1, 10)
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

y0 = lines["centreline"][-1].y.magnitude
Ab = np.pi*((lines["bypass_iface"][0].y.magnitude-y0)**2-(lines["bypass_iface"][-1].y.magnitude-y0)**2)*unit('m^2')
Ac = np.pi*((lines["core_iface"][-1].y.magnitude-y0)**2-(lines["core_iface"][0].y.magnitude-y0)**2)*unit('m^2')
# POWER OF 10
print(f"{Ab = }, {Ac = }")
y0 = lines["centreline"][0].y.magnitude
Ab = np.pi * ((lines["bypass_iface"][0].y.magnitude-y0)**2-(lines["bypass_iface"][-1].y.magnitude-y0)**2)
Ac = np.pi * ((lines["core_iface"][-1].y.magnitude-y0)**2-(lines["core_iface"][0].y.magnitude-y0)**2)
print(f"{Ab = :f}\n{Ac = :f}")
exit()

for i, key in enumerate(lines.keys()):
    plt.plot([point.x.magnitude for point in lines[key]], [point.y.magnitude for point in lines[key]], '-', label=key, color=colors[i])
plt.legend(loc="upper right")
plt.axis("equal")
plt.ylim(1.6,2.4)
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

import numpy as np
from flightcondition import unit
from matplotlib import pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from scipy.stats import skewnorm
from svgpathtools import svg2paths
from ansys.geometry.core import __version__, launch_modeler
from ansys.geometry.core.sketch import Sketch
from ansys.geometry.core.math import Point2D, Vector3D, Point3D
from ansys.geometry.core.shapes.curves.curve import Curve

print(f"PyAnsys Geometry version: {__version__}")

paths, attributes = svg2paths("exhaust_traced.svg")
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

t = np.linspace(0, 1, 10)
colors = [c for c in TABLEAU_COLORS.values()] + ['black']

sketch = Sketch()
print(paths)

for i, key in enumerate(lines.keys()):
    color = colors[i]
    for j, segment in enumerate(paths[i]):
        # [print(j+tt) for tt in t[j>0:]]
        pts = segment.poly()(t[j > 0:])
        [lines[key].append(Point2D(pt)) for pt in zip(pts.real, pts.imag)]

for points in lines.values():
    sketch.segment(points[0], points[1])
    for i in range(len(points) - 2):

        sketch.segment_to_point(points[i + 2])

print([lines["nosecone"][-1].x,lines["nosecone"][-1].y,0*unit('m')])
nose = Point3D((lines["nosecone"][-1].x.magnitude,lines["nosecone"][-1].y.magnitude,0))
sketch.translate_sketch_plane(-Vector3D(nose))
sketch.plot()

# has to be run on windows duh
modeler = launch_modeler()
design =modeler.create_design(f"nacelle")
design.revolve_sketch("Nacelle",sketch,Vector3D((1,0,0)),360,Point3D((0,0,0)))
#
# handles, labels = plt.gca().get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# plt.legend(by_label.values(), by_label.keys())
# plt.axis("equal")
# plt.show()

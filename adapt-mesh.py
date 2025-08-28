import re

import ansys.meshing.prime as prime
import numpy as np
import pyvista as pv
import h5py as h5

XF_COMMENT = 0
XF_HEADER = 1
XF_DIMENSION = 2
XF_NODE = 10
XF_PERIODIC_FACE = 18
XF_CELL = 12
XF_FACE = 13
XF_FACE_TREE = 59
XF_CELL_TREE = 58
XF_FACE_PARENTS = 61
XF_RP_TV = 39
XF_RP_TV_BC = 45
XF_PARTITION = 40

with open("geom/nacelle-2m.psf", "rb") as f:
    matches = re.search(
        rb"\(2319 \((?P<blocksize>\d+) [\D]+\n(?P<origin>(?:[\S]+\s){3})(?P<scale>(?:[\S]+\s){3})(?:(?:[\d ]+\n){4})[\d]+ [\d]+ (?P<maxsize>\d+)[^)]+\)\n\((?P<binary>[\s\S]+)\)\nEnd of Binary Section {3}2319\)",
        f.read())
meshes = []
with h5.File("geom/flow_2.vtkhdf", "r") as f:
    scale = 1000 * (np.max(f['VTKHDF/nacelle_freestream_1/Points'][:], axis=0) - np.min(
        f['VTKHDF/nacelle_freestream_1/Points'][:], axis=0))
    origin = np.min(f['VTKHDF/nacelle_freestream_1/Points'][:], axis=0) * 1000

    for surf in f['VTKHDF/Assembly/Boundary Cells/Domain 0'].keys():
        if 'freestream1' not in surf:
            try:
                meshes.append(pv.make_tri_mesh(f[f'VTKHDF/{surf}/Points'][:] * 1000,
                                               f[f'VTKHDF/{surf}/Connectivity'][:].reshape((-1, 3))))
            except:
                pass
print(len(meshes))

binary_data = matches.group("binary")
blocksize = int(matches.group("blocksize"))
nblocks = len(binary_data) // (4 * blocksize * 4)
maxsize = float(matches.group("maxsize"))
origin = np.float64(matches.group("origin").split(b" "))
scale = np.float64(matches.group("scale").split(b" "))
overlap = 1 + 0.5*maxsize/scale
scale *= overlap # add overlap (guess, its 1.05*scale empirically)
origin *= overlap
# BLOCKS OF 2048 POSITIONS
# read single precision floating point numbers from binary datas
print(f"origin: {origin}\n"
      f"scale: {scale}\n"
      f"block size: {blocksize}\n"
      f"number of blocks: {nblocks}")

gridsize = np.frombuffer(binary_data, dtype=np.float32)
pos = np.frombuffer(binary_data, dtype=np.int32)

gridsize = gridsize.reshape((-1, 4, blocksize))[:, 3, :]
pos = pos.reshape((-1, 4, blocksize))[:, 0:3, :].reshape((-1, blocksize, 3))
nonneg = np.all(pos >= 0, axis=2)  # pos=-1,-1,-1 for unused slots
pos = np.where(pos >= 0, pos, np.nan)
gridsize = np.where(nonneg, gridsize, np.nan)
print(f"unused slots: {nonneg.size - np.sum(np.sum(nonneg))}")

pos = np.float64(pos)*scale/2**29+origin
# pos = np.float64(pos - 2 ** 28) / 2 ** 29 * scale

plotter = pv.Plotter()
plotter.add_points(pos.reshape((-1, 3)),
                   scalars=gridsize.reshape((-1)), show_scalar_bar=True,
                   cmap='jet_r', log_scale=True, opacity=0.5
                   )
for mesh in meshes:
    plotter.add_mesh(mesh, color='white', opacity=0.99)
# plotter.show_bounds(bounds=(-6000,1000,-2000,2000,-2000,2000))
# plotter.show_bounds(bounds=(-10,10,-10,10,-10,10))
plotter.zoom_camera(100)
plotter.show_axes()
# plotter.show_grid()

plotter.show()

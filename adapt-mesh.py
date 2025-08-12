import re

import ansys.meshing.prime as prime
import numpy as np
import pyvista as pv


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

with open("geom/nacelle.psf", "rb") as f:
    matches = re.search(rb"\(2319 \((?P<blocksize>\d+) [\D]+\n(?P<origin>(?:[\S]+\s){3})(?P<scale>(?:[\S]+\s){3})[^)]+\)\n\((?P<binary>[\s\S]+)\)\nEnd of Binary Section {3}2319\)",f.read())


binary_data = matches.group("binary")
blocksize = int(matches.group("blocksize"))
nblocks = len(binary_data) // (4 * blocksize * 4)
origin = np.float64(matches.group("origin").split(b" "))
scale = np.float64(matches.group("scale").split(b" "))
# BLOCKS OF 2048 POSITIONS
# read single precision floating point numbers from binary datas
print(f"origin: {origin}\n"
      f"scale: {scale}\n"
      f"block size: {blocksize}\n"
      f"number of blocks: {nblocks}")

gridsize = np.frombuffer(binary_data, dtype=np.float32)
pos = np.frombuffer(binary_data, dtype=np.int32)

gridsize = gridsize.reshape((-1, 4, blocksize))[:,-1,:]
pos = pos.reshape((-1, 4, blocksize))[:, :3, :].reshape((-1,blocksize,3))
nonneg = np.all(pos>=0, axis=2)
pos = np.where(pos>=0, pos, np.nan)
gridsize = np.where(nonneg, gridsize, np.nan)
print(f"unused slots: {nonneg.size-np.sum(np.sum(nonneg))}")
pos = np.float64(pos)*scale/2**29+origin

plotter = pv.Plotter()
plotter.add_points(pos.reshape((-1,3)),
        scalars=gridsize.reshape((-1)),show_scalar_bar=True,
        cmap='jet_r',log_scale=False,opacity=0.8
        )
plotter.show_bounds(bounds=(-6000,1000,-2000,2000,-2000,2000))
plotter.show_axes()
# plotter.show_grid()

plotter.show()






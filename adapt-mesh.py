import numpy as np
import pyvista as pv
import h5py as h5
from pyvista import UnstructuredGrid
from vtkmodules.vtkIOHDF import vtkHDFReader
from vtkmodules.vtkCommonDataModel import (
    VTK_CUBIC_LINE,
    VTK_HEXAHEDRON,
    VTK_LINE,
    VTK_PYRAMID,
    VTK_QUAD,
    VTK_QUADRATIC_EDGE,
    VTK_QUADRATIC_HEXAHEDRON,
    VTK_QUADRATIC_PYRAMID,
    VTK_QUADRATIC_QUAD,
    VTK_QUADRATIC_TETRA,
    VTK_QUADRATIC_TRIANGLE,
    VTK_QUADRATIC_WEDGE,
    VTK_TETRA,
    VTK_TRIANGLE,
    VTK_WEDGE,
    vtkCellTypes, vtkCompositeDataSet, vtkUnstructuredGrid
)
from vtkmodules.numpy_interface import dataset_adapter as dsa

from ansys_formats import PrimeSizeField

plotter = pv.Plotter()

with h5.File("geom/flow_0.vtkhdf", 'r') as f:
    names = list(f["VTKHDF"].keys())
    meshes=[]
    for name in names:
        if 'wall' in name and "Nozzle" not in name:

            points = f[f"VTKHDF/{name}/Points"][:] * 1000
            connectivity = f[f"VTKHDF/{name}/Connectivity"][:].reshape((-1, 3))
            meshes.append(pv.make_tri_mesh(points, connectivity))
    pos = f[f"VTKHDF/Domain 0/Points"][::30] * 1000
    wdist = f[f"VTKHDF/Domain 0/PointData/wdist"][::30] * 1000

    wdist = np.clip(2 * wdist, 3, 80000)

    print(f[f"VTKHDF/Domain 0/PointData"].keys())

for mesh in meshes:
    plotter.add_mesh(mesh, color='white', opacity=1.0)


psf = PrimeSizeField("geom/test.psf")
psf.write("geom/test-orig.psf")

plotter.add_points(psf.pos,
                   scalars=psf.gridsize, show_scalar_bar=True,
                   cmap='jet_r', log_scale=True, opacity=0.1
                   )

psf.pos = pos
psf.gridsize = wdist
psf.write("geom/test-adapt.psf")



# plotter.show_bounds(bounds=(-6000,1000,-2000,2000,-2000,2000))
# plotter.show_bounds(bounds=(-10,10,-10,10,-10,10))
plotter.zoom_camera(100)
plotter.show_axes()
# plotter.show_grid()

plotter.show()

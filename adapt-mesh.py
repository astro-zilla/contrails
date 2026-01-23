import gc
from os import PathLike
from pathlib import Path

import numpy as np
import pyvista as pv
import h5py as h5
from pyvista import CameraPosition, MultiBlock, PolyData, UnstructuredGrid
from vtkmodules.util.data_model import MultiBlockDataSet
from vtkmodules.vtkFiltersGeneral import vtkGradientFilter
from vtkmodules.vtkIOHDF import vtkHDFReader, vtkHDFWriter
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


def calculate_Qc1(fname: Path):
    tmp = Path("tmp.vtkhdf")
    fname.replace(tmp)
    grid = pv.read(tmp)

    domain0: UnstructuredGrid = grid.get_block(0).get_block(0)

    if 'vol' not in domain0.point_data.keys():
        grid = grid.compute_cell_sizes(length=False, area=False, volume=True, progress_bar=True, )

    if 'd_sigma0' not in domain0.point_data.keys():
        domain0.point_data.update({'sigma0': domain0['scalar0'] / domain0['ro']})
        domain0.point_data.update({'d_sigma0':
                                       domain0.compute_derivative(scalars="sigma0", gradient="d_sigma0", preference='point',
                                                                  progress_bar=True, faster=True)['d_sigma0']})

    if 'd2_sigma0' not in domain0.point_data.keys():
        domain0.point_data.update({'d2_sigma0':
                                       domain0.compute_derivative(scalars="d_sigma0", gradient="d2_sigma0",
                                                                  preference='point',
                                                                  progress_bar=True, faster=True)['d2_sigma0']})
    if 'Delta' not in domain0.point_data.keys():
        domain0.point_data.update({'Delta': (domain0['vol'] * 6. * np.sqrt(2)) ** (1 / 3)})

    if "Q_c1" not in domain0.point_data.keys():
        domain0.point_data.update({'Q_c1':
                                       domain0["Delta"] ** 2 * np.maximum.reduce(
                                           np.abs(domain0['d2_sigma0'][:, [0, 4, 8]]),
                                           axis=1)})


    grid.save(fname)
    del grid, domain0

def calculate_Qc2(fname: Path):
    grid = pv.read(fname)
    domain0: UnstructuredGrid = grid.get_block(0).get_block(0)

fname = Path("geom/slice_unsteady.vtkhdf")
calculate_Qc2(fname)
exit(0)

fname = Path("geom/nacelle-2m.vtkhdf")
calculate_Qc1(fname)

grid: MultiBlock = pv.read(fname)
grid.scale((1000,1000,1000),inplace=True)
bcs = grid.get_block(1).get_block(0)
bcs.pop("nacelle_freestream_1")
plotter.add_mesh(bcs, color='white', opacity=1.0)

# psf = PrimeSizeField("geom/nacelle-2m-surf.psf")
# # psf.delete_farfield(fname)
# print(psf.nblocks)
# psf.append(fname, 6)
# print(psf.nblocks)
# psf.write("geom/nacelle-2m-adapt.psf")
psf=PrimeSizeField("geom/nacelle-2m-adapt.psf")
plotter.add_points(psf.pos,
                   scalars=psf.gridsize, show_scalar_bar=True,
                   cmap='jet_r', log_scale=True, opacity=0.1
                   )

# plotter.show_bounds(bounds=(-6000,1000,-2000,2000,-2000,2000))
# plotter.show_bounds(bounds=(-10,10,-10,10,-10,10))
plotter.zoom_camera(100)
plotter.camera_position = CameraPosition((800000, 0, 0), (0, 0, 0), (0, 1, 0))
plotter.show_axes()
# plotter.show_grid()

plotter.show()

import re

import numpy as np
import h5py as h5
import pyvista as pv
from engineering_notation import EngNumber
from pyvista import MultiBlock
from scipy.interpolate import NearestNDInterpolator, RBFInterpolator
from vtkmodules.vtkCommonDataModel import vtkMultiBlockDataSet, vtkStaticPointLocator, vtkUnstructuredGrid
from vtkmodules.vtkFiltersPoints import vtkGaussianKernel, vtkPointInterpolator

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

class PrimeSizeField:
    @property
    def nblocks(self):
        return int(np.ceil((self.pos.size + self.gridsize.size ) / (4*self.blocksize)))
    def _parse(self):
        with open(self.filename, 'rb') as f:
            data = f.read()
        if not data:
            raise ValueError("File is empty")

        matches = re.search(
            rb"\(2319 \((?P<blocksize>\d+) [\D]+\n(?P<corner>(?:[\S]+\s){3})(?P<extent>(?:[\S]+\s){3})(?:(?:[\d ]+\n){4})[\d]+ (?P<minsize>\d+) (?P<maxsize>\d+)[^)]+\)\n\((?P<binary>[\s\S]+)\)\nEnd of Binary Section {3}2319\)",
            data)

        self.blocksize = int(matches.group("blocksize"))
        corner = np.float64(matches.group("corner").split(b" "))
        extent = np.float64(matches.group("extent").split(b" "))
        self.minsize = float(matches.group("minsize"))
        self.maxsize = float(matches.group("maxsize"))
        binary_data = matches.group("binary")

        nblocks = len(binary_data) // (4 * self.blocksize * 4)
        self.scale_mult = 1 + 0.5 * self.maxsize / extent
        self.scale  = extent * self.scale_mult  # add overlap (guess, its 1.05*scale empirically)
        self.origin_mult = 1 - 0.25 * self.maxsize / corner
        self.origin = corner * self.origin_mult

        print(f"origin: {self.origin}\n"
              f"scale: {self.scale}\n"
              f"block size: {self.blocksize}\n"
              f"number of blocks: {nblocks}")

        floats = np.frombuffer(binary_data, dtype=np.float32).astype(float)
        ints = np.frombuffer(binary_data, dtype=np.int32).astype(float)

        gridsize = floats.reshape((-1, 4, self.blocksize))[:, 3, :]
        pos = ints.reshape((-1, 4, self.blocksize))[:, 0:3, :].reshape((-1,3))

        nonneg = np.all(pos >= 0, axis=1)
        self.gridsize = gridsize.reshape((-1,))[nonneg]
        self.pos = (pos[nonneg]) / 2 ** 29  * self.scale + self.origin
        # self.pos = np.float64(pos - 2 ** 28) / 2 ** 29 * scale


        print(f"unused slots: {nonneg.size - np.sum(np.sum(nonneg))}")

    def write(self, filename):
        with open(filename, 'wb') as f:


            pad_width = ((0,self.nblocks*self.blocksize-self.pos.shape[0]), (0,0))

            pos = np.pad(self.pos, pad_width=pad_width, mode="constant", constant_values=np.nan)
            pos = ((pos.reshape((-1,3)) - self.origin)  * 2 ** 29 / self.scale).reshape((-1, 3, self.blocksize))
            pos = np.where(np.isnan(pos), -1, pos).astype(np.int32)

            gridsize = np.pad(self.gridsize, pad_width=pad_width[0], mode="constant", constant_values=-1)
            gridsize = gridsize.reshape((-1,self.blocksize)).astype(np.float32)

            origin = self.origin / self.origin_mult
            scale = self.scale / self.scale_mult

            f.write(f"""(1 "ANSYS(R) TGLib(TM) 3D, revision 24.2.0")
(0 "Size Field File")
(0 "Machine Config:")
(4 (60 0 0 1 2 4 4 4 8 8 8 4))
(704 ( 1   1
  1   1          0))
(2319 ({self.blocksize} glob-sf-bgrid
{origin[0]:g} {origin[1]:g} {origin[2]:g}
{scale[0]:g} {scale[1]:g} {scale[2]:g}
1 0 0
0 1 0
0 0 1
10 1
2 3 {self.maxsize:g} 1.2
1
0 0 0 0
1 0 0 0
0 1 0 0
0 0 1 0
0 0 0 1
1 0 0 0
0 1 0 0
0 0 1 0
0 0 0 1
0
0 0 0
0 0 0
0 0 0
0)
(""".encode('utf-8')
                    )
            for block in range(self.nblocks):
                f.write(pos[block,:,:].tobytes())
                f.write(gridsize[block, :].tobytes())
            f.write(b")\nEnd of Binary Section   2319)\n")


    def adapt(self, fname_vtkhdf):
        with h5.File(fname_vtkhdf, 'r') as f:
            step = 1

            pos = f[f"VTKHDF/Assembly/Domains/Domain 0/Points"][::step] * 1000
            data = f[f"VTKHDF/Assembly/Domains/Domain 0/PointData"]

            pos, indices = np.unique(pos, axis=0, return_index=True)

            wdist = data["wdist"][::step][indices] * 1000
            Q_c1 = data["Q_c1"][::step][indices]

            wdist = np.clip(2 * wdist, 6, 80000)

            grid_adapter = np.clip(np.sqrt(0.001 / Q_c1), 0, 1)  # *Delta

            grid_adapter[wdist < 48] = 1  # Delta[wdist<48]
            mask = wdist > 48

            nblocks = self.nblocks

            print(f"starting interpolation on pos:{pos[mask].shape} and grid_adapter:{grid_adapter[mask].shape}")
            adapt_interpolator = NearestNDInterpolator(pos[mask], grid_adapter[mask])
            self.gridsize *= adapt_interpolator(self.pos)
            self.gridsize=np.clip(self.gridsize, self.minsize, self.maxsize)

            print(f"Adapted size field using {EngNumber(pos[mask].shape[0])} points, added {self.nblocks - nblocks} blocks")

    def append(self, fname_vtkhdf, downsample=1):
        with h5.File(fname_vtkhdf, 'r') as f:

            pos = f[f"VTKHDF/Assembly/Domains/Domain 0/Points"][::downsample] * 1000
            data = f[f"VTKHDF/Assembly/Domains/Domain 0/PointData"]

            pos, indices = np.unique(pos, axis=0, return_index=True)

            wdist = data["wdist"][::downsample][indices] * 1000
            Q_c1 = data["Q_c1"][::downsample][indices]
            vol = data["vol"][::downsample][indices] * 1e9

            wdist = np.clip(2 * wdist, 6, 80000)

            grid_adapter = np.clip(np.sqrt(0.001 / Q_c1), 0, 1)  # *Delta
            gridsize = np.clip(grid_adapter * (6*np.sqrt(2)*vol)**(1/3), self.minsize, self.maxsize)

            mask = wdist > 100

            self.pos = np.concatenate((self.pos, pos[mask]))
            self.gridsize = np.concatenate((self.gridsize, gridsize[mask]))


    def delete_farfield(self, fname_vtkhdf):
        with h5.File(fname_vtkhdf, 'r') as f:
            pos = f[f"VTKHDF/Assembly/Domains/Domain 0/Points"][:] * 1000
            data = f[f"VTKHDF/Assembly/Domains/Domain 0/PointData"]
            wdist = data["wdist"][:] * 1000

            wdist_interpolator = NearestNDInterpolator(pos,wdist)
            mask = wdist_interpolator(self.pos) < 100
            self.pos = self.pos[mask]
            self.gridsize = self.gridsize[mask]



    def __init__(self,filename=None):
        self.filename = filename

        self.blocksize = None
        self.origin = None
        self.scale = None
        self.minsize = None
        self.maxsize = None

        self.pos = None
        self.gridsize = None

        if self.filename:
            self._parse()

import re

import numpy as np


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
            rb"\(2319 \((?P<blocksize>\d+) [\D]+\n(?P<corner>(?:[\S]+\s){3})(?P<extent>(?:[\S]+\s){3})(?:(?:[\d ]+\n){4})[\d]+ [\d]+ (?P<maxsize>\d+)[^)]+\)\n\((?P<binary>[\s\S]+)\)\nEnd of Binary Section {3}2319\)",
            data)

        self.blocksize = int(matches.group("blocksize"))
        corner = np.float64(matches.group("corner").split(b" "))
        extent = np.float64(matches.group("extent").split(b" "))
        self.maxsize = float(matches.group("maxsize"))
        binary_data = matches.group("binary")

        nblocks = len(binary_data) // (4 * self.blocksize * 4)
        self.overlap = 1 + 0.5 * self.maxsize / extent
        self.scale  = extent * self.overlap  # add overlap (guess, its 1.05*scale empirically)
        self.origin = corner * self.overlap

        print(corner,extent)

        print(f"origin: {self.origin}\n"
              f"scale: {self.scale}\n"
              f"block size: {self.blocksize}\n"
              f"number of blocks: {nblocks}")

        floats = np.frombuffer(binary_data, dtype=np.float32)
        ints = np.frombuffer(binary_data, dtype=np.int32)

        gridsize = floats.reshape((-1, 4, self.blocksize))[:, 3, :]
        pos = ints.reshape((-1, 4, self.blocksize))[:, 0:3, :].reshape((-1,3))

        nonneg = np.all(pos >= 0, axis=1)
        self.gridsize = gridsize.reshape((-1,))[nonneg]
        self.pos = (pos.astype(float)[nonneg] * self.scale / 2 ** 29 + self.origin)
        # self.pos = np.float64(pos - 2 ** 28) / 2 ** 29 * scale


        print(f"unused slots: {nonneg.size - np.sum(np.sum(nonneg))}")

    def write(self, filename):
        with open(filename, 'wb') as f:
            origin = self.origin / self.overlap
            scale = self.scale / self.overlap

            pad_width = ((0,self.nblocks*self.blocksize-self.pos.shape[0]), (0,0))

            pos = np.pad(self.pos, pad_width=pad_width, mode="constant", constant_values=np.nan)
            pos = ((pos.reshape((-1,3)) - self.origin) / self.scale * 2 ** 29)
            pos = np.where(np.isnan(pos), -1, pos).astype(np.int32)


            gridsize = np.pad(self.gridsize, pad_width=pad_width[0], mode="constant", constant_values=-1)
            gridsize = gridsize.reshape((-1,self.blocksize)).astype(np.float32)

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
                f.write(pos.reshape((-1, 3, self.blocksize))[block,:,:].tobytes())
                f.write(gridsize[block, :].tobytes())
            f.write(b")\nEnd of Binary Section   2319)\n")






    def __init__(self,filename=None):
        self.filename = filename

        self.blocksize = None
        self.origin = None
        self.scale = None
        self.maxsize = None

        self.pos = None
        self.gridsize = None

        if self.filename:
            self._parse()

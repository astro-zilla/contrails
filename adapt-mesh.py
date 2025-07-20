import re
from io import BufferedReader
from operator import invert

import ansys.meshing.prime as prime
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.pyplot import viridis
from mpl_toolkits.mplot3d.art3d import Line3DCollection

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


def tokenize(data: bytes):
    """
    Tokenizes a line of bytes into a list of tokens.
    """
    yield from re.findall(rb'\(|\)|"[^"]*"|[^\s()]+', data)


with open("geom/test.psf", "rb") as f:
    binary_data = re.search(rb"\(2319 \([^)]+\)\n\(([\s\S]+)\)\nEnd of Binary Section {3}2319\)",f.read()).group(1)
    # BLOCKS OF 2048 POSITIONS
    # read single precision floating point numbers from binary datas
    print(len(binary_data))  # print first 100 bytes for inspection

    gridsize = np.frombuffer(binary_data, dtype=np.float32)
    pos = np.frombuffer(binary_data, dtype=np.int32)

    gridsize = gridsize.reshape((-1, 4, 2048))[:,-1,:]
    pos = pos.reshape((-1, 4, 2048))[:, :3, :].reshape((-1,2048,3))-2**28
    pos = np.float64(pos)*800000/2**29




    fig=plt.figure()

    ax1=fig.add_subplot(1,1,1,projection='3d')
    # ax2=fig.add_subplot(1,1,1)
    for i,int_subset in enumerate(pos):
        if 1:
            offsets = pos[i]
            size = gridsize[i]
            ax1.scatter(offsets[:,0], offsets[:,1], offsets[:,2],marker='.',linewidth=0,c=-size,cmap='jet')

    plt.axis('equal')
    plt.show()





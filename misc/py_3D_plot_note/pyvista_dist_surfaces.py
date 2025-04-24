#https://docs.pyvista.org/examples/01-filter/distance-between-surfaces.html

import numpy as np

import pyvista as pv


def hill(seed):
    """A helper to make a random surface."""
    mesh = pv.ParametricRandomHills(randomseed=seed, u_res=50, v_res=50, hillamplitude=0.5)
    mesh.rotate_y(-10, inplace=True)  # give the surfaces some tilt

    return mesh


h0 = hill(1).elevation()
h1 = hill(10)
# Shift one surface
h1.points[:, -1] += 5
h1 = h1.elevation()


p = pv.Plotter()
p.add_mesh(h0, smooth_shading=True)
p.add_mesh(h1, smooth_shading=True)
p.show_grid()
p.show()

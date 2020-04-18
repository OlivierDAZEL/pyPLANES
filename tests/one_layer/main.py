import sys
sys.path.insert(0,"../../pyPLANES")

import numpy as np

from pyPLANES.model.model import Model
from pyPLANES.classes.model_classes import ModelParameter
from pyPLANES.utils.utils_PW import Solver_PW
from pyPLANES.gmsh.write_geo_file import Gmsh as Gmsh

from pymls import from_yaml, Solver, Layer, backing



p = ModelParameter()
p.frequency = (10., 50., 1)
p.name_project = "one_layer"
p.theta_d = 30
L = 0.1
d = 0.1
lcar = 0.02

p.order = 5
p.plot = [False, False, False, False, False, True]


G = Gmsh(p.name_project)

p_0 = G.new_point(0, 0, lcar/2)
p_1 = G.new_point(L, 0,lcar*2)
p_2 = G.new_point(L, d, lcar*2)
p_3 = G.new_point(0, d, lcar/2)
l_0 = G.new_line(p_0, p_1)
l_1 = G.new_line(p_1, p_2)
l_2 = G.new_line(p_2, p_3)
l_3 = G.new_line(p_3, p_0)
ll_0 = G.new_line_loop([l_0, l_1, l_2, l_3])

matrice = G.new_surface([ll_0.tag])

G.new_physical(l_2, "condition=Transmission")
G.new_physical([l_1, l_3], "condition=Periodicity")
G.new_physical(l_0, "condition=Incident_PW")
G.new_physical(matrice, "mat=foam2")
G.new_physical([l_0, l_1, l_3, l_2], "model=FEM1D")
G.new_physical([matrice], "model=FEM2D")
G.new_periodicity(l_1, l_3, (L, 0, 0))

option = "-2 "
G.run_gmsh(option)



foam2 = from_yaml("foam2.yaml")
p.solver_pymls = Solver()
p.solver_pymls.layers = [
    Layer(foam2, d),
]
p.solver_pymls.backing = backing.transmission
p.S_PW = Solver_PW(p.solver_pymls, p)


p.gmsh_file = p.name_project


model = Model(p)
model.resolution(p)




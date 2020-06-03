import sys
import platform

sys.path.insert(0, "../..")

import numpy as np
import matplotlib.pyplot as plt

from pymls import from_yaml, Solver, Layer, backing
from mediapack import Air

from pyPLANES.model.model import Model
from pyPLANES.classes.model_classes import ModelParameter
from pyPLANES.utils.utils_PW import Solver_PW
from pyPLANES.gmsh.write_geo_file import Gmsh as Gmsh

from pyPLANES.utils.utils_io import print_entities

param = ModelParameter()
param.theta_d = 60.0000001
param.frequencies = (30., 5010., 1)
param.name_project = "one_layer"

pem = from_yaml('foam2.yaml')
Wwood = from_yaml('Wwood.yaml')

L = 0.1
d_1 = 0.1
d_2 = 0.1
d_3 = 0.1
lcar = 0.1

param.order = 3
param.plot = [True, True, True, False, False, False]
# param.plot = [False]*6

G = Gmsh(param.name_project)

# p_7 <---l_8 --- p_6
#  |               ^
# l_9             l_7
#  v               |
# p_4 <---l_5 --- p_5
#  |               ^
# l_6               l_4
#  v               |
# p_3 <---l_2---- p_2
#  |               ^
# l_3             l_1
#  v               |
# p_O ---l_0 ---> p_1

p_0 = G.new_point(0, 0, lcar)
p_1 = G.new_point(L, 0, lcar)
p_2 = G.new_point(L, d_1, lcar)
p_3 = G.new_point(0, d_1, lcar)
p_4 = G.new_point(0, d_1+d_2, lcar)
p_5 = G.new_point(L, d_1+d_2, lcar)
p_6 = G.new_point(L, d_1+d_2+d_3, lcar)
p_7 = G.new_point(0, d_1+d_2+d_3, lcar)

l_0 = G.new_line(p_0, p_1)
l_1 = G.new_line(p_1, p_2)
l_2 = G.new_line(p_2, p_3)
l_3 = G.new_line(p_3, p_0)
l_4 = G.new_line(p_2, p_5)
l_5 = G.new_line(p_5, p_4)
l_6 = G.new_line(p_4, p_3)
l_7 = G.new_line(p_5, p_6)
l_8 = G.new_line(p_6, p_7)
l_9 = G.new_line(p_7, p_4)

# p_7 <---l_8 --- p_6
#  |               ^
# l_9             l_7
#  v               |
# p_4 <---l_5 --- p_5
#  |               ^
# l_6               l_4
#  v               |
# p_3 <---l_2---- p_2
#  |               ^
# l_3             l_1
#  v               |
# p_O ---l_0 ---> p_1

ll_0 = G.new_line_loop([l_0, l_1, l_2, l_3])
ll_1 = G.new_line_loop([l_4, l_5, l_6, l_2.inverted()])
ll_2 = G.new_line_loop([l_7, l_8, l_9, l_5.inverted()])



layer_1 = G.new_surface([ll_0.tag])
layer_2 = G.new_surface([ll_1.tag])
layer_3 = G.new_surface([ll_2.tag])


# G.new_physical(l_8, "condition=Transmission")
G.new_physical(l_8, "condition=Rigid Wall")
G.new_physical([l_1, l_4, l_7, l_3, l_6, l_9], "condition=Periodicity")
G.new_physical(l_0, "condition=Incident_PW")
# G.new_physical(matrice, "mat=Air")
G.new_physical(layer_1, "mat=foam2")
G.new_physical(layer_2, "mat=Wwood")
G.new_physical(layer_3, "mat=Air")
G.new_physical([l_5], "condition=Fluid_Structure")
# G.new_physical(matrice, "mat=Wwood")
G.new_physical([l_0, l_1, l_3, l_4, l_6, l_7, l_9, l_8, l_5], "model=FEM1D")
G.new_physical([layer_1, layer_2, layer_3], "model=FEM2D")
G.new_periodicity(l_1, l_3, (L, 0, 0))
G.new_periodicity(l_4, l_6, (L, 0, 0))
G.new_periodicity(l_7, l_9, (L, 0, 0))
option = "-2 -v 0 "
G.run_gmsh(option)

# param.incident_ml = [Layer(pem, d)] ; param.shift_pw = -param.incident_ml[0].thickness
# param.transmission_ml = [Layer(pem, d)]

model = Model(param)
result_pyPLANES = model.resolution(param)

param.solver_pymls = Solver()
param.solver_pymls.layers = [
    Layer(pem, d_1),
    Layer(Wwood, d_2),
    Layer(Air, d_3),
    # Layer(Air,d)
    # Layer(pem, d),
    # Layer(Wwood,d/10),
]

param.solver_pymls.backing = backing.rigid
# param.solver_pymls.backing = backing.transmission


param.S_PW = Solver_PW(param.solver_pymls, param)
result_pyPLANESPW = param.S_PW.resolution(param.theta_d)

# result_pymls =  param.solver_pymls.solve(param.frequencies[0], param.theta_d)

# print("result_pymls     = {}".format(result_pymls["R"][0]))
print("result_pyPLANESPW= {}".format(result_pyPLANESPW["R"]))
print("result_pyPLANES  = {}".format(result_pyPLANES["R"]))
# print("result_pymls  T  = {}".format(result_pymls["T"][0]))

if any(param.plot):
    plt.show()
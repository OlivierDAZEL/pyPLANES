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
param.theta_d = 0.0000001
param.frequencies = (1., 5000., 201)
param.name_project = "two_layers"

rubber = from_yaml('rubber.yaml')
pem = from_yaml('pem_benchmark_1.yaml')
Wwood = from_yaml('Wwood.yaml')
Air_Elastic = from_yaml('Air_Elastic.yaml')


L = 0.2
d_1 = 0.0002
d_2 = 0.1
lcar = 0.1
param.verbose = False


param.order = 2
# param.plot = [True, True, True, False, False, False]
param.plot = [False]*6

G = Gmsh(param.name_project)

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

l_0 = G.new_line(p_0, p_1)
l_1 = G.new_line(p_1, p_2)
l_2 = G.new_line(p_2, p_3)
l_3 = G.new_line(p_3, p_0)
l_4 = G.new_line(p_2, p_5)
l_5 = G.new_line(p_5, p_4)
l_6 = G.new_line(p_4, p_3)

ll_0 = G.new_line_loop([l_0, l_1, l_2, l_3])
ll_1 = G.new_line_loop([l_4, l_5, l_6, l_2.inverted()])

layer_1 = G.new_surface([ll_0.tag])
layer_2 = G.new_surface([ll_1.tag])

G.new_physical(l_5, "condition=Rigid Wall")
G.new_physical([l_1, l_4, l_3, l_6], "condition=Periodicity")
G.new_physical(l_0, "condition=Incident_PW")
# G.new_physical(matrice, "mat=Air")
G.new_physical(layer_1, "mat=rubber")
G.new_physical(layer_2, "mat=pem_benchmark_1")

# G.new_physical([l_2], "condition=Fluid_Structure")
# G.new_physical([l_0, l_1, l_3, l_4, l_6, l_5, l_2], "model=FEM1D")

G.new_physical([l_0, l_1, l_3, l_4, l_6, l_5], "model=FEM1D")

G.new_physical([layer_1, layer_2], "model=FEM2D")
G.new_periodicity(l_1, l_3, (L, 0, 0))
G.new_periodicity(l_4, l_6, (L, 0, 0))
option = "-2 -v 0 "
G.run_gmsh(option)

# param.incident_ml = [Layer(pem, d)] ; param.shift_pw = -param.incident_ml[0].thickness
# param.transmission_ml = [Layer(pem, d)]

model = Model(param)
result_pyPLANES = model.resolution(param)

param.solver_pymls = Solver()
param.solver_pymls.layers = [
    Layer(rubber, d_1),
    Layer(pem, d_2),
]

param.solver_pymls.backing = backing.rigid
# param.solver_pymls.backing = backing.transmission

param.S_PW = Solver_PW(param.solver_pymls, param)
result_pyPLANESPW = param.S_PW.resolution(param.theta_d)

result_pymls =  param.solver_pymls.solve(param.frequencies[0], param.theta_d)

print("result_pymls     = {}".format(result_pymls["R"][0]))
print("result_pyPLANESPW= {}".format(result_pyPLANESPW["R"]))
print("result_pyPLANES  = {}".format(result_pyPLANES["R"]))


if any(param.plot):
    plt.show()

# print(Air.K)
# print(Air.rho)

# k = 2*np.pi*param.frequencies[0]/Air.c
# Z_s = -1j*Air.Z/np.tan(k*(d_1+d_2))
# R = (Z_s-Air.Z)/(Z_s+Air.Z)
# print(R)


FEM=np.loadtxt("two_layers_out.txt")
PW=np.loadtxt("two_layers_PW_out.txt")
plt.plot(PW[:,0],PW[:,1], "b", label="PW")
plt.plot(FEM[:,0],FEM[:,1], "r.", label="FEM")
plt.show()
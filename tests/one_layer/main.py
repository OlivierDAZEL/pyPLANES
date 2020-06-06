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

name_server = platform.node()

param = ModelParameter()
param.theta_d = 60.000000
param.frequencies = (20., 5010., 201)
param.name_project = "one_layer"

L = 0.01
d = 0.02
lcar = 0.01

param.order = 5
# param.plot = [True, True, True, False, False, False]
param.plot = [False]*6

if name_server in ["oliviers-macbook-pro.home", "Oliviers-MacBook-Pro.local"]:
    param.verbose = True

param.verbose = False

G = Gmsh(param.name_project)

p_0 = G.new_point(0, 0, lcar)
p_1 = G.new_point(L, 0,lcar)
p_2 = G.new_point(L, d, lcar)
p_3 = G.new_point(0, d, lcar)
l_0 = G.new_line(p_0, p_1)
l_1 = G.new_line(p_1, p_2)
l_2 = G.new_line(p_2, p_3)
l_3 = G.new_line(p_3, p_0)
ll_0 = G.new_line_loop([l_0, l_1, l_2, l_3])
matrice = G.new_surface([ll_0.tag])
# G.new_physical(l_2, "condition=Transmission")
G.new_physical(l_2, "condition=Rigid Wall")
G.new_physical([l_1, l_3], "condition=Periodicity")
G.new_physical(l_0, "condition=Incident_PW")
G.new_physical(matrice, "mat=pem_benchmark_1")
G.new_physical([l_0, l_1, l_3, l_2], "model=FEM1D")
G.new_physical([matrice], "model=FEM2D")
G.new_periodicity(l_1, l_3, (L, 0, 0))
option = "-2 -v 0 "
G.run_gmsh(option)

rubber = from_yaml('rubber.yaml')
pem = from_yaml('pem_benchmark_1.yaml')
Wwood = from_yaml('Wwood.yaml')
Air_Elastic = from_yaml('Air_Elastic.yaml')


param.incident_ml = [Layer(rubber, 0.0002)] ; param.shift_pw = -param.incident_ml[0].thickness

# model = Model(param)
# result_pyPLANES = model.resolution(param)


param.solver_pymls = Solver()
param.solver_pymls.layers = [
    Layer(rubber, 0.0002),
    Layer(pem, d),
]
param.solver_pymls.backing = backing.rigid

param.S_PW = Solver_PW(param.solver_pymls, param)
result_pyPLANESPW = param.S_PW.resolution(param.theta_d)

# print("result_pyPLANESPW= {}".format(result_pyPLANESPW["R"]))
# print("result_pyPLANES  = {}".format(result_pyPLANES["R"]))
# print("result_pymls  R  = {}".format(result_pymls["R"][0]))

if any(param.plot):
    plt.show()


# print(Air.K)
# print(Air.rho)
# k = 2*np.pi*param.frequencies[0]/Air.c
# Z_s = -1j*Air.Z/np.tan(k*(d))
# R = (Z_s-Air.Z)/(Z_s+Air.Z)
# print(R)


FEM = np.loadtxt("one_layer_out.txt")
PW = np.loadtxt("one_layer_PW_out.txt")
plt.plot(PW[:,0],PW[:,1], "b", label="PW")
plt.plot(FEM[:,0],FEM[:,1], "r.", label="FEM")
plt.show()
import sys
import platform

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "../..")

from pyPLANES.model.model import Model
from pyPLANES.classes.model_classes import ModelParameter
from pyPLANES.utils.utils_PW import Solver_PW
from pyPLANES.gmsh.write_geo_file import Gmsh as Gmsh

from pymls import from_yaml, Solver, Layer, backing
from mediapack import Air

name_server = platform.node()

param = ModelParameter()
theta_d = 0.
param.frequencies = (3400., 5010., 1)
param.name_project = "one_layer"

param.theta_d = theta_d
L = 0.02
d = 0.02
# a = 0.008
lcar = 0.0005

param.order = 5
param.plot = [True, True, True, False, False, False]
param.plot = [False]*6
# print(name_server)
# if name_server in ["oliviers-macbook-pro.home","Oliviers-MacBook-Pro.local"]:
#     param.verbose = True

p = param
G = Gmsh(p.name_project)

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

G.new_physical(l_2, "condition=Rigid Wall")
# G.new_physical(l_2, "condition=Transmission")
G.new_physical([l_1, l_3], "condition=Periodicity")
G.new_physical(l_0, "condition=Incident_PW")
G.new_physical(matrice, "mat=Air")
G.new_physical([l_0, l_1, l_3, l_2], "model=FEM1D")
G.new_physical([matrice], "model=FEM2D")
G.new_periodicity(l_1, l_3, (L, 0, 0))

option = "-2 -v 0 "
G.run_gmsh(option)

model = Model(param)

model.resolution(param)

pem = from_yaml('pem_benchmark_1.yaml')
param.solver_pymls = Solver()
param.solver_pymls.layers = [
    Layer(Air, L),
]
param.solver_pymls.backing = backing.rigid
param.S_PW = Solver_PW(param.solver_pymls, param)
param.S_PW.resolution(param.theta_d)

k = 2*np.pi*param.frequencies[0]/Air.c
k3 = k * np.cos(theta_d*np.pi/180)

Z = -1j*Air.Z*(k/k3)/np.tan(k3*L)
R = (Z*k3-Air.Z*k)/(Z*k3+Air.Z*k)
# print("R={}".format(R))



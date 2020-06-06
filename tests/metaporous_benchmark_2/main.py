import sys
import numpy as np
import matplotlib.pyplot as plt

from pymls import from_yaml, Solver, Layer, backing


sys.path.insert(0, "../../")
from pyPLANES.model.model import Model
from pyPLANES.classes.model_classes import ModelParameter
from pyPLANES.utils.utils_PW import Solver_PW
from pyPLANES.gmsh.write_geo_file import Gmsh as Gmsh


param = ModelParameter()
theta_d = 00.
param.frequencies = (10., 15., 2)
param.name_project = "metaporous_benchmark_2"

param.theta_d = theta_d
L = 0.02
d = 0.02
a = 0.008
r_i = 0.0078
lcar = 0.008
param.verbose = True
param.order = 2
# param.plot = [False, False, False, True, True, True]
param.plot = [False]*6

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
c_0 = G.new_circle(L/2, d/2, a, lcar/2)
c_1 = G.new_circle(L/2, d/2, r_i, lcar/2)

matrice = G.new_surface([ll_0.tag, -c_0.tag])
rubber = G.new_surface([c_0.tag, -c_1.tag])
cavity = G.new_surface([c_1.tag])

G.new_physical(l_2, "condition=Rigid Wall")
G.new_physical([l_1, l_3], "condition=Periodicity")
G.new_physical(l_0, "condition=Incident_PW")
G.new_physical(matrice, "mat=pem_benchmark_1")
G.new_physical(rubber, "mat=rubber")
G.new_physical(cavity, "mat=Air")
G.new_physical_curve(c_1.tag_arcs, "condition=Fluid_Structure")

G.new_physical_curve([l_0.tag, l_1.tag, l_3.tag, l_2.tag] +c_1.tag_arcs, "model=FEM1D")
G.new_physical([matrice, rubber, cavity], "model=FEM2D")
G.new_periodicity(l_1, l_3, (L, 0, 0))

option = "-2 -v 0 "
G.run_gmsh(option)

model = Model(param)
model.resolution(param)

pem = from_yaml('pem_benchmark_1.yaml')

param.solver_pymls = Solver()
param.solver_pymls.layers = [
    Layer(pem, d),
]

param.solver_pymls.backing = backing.rigid

param.S_PW = Solver_PW(param.solver_pymls, param)
result_pyPLANESPW = param.S_PW.resolution(param.theta_d)


FEM = np.loadtxt("metaporous_benchmark_2_out.txt")
PW = np.loadtxt("metaporous_benchmark_2_PW_out.txt")
plt.plot(PW[:, 0],PW[:, 1], "b", label="PW")
plt.plot(FEM[:, 0],FEM[:, 1], "r.", label="FEM")
plt.show()

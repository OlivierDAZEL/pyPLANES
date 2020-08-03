import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "../../")
from pyPLANES.model.model import Model
from pyPLANES.classes.model_classes import ModelParameter
from pyPLANES.utils.utils_PW import Solver_PW
from pyPLANES.gmsh.write_geo_file import Gmsh as Gmsh

param = ModelParameter()
theta_d = 0
param.frequencies = (3235., 5010., 1)
param.name_project = "metaporous_benchmark_7"

param.theta_d = theta_d
L = 0.02
d = 0.02
a = 0.008
d_air = 0.04
lcar = 0.004
param.verbose = True
param.order = 4
param.plot = [False, False, True, True, True, True]
param.plot = [False]*6

p = param
G = Gmsh(p.name_project)

p_0 = G.new_point(0, 0, lcar)
p_1 = G.new_point(L, 0,lcar)
p_2 = G.new_point(L, d, lcar)
p_3 = G.new_point(0, d, lcar)

p_4 = G.new_point(L, -d_air, lcar)
p_5 = G.new_point(0, -d_air, lcar)
p_6 = G.new_point(L, d+d_air, lcar)
p_7 = G.new_point(0, d+d_air, lcar)



l_0 = G.new_line(p_0, p_1)
l_1 = G.new_line(p_1, p_2)
l_2 = G.new_line(p_2, p_3)
l_3 = G.new_line(p_3, p_0)
l_4 = G.new_line(p_0, p_5)
l_5 = G.new_line(p_5, p_4)
l_6 = G.new_line(p_4, p_1)
l_7 = G.new_line(p_2, p_6)
l_8 = G.new_line(p_6, p_7)
l_9 = G.new_line(p_7, p_3)

ll_0 = G.new_line_loop([l_0, l_1, l_2, l_3])
c_0 = G.new_circle(L/2, d/2, a, lcar/2)

matrice = G.new_surface([ll_0.tag, -c_0.tag])
inclusion = G.new_surface([c_0.tag])


ll_1 = G.new_line_loop([l_4, l_5, l_6, l_0.inverted()])
ll_2 = G.new_line_loop([l_7, l_8, l_9, l_2.inverted()])

air_incident = G.new_surface([ll_1.tag])
air_transmission = G.new_surface([ll_2.tag])

G.new_physical(l_8, "condition=Transmission")
G.new_physical([l_1, l_3, l_4, l_6, l_9, l_7], "condition=Periodicity")
G.new_physical(l_5, "condition=Incident_PW")
G.new_physical(matrice, "mat=pem_benchmark_1")
G.new_physical(inclusion, "mat=pem_benchmark_2")
G.new_physical(air_incident, "mat=Air 0")
G.new_physical(air_transmission, "mat=Air 2")
G.new_physical([l_1, l_3, l_4, l_5, l_8, l_6, l_9, l_7], "model=FEM1D")
G.new_physical([matrice, inclusion, air_incident, air_transmission], "model=FEM2D")
G.new_periodicity(l_1, l_3, (L, 0, 0))
G.new_periodicity(l_6, l_4, (L, 0, 0))
G.new_periodicity(l_7, l_9, (L, 0, 0))


option = "-2 "
G.run_gmsh(option)

model = Model(param)

model.resolution(param)


# output = np.loadtxt(model.outfile_name)
# plt.plot(output[:, 0], output[:, 1])
# plt.savefig(param.name_project+".pdf")


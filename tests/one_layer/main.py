import sys
sys.path.insert(0, "../..")

import numpy as np
import matplotlib.pyplot as plt
from pyPLANES.core.pw_problem import PwProblem
from pyPLANES.gmsh.templates.layers import one_layer

from pyPLANES.core.periodic_fem_problem import FemProblem, PeriodicFemProblem
from pyPLANES.core.periodic_pw_problem import PeriodicPwProblem
from pyPLANES.core.dgm_problem import DgmProblem

# Parameters of the simulation 
frequencies = np.linspace(1., 5000., 1)
theta_d = 50
L = 5e-2
d = 5e-2
lcar = 5e-2
material = "Wwood"
material = "melamine"
# material = "melamine_eqf"
# material = "Air"

name_project = "one_layer"
ml = [(material, d)]
termination = "transmission"
termination = "rigid"

plot_solution = [True, True, True, False, False, True]
# plot_solution = [False]*6
# plot_solution = [True]*6

method = "FEM"

bc_bottom = "bottom"
bc_right = "Periodicity"
bc_left = "Periodicity"
bc_top = "top"

ml = [(material, d), (material, d), (material, d)]
global_method = PwProblem(ml=ml, name_project=name_project, theta_d=theta_d, frequencies=frequencies, plot_solution=plot_solution,termination=termination, method="global", verbose=False, print_result=True)
global_method.resolution()

recursive_method = PwProblem(ml=ml, name_project=name_project, theta_d=theta_d, frequencies=frequencies, plot_solution=[False]*6,termination=termination, method="JAP", verbose=False, print_result=True)
recursive_method.resolution()


one_layer(name_mesh="one_layer_TMM", L=L, d=d, lcar=lcar, mat=material, method="FEM",  BC=[bc_bottom, bc_right, bc_top, bc_left])
ml = [(material, d), ("one_layer_TMM", d), (material, d)]
# ml = [("one_layer_TMM", d)]
eTMM_method = PeriodicPwProblem(ml=ml, name_project=name_project, theta_d=theta_d, order=4, frequencies=frequencies, plot_solution=[False]*6,termination=termination, verbose=False, print_result=True)
eTMM_method.resolution()
# print((Z-Air.Z)/(Z+Air.Z))

# one_layer(name_mesh=name_project, L=L, d=d, lcar=lcar, mat=material, method="FEM",  BC=[bc_bottom, bc_right, bc_top, bc_left])
# fem = PeriodicFemProblem(name_project=name_project, name_mesh=name_project, order=3, theta_d=theta_d, frequencies=frequencies, plot_solution=plot_solution,termination=termination, verbose=False)
# fem.resolution()
# plt.show()


# om = 2*np.pi*frequencies[0]
# from mediapack import Air
# k = om/Air.c
# T = np.zeros((2, 2), dtype=complex)
# T[0, 0] = np.cos(k*d)
# T[1, 0] = (om**2*Air.rho/k)*np.sin(k*d)
# T[0, 1] = -(k/(om**2*Air.rho))*np.sin(k*d)
# T[1, 1] = np.cos(k*d)
# # print(T)

# Z = -1j*Air.Z/np.tan(k*d)




# one_layer(name_mesh=name_project, L=L, d=d, lcar=lcar, mat=material, method="DGM",  BC=[bc_bottom, bc_right, bc_top, bc_left])
# dgm = DgmProblem(name_project=name_project, name_mesh=name_project, order=3, theta_d=theta_d, frequencies=frequencies, plot_solution=plot_solution,termination=termination, verbose=False)
# dgm.resolution()

# for _ent in dgm.entities:
#     print(_ent)



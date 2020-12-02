import sys
import platform

sys.path.insert(0, "../..")

import numpy as np
import matplotlib.pyplot as plt

from pyPLANES.core.periodic_fem_problem import FemProblem, PeriodicFemProblem
from pyPLANES.core.dgm_problem import DgmProblem
from pyPLANES.core.pw_problem import PwProblem
from pyPLANES.gmsh.templates.layers import one_layer

# Parameters of the simulation 
frequencies = np.linspace(1000., 5000., 1)
theta_d = 0.000

L = 5e-2
d = 5e-2
lcar = 5e-2
material = "Wwood"
material = "melamine"
# material = "melamine_eqf"
material = "Air"

name_project = "one_layer"
ml = [(material, d)]
termination = "transmission" 
termination = "rigid" 

plot_solution = [True, True, True, False, False, False]
# plot_solution = [False]*6
# plot_solution = [True]*6

method = "DGM"


bc_bottom = "Incident_PW"
bc_bottom = "Imposed displacement"
bc_right = "Periodicity"
bc_left = "Periodicity"
bc_right = "rigid"
bc_left = "rigid"
bc_top = "rigid"


# global_method = PwProblem(ml=ml, name_project=name_project, theta_d=theta_d, frequencies=frequencies, plot_solution=plot_solution,termination=termination, method="global", verbose=False)
# global_method.resolution()

one_layer(name_mesh=name_project, L=L, d=d, lcar=lcar, mat=material, method="FEM",  BC=[bc_bottom, bc_right, bc_top, bc_left])
fem = FemProblem(name_project=name_project, name_mesh=name_project, order=3, theta_d=theta_d, frequencies=frequencies, plot_solution=plot_solution,termination=termination, verbose=False)
fem.resolution()


one_layer(name_mesh=name_project, L=L, d=d, lcar=lcar, mat=material, method="DGM",  BC=[bc_bottom, bc_right, bc_top, bc_left])
dgm = DgmProblem(name_project=name_project, name_mesh=name_project, order=3, theta_d=theta_d, frequencies=frequencies, plot_solution=plot_solution,termination=termination, verbose=False)
dgm.resolution()

# print("abs                = {}".format(1-np.abs(global_method.R)**2-np.abs(global_method.T)**2))
# print("abs                = {}".format(1-np.abs(global_method.R)**2))


from mediapack import Air
omega = 2*np.pi*frequencies[0]
k = omega/Air.c
A = -1/np.sin(k*d)
x = np.linspace(0,d, 500)
p = -k*A*Air.K*np.cos(k*(x-d))
plt.plot(x, p, 'k')

if any(plot_solution):
    plt.show()

   

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
lcar = 1e-2
material = "Wwood"
material = "melamine"
# material = "melamine_eqf"
material = "Air"

name_project = "one_layer"
ml = [(material, d)]
termination = "transmission" 
termination = "rigid" 

plot_solution = [True, True, True, False, False, True]
# plot_solution = [False]*6
# plot_solution = [True]*6

method = "FEM"

bc_bottom = "Incident_PW"
# bc_bottom = "Imposed displacement"
bc_right = "Periodicity"
bc_left = "Periodicity"
# bc_right = "rigid"
# bc_left = "rigid"
bc_top = "rigid"


global_method = PwProblem(ml=ml, name_project=name_project, theta_d=theta_d, frequencies=frequencies, plot_solution=plot_solution,termination=termination, method="global", verbose=False)
global_method.resolution()

one_layer(name_mesh=name_project, L=L, d=d, lcar=lcar, mat=material, method="FEM",  BC=[bc_bottom, bc_right, bc_top, bc_left])
fem = PeriodicFemProblem(name_project=name_project, name_mesh=name_project, order=3, theta_d=theta_d, frequencies=frequencies, plot_solution=plot_solution,termination=termination, verbose=False)
fem.resolution()
plt.show()

# one_layer(name_mesh=name_project, L=L, d=d, lcar=lcar, mat=material, method="DGM",  BC=[bc_bottom, bc_right, bc_top, bc_left])
# dgm = DgmProblem(name_project=name_project, name_mesh=name_project, order=3, theta_d=theta_d, frequencies=frequencies, plot_solution=plot_solution,termination=termination, verbose=False)
# dgm.resolution()

# for _ent in dgm.entities:
#     print(_ent)



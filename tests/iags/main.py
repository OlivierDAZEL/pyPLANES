import sys
import platform

sys.path.insert(0, "../..")

import numpy as np
import matplotlib.pyplot as plt

from pyPLANES.core.periodic_fem_problem import FemProblem, PeriodicFemProblem
from pyPLANES.gmsh.templates.iags import iags

# Parameters of the simulation 
frequencies = np.linspace(0.1, 1.5, 150)
theta_d = 45.000

L = 5e-2
d = 5e-2
lcar = 5e-2
material = "Wwood"
material = "melamine"
# material = "melamine_eqf"
material = "Air"

name_project = "iags-2021"

bc_bottom = "Incident_PW"
bc_bottom = "Imposed displacement"
bc_right = "Periodicity"
bc_left = "Periodicity"
bc_right = "rigid"
bc_left = "rigid"
bc_top = "rigid"

BC = ["Incident_PW", "Periodicity", "rigid", "Periodicity"]
BC = ["Imposed displacement"]+["rigid"]*3

plot_solution = [False]*6

iags(name_project=name_project, lcar=100., BC = BC )
fem = FemProblem(name_project=name_project, name_mesh=name_project, order=2, frequencies=frequencies, plot_solution=plot_solution, export_plots=plot_solution, verbose=True, export_paraview=True)
fem.resolution()




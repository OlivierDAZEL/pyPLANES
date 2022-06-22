import sys
sys.path.insert(0, "../..")
import pickle



import numpy as np
import matplotlib.pyplot as plt

from pyPLANES.gmsh.templates.layers import one_layer
from pyPLANES.core.pw_problem import PwProblem
from pyPLANES.core.periodic_pw_problem import PeriodicPwProblem
from pyPLANES.core.result import Results, Result, Test


plot_solution = [True, True, True, False, False, False]
# plot_solution = [False]*6
verbose = [True, False][0]
# Parameters of the simulation
theta_d = 0.00000
nb_layers = 1
L = 5e-2
d = 5e-2
lcar = 5e-2/10
from mediapack import Air

frequency = Air.c /d

name_project="solution"

termination = ["rigid", "transmission"][0]

# ml = [("Wwood", d),  ("melamine", d), ("Wwood", d)]
# ml = [("melamine", d)]*nb_layers
# ml = [("WWood", d), ("melamine",d)]
ml = [("Air", d)]*nb_layers

# ml = [("Air", d), ("melamine",d), ("Air", d)]
global_method = PwProblem(ml=ml, name_project=name_project, theta_d=theta_d, frequency=frequency, plot_solution=plot_solution,termination=termination, method="global", verbose=False, print_result=True)
global_method.resolution()

recursive_method = PwProblem(ml=ml, name_project=name_project, theta_d=theta_d, frequency=frequency, plot_solution=plot_solution,termination=termination, method="JAP", verbose=verbose,save_append="a", print_result=True)
recursive_method.resolution()

material = ["Air", "Wwood", "melamine"][0]
one_layer(name_mesh="mesh", L=L, d=d, lcar=lcar, mat=material)
# material = ["Air", "Wwood", "melamine"][2]
# one_layer(name_mesh="mesh_2", L=L, d=d, lcar=lcar, mat=material)

ml_fem = [ ("mesh", None)]*nb_layers
ml_fem = ml.copy()
# ml_fem[1] = ("mesh", None)

eTMM_method = PeriodicPwProblem(ml=ml_fem, name_project=name_project, theta_d=theta_d, order=3, nb_bloch_waves=0, frequency=frequency, plot_solution=plot_solution,termination=termination, verbose=verbose, save_append="a",print_result=True)
eTMM_method.resolution()







if any(plot_solution):
    plt.show()


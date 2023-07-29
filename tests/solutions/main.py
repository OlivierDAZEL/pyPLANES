import sys
sys.path.insert(0, "../..")
import pickle

from mediapack import Air
import numpy as np
import matplotlib.pyplot as plt

from pyPLANES.gmsh.templates.layers import one_layer
from pyPLANES.core.pw_problem import PwProblem
from pyPLANES.core.periodic_pw_problem import PeriodicPwProblem
from pyPLANES.core.fem_problem import FemProblem
from pyPLANES.core.result import Results, Result, Test


plot_solution = [True, True, True, False, False, False]
# plot_solution = [False]*6
verbose = [True, False][1]
# Parameters of the simulation
theta_d = 0.00000
nb_layers = 4
L = 5e-2
d = 5e-2
lcar = 5e-2/5


frequency = 1e3

name_project="solution"
termination = ["rigid", "transmission"][0]
material = ["Air", "Wwood", "melamine"][0]

ml = [(material, d)]*nb_layers

global_method = PwProblem(ml=ml, name_project=name_project, theta_d=theta_d, frequencies=frequency, plot_solution=plot_solution,termination=termination, method="global", verbose=verbose, print_result=True)
global_method.resolution()
RG = global_method.result.R0 
print(global_method.result.R0)
recursive_method = PwProblem(ml=ml, name_project=name_project, theta_d=theta_d, frequencies=frequency, plot_solution=plot_solution,termination=termination, method="JAP", verbose=verbose,save_append="a", print_result=True)
recursive_method.resolution()
print(recursive_method.result.R0)

one_layer(name_mesh="mesh", L=L, d=d, lcar=lcar, mat=material)
ml_fem = [ ("mesh", None)]*nb_layers

eTMM_method = PeriodicPwProblem(ml=ml_fem, name_project=name_project, theta_d=theta_d, order=2, nb_bloch_waves=0, frequencies=frequency, plot_solution=plot_solution,termination=termination, verbose=verbose, save_append="a", print_result=True)
eTMM_method.resolution()
print(eTMM_method.result.R0)

print(1+np.array(RG))

if any(plot_solution):
    plt.show() 
    

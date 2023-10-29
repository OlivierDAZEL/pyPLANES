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
plot_solution = [False]*6
verbose = [True, False][1]
# Parameters of the simulation
theta_d = 60.00000
nb_layers = 1
L = 1.
d = 1.
lcar = d/10


frequency = np.linspace( 5e0, 1e4,1)

name_project="solution"
method_FEM = ["jap", "characteristics"][1]
termination = ["rigid", "transmission"][0]
material = ["Air", "Wwood", "melamine"][0]
# material = ["Air", "Wwood", "melamine"][2]

ml = [(material, d)]*nb_layers

global_method = PwProblem(ml=ml, name_project=name_project+"_GM", theta_d=theta_d, frequencies=frequency, plot_solution=plot_solution,termination=termination, method="global", verbose=verbose, print_result=True)
global_method.resolution()

RG = global_method.result.R0 
recursive_method = PwProblem(ml=ml, name_project=name_project+"_JAP", theta_d=theta_d, frequencies=frequency, plot_solution=plot_solution,termination=termination, method="JAP", verbose=verbose,save_append="a", print_result=True)
recursive_method.resolution()

characteristic_method = PwProblem(ml=ml, name_project=name_project, theta_d=theta_d, frequencies=frequency, plot_solution=plot_solution,termination=termination, method="characteristics", verbose=verbose,save_append="a", print_result=True)
characteristic_method.resolution()

one_layer(name_mesh="mesh", L=L, d=d, lcar=lcar, mat=material)
ml_fem = [ ("mesh", None)]*nb_layers

eTMM_method = PeriodicPwProblem(ml=ml_fem, name_project=name_project, theta_d=theta_d, order=2, nb_bloch_waves=2, frequencies=frequency, plot_solution=plot_solution,termination=termination, verbose=verbose, save_append="a", print_result=True, method=method_FEM)
eTMM_method.resolution()


print(f"R GM ={global_method.result.R0}")
print(f"R RM ={recursive_method.result.R0}")
print(f"R CM ={characteristic_method.result.R0}")
print(f"R FEM={eTMM_method.result.R0}")

# print(f"xxxxxxxxxxxx")
# print(f"\t\tER RM={np.abs(recursive_method.result.R0[0]-global_method.result.R0[0])}")
# print(f"\t\tER CM={np.abs(characteristic_method.result.R0[0]-global_method.result.R0[0])}")
# if len(characteristic_method.result.T0) !=0:
#     print(f"T GM={global_method.result.T0}")
#     print(f"T RM={recursive_method.result.T0}")
#     print(f"T CM={characteristic_method.result.T0}")
#     print(f"xxxxxxxxxxxx")
#     print(f"\t\tET RM={np.abs(recursive_method.result.T0[0]-global_method.result.T0[0])}")
#     print(f"\t\tET CM={np.abs(characteristic_method.result.T0[0]-global_method.result.T0[0])}")

if any(plot_solution):
    # plt.legend()
    plt.show() 
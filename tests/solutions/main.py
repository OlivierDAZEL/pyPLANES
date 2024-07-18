import sys
sys.path.insert(0, "../..")

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
theta_d = 50.00000
nb_layers = 1
L = 2.e-2
d = 2.e-2
lcar = d
nb_bloch_waves = 4
order = 2

frequency = 3e1

name_project="solution"
case = ["layer", "sandwich"][0]
method_FEM = ["jap", "characteristics", "global"][2]
termination = ["rigid", "transmission"][1]
material = ["Air", "Wwood", "melamine"][2]

if case == "layer":
    ml = [(material, d)]*nb_layers
    one_layer(name_mesh="mesh", L=L, d=d, lcar=lcar, mat=material)
    ml_fem = [ ("mesh", None)]*nb_layers
if case == "sandwich":
    ml = [("rubber",0.2e-3), ["melamine" , d], ("rubber",0.2e-3)]
    # one_layer(name_mesh="mesh", L=L, d=d, lcar=lcar, mat=ml[1][0])
    ml_fem = [("rubber",0.2e-3), ["mesh" , None], ("rubber",0.2e-3)]

global_method = PwProblem(ml=ml, name_project=name_project+"_GM", theta_d=theta_d, frequencies=frequency, plot_solution=plot_solution,termination=termination, method="global", verbose=verbose, print_result=True)
global_method.resolution()

recursive_method = PwProblem(ml=ml, name_project=name_project+"_JAP", theta_d=theta_d, frequencies=frequency, plot_solution=plot_solution,termination=termination, method="JAP", verbose=verbose,print_result=True)
recursive_method.resolution()

characteristic_method = PwProblem(ml=ml, name_project=name_project, theta_d=theta_d, frequencies=frequency, plot_solution=plot_solution,termination=termination, method="characteristics", verbose=verbose, print_result=True)
characteristic_method.resolution()

eTMM_method = PeriodicPwProblem(ml=ml_fem, name_project=name_project, theta_d=theta_d, order=order, nb_bloch_waves=nb_bloch_waves, frequencies=frequency, plot_solution=plot_solution,termination=termination, verbose=verbose, save_append="a", print_result=True, method=method_FEM)
eTMM_method.resolution()


# eTMM_method = PeriodicPwProblem(ml=ml_fem, name_project=name_project, theta_d=theta_d, order=order, nb_bloch_waves=nb_bloch_waves, frequencies=frequency, plot_solution=plot_solution,termination=termination, verbose=verbose, save_append="a", print_result=True, method=method_FEM)
# eTMM_method.resolution()

# rTMM_method = PeriodicPwProblem(ml=ml_fem, name_project=name_project, theta_d=theta_d, order=order, nb_bloch_waves=nb_bloch_waves, frequencies=frequency, plot_solution=plot_solution,termination=termination, verbose=verbose, save_append="a", print_result=True, method="characteristics")
# rTMM_method.resolution()

print(f"R GM ={global_method.result.R0}")
print(f"R RM ={recursive_method.result.R0}")
print(f"R CM ={characteristic_method.result.R0}")
# print(f"R FEM={eTMM_method.result.R0}")
# print(f"R CFE={rTMM_method.result.R0}")

# print(f"xxxxxxxxxxxx")
# print(f"\t\tER RM={np.abs(recursive_method.result.R0[0]-global_method.result.R0[0])}")
# print(f"\t\tER CM={np.abs(characteristic_method.result.R0[0]-global_method.result.R0[0])}")
# print(f"\t\tER FEM={np.abs(eTMM_method.result.R0[0]-global_method.result.R0[0])}")
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
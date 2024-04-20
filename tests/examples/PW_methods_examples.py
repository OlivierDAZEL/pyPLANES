import sys
sys.path.insert(0, "../..")


from mediapack import Air
import numpy as np
import matplotlib.pyplot as plt

from pyPLANES.core.pw_problem import PwProblem
from pyPLANES.core.result import Results, Result, Test


plot_solution = [True, True, True, False, False, False]
verbose = [True, False][1]
# Parameters of the simulation
theta_d = 60.00000

nb_layers = 2
L = 2.e-2
d = 2.e-2
lcar = d/5
nb_bloch_waves = 1
order = 2

frequency = 5e3

name_project="solution"
case = ["layer", "sandwich"][0]
method_FEM = ["jap", "characteristics"][1]
termination = ["rigid", "transmission"][1]
material = ["Air", "Wwood", "melamine"][2]
# material = ["Air", "Wwood", "melamine"][0]


if case == "layer":
    ml = [(material, d)]*nb_layers
if case == "sandwich":
    ml = [("rubber",0.2e-3), ["melamine" , d], ("rubber",0.2e-3)]


global_method = PwProblem(ml=ml, name_project=name_project+"_GM", theta_d=theta_d, frequencies=frequency, plot_solution=plot_solution,termination=termination, method="global", verbose=verbose, print_result=True)
global_method.resolution()


recursive_method = PwProblem(ml=ml, name_project=name_project+"_JAP", theta_d=theta_d, frequencies=frequency, plot_solution=plot_solution,termination=termination, method="JAP", verbose=verbose,save_append="a", print_result=True)
recursive_method.resolution()

characteristic_method = PwProblem(ml=ml, name_project=name_project, theta_d=theta_d, frequencies=frequency, plot_solution=plot_solution,termination=termination, method="characteristics", verbose=verbose,save_append="a", print_result=True)
characteristic_method.resolution()

# print(f"R GM ={global_method.result.R0}")
# print(f"R RM ={recursive_method.result.R0}")
# print(f"R CM ={characteristic_method.result.R0}")


if any(plot_solution):
    # plt.legend()
    plt.show() 
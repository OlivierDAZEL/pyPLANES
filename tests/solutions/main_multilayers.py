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
plot_solution = [False]*6
verbose = [True, False][1]
# Parameters of the simulation
theta_d = 45.00000



nb_bloch_waves = 0
order = 2
frequency = 3e3
# frequency = np.linspace(200, 1e3, 30)

name_project="solution"
method_FEM = ["jap", "global"][0]
termination = ["rigid", "transmission"][1]

methods = ["layer","FEM", "layer"]
methods = ["FEM"]*3
materials = ["rubber", "melamine", "rubber"]

ds = [2e-4, 2e-2, 2e-4]
L = np.max(ds)
lcar = L/10
ml, ml_fem = [], []
for i, mat in enumerate(materials):
    ml.append((mat, ds[i]))
    if methods[i]== "FEM":
        one_layer(name_mesh=f"Air_FEM_{i}", L=L, d=ds[i], lcar=lcar, mat=mat)
        ml_fem.append((f"Air_FEM_{i}", None))
    else:
        ml_fem.append(ml[i])

print(ml)
print(ml_fem)

global_method = PwProblem(ml=ml, name_project=name_project+"_GM", theta_d=theta_d, frequencies=frequency, plot_solution=plot_solution,termination=termination, method="global", verbose=verbose, print_result=True)
global_method.resolution()


eTMM_method = PeriodicPwProblem(ml=ml_fem, name_project=name_project, theta_d=theta_d, order=order, nb_bloch_waves=nb_bloch_waves, frequencies=frequency, plot_solution=plot_solution,termination=termination, verbose=verbose, save_append="a", print_result=True, method=method_FEM)
eTMM_method.resolution()



# plt.plot(frequency,np.real(global_method.result.T0), 'b')
# plt.plot(frequency,np.real(recursive_method.result.T0), 'b.')
# plt.plot(frequency,np.real(eTMM_method.result.T0), 'm+')
# # plt.plot(frequency,np.imag(global_method.result.T0), 'r')
# # plt.plot(frequency,np.imag(recursive_method.result.T0), 'r.')
# # plt.plot(frequency,np.imag(eTMM_method.result.T0), 'm+')


# plt.show()
# exit()

# print(eTMM_method.layers)
# print(eTMM_method.interfaces)
# exit()


# eTMM_method = PeriodicPwProblem(ml=ml_fem, name_project=name_project, theta_d=theta_d, order=order, nb_bloch_waves=nb_bloch_waves, frequencies=frequency, plot_solution=plot_solution,termination=termination, verbose=verbose, save_append="a", print_result=True, method=method_FEM)
# eTMM_method.resolution()

# rTMM_method = PeriodicPwProblem(ml=ml_fem, name_project=name_project, theta_d=theta_d, order=order, nb_bloch_waves=nb_bloch_waves, frequencies=frequency, plot_solution=plot_solution,termination=termination, verbose=verbose, save_append="a", print_result=True, method="characteristics")
# rTMM_method.resolution()

print(f"T GM ={global_method.result.T0}")
# print(f"T RM ={recursive_method.result.T0}")
# print(f"T CM ={characteristic_method.result.T0}")
print(f"T FEM={eTMM_method.result.T0}")
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
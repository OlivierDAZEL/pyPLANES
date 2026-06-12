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
energetic_balance = [False, True][1]
# Parameters of the simulation
theta_d = 89.000
nb_layers = 1
L = 2.e-2
d = 2.e-2
lcar = d/5
nb_bloch_waves = 3
order = 2

frequency = 3e3
# frequency = np.linspace(200, 1e3, 30)

name_project="solution"
case = ["layer", "sandwich"][0]
method_FEM = ["jap", "global", "TMM"][0]
termination = ["rigid", "transmission"][0]
material = ["Air", "Wwood", "melamine", "rubber", "melamine_eqf"][2]

if case == "layer":
    ml = [(material, d)]*nb_layers
    one_layer(name_mesh="mesh", L=L, d=d, lcar=lcar, mat=material)
    ml_fem = [ ("mesh", None)]*nb_layers
if case == "sandwich":
    ml = [("rubber",0.2e-3), [material , d], ("rubber",0.2e-3)]
    one_layer(name_mesh="mesh", L=L, d=d, lcar=lcar, mat=material)
    ml_fem = [("rubber",0.2e-3), ["mesh" , None], ("rubber",0.2e-3)]

global_method = PwProblem(ml=ml, name_project=name_project+"_GM", theta_d=theta_d, frequencies=frequency, plot_solution=plot_solution,termination=termination, method="global", verbose=verbose, print_result=True,energetic_balance=energetic_balance)
global_method.resolution()







# H_method = PwProblem(ml=ml, name_project=name_project+"_H", theta_d=theta_d, frequencies=frequency, plot_solution=plot_solution,termination=termination, method="H", verbose=verbose, print_result=True,energetic_balance=energetic_balance)
# H_method.resolution()


# print(f"Z_global   ={global_method.result.Z_prime[0]*Air.Z}")
# print(f"T_global   ={global_method.result.T0[0]}")
# print(global_method.result.R0[0], "R0   global")
# recursive_method = PwProblem(ml=ml, name_project=name_project+"_JAP", theta_d=theta_d, frequencies=frequency, plot_solution=plot_solution,termination=termination, method="JAP", verbose=verbose,print_result=True)
# recursive_method.resolution()
# print(f"Z_recursive={recursive_method.result.Z_prime[0]*Air.Z}")

# characteristic_method = PwProblem(ml=ml, name_project=name_project, theta_d=theta_d, frequencies=frequency, plot_solution=plot_solution,termination=termination, method="characteristics", verbose=verbose, print_result=True)
# characteristic_method.resolution()


# TMM_method = PwProblem(ml=ml, name_project=name_project+"_TMM", theta_d=theta_d, frequencies=frequency, plot_solution=plot_solution,termination=termination, method="TMM", verbose=verbose, print_result=True)
# TMM_method.resolution()
# # print(f"R_TMM      ={TMM_method.result.R0[0]}")
# print(f"Z_TMM      ={TMM_method.result.Z_prime[0]*Air.Z}")


# Z_method = PwProblem(ml=ml, name_project=name_project+"_Z", theta_d=theta_d, frequencies=frequency, plot_solution=plot_solution,termination=termination, method="Z", verbose=verbose, print_result=True)
# Z_method.resolution()
# print(f"Z_Z        ={Z_method.result.Z_prime[0]*Air.Z}")
# print(f"T_Z.       ={Z_method.result.T0[0]}")


# eTMM_method = PeriodicPwProblem(ml=ml_fem, name_project=name_project, theta_d=theta_d, order=order, nb_bloch_waves=nb_bloch_waves, frequencies=frequency, plot_solution=plot_solution,termination=termination, verbose=verbose, save_append="a", print_result=True, method=method_FEM)
# eTMM_method.resolution()
# print(f"R_eTMM     ={eTMM_method.result.Z_prime[0]*Air.Z}")



# print(f"R GM ={global_method.result.R0}")
# print(f"R CM ={characteristic_method.result.R0}")
# print(f"R TM ={TMM_method.result.R0}")
# print(f"R FEM={eTMM_method.result.R0}")


# print(f"T GM ={global_method.result.T0}")
# print(f"T RM ={recursive_method.result.T0}")
# print(f"T CM ={characteristic_method.result.T0}")
# print(f"T TM ={TMM_method.result.T0}")
# print(f"T FEM={eTMM_method.result.T0}")


# # plt.plot(frequency,np.real(global_method.result.T0), 'b')
# # plt.plot(frequency,np.real(recursive_method.result.T0), 'b.')
# # plt.plot(frequency,np.real(eTMM_method.result.T0), 'm+')
# # # plt.plot(frequency,np.imag(global_method.result.T0), 'r')
# # # plt.plot(frequency,np.imag(recursive_method.result.T0), 'r.')
# # # plt.plot(frequency,np.imag(eTMM_method.result.T0), 'm+')


# # plt.show()
# # exit()

# # print(eTMM_method.layers)
# # print(eTMM_method.interfaces)
# # exit()


# # eTMM_method = PeriodicPwProblem(ml=ml_fem, name_project=name_project, theta_d=theta_d, order=order, nb_bloch_waves=nb_bloch_waves, frequencies=frequency, plot_solution=plot_solution,termination=termination, verbose=verbose, save_append="a", print_result=True, method=method_FEM)
# # eTMM_method.resolution()

# # rTMM_method = PeriodicPwProblem(ml=ml_fem, name_project=name_project, theta_d=theta_d, order=order, nb_bloch_waves=nb_bloch_waves, frequencies=frequency, plot_solution=plot_solution,termination=termination, verbose=verbose, save_append="a", print_result=True, method="characteristics")
# # rTMM_method.resolution()

# print(f"T GM ={global_method.result.T0}")
# print(f"T RM ={recursive_method.result.T0}")
# print(f"T CM ={characteristic_method.result.T0}")
# print(f"T FEM={eTMM_method.result.T0}")
# # print(f"R CFE={rTMM_method.result.R0}")

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

# k = 2*np.pi*frequency/Air.c
# print(-1j*Air.Z/np.tan(k*ml[0][1]))



if any(plot_solution):
    # plt.legend()
    plt.show() 
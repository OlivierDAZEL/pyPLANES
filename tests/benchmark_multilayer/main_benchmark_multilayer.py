import sys
import platform
sys.path.insert(0, "../..")

import numpy as np

import matplotlib.pyplot as plt

from pyPLANES.core.pw_problem import PwProblem
from pyPLANES.utils.io import result_pymls

# General parameters of the benchmark 
frequencies = np.linspace(100., 500., 100)
theta_d = 45.000
plot_RT = True
# plot_RT = False

name_project_list = []
ml_list = []
termination_list = []

name_project_list.append("Melamine - Rigid Backing")
ml_list.append([("melamine", 3e-2)])
termination_list.append("rigid")

name_project_list.append("Melamine - Transmission")
ml_list.append([("melamine", 3e-2)])
termination_list.append("transmission")

name_project_list.append("Melamine *2 - Transmission")
ml_list.append([("melamine", 3e-2)]*2)
termination_list.append("transmission")


name_project_list.append("Melamine_eqf - Rigid Backing")
ml_list.append([("melamine_eqf", 3e-2)])
termination_list.append("rigid")

name_project_list.append("Melamine_eqf - Transmission")
ml_list.append([("melamine_eqf", 3e-2)])
termination_list.append("transmission")

name_project_list.append("Melamine_eqf *2 - Transmission")
ml_list.append([("melamine_eqf", 3e-2)]*2)
termination_list.append("transmission")


name_project_list.append("Wwood Plate - Rigid Backing")
ml_list.append([("Wwood", 1e-1)])
termination_list.append("rigid")

name_project_list.append("Wwood Plate - Transmission")
ml_list.append([("Wwood", 1e-2)])
termination_list.append("transmission")

name_project_list.append("Wwood Plate *2- Transmission")
ml_list.append([("Wwood", 1e-2)]*2)
termination_list.append("transmission")

name_project_list.append("Aluminium - Melamine - Rigid")
ml_list.append([("Aluminium", 1e-3), ("melamine", 4e-2)])
termination_list.append("rigid")

name_project_list.append("Aluminium - Melamine - Transmission")
ml_list.append([("Aluminium", 1e-3), ("melamine", 4e-2)])
termination_list.append("transmission")

name_project_list.append("Sandwich -  Transmission")
ml_list.append([("Aluminium", 1e-3), ("melamine", 4e-2), ("Aluminium", 1e-3)])
termination_list.append("transmission")

name_project_list.append("Test Elastic - Fluid interface -  Transmission")
ml_list.append([("Aluminium", 1e-3), ("Air", 1e-3), ("Aluminium", 1e-3)])
termination_list.append("transmission")

name_project_list.append("Test_PEM - Fluid interface -  Transmission")
ml_list.append([("melamine_eqf", 4e-2), ("Air", 1e-3), ("Aluminium", 1e-3)])
termination_list.append("transmission")


for ii, name_project in enumerate(name_project_list):
    ml = ml_list[ii]
    print(name_project)
    termination= termination_list[ii]
    f_ref, R_ref, T_ref  = result_pymls(name_project=name_project, ml=ml, termination=termination, theta_d=theta_d, frequencies=frequencies, plot_RT=plot_RT)
    # test_method = PwProblem(name_project=name_project, ml=ml, theta_d=theta_d , termination=termination, frequencies=frequencies, method="recursive").resolution()
    global_method = PwProblem(name_project=name_project, ml=ml, theta_d=theta_d , termination=termination, frequencies=frequencies, method="global").compute_error(f_ref, R_ref, T_ref, plot_RT)
    recursive_method = PwProblem(name_project=name_project, ml=ml, theta_d=theta_d , termination=termination, frequencies=frequencies, method="recursive").compute_error(f_ref, R_ref, T_ref, plot_RT)

if plot_RT:
    plt.show()

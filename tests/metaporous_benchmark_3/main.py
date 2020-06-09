import sys
import platform

sys.path.insert(0, "../..")

import numpy as np
import matplotlib.pyplot as plt

from pyPLANES.classes.fem_problem import FemProblem
from pyPLANES.utils.utils_PW import Solver_PW
from pyPLANES.gmsh.inclusions import one_inclusion


frequencies = (10., 5010., 201)
name_mesh = "one_inclusion"
L = 0.02
d = 0.02
a = 8e-3
lcar = 0.001
order = 4


# # Homogeneous layer
ml = [("pem_benchmark_1", d)]

theta_d = 00.000000
name_project = "CASE3_normal"
S_PW = Solver_PW(ml=ml, name_project=name_project, theta_d=theta_d, frequencies=frequencies).resolution(theta_d)
theta_d = 30.000000
name_project = "CASE3_pi3"
S_PW = Solver_PW(ml=ml, name_project=name_project, theta_d=theta_d, frequencies=frequencies).resolution(theta_d)


# # FEM problem
one_inclusion(name_mesh, L, d, a, lcar, "pem_benchmark_1", "pem_benchmark_2", "Rigid Wall")

theta_d = 00.000000
name_project = "CASE3_normal"
problem = FemProblem(name_mesh=name_mesh, theta_d=theta_d, name_project=name_project, order=order, frequencies=frequencies).resolution()

theta_d = 30.000000
name_project = "CASE3_pi3"
problem = FemProblem(name_mesh=name_mesh, theta_d=theta_d, name_project=name_project, order=order, frequencies=frequencies).resolution()


import sys

sys.path.insert(0, "../../")

from pyPLANES.classes.fem_problem import FemProblem
from pyPLANES.utils.utils_PW import Solver_PW
from pyPLANES.gmsh.inclusions import one_inclusion_bicomposite

frequencies = (10., 5010., 201)

name_mesh = "bicomposite_inclusion"

L = 0.02
d = 0.02
a = 0.008
r_i = 0.002
lcar = 0.0005

order = 3

# Homogeneous layer
ml = [("pem_benchmark_2", d)]
theta_d = 00.000000
name_project = "CASE4_normal"
S_PW = Solver_PW(ml=ml, name_project=name_project, theta_d=theta_d, frequencies=frequencies).resolution(theta_d)
theta_d = 30.000000
name_project = "CASE4_pi3"
S_PW = Solver_PW(ml=ml, name_project=name_project, theta_d=theta_d, frequencies=frequencies).resolution(theta_d)

one_inclusion_bicomposite(name_mesh, L, d, a, r_i, lcar, "pem_benchmark_2", "pem_benchmark_1", "steel", "Rigid Wall")

theta_d = 00.000000
name_project = "CASE4_normal"
problem = FemProblem(name_mesh=name_mesh, theta_d=theta_d, name_project=name_project, order=order, frequencies=frequencies).resolution()

theta_d = 30.000000
name_project = "CASE4_pi3"
problem = FemProblem(name_mesh=name_mesh, theta_d=theta_d, name_project=name_project, order=order, frequencies=frequencies).resolution()


import numpy as np
import matplotlib.pyplot as plt



from pymls import from_yaml, Solver, Layer, backing
from pymls.media import Air, PEM, EqFluidJCA

import src.gmsh.write_geo_file as geo_write
from src.model.model import Model
from src.classes.model_classes import ModelParameter
from src.utils.utils_PW import Solver_PW


if __name__ == "__main__":

    param = ModelParameter()
    theta_d = 30
    param.frequency = (10., 50., 1)
    param.name_project = "metaporous_benchmark_7"
    param.subproject = None
    theta = theta_d*np.pi/180.

    param.theta_d = theta_d
    param.L = 0.1
    param.d = 0.1
    param.a = 0.02
    param.period = param.L
    param.lcar = 0.03
    param.period = param.L
    param.l1 = param.d
    param.excitation = "Imposed_PW"
    param.transmission = "Rigid Wall"
    param.pem1 = "foam2 98 1"
    # param.pem1 = "Air"
    param.pem2 = "foam2 98 2"
    param.NbPartition = 1
    param.order = 5
    param.plot = [True, True, True, False, False, False]
    param.plot = [False]*6



    foam2 = from_yaml('materials/foam2.yaml')
    param.solver_pymls = Solver()
    param.solver_pymls.layers = [
        Layer(foam2, param.d),
    ]
    param.solver_pymls.backing = backing.transmission
    # param.solver_pymls.backing = backing.rigid
    param.S_PW = Solver_PW(param.solver_pymls, param)


    param.gmsh_file = "GMSH/toto"

    geo_write.one_layer(param)

    model = Model(param)

    model.resolution(param)




    plt.show()

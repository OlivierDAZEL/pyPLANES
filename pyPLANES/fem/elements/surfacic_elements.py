import numpy as np
import numpy.linalg as LA


def imposed_Neumann(_elem):
    coord_e = _elem.get_coordinates()
    K_ref = _elem.reference_element
    n, m = K_ref.Phi.shape
    F = np.zeros(n)
    for ipg in range(m):
        _Phi = K_ref.Phi[:, ipg].reshape(n)
        F += K_ref.w[ipg]*_Phi
    F *= LA.norm(coord_e[:, 1]-coord_e[:, 0])/2.
    return F


def imposed_pw_elementary_vector(_elem, k):
    coord_e = _elem.get_coordinates()
    a = min(coord_e[0, :])
    K_ref = _elem.reference_element
    length = LA.norm(coord_e[:, 1]-coord_e[:, 0])
    n, m = K_ref.Phi.shape
    F = np.zeros(n, dtype=complex)
    for ipg in range(m):
        _Phi = K_ref.Phi[:, ipg].reshape(n)
        F += K_ref.w[ipg]*_Phi*np.exp(-1j*k*(length*K_ref.xi[ipg]/2.))
    F *= (length/2.)*np.exp(-1j*k*(a+length/2.))
    return F

def fluid_structure_interaction_elementary_matrix(_elem):
    coord_e = _elem.get_coordinates()
    K_ref = _elem.reference_element
    n, m = K_ref.Phi.shape
    M = np.zeros((n, n))
    for ipg in range(m):
        _Phi = K_ref.Phi[:, ipg].reshape(n)
        M += K_ref.w[ipg]*np.dot(_Phi.reshape((n, 1)), _Phi.reshape((1, n)))
    M *= LA.norm(coord_e[:, 1]-coord_e[:, 0])/2.

    return M
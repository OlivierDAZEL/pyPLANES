import numpy as np
import numpy.linalg as LA

def fluid_elementary_matrices(elem):

    coord_e = elem.get_coordinates()
    K_ref = elem.reference_element

    X1X2 = coord_e[:,1]- coord_e[:,0]
    e_x = X1X2/LA.norm(X1X2)
    X1X3 = coord_e[:,2]- coord_e[:,0]
    e_z = np.cross(X1X2, X1X3)
    e_z /= LA.norm(e_z)
    e_y = np.cross(e_z, e_x)

    Coord_e = np.zeros((2,3))
    Coord_e[0, 1] = X1X2.dot(e_x)
    Coord_e[0, 2] = X1X3.dot(e_x)
    Coord_e[1, 2] = X1X3.dot(e_y)

    n, m = K_ref.Phi.shape
    vh = np.zeros((n ,n))
    vq = np.zeros((n, n))
    for ipg in range(m):
        _Phi = K_ref.Phi[:,ipg].reshape((1, n))
        _dPhi  = np.array([K_ref.dPhi[0][:, ipg], K_ref.dPhi[1][:, ipg]])
        J = _dPhi[:, 0:3].dot(Coord_e.T)
        _weight = K_ref.w[ipg] * LA.det(J)

        Gd = LA.solve(J, _dPhi)

        vh += _weight*np.dot(Gd.T, Gd)
        vq += _weight*np.dot(_Phi.T, _Phi)

    return vh, vq

def elas_elem(coord_e, K_ref):

    n, m = K_ref.Phi.shape
    vm = np.zeros((3*n, 3*n))
    vk0 = np.zeros((3*n, 3*n))
    vk1 = np.zeros((3*n, 3*n))
    vc = np.zeros((3*n, n))
    vh = np.zeros((n, n))
    vq = np.zeros((n, n))

    X1X2 = coord_e[:,1]- coord_e[:,0]
    e_x = X1X2/LA.norm(X1X2)
    X1X3 = coord_e[:,2]- coord_e[:,0]
    e_z = np.cross(X1X2, X1X3)
    e_z /= LA.norm(e_z)
    e_y = np.cross(e_z, e_x)

    Coord_e = np.zeros((2,3))
    Coord_e[0, 1] = X1X2.dot(e_x)
    Coord_e[0, 2] = X1X3.dot(e_x)
    Coord_e[1, 2] = X1X3.dot(e_y)

    Coord_e = coord_e[0:2,:]

    for ipg in range(m):
        _Phi = K_ref.Phi[:,ipg].reshape((1, n))
        _dPhi  = np.array([K_ref.dPhi[0][:, ipg], K_ref.dPhi[1][:, ipg]])
        J = _dPhi[:, 0:3].dot(Coord_e.T)
        _weight = K_ref.w[ipg] * LA.det(J)

        Gd = LA.solve(J, _dPhi)
        eps = np.zeros((3, 3*n))
        eps[0, 0:n] = Gd[0, :] # eps_xx = \dpar{ux}{x}
        eps[1, n:2*n] = Gd[1, :] # eps_yy = \dpar{uy}{y}
        eps[2, 0:n] = Gd[1, :]
        eps[2, n:2*n] = Gd[0, :] # eps_yz = \dpar{uz}{y}+ \dpar{uy}{z}

        Phi_u = np.zeros((2, 3*n))
        Phi_u[0, 0:n] = _Phi
        Phi_u[1, n:2*n] = _Phi

        D_0, D_1 = np.zeros((3, 3)), np.zeros((3, 3))
        D_0[0, 0:2] = 1.
        D_0[1, 0:2] = 1.
        D_1[0, 1] = -2.
        D_1[1, 0] = -2.
        D_1[2, 2] = 1.

        vk0 += _weight*LA.multi_dot([eps.T, D_0, eps])
        vk1 += _weight*LA.multi_dot([eps.T, D_1, eps])
        vm += _weight*np.dot(Phi_u.T, Phi_u)

    return vm, vk0, vk1




def pem98_elementary_matrices(elem):

    coord_e = elem.get_coordinates()
    K_ref = elem.reference_element
    n, m = K_ref.Phi.shape
    vm = np.zeros((3*n, 3*n))
    vk0 = np.zeros((3*n, 3*n))
    vk1 = np.zeros((3*n, 3*n))
    vc = np.zeros((3*n, n))
    vh = np.zeros((n, n))
    vq = np.zeros((n, n))

    X1X2 = coord_e[:,1]- coord_e[:,0]
    e_x = X1X2/LA.norm(X1X2)
    X1X3 = coord_e[:,2]- coord_e[:,0]
    e_z = np.cross(X1X2, X1X3)
    e_z /= LA.norm(e_z)
    e_y = np.cross(e_z, e_x)

    Coord_e = np.zeros((2,3))
    Coord_e[0, 1] = X1X2.dot(e_x)
    Coord_e[0, 2] = X1X3.dot(e_x)
    Coord_e[1, 2] = X1X3.dot(e_y)

    Coord_e = coord_e[0:2,:]

    for ipg in range(m):
        _Phi = K_ref.Phi[:,ipg].reshape((1, n))
        _dPhi  = np.array([K_ref.dPhi[0][:, ipg], K_ref.dPhi[1][:, ipg]])
        J = _dPhi[:, 0:3].dot(Coord_e.T)
        _weight = K_ref.w[ipg] * LA.det(J)

        Gd = LA.solve(J, _dPhi)
        eps = np.zeros((3, 3*n))
        eps[0, 0:n] = Gd[0, :] # eps_xx = \dpar{ux}{x}
        eps[1, n:2*n] = Gd[1, :] # eps_yy = \dpar{uy}{y}
        eps[2, 0:n] = Gd[1, :]
        eps[2, n:2*n] = Gd[0, :] # eps_yz = \dpar{uz}{y}+ \dpar{uy}{z}

        Phi_u = np.zeros((2, 3*n))
        Phi_u[0, 0:n] = _Phi
        Phi_u[1, n:2*n] = _Phi

        D_0, D_1 = np.zeros((3, 3)), np.zeros((3, 3))
        D_0[0, 0:2] = 1.
        D_0[1, 0:2] = 1.
        D_1[0, 1] = -2.
        D_1[1, 0] = -2.
        D_1[2, 2] = 1.

        vk0 += _weight*LA.multi_dot([eps.T, D_0, eps])
        vk1 += _weight*LA.multi_dot([eps.T, D_1, eps])
        vm += _weight*np.dot(Phi_u.T, Phi_u)
        vc += _weight*np.dot(Phi_u.T, Gd)
        vh += _weight*np.dot(Gd.T, Gd)
        vq += _weight*np.dot(_Phi.T, _Phi)

    return vm, vk0, vk1, vh, vq, vc

def pem01_elem(coord_e, K_ref):

    n, m = K_ref.Phi.shape
    vm = np.zeros((3*n, 3*n))
    vk0 = np.zeros((3*n, 3*n))
    vk1 = np.zeros((3*n, 3*n))
    vc = np.zeros((3*n, n))
    vh = np.zeros((n, n))
    vq = np.zeros((n, n))

    X1X2 = coord_e[:,1]- coord_e[:,0]
    e_x = X1X2/LA.norm(X1X2)
    X1X3 = coord_e[:,2]- coord_e[:,0]
    e_z = np.cross(X1X2, X1X3)
    e_z /= LA.norm(e_z)
    e_y = np.cross(e_z, e_x)

    Coord_e = np.zeros((2,3))
    Coord_e[0, 1] = X1X2.dot(e_x)
    Coord_e[0, 2] = X1X3.dot(e_x)
    Coord_e[1, 2] = X1X3.dot(e_y)

    Coord_e = coord_e[0:2,:]

    for ipg in range(m):
        _Phi = K_ref.Phi[:,ipg].reshape((1, n))
        _dPhi  = np.array([K_ref.dPhi[0][:, ipg], K_ref.dPhi[1][:, ipg]])
        J = _dPhi[:, 0:3].dot(Coord_e.T)
        _weight = K_ref.w[ipg] * LA.det(J)

        Gd = LA.solve(J, _dPhi)
        eps = np.zeros((3, 3*n))
        eps[0, 0:n] = Gd[0, :] # eps_xx = \dpar{ux}{x}
        eps[1, n:2*n] = Gd[1, :] # eps_yy = \dpar{uy}{y}
        eps[2, 0:n] = Gd[1, :]
        eps[2, n:2*n] = Gd[0, :] # eps_yz = \dpar{uz}{y}+ \dpar{uy}{z}

        Phi_u = np.zeros((2, 3*n))
        Phi_u[0, 0:n] = _Phi
        Phi_u[1, n:2*n] = _Phi

        D_0, D_1 = np.zeros((3, 3)), np.zeros((3, 3))
        D_0[0, 0:2] = 1.
        D_0[1, 0:2] = 1.
        D_1[0, 1] = -2.
        D_1[1, 0] = -2.
        D_1[2, 2] = 1.

        vk0 += _weight*LA.multi_dot([eps.T, D_0, eps])
        vk1 += _weight*LA.multi_dot([eps.T, D_1, eps])
        vm += _weight*np.dot(Phi_u.T, Phi_u)
        vc += _weight*np.dot(Phi_u.T, Gd)
        vh += _weight*np.dot(Gd.T, Gd)
        vq += _weight*np.dot(_Phi.T, _Phi)

    return vm, vk0, vk1, vh, vq, vc
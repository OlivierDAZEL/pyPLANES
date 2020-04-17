from itertools import product
import numpy as np

def dof_p_element(_elem):
    dof, orient = dof_element(_elem, 3)
    return dof, orient


def dof_u_element(_elem):
    dof_ux, orient_ux = dof_element(_elem, 0)
    dof_uy, orient_uy = dof_element(_elem, 1)
    dof_uz, orient_uz = dof_element(_elem, 2)
    dof = dof_ux + dof_uy + dof_uz
    orient = orient_ux + orient_uy + orient_uz
    return dof, orient


def dof_element(_elem, i_field):
    order = _elem.reference_element.order
    if _elem.typ == 2:
        # Pressure dofs of the vertices
        dof = [_elem.vertices[i].dofs[i_field] for i in range(3)]
        # Pressure dofs of the three edges
        dof = dof + _elem.edges[0].dofs[i_field] + _elem.edges[1].dofs[i_field] +_elem.edges[2].dofs[i_field]
        # Pressure dofs of the face functions
        if _elem.faces[0].dofs != []:
            dof.extend(_elem.faces[0].dofs[i_field])
        # Orientation of the vertices
        orient = [1, 1, 1]
        # Orientation of the edges
        for _e, k in product(range(3), range(order-1)):
            orient.append(_elem.edges_orientation[_e]**k)
        # Orientation of the (unique) face
        orient += [1] * int((order-1)*(order-2)/2)
    elif _elem.typ == 1:
        # dof = dofs of the 2 vertices + of the edge
        dof = _elem.dofs[i_field][0:2]+_elem.dofs[i_field][2]
        orient = [1]*2 # Orientation of the vertices
        # Orientation of the edges
        orient.extend(_elem.edges_orientation[0]**np.arange(order-1))
    return dof, orient
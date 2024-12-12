import numpy
import sympy
import os
import sys
import math
import itkwidgets
import vtk
import myVTKPythonLibrary
import ipywidgets
import pdb
import pyvista as pv

import TestUtils


# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import LIB552 as lib
#On crée un mesh 2D et on va tester de faire un champs de vecteur 3D sur ce mesh

# Values
n_cells_y_ = 1
n_cells_x_ = 1
LX_ = 1.
LY_ = 1.
h = 1.

n_vector_components = 3

E  = 1.
nu = 0.
F  = 1.

lmbda = (E*nu)/((1+nu)*(1-nu))
mu    = E/(2*(1+nu))

#J'ai changé pour le cas 3d vector field mais je suis pas du tout sur de la valeur que ça doit réellement prendre
C_p = numpy.array([[1, nu, 0],
                    [nu, 1, 0],
                    [0, 0, (1-nu)/2]])

A = ((h*E)/(1- nu**2))*C_p
D = (((h**3)*E)/(12*(1- nu**2)))*C_p


# We test the function
mesh = lib.create_unit_triangle_mesh()

#Finite element
finite_element = lib.FiniteElement_Triangle_P2() #This is a 2D structure by nature
print("""Finite element:""",finite_element)

import sympy


def repr_sym_expr_unit_triangle(expr, matrix=False):
    """
    Évalue et affiche une expression, une liste d'expressions ou une matrice SymPy.
    Si une matrice est fournie, elle affiche aussi sa forme (shape).
    
    Parameters:
    - expr : sympy.Expr, list[sympy.Expr], or sympy.Matrix
    - matrix : bool, optionnel (True si expr est une matrice)
    
    Returns:
    - None
    """
    # Déclaration des variables symboliques
    n00, n01, n10, n11, n20, n21 = sympy.symbols('n00 n01 n10 n11 n20 n21')

    # Définir des valeurs spécifiques pour les variables
    values = {n00: 0, n01: 0, n10: 1, n11: 0, n20: 0, n21: 1}

    if matrix:
        # Si l'entrée est une matrice, appliquez la substitution à chaque élément
        evaluated_matrix = expr.subs(values).applyfunc(lambda x: x.evalf())
        
        # Affichage de la forme de la matrice
        rows, cols = evaluated_matrix.shape
        print(f"Shape: {rows}x{cols}")
        print()
        
        # Affichage propre de la matrice
        print("Matrice évaluée après substitution :")
        sympy.pprint(evaluated_matrix, use_unicode=True)
        print()
    else:
        # Si l'entrée est une expression ou une liste d'expressions
    
        evaluated_expr = [e.subs(values) for e in expr]
        numerical_results = [e.evalf() for e in evaluated_expr]
        
        # Affichage des résultats pour chaque expression
        for i, (eval_expr, num_res) in enumerate(zip(evaluated_expr, numerical_results)):
            print(eval_expr)
            # print("\n")
            

# repr_sym_ex(finite_element)

#Vector element
vector_element = lib.VectorElement(finite_element, n_components = n_vector_components, ordering="point-wise")
print("""Vector element:""",vector_element)

# Dof manager
dof_manager = lib.DofManager(
mesh=mesh,
finite_element=vector_element)

# Number of dofs
dof_manager.n_dofs = 6*mesh.n_nodes

# Dofs connectivity, i.e., for each cell, the global dofs indexes
# Update to handle P2 interpolation (including midpoints for edges)
dof_manager.local_to_global = numpy.empty((mesh.n_cells, 6*3), dtype=int)  # 6 nodes per cell for P2 interpolation

for i in range(mesh.n_cells):
    # Get the indices of the nodes for the current cell (triangle)
    p, q, r = tuple(mesh.cells_nodes[i])
    
    # Midpoint nodes for each edge of the triangle (assuming cells_mid_edges stores midpoint indices)
    mp_pq = mesh.cells_mid_edges[i, 0]  # Midpoint of edge (p, q)
    mp_qr = mesh.cells_mid_edges[i, 1]  # Midpoint of edge (q, r)
    mp_rp = mesh.cells_mid_edges[i, 2]  # Midpoint of edge (r, p)
    
    # Local-to-global mapping for the P2 elements (3 vertices + 3 midpoints)
    dof_manager.local_to_global[i, :] = [
        3*p, 3*p + 1, 3*p + 2,  # Vertex p
        3*q, 3*q + 1, 3*q + 2,  # Vertex q
        3*r, 3*r + 1, 3*r + 2,  # Vertex r
        3*mp_pq, 3*mp_pq + 1, 3*mp_pq + 2,  # Midpoint (p, q)
        3*mp_qr, 3*mp_qr + 1, 3*mp_qr + 2,  # Midpoint (q, r)
        3*mp_rp, 3*mp_rp + 1, 3*mp_rp + 2   # Midpoint (r, p)
    ]


# # Dofs coordinates (you can comment out these lines after verifying your code)
dof_manager.set_dofs_coords()
# print("Dofs coordinates",dof_manager.dofs_coords)

#################
##Boundary conditions
#################

#Dirichlet boundary conditions
# We initialize the list of dofs to constrain
boundary_dofs_idx = []

# As well as the list of associated imposed values
boundary_dofs_vals = []

# We loop over all dofs
for k_dof in range(dof_manager.n_dofs):
    # We get the position of the current dof (in cartesian coordinates)
    x,y = dof_manager.dofs_coords[k_dof]
    
    # If the current dof is on the x = 0 side
    if math.isclose(x, 0.):
        if k_dof % 3 in [0,1,2]: #ux = 0, uy =1, uz = 2
            # We add it to the list of dofs to constrain       
            boundary_dofs_idx+=[k_dof]
        
            # And we add the imposed value to the associated list
            boundary_dofs_vals+=[0.]


print("boundary_dofs_idx:", boundary_dofs_idx)
# print("boundary_dofs_vals:", boundary_dofs_vals)

#Neumann boundary conditions
#I want to push vertically on top edge and right edge

#composante en x du 1er noeud de l'edge
#mesh.nodes[mesh.edges_nodes[k_edge,0],0]
#composante en x du 2eme noeud de l'edge
#mesh.nodes[mesh.edges_nodes[k_edge,1],0]

imposed_edges_idx = []

# # Top edge
# top_edges_idx = [k_edge for k_edge in range(mesh.n_edges)
#                  if (math.isclose(mesh.nodes[mesh.edges_nodes[k_edge, 0], 1], LY_)
#                      and math.isclose(mesh.nodes[mesh.edges_nodes[k_edge, 1], 1], LY_))]
# imposed_edges_idx.extend(top_edges_idx)

# Right edge
right_edges_idx = [k_edge for k_edge in range(mesh.n_edges)
                   if (math.isclose(mesh.nodes[mesh.edges_nodes[k_edge, 0], 0], LX_)
                       and math.isclose(mesh.nodes[mesh.edges_nodes[k_edge, 1], 0], LX_))]
imposed_edges_idx.extend(right_edges_idx)
print("imposed_edges_idx:", imposed_edges_idx)

def main():
    #Visualization of the mesh
    # TestUtils.visualize(mesh, edges_ids = False, point_coords = False, nodes_ids = True)

    # print("""P:""",vector_element.sym_P)
    # print("""B:""",vector_element.sym_B)
    # print("phi")
    # print(len(finite_element.sym_phi))
    # # repr_sym_expr_unit_triangle(finite_element.sym_phi)
    # print("dphi")
    # repr_sym_expr_unit_triangle(finite_element.sym_dphi.tomatrix(), matrix = True)
    # print("P")
    # repr_sym_expr_unit_triangle(vector_element.sym_P.tomatrix().T, matrix = True)
    # print("B")
    # repr_sym_expr_unit_triangle(vector_element.sym_B.tomatrix().T, matrix = True)

    

    #################
    ##ASSEMBLY
    #################

    # Symbolic integration and compilation (for fast execution) of the elementary stiffness matrix
    vector_element.init_get_B_B_and_P_P_int(coeff_B = A, coeff_P = D)

    # print("sym_B_B_and_P_P")
    # repr_sym_expr_unit_triangle(vector_element.sym_B_B_and_P_P.tomatrix(), matrix = True)

    # Symbolic integration and compilation (for fast execution) of a null volume force
    vector_element.init_get_phi_int(coeff=[0.,0.,0.]) #adapt it to have a 3d volume force ?

    # Assembly
    KK = numpy.zeros((dof_manager.n_dofs, dof_manager.n_dofs))
    FF = numpy.zeros(dof_manager.n_dofs)
    lib.assemble_system_w_constraints(
        mesh=mesh,
        finite_element=vector_element,
        get_loc_mat=vector_element.get_B_B_and_P_P_int,
        get_loc_vec=vector_element.get_phi_int,
        dof_manager=dof_manager,
        prescribed_dofs_idx=boundary_dofs_idx,
        prescribed_dofs_vals=boundary_dofs_vals,
        mat=KK,
        vec=FF)
    
    print("KK:",KK)
    print("FF:", FF)

    #Plot the sparsity pattern
    TestUtils.plot_sparsity_pattern(KK, title=f"Sparsity Pattern of KK\n({KK.shape[0]} x {KK.shape[1]})")


if __name__ == "__main__":
    main()
    print("End.")
    # pass
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
h = 0.1

n_vector_components = 3

E  = 1.
nu = 0.3
F  = 1.

lmbda = (E*nu)/((1+nu)*(1-nu))
mu    = E/(2*(1+nu))

#J'ai changé pour le cas 3d vector field mais je suis pas du tout sur de la valeur que ça doit réellement prendre
C_p = numpy.array([[1, nu, 0],
                    [nu, 1, 0],
                    [0, 0, (1-nu)/2]])

A = ((h*E)/(1- nu**2))*C_p
D = (((h**3)*E)/(12*(1- nu**2)))*C_p

#2D mesh
def create_plate_triangular_mesh(
        LX,
        LY,
        n_cells_x=1,
        n_cells_y=1):
    """
    Creates a plate mesh of triangles.

    Args:
        n_cells_x (uint): The number of cells in x direction.
        n_cells_y (uint): The number of cells in y direction.
        LX (float): The length of the plate in x direction.
        LY (float): The length of the plate in y direction.

    Returns:
        mesh (LIB552.Mesh): The mesh.
    """
    # We only consider the case of a disc embedded in the 2D plane, not for instance in the 3D space.
    dim = 2
    
    # How many nodes are there in the mesh?
    n_nodes = (n_cells_x+1)*(n_cells_y+1) 
    
    # We create the list of nodes coordinates
    nodes = numpy.empty(
        (n_nodes, dim),
        dtype=float)

    # We define the first node
    k_node = 0
    
    # We define the following nodes, by looping over the x positions
    for k_x in range(0, n_cells_x+1):
        # print("k_x = "+str(k_x))
        
        # What is the current x position?
        x =k_x * LX/n_cells_x

        # We define the following nodes, by looping over the y positions        
        for k_y in range(0, n_cells_y+1):
            # print("k_c = "+str(k_c))
            
            # What is the current circumferential position (in radians)?
            y = k_y * LY/n_cells_y
            
            # What is the current node number?
            k_node = k_x*(n_cells_y+1) + k_y
            # print("k_node = "+str(k_node))

            # We define the current node
            nodes[k_node,:] = [x,y]
    
    # print(nodes)
    
    # We define the cell structure (see LIB552/Mesh.py)
    cell = lib.Cell_Triangle()
    
    # How many cells are there in the mesh?
    n_cells = 2*n_cells_x*n_cells_y

    # We create the mesh connectivity, i.e., the list of node numbers of each cell
    cells_nodes = numpy.empty(
        (n_cells, cell.n_nodes),
        dtype=int)

    
    for k_x in range(0, n_cells_x):
        for k_y in range(0, n_cells_y):
            # print("k_c = "+str(k_c))
            
            # What is the current cell number?
            k_cell = k_x*n_cells_y + k_y
            # print("k_cell = "+str(k_cell))

            # What are the numbers of the four nodes of the quadrangle?
            n1 = k_x*(n_cells_y+1) + k_y
            n2 = k_x*(n_cells_y+1) + k_y + 1

            n3 = (k_x+1)*(n_cells_y+1) + k_y
            n4 = (k_x+1)*(n_cells_y+1) + k_y + 1
        
            cells_nodes[2*k_cell  , :] = [n1,n3,n4]
            cells_nodes[2*k_cell+1, :] = [n1,n4,n2]

    # print(cells_nodes)
    
    # We return the mesh structure
    return lib.Mesh(
        dim=dim,
        nodes=nodes,
        cell=cell,
        cells_nodes=cells_nodes)

# We test the function
mesh = create_plate_triangular_mesh(
    LX=LX_,
    LY=LY_,
    n_cells_x=n_cells_x_,
    n_cells_y=n_cells_y_)

# #2D vector field
# finite_element = lib.FiniteElement_Triangle_P1()
# vector_element = lib.VectorElement(finite_element, ordering="point-wise")

#3D vector field
finite_element = lib.FiniteElement_Triangle_P1() #This is a 2D structure by nature
print("""Finite element:""",finite_element)
vector_element = lib.VectorElement(finite_element, n_components = n_vector_components, ordering="point-wise")
print("""Vector element:""",vector_element)

# Dof manager
dof_manager = lib.DofManager(
mesh=mesh,
finite_element=vector_element)

# Number of dofs
dof_manager.n_dofs = 3*mesh.n_nodes

# Dofs connectivity, i.e., for each cell, the global dofs indexes
dof_manager.local_to_global = numpy.empty((mesh.n_cells,3*3), dtype=int)
for i in range(mesh.n_cells):
    p,q,r = tuple(mesh.cells_nodes[i])
    dof_manager.local_to_global[i,:] = [3*p,3*p + 1, 3*p + 2,         #Due to the point-wise ordering
                                        3*q,3*q + 1, 3*q + 2,
                                        3*r,3*r + 1, 3*r + 2]


# # Nodes vs Dofs connectivity (you can comment out these lines after verifying your code)
# print("Nodes connectivity",mesh.cells_nodes)
# print("Dofs connectivity",dof_manager.local_to_global)
# print("Edges connectivity",dof_manager.edges_connectivity)

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

    #################
    ##ASSEMBLY
    #################

    # Symbolic integration and compilation (for fast execution) of the elementary stiffness matrix
    vector_element.init_get_B_B_and_P_P_int(coeff_B = A, coeff_P = D)
    
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

    # Symbolic integration and compilation (for fast execution) of the surface force
    vector_element.init_get_phi_edge_int(coeff=[0.,0.,F])

    # Assembly
    lib.assemble_vector_from_edge_integral(
        mesh=mesh,
        finite_element=vector_element,
        get_loc_vec=vector_element.get_phi_edge_int,
        dof_manager=dof_manager,
        imposed_edges_idx=imposed_edges_idx,
        vec=FF)
    print("FF:",FF)

    ##########
    # Solve
    ##########
    det = numpy.linalg.det(KK)
    print("det(KK):",det)
    if det == 0:
        raise ValueError("KK is singular")
    else:
        U = numpy.linalg.solve(KK, FF)
        print("U:",U) 

    ##############
    ##Visualization
    ##############

    TestUtils.visualize_U(U, mesh, vector_element, dof_manager)

if __name__ == "__main__":
    main()
    print("End.")
    # pass
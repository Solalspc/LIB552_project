import numpy
import sympy
import os
import sys
import math
import itkwidgets
import vtk
import myVTKPythonLibrary
import ipywidgets


# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import LIB552 as lib
#On crée un mesh 2D et on va tester de faire un champs de vecteur 3D sur ce mesh

# Values
n_cells_r_ = 2
n_cells_c_ = 2
R_ = 1
n_vector_components = 3

R  = 1.
L  = 3.
E  = 1.
nu = 0.3
F  = 0.5

lmbda = (E*nu)/((1+nu)*(1-nu))
mu    = E/(2*(1+nu))

#J'ai changé pour le cas 3d vector field mais je suis pas du tout sur de la valeur que ça doit réellement prendre
K = numpy.zeros((n_vector_components,n_vector_components,n_vector_components,n_vector_components))
# K[0,0,:,:] = [[lmbda + 2*mu,0],[0,lmbda]]
# K[1,1,:,:] = [[0,lmbda],[lmbda + 2*mu,0]]
# K[0,1,:,:] = [[0,mu],[mu,0]]
# K[1,0,:,:] = [[0,mu],[mu,0]]
for i in range(n_vector_components):
    for j in range(n_vector_components):
        for k in range(n_vector_components):
            for l in range(n_vector_components):
                K[i, j, k, l] = (
                    lmbda * (i == j) * (k == l) +
                    mu * ((i == k) * (j == l) + (i == l) * (j == k))
                )
#2D mesh
def create_quarter_disc_triangular_mesh(
        R=1.,
        n_cells_r=1,
        n_cells_c=1):
    """
    Creates a quarter disc mesh of triangles.

    Args:
        R (float): The disc radius.
        n_cells_r (uint): The number of cells in radial direction.
        n_cells_c (uint): The number of cells in circumferential direction.

    Returns:
        mesh (LIB552.Mesh): The mesh.
    """
    # We only consider the case of a disc embedded in the 2D plane, not for instance in the 3D space.
    dim = 2
    
    # How many nodes are there in the mesh?
    n_nodes = n_cells_r*(n_cells_c+1)+1
    
    # We create the list of nodes coordinates
    nodes = numpy.empty(
        (n_nodes, dim),
        dtype=float)

    # We define the first node
    k_node = 0
    nodes[k_node,:] = [0.,0.]
    
    # We define the following nodes, by looping over the radial positions
    for k_r in range(1, n_cells_r+1):
        # print("k_r = "+str(k_r))
        
        # What is the current radial position?
        r = k_r * R/n_cells_r
        # print("r = "+str(r))
        
        # And looping over the circumferential positions 
        for k_c in range(0, n_cells_c+1):
            # print("k_c = "+str(k_c))
            
            # What is the current circumferential position (in radians)?
            c = k_c*numpy.pi/(2*n_cells_c)
            # print("c = "+str(c*180/math.pi))
            
            # What are the current cartesian coordinates?
            x = r*numpy.cos(c)
            y = r*numpy.sin(c)

            # What is the current node number?
            k_node = (k_r-1)*(n_cells_c+1)+k_c+1
            # print("k_node = "+str(k_node))

            # We define the current node
            nodes[k_node,:] = [x,y]
    
    # print(nodes)
    
    # We define the cell structure (see LIB552/Mesh.py)
    cell = lib.Cell_Triangle()
    
    # How many cells are there in the mesh?
    n_cells = (2*n_cells_r-1)*n_cells_c

    # We create the mesh connectivity, i.e., the list of node numbers of each cell
    cells_nodes = numpy.empty(
        (n_cells, cell.n_nodes),
        dtype=int)

    # We define the first row of cells, which has only n_cells_c cells,
    #  by looping over the circumferential positions
    for k_c in range(0, n_cells_c):
        # print("k_c = "+str(k_c))
        
        # What is the current cell number?
        k_cell = k_c
        # print("k_cell = "+str(k_cell))

        # We define the current cell
        cells_nodes[k_cell, :] = [0,k_c+1,k_c+2]

    # We define the following rows of cells, which have 2 x n_cells_c cells,
    #  by looping over the radial positions
    for k_r in range(1, n_cells_r):
        # print("k_r = "+str(k_r))

        # And looping over the circumferential positions
        for k_c in range(0, n_cells_c):
            # print("k_c = "+str(k_c))
            
            # For each quadrangle, we will create two triangle
            # What is the first current cell number?
            k_cell = n_cells_c + (k_r-1)*2*n_cells_c + 2*k_c
            # print("k_cell = "+str(k_cell))
            
            # What are the numbers of the four nodes of the quadrangle?
            n1 = (k_r-1)*(n_cells_c+1)+k_c+1
            n2 = (k_r-1)*(n_cells_c+1)+k_c+2
            n3 = k_r*(n_cells_c+1)+k_c+1
            n4 = k_r*(n_cells_c+1)+k_c+2
            
            # We define the two current cells
            cells_nodes[k_cell  , :] = [n1,n3,n4]
            cells_nodes[k_cell+1, :] = [n1,n4,n2]

    # print(cells_nodes)
    
    # We return the mesh structure
    return lib.Mesh(
        dim=dim,
        nodes=nodes,
        cell=cell,
        cells_nodes=cells_nodes)

# We test the function
mesh = create_quarter_disc_triangular_mesh(
    R=R_,
    n_cells_r=n_cells_r_,
    n_cells_c=n_cells_c_)

 # number of vector components
    

# #2D vector field
# finite_element = lib.FiniteElement_Triangle_P1()
# vector_element = lib.VectorElement(finite_element, ordering="point-wise")


def main():
    #3D vector field
    finite_element = lib.FiniteElement_Triangle_P1() #This is a 2D structure by nature
    finite_element

    vector_element = lib.VectorElement(finite_element, n_components = n_vector_components, ordering="point-wise")
    vector_element

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

    # # Dofs coordinates (you can comment out these lines after verifying your code)
    dof_manager.set_dofs_coords()
    # print("Dofs coordinates",dof_manager.dofs_coords)

    #################
    ##Boundary conditions
    #################

    #Neumann boundary conditions
    # We initialize the list of dofs to constrain
    boundary_dofs_idx = []

    # As well as the list of associated imposed values
    boundary_dofs_vals = []

    # We loop over all dofs
    for k_dof in range(dof_manager.n_dofs):
        # We get the position of the current dof (in cartesian coordinates)
        x_ = dof_manager.dofs_coords[k_dof]
        
        # We compute its radial position
        r_ = numpy.sqrt(x_[0]**2+x_[1]**2)

        # If the current dof is on the external boundary
        if math.isclose(r_, R_):
            # We add it to the list of dofs to constrain
            boundary_dofs_idx+=[k_dof]
            
            # And we add the imposed value to the associated list
            boundary_dofs_vals+=[0]

    print("boundary_dofs_idx:", boundary_dofs_idx)
    print("boundary_dofs_vals:", boundary_dofs_vals)


    #################
    ##ASSEMBLY
    #################

    # Symbolic integration and compilation (for fast execution) of the elementary stiffness matrix
    vector_element.init_get_B_B_int(coeff=K)

    # Symbolic integration and compilation (for fast execution) of a null volume force
    vector_element.init_get_phi_int(coeff=[0.,0.,0.]) #adapt it to have a 3d volume force ?

    # Assembly
    KK = numpy.zeros((dof_manager.n_dofs, dof_manager.n_dofs))
    FF = numpy.zeros(dof_manager.n_dofs)
    lib.assemble_system_w_constraints(
        mesh=mesh,
        finite_element=vector_element,
        get_loc_mat=vector_element.get_B_B_int,
        get_loc_vec=vector_element.get_phi_int,
        dof_manager=dof_manager,
        prescribed_dofs_idx=boundary_dofs_idx,
        prescribed_dofs_vals=boundary_dofs_vals,
        mat=KK,
        vec=FF)
    print("KK:",KK)
    print("FF:", FF)

    # Symbolic integration and compilation (for fast execution) of the surface force
    vector_element.init_get_phi_edge_int(coeff=[F,0.,0.])

    # Assembly
    lib.assemble_vector_from_edge_integral(
        mesh=mesh,
        finite_element=vector_element,
        get_loc_vec=vector_element.get_phi_edge_int,
        dof_manager=dof_manager,
        imposed_edges_idx=[1], #je veux juste charger le centre
        vec=FF)
    print("FF:",FF)

    ##########
    # Solve
    ##########

    U = numpy.linalg.solve(KK, FF)
    # print("U:",U) 

    ##############
    ##Visualization
    ##############

    # Convert to VTK
    dof_manager.set_inverse_connectivity()
    disp_ugrid = lib.field_to_ugrid(U, mesh, vector_element, dof_manager, "disp") 
    #Pas encore adapté pour la 3D

    # Warp
    warp_filter = vtk.vtkWarpVector()
    warp_filter.SetInputData(disp_ugrid)
    warp_filter.SetScaleFactor(1.)
    warp_filter.Update()
    warp_ugrid = warp_filter.GetOutput()

    # itkwidget viewer
    viewer = itkwidgets.view(geometries=[warp_ugrid])

    # ipywidget slider
    def warp(factor=1.):
        warp_filter.SetScaleFactor(factor)
        warp_filter.Update()
        viewer.geometries = [warp_ugrid]
    slider = ipywidgets.interactive(warp, factor=(0., 10., 0.1), continuous_update=True)

    ipywidgets.VBox([viewer, slider])
    print("END")

if __name__ == "__main__":
    main()
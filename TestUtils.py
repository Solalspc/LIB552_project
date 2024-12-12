import numpy as np
import matplotlib.pyplot as plt
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import LIB552 as lib

def plot_sparsity_pattern(matrix, title="Sparsity Pattern"):
    """
    Plots the sparsity pattern of a given NumPy matrix.

    Parameters:
        matrix (np.ndarray): The NumPy matrix to visualize.
        title (str): The title of the plot.
    """
    if not isinstance(matrix, np.ndarray):
        raise ValueError("Input must be a NumPy array.")
    
    # Get the full grid of indices
    rows, cols = np.indices(matrix.shape)
    rows, cols = rows.ravel(), cols.ravel()

    # Get non-zero positions
    non_zero_rows, non_zero_cols = np.nonzero(matrix)

    # Create the sparsity pattern plot
    plt.figure(figsize=(8, 8))
    plt.scatter(cols, rows, color="lightgray", s=10, label="Zero elements")
    plt.scatter(non_zero_cols, non_zero_rows, color="black", s=10, label="Non-zero elements")
    plt.title(title)
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    plt.gca().invert_yaxis()  # Invert y-axis for better matrix view
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.show()

def visualize_U(U, mesh, vector_element, dof_manager):
    # Convert to VTK (get ugrid)
    dof_manager.set_inverse_connectivity()
    disp_ugrid = lib.field_to_ugrid(U, mesh, vector_element, dof_manager, "disp")

    # Check if ugrid is vtkUnstructuredGrid
    print(type(disp_ugrid))  # Should be vtkUnstructuredGrid
    if isinstance(disp_ugrid, vtk.vtkUnstructuredGrid):
        surface_filter = vtk.vtkDataSetSurfaceFilter()
        surface_filter.SetInputData(disp_ugrid)
        surface_filter.Update()
        warp_poly_data = surface_filter.GetOutput()
    else:
        warp_poly_data = disp_ugrid  # If already vtkPolyData

    # Warp (scale factor for visualization)
    warp_filter = vtk.vtkWarpVector()
    warp_filter.SetInputData(warp_poly_data)
    warp_filter.SetScaleFactor(1.)
    warp_filter.Update()
    warp_ugrid = warp_filter.GetOutput()

    # VTK Renderer and RenderWindow setup
    ren = vtk.vtkRenderer()
    ren_win = vtk.vtkRenderWindow()
    ren_win.AddRenderer(ren)
    
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(ren_win)
    
    # Mapper and Actor
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(warp_ugrid)
    
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    
    ren.AddActor(actor)
    ren.SetBackground(1.0, 1.0, 1.0)  # Set background to white
    ren_win.Render()
    
    iren.Start()

def visualize(mesh, nodes_ids = False, cells_ids = False, dofs_ids = False, edges_ids = False, point_coords = False): 
    ugrid = lib.mesh_to_ugrid(mesh)
    pv_ugrid = pv.wrap(ugrid)


    edges_nodes_coords = [(mesh.nodes[mesh.edges_nodes[k_edge, 0]],mesh.nodes[mesh.edges_nodes[k_edge, 1]]) for k_edge in range(mesh.n_edges)]

    plotter = pv.Plotter()
    plotter.add_mesh(pv_ugrid, color="lightblue", show_edges=True)
    
    if nodes_ids :
        plotter.add_point_labels(pv_ugrid.points, 
                                labels=[str(i) for i in range(pv_ugrid.n_points)], 
                                point_size=8, 
                                font_size=10)
        
    if cells_ids :    
        plotter.add_point_labels(pv_ugrid.cell_centers().points, 
                                labels=[str(i) for i in range(pv_ugrid.n_cells)], 
                                point_size=8, 
                                font_size=10)
        
    if edges_ids :
        # Compute midpoints of edges
        edge_midpoints = [((edge[0] + edge[1]) / 2).tolist() + [0] for edge in edges_nodes_coords]
        edge_midpoints = np.array(edge_midpoints)
        
        # Add labels for edges
        plotter.add_point_labels(edge_midpoints, 
                                 labels=[str(i) for i in range(len(edges_nodes_coords))], 
                                 point_size=8, 
                                 font_size=10)
        
    if point_coords:
        # Add labels with coordinates for each point
        coords = [f"({x:.2f}, {y:.2f}, {z:.2f})" for x, y, z in pv_ugrid.points]
        plotter.add_point_labels(pv_ugrid.points, 
                                 labels=coords, 
                                 point_size=8, 
                                 font_size=10, 
                                 text_color="yellow")

    plotter.show_axes()
    plotter.set_background("black")
    plotter.add_bounding_box(color="white")
    plotter.show()

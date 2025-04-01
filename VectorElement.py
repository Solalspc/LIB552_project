#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2020-2022                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################


import numpy
import sympy
import os
import sys
# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import LIB552 as lib


################################################################################


class VectorElement(lib.FiniteElement):
    def __init__(self, finite_element, n_components=None, ordering="component-wise"):
        self.finite_element = finite_element
        self.dim = self.finite_element.dim
        self.n_components = n_components if n_components is not None else self.dim
        self.sym_x = self.finite_element.sym_x
        self.shape = self.finite_element.shape
        self.n_nodes = self.finite_element.n_nodes
        self.sym_nodes = self.finite_element.sym_nodes
        self.n_edges = self.finite_element.n_edges
        self.interpolation = self.finite_element.interpolation
        self.n_points = self.finite_element.n_points
        self.sym_points = self.finite_element.sym_points

        self.n_dofs = self.n_components * self.finite_element.n_dofs
        self.dofs_component     = numpy.empty((self.n_dofs), dtype=int)
        self.dofs_component_dof = numpy.empty((self.n_dofs), dtype=int)
        self.ordering = ordering
        if (self.ordering == "component-wise"):
            for k_component in range(self.n_components):
                for k_component_dof in range(self.finite_element.n_dofs):
                    k_dof = k_component*self.finite_element.n_dofs + k_component_dof
                    self.dofs_component[k_dof]     = k_component
                    self.dofs_component_dof[k_dof] = k_component_dof
        elif (self.ordering == "point-wise"):
            for k_component_dof in range(self.finite_element.n_dofs):
                for k_component in range(self.n_components):
                    k_dof = k_component_dof*self.n_components + k_component
                    self.dofs_component[k_dof]     = k_component
                    self.dofs_component_dof[k_dof] = k_component_dof
        # print (self.dofs_component)
        # print (self.dofs_component_dof)
        self.dofs_attachement     = [self.finite_element.dofs_attachement    [self.dofs_component_dof[k_dof]] for k_dof in range(self.n_dofs)]
        self.dofs_attachement_idx = [self.finite_element.dofs_attachement_idx[self.dofs_component_dof[k_dof]] for k_dof in range(self.n_dofs)]

        self.sym_phi = self.scalar_to_vector_sympy_array(self.finite_element.sym_phi)

    def scalar_to_vector_sympy_array(self, scalar_array):
        assert (scalar_array.shape == (self.finite_element.n_dofs,))
        vector_array = sympy.MutableDenseNDimArray.zeros(self.n_dofs, self.n_components)
        for k_dof in range(self.n_dofs):
            k_component     = self.dofs_component[k_dof]
            k_component_dof = self.dofs_component_dof[k_dof]
            vector_array[k_dof, k_component] = scalar_array[k_component_dof,]
        return vector_array

    def init_get_dofs_coords(self):
        """Initializes the (efficient) computation of dofs coordinates."""
        self.finite_element.init_get_dofs_coords()

    def get_dof_coords(self, mesh, k_cell, k_cell_dof):
        """
        Returns the coordinates of a given dof.

        Args:
            mesh (LIB552.Mesh) The mesh.
            k_cell (uint): The cell index.
            k_cell_dof (uint): The local dof index.
        """
        return self.finite_element.get_dof_coords(mesh, k_cell, self.dofs_component_dof[k_cell_dof])

    def init_get_phi_int(self, coeff, n=0):
        """
        Initializes the (efficient) computation of the shape functions element integral.
        This function directly uses the equivalent function within the finite element.
        The coefficient, which is a vector here, is stored and used within the get_phi_int function.
        """
        self.finite_element.init_get_phi_int(coeff=1, n=n)
        self.phi_int_coeff = coeff

    def get_phi_int(self, mesh, k_cell, loc_vec):
        """(Efficient) computation of shape functions element integral."""
        if   (self.ordering == "component-wise"):
            loc_vec[0:self.finite_element.n_dofs] = self.finite_element._get_phi_int(*mesh.get_cell_nodes_coords(k_cell))
            for k_component in range(1,self.n_components):
                loc_vec[k_component*self.finite_element.n_dofs:(k_component+1)*self.finite_element.n_dofs] = loc_vec[0:self.finite_element.n_dofs]
            for k_component in range(self.n_components):
                loc_vec[k_component*self.finite_element.n_dofs:(k_component+1)*self.finite_element.n_dofs] *= self.phi_int_coeff[k_component]
        
        elif (self.ordering == "point-wise"):
            loc_vec[0::self.n_components] = self.finite_element._get_phi_int(*mesh.get_cell_nodes_coords(k_cell)).reshape(-1)
            # loc_vec[0::self.n_components] = self.finite_element._get_phi_int(*mesh.get_cell_nodes_coords(k_cell))
            for k_component in range(1,self.n_components):
                loc_vec[k_component::self.n_components] = loc_vec[0::self.n_components]
            for k_component in range(0,self.n_components):
                loc_vec[k_component::self.n_components] *= self.phi_int_coeff[k_component]

    def init_get_phi_edge_int(self, coeff, n=0):
        """
        Initializes the (efficient) computation of the shape functions edges integrals.
        This function directly uses the equivalent function within the finite element.
        The coefficient, which is a vector here, is stored and used within the get_phi_int function.
        """
        self.finite_element.init_get_phi_edge_int(coeff=1, n=n)
        self.phi_edge_int_coeff = coeff

    def get_phi_edge_int(self, mesh, k_cell, k_cell_edge, loc_vec):
        """(Efficient) computation of shape functions element integral."""
        if   (self.ordering == "component-wise"):
            loc_vec[0:self.finite_element.n_dofs] = self.finite_element._get_phi_edge_int[k_cell_edge](*mesh.get_cell_nodes_coords(k_cell))
            for k_component in range(1,self.n_components):
                loc_vec[k_component*self.finite_element.n_dofs:(k_component+1)*self.finite_element.n_dofs] = loc_vec[0:self.finite_element.n_dofs]
            for k_component in range(self.n_components):
                loc_vec[k_component*self.finite_element.n_dofs:(k_component+1)*self.finite_element.n_dofs] *= self.phi_edge_int_coeff[k_component]
        
        elif (self.ordering == "point-wise"):
            loc_vec[0::self.n_components] = self.finite_element._get_phi_edge_int[k_cell_edge](*mesh.get_cell_nodes_coords(k_cell))
            for k_component in range(1,self.n_components):
                loc_vec[k_component::self.n_components] = loc_vec[0::self.n_components]
            for k_component in range(0,self.n_components):
                loc_vec[k_component::self.n_components] *= self.phi_edge_int_coeff[k_component]
    
    def init_get_phi_phi_int(self, coeff=1, n=0):
        """
        Initializes the (efficient) computation of the shape functions products element integral.
        This function directly uses the equivalent function within the finite element.
        """
        self.finite_element.init_get_phi_phi_int(coeff=coeff, n=n)

    def get_phi_phi_int(self, mesh, k_cell, loc_mat):
        """(Efficient) computation of shape functions products element integral."""
        if   (self.ordering == "component-wise"):
            loc_mat[0:self.finite_element.n_dofs, 0:self.finite_element.n_dofs] = self.finite_element._get_phi_phi_int(*mesh.get_cell_nodes_coords(k_cell))
            for k_component in range(1,self.n_components):
                loc_mat[k_component*self.finite_element.n_dofs:(k_component+1)*self.finite_element.n_dofs, k_component*self.finite_element.n_dofs:(k_component+1)*self.finite_element.n_dofs] = loc_mat[0:self.finite_element.n_dofs, :self.finite_element.n_dofs]
        elif (self.ordering == "point-wise"):
            loc_mat[0::self.n_components,0::self.n_components] = self.finite_element._get_phi_phi_int(*mesh.get_cell_nodes_coords(k_cell))
            for k_component in range(1,self.n_components):
                loc_mat[k_component::self.n_components,k_component::self.n_components] = loc_mat[0::self.n_components,0::self.n_components]

    def _init_sym_B(self):
        """Computes the (symbolic) symmetric gradient of the shape functions, and stores them as a (n_dofs x dim x dim) sympy Array."""
        if (self.dim == 2) and (self.n_components == 2) :
            print("Only implemented for 2D vectors on 2D elements.")
            self.finite_element._init_sym_dphi()
            self.sym_B = sympy.MutableDenseNDimArray.zeros(self.n_dofs, self.dim, self.dim)
            if   (self.ordering == "component-wise"):
                for k_component in range(self.n_components):
                    self.sym_B[:self.finite_element.n_dofs,k_component,k_component] = self.finite_element.sym_dphi[:,0]
                self.sym_B[self.finite_element.n_dofs:,1,1] = self.finite_element.sym_dphi[:,1]
                self.sym_B[self.finite_element.n_dofs:,2,1] = self.finite_element.sym_dphi[:,1]
                self.sym_B[:self.finite_element.n_dofs,0,1] = self.finite_element.sym_dphi[:,1]/2
                self.sym_B[self.finite_element.n_dofs:,0,1] = self.finite_element.sym_dphi[:,0]/2
                self.sym_B[:self.finite_element.n_dofs,1,0] = self.finite_element.sym_dphi[:,1]/2
                self.sym_B[self.finite_element.n_dofs:,1,0] = self.finite_element.sym_dphi[:,0]/2
            elif (self.ordering == "point-wise"):
                # MG20201111: This should work, right? Cf. https://github.com/sympy/sympy/issues/20410
                # self.sym_B[0::2,0,0] = self.finite_element.sym_dphi[:,0]
                # self.sym_B[1::2,1,1] = self.finite_element.sym_dphi[:,1]
                # self.sym_B[0::2,0,1] = self.finite_element.sym_dphi[:,1]/2
                # self.sym_B[1::2,0,1] = self.finite_element.sym_dphi[:,0]/2
                # self.sym_B[0::2,1,0] = self.finite_element.sym_dphi[:,1]/2
                # self.sym_B[1::2,1,0] = self.finite_element.sym_dphi[:,0]/2
                for k_dof in range(self.finite_element.n_dofs):
                    self.sym_B[2*k_dof  ,0,0] = self.finite_element.sym_dphi[k_dof,0]
                    self.sym_B[2*k_dof+1,1,1] = self.finite_element.sym_dphi[k_dof,1]
                    self.sym_B[2*k_dof  ,0,1] = self.finite_element.sym_dphi[k_dof,1]/2
                    self.sym_B[2*k_dof+1,0,1] = self.finite_element.sym_dphi[k_dof,0]/2
                    self.sym_B[2*k_dof  ,1,0] = self.finite_element.sym_dphi[k_dof,1]/2
                    self.sym_B[2*k_dof+1,1,0] = self.finite_element.sym_dphi[k_dof,0]/2

        if (self.dim == 2) and (self.n_components == 3) :
            print("Only implemented for 3D vectors on 2D elements.")
            self.finite_element._init_sym_dphi() #Même dphi, les shape functions ne sont toujours que fonctions de x et y
            self.sym_B = sympy.MutableDenseNDimArray.zeros(self.n_dofs, self.n_components, self.n_components)
            if self.ordering == "component-wise":
                print("Not implemented for 3D vectors on 2D elements  with component-wise ordering.")
                # # Assigning components for component-wise DoF ordering
                # self.sym_B[:self.finite_element.n_dofs, 0, 0] = self.finite_element.sym_dphi[:, 0]  # ∂u_x / ∂x
                # self.sym_B[self.finite_element.n_dofs:2*self.finite_element.n_dofs, 1, 1] = self.finite_element.sym_dphi[:, 1]  # ∂u_y / ∂y
                # self.sym_B[2*self.finite_element.n_dofs:, 0, 2] = self.finite_element.sym_dphi[:, 0] / 2  # ∂u_z / ∂x
                # self.sym_B[2*self.finite_element.n_dofs:, 1, 2] = self.finite_element.sym_dphi[:, 1] / 2  # ∂u_z / ∂y
                
                # # Symmetric off-diagonal terms
                # self.sym_B[:self.finite_element.n_dofs, 0, 1] = self.finite_element.sym_dphi[:, 1] / 2  # (∂u_x / ∂y + ∂u_y / ∂x)/2
                # self.sym_B[self.finite_element.n_dofs:2*self.finite_element.n_dofs, 1, 0] = self.finite_element.sym_dphi[:, 0] / 2  # (∂u_y / ∂x + ∂u_x / ∂y)/2
        
            elif self.ordering == "point-wise":
            # Assigning components for point-wise DoF ordering
                for k_dof in range(self.finite_element.n_dofs):
                    # For each DoF, construct a 3x3 matrix (B_sub)
                    for k_component in range(self.n_components):
                        # Initialize the 3x3 sub-matrix for the current DoF
                        B_sub = sympy.MutableDenseNDimArray.zeros(self.n_components, self.n_components)
                        # print(f"k_dof = {k_dof}, k_component = {k_component}")
                        # Set the third row (corresponding to derivatives in the z-direction) to zero
                        B_sub[2, 0] = 0 
                        B_sub[2, 1] = 0
                        B_sub[2, 2] = 0    
                        # Diagonal terms
                        if k_component != 2: 
                            # For the diagonal components different than z, assign the corresponding derivative
                            B_sub[k_component, k_component] = self.finite_element.sym_dphi[k_dof, k_component]

                        # Shear terms (off-diagonal) - averaged terms for 2D strain
                        if k_component == 0:  # x-component (u_x)
                            # Averaged shear terms
                            B_sub[0, 1] = B_sub[1, 0] = self.finite_element.sym_dphi[k_dof, 1] / 2  # (∂u_x / ∂y + ∂u_y / ∂x)/2
                            B_sub[0, 2] = self.finite_element.sym_dphi[k_dof, 0] / 2  # ∂u_z / ∂x
                        elif k_component == 1:  # y-component (u_y)
                            # Averaged shear terms
                            B_sub[1, 0] = B_sub[0, 1] = self.finite_element.sym_dphi[k_dof, 0] / 2  # (∂u_y / ∂x + ∂u_x / ∂y)/2
                            B_sub[1, 2] = self.finite_element.sym_dphi[k_dof, 1] / 2  # ∂u_z / ∂y

                        # Assign the filled sub-matrix to the full B matrix
                        self.sym_B[3 * k_dof + k_component, :, :] = B_sub[:, :]
                
                # print("sym_B shape = ",self.sym_B.shape)
                # print("For the first node of the element:\nsym_B = ",self.sym_B[:3,:,:])

    def _init_sym_B_B(self, coeff):
        """Computes the (symbolic) products of shape functions symmetric gradients,
        and stores them as a (n_dofs x n_dofs) sympy Array."""
        
        # For 2D mesh, 2D vector case
        if self.dim == 2 and self.n_components == 2:
            assert (self.sym_B.shape == (self.n_dofs, self.dim, self.dim))
            assert (coeff.shape == (self.dim, self.dim, self.dim, self.dim))
            self.sym_B_B = sympy.tensorcontraction(sympy.tensorcontraction(
                sympy.tensorproduct(
                    self.sym_B,
                    sympy.tensorcontraction(sympy.tensorcontraction(
                        sympy.tensorproduct(
                            sympy.Array(coeff),
                            sympy.permutedims(
                                self.sym_B,
                                (1, 2, 0)
                            )
                        ),
                        (2, 4)
                    ),
                    (2, 3))),
                (1, 3)), (1, 2))
            # print("sym_B_B shape = ",self.sym_B_B.shape)
        
        # For 2D mesh, 3D vector case
        if self.dim == 2 and self.n_components == 3:
            assert (self.sym_B.shape == (self.n_dofs, self.n_components, self.n_components))
            assert (coeff.shape == (self.n_components, self.n_components, self.n_components, self.n_components))
            
            # Compute sym_B_B
            self.sym_B_B = sympy.tensorcontraction(sympy.tensorcontraction(
                sympy.tensorproduct(
                    self.sym_B,
                    sympy.tensorcontraction(sympy.tensorcontraction(
                        sympy.tensorproduct(
                            sympy.Array(coeff),
                            sympy.permutedims(
                                self.sym_B,
                                (1, 2, 0)
                            )
                        ),
                        (2, 4)
                    ),
                    (2, 3))),
                (1, 3)), (1, 2))
            print("sym_B_B shape = ",self.sym_B_B.shape)
            
        # Validate final shape
        assert (self.sym_B_B.shape == (self.n_dofs, self.n_dofs))

    def _init_sym_B_B_int(self, n=0):
        """Computes the (symbolic) integrals over the element of the of shape functions symmetric gradients products (stiffness matrix)."""
        self.sym_B_B_int = self.finite_element._integrate_array(array=self.sym_B_B, coeff=1, n=n)

    def init_get_B_B_int(self, coeff, n=0):
        """Initializes the (efficient) computation of the shape functions symmetric gradients products element integral."""
        self._init_sym_B()
        self._init_sym_B_B(coeff)
        self._init_sym_B_B_int(n=n)
        self._get_B_B_int = sympy.lambdify(
            args=self.sym_nodes.tolist(),
            expr=self.sym_B_B_int,
            modules="numpy")
        print("B_B_int initialized")

    def get_B_B_int(self, mesh, k_cell, loc_mat):
        """(Efficient) computation of shape functions derivatives products element integral."""
        loc_mat[:,:] = self._get_B_B_int(*mesh.get_cell_nodes_coords(k_cell))

    # ADDITIONS 

    def get_B_B_and_P_P_int(self,mesh, k_cell, loc_mat):
        """(Efficient) computation of shape functions derivatives products element integral."""
        loc_mat[:,:] = self._get_B_B_and_P_P_int(*mesh.get_cell_nodes_coords(k_cell))

    def init_get_B_B_and_P_P_int(self, coeff_B, coeff_P, n=0):
        """Initializes the (efficient) computation of the shape functions symmetric gradients products element integral."""
        self._init_sym_B_and_P() #Compute the elementary B and P
        print("B and P initialized")
        self._init_sym_B_B_and_P_P(coeff_B,coeff_P) #Initializes the products
        print("BB and PP initialized")
        self._init_sym_B_B_and_P_P_int(n=n) #Initializes the integral of the array
        print("BB and PP int initialized")

        self._get_B_B_and_P_P_int = sympy.lambdify(
            args=self.sym_nodes.tolist(),
            expr=self.sym_B_B_and_P_P_int,
            modules="numpy")
        print("B_B_and_P_P_int initialized")

    def _init_sym_B_B_and_P_P(self,coeff_B,coeff_P):
            
        """Computes the (symbolic) products of shape functions symmetric gradients,
        and stores them as a (n_dofs x n_dofs) sympy Array."""

        # For 2D mesh, 3D vector case
        if self.dim == 2 and self.n_components == 3:
            # assert (self.sym_B.shape == (self.n_dofs, self.n_components))
            # assert (coeff_B.shape == (self.n_components, self.n_components))

            #La somme des deux termes
            self.sym_B_B_and_P_P = sympy.MutableDenseNDimArray.zeros(self.n_dofs, self.n_dofs)

            # B*A*B.T
            t_sym_B = sympy.transpose(self.sym_B)
            coeff_B = sympy.Array(coeff_B)
            

            AtB = sympy.tensorcontraction(
                                        sympy.tensorproduct(
                                            coeff_B,
                                            t_sym_B),
                                        (1, 2)
                                    )
            # print("Atb shape", AtB.shape)
            # print("sym_B shape:", self.sym_B.shape)

            self.sym_B_B = sympy.tensorcontraction(
                                sympy.tensorproduct(
                                    self.sym_B,
                                    AtB    
                                ),
                                (1, 2)
                            )

            # P*D*P.T
            t_sym_P = sympy.transpose(self.sym_P)
            coeff_P = sympy.Array(coeff_P)
            

            DtP = sympy.tensorcontraction(
                                        sympy.tensorproduct(
                                            coeff_P,
                                            t_sym_P),
                                        (1, 2)
                                    )
            # print("Atb shape", DtP.shape)
            # print("sym_B shape:", self.sym_P.shape)

            self.sym_P_P = sympy.tensorcontraction(
                                sympy.tensorproduct(
                                    self.sym_P,
                                    DtP    
                                ),
                                (1, 2)
                            )
            
            #On les somme
            self.sym_B_B_and_P_P = self.sym_B_B + self.sym_P_P
     
    def _init_sym_B_B_and_P_P_int(self, n):
        """Computes the (symbolic) integrals over the element of the of shape functions symmetric gradients products (stiffness matrix)."""
        # for i in range(self.sym_B_B_and_P_P.shape[0]):
        #     for j in range(self.sym_B_B_and_P_P.shape[1]):
        #         self.sym_B_B_and_P_P[i,j].
        self.sym_B_B_and_P_P_int = self.finite_element._integrate_array(array=self.sym_B_B_and_P_P, coeff=1, n=n)

    # TO COMPLETE
    def _init_sym_B_and_P(self):
        #Initializes the shape functions derivatives
        self.finite_element._init_sym_dphi()
        self.finite_element._init_sym_ddphi()

        #Initializes arrays
        self.sym_B = sympy.MutableDenseNDimArray.zeros(self.n_dofs, self.n_components)
        self.sym_P = sympy.MutableDenseNDimArray.zeros(self.n_dofs, self.n_components)

        #Fill the arrays
        self._init_only_sym_P()
        self._init_only_sym_B()

    def _init_only_sym_P(self):
        assert self.dim == 2 and self.n_components == 3
        # if   (self.ordering == "component-wise"):
    #         for k_component in range(self.n_components):
    #             self.sym_B[:self.finite_element.n_dofs,k_component,k_component] = self.finite_element.sym_dphi[:,0]
    #         self.sym_B[self.finite_element.n_dofs:,1,1] = self.finite_element.sym_dphi[:,1]
    #         self.sym_B[self.finite_element.n_dofs:,2,1] = self.finite_element.sym_dphi[:,1]
    #         self.sym_B[:self.finite_element.n_dofs,0,1] = self.finite_element.sym_dphi[:,1]/2
    #         self.sym_B[self.finite_element.n_dofs:,0,1] = self.finite_element.sym_dphi[:,0]/2
    #         self.sym_B[:self.finite_element.n_dofs,1,0] = self.finite_element.sym_dphi[:,1]/2
    #         self.sym_B[self.finite_element.n_dofs:,1,0] = self.finite_element.sym_dphi[:,0]/2

        if (self.ordering == "point-wise"):
            #self.finite_element.sym_ddphi
            for k_dof in range(self.finite_element.n_dofs): #self.finite_element.n_dofs : they're at the 3 vertices for a P1 triangle element
                self.sym_P[3*k_dof+2,0] = - self.finite_element.sym_ddphi[k_dof ,0,0]
                self.sym_P[3*k_dof+2,1] = - self.finite_element.sym_ddphi[k_dof,1,1]
                self.sym_P[3*k_dof+2,2] = - (self.finite_element.sym_ddphi[k_dof,0,1] + self.finite_element.sym_ddphi[k_dof,1,0] )*numpy.sqrt(2) #the two terms are supposed to be equal
        

    def _init_only_sym_B(self):
        assert self.dim == 2 and self.n_components == 3

        # if   (self.ordering == "component-wise"):
        #         for k_component in range(self.n_components):
        #             self.sym_B[:self.finite_element.n_dofs,k_component,k_component] = self.finite_element.sym_dphi[:,0]
        #         self.sym_B[self.finite_element.n_dofs:,1,1] = self.finite_element.sym_dphi[:,1]
        #         self.sym_B[self.finite_element.n_dofs:,2,1] = self.finite_element.sym_dphi[:,1]
        #         self.sym_B[:self.finite_element.n_dofs,0,1] = self.finite_element.sym_dphi[:,1]/2
        #         self.sym_B[self.finite_element.n_dofs:,0,1] = self.finite_element.sym_dphi[:,0]/2
        #         self.sym_B[:self.finite_element.n_dofs,1,0] = self.finite_element.sym_dphi[:,1]/2
        #         self.sym_B[self.finite_element.n_dofs:,1,0] = self.finite_element.sym_dphi[:,0]/2

        if (self.ordering == "point-wise"):
            for k_dof in range(self.finite_element.n_dofs): #self.finite_element.n_dofs : they're at the 3 vertices for a P1 triangle element
                self.sym_B[3*k_dof , 0] = self.finite_element.sym_dphi[k_dof,0]
                self.sym_B[3*k_dof,2] = self.finite_element.sym_dphi[k_dof,1]/numpy.sqrt(2)

                self.sym_B[3*k_dof+1,1] = self.finite_element.sym_dphi[k_dof,1]
                self.sym_B[3*k_dof+1,2] = self.finite_element.sym_dphi[k_dof,0]/numpy.sqrt(2)


###########
# TEST PART
###########
import unittest
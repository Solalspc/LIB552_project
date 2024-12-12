            self.n_edges = 0
            self.edges_nodes = []
            self.cells_edges = []
        elif (self.dim == 2):
            self.n_edges = self.n_nodes + (self.n_cells+1) - 2 # Euler's formula (only works if mesh has no hole, see later)
            # print ("n_edges = "+str(self.n_edges))
            self.edges_nodes = numpy.empty((self.n_edges,                 2), dtype=int)
            self.cells_edges = numpy.empty((self.n_cells, self.cell.n_edges), dtype=int)
            nodes_edges = [set() for k_node in range(self.n_nodes)]
            cell_nodes = numpy.empty(self.cell.n_nodes, dtype=int)
            edge_nodes = numpy.empty(                2, dtype=int)
            k_edge = 0
            for k_cell in range(self.n_cells):
                # print ("k_cell = "+str(k_cell))
                cell_nodes[:] = self.cells_nodes[k_cell]
                # print ("cell_nodes = "+str(cell_nodes))
                for k_cell_edge in range(self.cell.n_edges):
                    # print ("k_cell_edge = "+str(k_cell_edge))
                    edge_nodes = cell_nodes[self.cell.get_edge_nodes(k_cell_edge)]
                    # print ("edge_nodes = "+str(edge_nodes))
                    intersection = nodes_edges[edge_nodes[0]] & nodes_edges[edge_nodes[1]]
                    if   (len(intersection) == 1):
                        self.cells_edges[k_cell, k_cell_edge] = intersection.pop()
                    elif (len(intersection) == 0):
                        if (k_edge >= len(self.edges_nodes)): # If mesh has holes, there are actually more edges than predicted by Euler's formula
                            self.n_edges += 1
                            self.edges_nodes = numpy.concatenate((self.edges_nodes, numpy.empty((1,2), dtype=int)))
                        self.cells_edges[k_cell, k_cell_edge] = k_edge
                        self.edges_nodes[k_edge] = numpy.sort(edge_nodes)
                        nodes_edges[edge_nodes[0]].add(k_edge)
                        nodes_edges[edge_nodes[1]].add(k_edge)
                        k_edge += 1
                    else: assert (0), "This should not happen. Aborting."
            assert (k_edge == self.n_edges)
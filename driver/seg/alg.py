import vtk
import numpy as np
from vtk.util import numpy_support
import networkx as nx
from scipy.sparse import csr_matrix

def split_mesh_by_sharp_edges(graph, flags, min_graph_size):

    # remove the nodes whose flags are 1
    nodes_to_remove = []
    for node in graph.nodes():
        if flags[node] == 1:
            nodes_to_remove.append(node)
    new_graph = graph.copy()
    new_graph.remove_nodes_from(nodes_to_remove)

    # find the seperated components and keep the large ones
    # return a list of sets of nodes
    all_graph_nodes = list(nx.connected_components(new_graph))  
    
    # if these nodes are just make the graph smaller but not split the graph
    if len(all_graph_nodes) == 1:
        return [graph]
    else:
        all_large_graphs = []
        for i in range(len(all_graph_nodes)):
            nodes = all_graph_nodes[i]
            if len(nodes) > min_graph_size:
                splited_subgraph = graph.subgraph(nodes).copy()
                all_large_graphs.append(splited_subgraph)
        # if all split graphs are too small, return the original graph
        if len(all_large_graphs) <= 1:
            return [graph]
        else:
            return all_large_graphs

def mark_sharp_edges(polydata, angle_threshold=20.0):
    poly = polydata  # your vtkPolyData surface
    angle_deg = angle_threshold  # threshold; increase for noisier meshes

    # (Optional) keep a mapping back to original point IDs
    idf = vtk.vtkIdFilter()
    idf.SetInputData(poly)
    idf.SetPointIds(True)
    idf.PointIdsOn()
    idf.SetPointIdsArrayName("OrigPointIds")  # Fixed method name
    idf.Update()

    fe = vtk.vtkFeatureEdges()
    fe.SetInputConnection(idf.GetOutputPort())
    fe.FeatureEdgesOn()
    fe.BoundaryEdgesOff()     # set On() if you also want open borders
    fe.NonManifoldEdgesOn()
    fe.SetFeatureAngle(angle_deg)
    fe.Update()

    edges_poly = fe.GetOutput()

    # Build (N,1) flag on original vertices
    N = poly.GetNumberOfPoints()
    flags = np.zeros((N, 1), dtype=np.uint8)

    # Use preserved original point IDs from the IdFilter
    orig_ids = numpy_support.vtk_to_numpy(edges_poly.GetPointData().GetArray("OrigPointIds"))

    # Each line cell references two points from edges_poly; map those points back
    lines = edges_poly.GetLines()
    idlist = vtk.vtkIdList()
    lines.InitTraversal()
    while lines.GetNextCell(idlist):
        for j in range(idlist.GetNumberOfIds()):
            pid_edgepoly = idlist.GetId(j)
            pid_orig = int(orig_ids[pid_edgepoly])
            flags[pid_orig, 0] = 1

    print(f"Feature edges detected: {np.sum(flags)} points marked as sharp edges")
    print(f"Sharp edge ratio: {np.sum(flags) / N * 100:.2f}%")

    return flags

def find_nearest_neighbors_from_mesh(polydata):

    # Extract cell connectivity
    max_node_id = 0
    cells = polydata.GetPolys()
    num_nodes = polydata.GetNumberOfPoints()
    if cells:
        # Get the connectivity array
        connectivity = cells.GetData()
        connectivity_array = numpy_support.vtk_to_numpy(connectivity)
        
        # Get cell information
        num_cells = polydata.GetNumberOfCells()
        
        # Extract edges from polygon connectivity
        nearest_neighbors = [[] for _ in range(num_nodes)]
        
        idx = 0
        for i in range(num_cells):
            num_vertices = connectivity_array[idx]
            vertices = connectivity_array[idx+1:idx+1+num_vertices]

            vertices = vertices.tolist()
            vertices = vertices + [vertices[0]]

            for j in range(num_vertices):
                nearest_neighbors[vertices[j]].append(vertices[(j+1)])
                nearest_neighbors[vertices[(j+1)]].append(vertices[j])

            max_node_id = max(max_node_id, max(vertices))
            
            idx += (1 + num_vertices)

    for i in range(len(nearest_neighbors)):
        nearest_neighbors[i] = list(set(nearest_neighbors[i]))
    
    print('max node id: ', max_node_id)

    return nearest_neighbors

def find_clusters_from_mesh(coords, org_graph):

    num_nodes = len(org_graph.nodes())
    all_graph_nodes = list(nx.connected_components(org_graph))  # List of sets of nodes
    all_graph_nodes = [np.array(list(comp)) for comp in all_graph_nodes]
    print(f"Number of connected components: {len(all_graph_nodes)}")
    print(f"Component sizes: {[len(comp) for comp in all_graph_nodes]}")

    '''
    classify the graph nodes into different clusters
        - the largest cluster is the vehicle surface -> cluster #1
        - 8 second largest cluster are the vehicle wheels
    '''
    flags = np.zeros(num_nodes)
    # Get the vehicle surface
    graph_sizes = [len(comp) for comp in all_graph_nodes]
    surface_index = np.argmax(graph_sizes)
    flags[all_graph_nodes[surface_index]] = 1

    # Get the vehicle wheels
    X_bar = 1.5
    Y_bar = 0.0
    size_orders = np.argsort(graph_sizes)[::-1]
    size_orders = size_orders[1:9]
    for index in size_orders:
        coors = coords[all_graph_nodes[index]]
        center = np.mean(coors, axis=0)
        # W1
        if center[0] > X_bar and center[1] > Y_bar:
            flags[all_graph_nodes[index]] = 2
        # W2
        elif center[0] > X_bar and center[1] < Y_bar:
            flags[all_graph_nodes[index]] = 3
        # W3
        elif center[0] < X_bar and center[1] > Y_bar:
            flags[all_graph_nodes[index]] = 4
        # W4
        elif center[0] < X_bar and center[1] < Y_bar:
            flags[all_graph_nodes[index]] = 5
    
    # for all the other clusters, set the flag to 6
    flags[flags == 0] = 6
    
    return flags


def create_seg_matrix(coords, polydata, threshold_angles, 
    min_graph_size_ratio = 0.0001, FIND_SEG = True):

    # extract basic information
    num_nodes = len(coords)

    # create neighbor information
    neighbor_indices = find_nearest_neighbors_from_mesh(polydata)

    # create a graph
    org_graph = nx.Graph()
    org_graph.add_nodes_from(range(len(coords)))
    for i in range(len(coords)):
        for j in neighbor_indices[i]:
            org_graph.add_edge(i, j)

    '''
    find_clusters_from_mesh(coords, org_graph)
    '''
    node_cluster_flags = find_clusters_from_mesh(coords, org_graph)

    if FIND_SEG:
        '''
        The graph is already a combination of connected components
        Only the main surface is changing, so we only need to extract physics tokens from it
        '''
        graph_nodes = list(nx.connected_components(org_graph))  # Convert generator to list
        print(f"Number of connected components: {len(graph_nodes)}")
        print(f"Component sizes: {[len(comp) for comp in graph_nodes]}")

        # Convert sets of nodes into actual NetworkX Graph objects
        # find the largest node set and only care about it
        largest_node_set = max(graph_nodes, key=len)
        subgraph = org_graph.subgraph(largest_node_set).copy()
        graph_still_need_split = [subgraph]

        # iteratively split the mesh based on the sharp edges
        # threshold_angles = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        all_flags = []
        all_sharp_edges = []
        identified_sharp_edges_last_iteration = set([])
        sharp_edge_subgraphs = []  # Store subgraphs created from newly detected sharp edges
        
        for threshold_angle in threshold_angles:
            print(f"Processing threshold angle: {threshold_angle}")
            flags = mark_sharp_edges(polydata, angle_threshold=threshold_angle)
            all_flags.append(flags)

            # store the identified sharp edges in this iteration
            sharp_edges_this_threshold = []
            for i in range(len(flags)):
                if flags[i] == 1:
                    sharp_edges_this_threshold.append(i)
            sharp_edges_this_threshold = set(sharp_edges_this_threshold)
            
            # not include the sharp edges that are already in the all_sharp_edges
            sharp_edges_this_threshold_after_pruning = sharp_edges_this_threshold - identified_sharp_edges_last_iteration
            all_sharp_edges.append(list(sharp_edges_this_threshold_after_pruning))
            
            # Create subgraphs from newly detected sharp edge nodes
            if len(sharp_edges_this_threshold_after_pruning) > 0:
                # Create a subgraph from the original graph containing only the newly detected sharp edge nodes
                sharp_edge_subgraph = org_graph.subgraph(list(sharp_edges_this_threshold_after_pruning))
                # Find connected components in this subgraph
                connected_components = list(nx.connected_components(sharp_edge_subgraph))
                # Add each connected component as a subgraph
                for component in connected_components:
                    if len(component) > 0:
                        component_subgraph = org_graph.subgraph(component).copy()
                        sharp_edge_subgraphs.append(component_subgraph)
                print(f"Found {len(connected_components)} connected components from newly detected sharp edges")
            
            identified_sharp_edges_last_iteration = sharp_edges_this_threshold
            
            # seperate the mesh by the sharp edges
            graph_still_need_split_this_iteration = []
            for graph in graph_still_need_split:
                result = split_mesh_by_sharp_edges(graph, flags, min_graph_size=int(min_graph_size_ratio * num_nodes))
                graph_still_need_split_this_iteration.extend(result)
            assert len(graph_still_need_split_this_iteration) >= len(graph_still_need_split), '# of split graphs should not be decreasing'
            
            # update the graph_still_need_split
            graph_still_need_split = graph_still_need_split_this_iteration
            print('current number of subgraphs: ', len(graph_still_need_split))
            if len(graph_still_need_split) == 0:
                break

        # combine the subgraphs from mesh splitting and sharp edge detection
        all_subgraphs = graph_still_need_split + sharp_edge_subgraphs
        print('number of subgraphs before deduplication: ', len(all_subgraphs))
        print(f'  - {len(graph_still_need_split)} from mesh splitting')
        print(f'  - {len(sharp_edge_subgraphs)} from sharp edge detection')
        
        # Remove duplicate subgraphs based on node sets (find unique separated graphs)
        unique_subgraphs = []
        seen_node_sets = set()
        for subgraph in all_subgraphs:
            node_set = frozenset(subgraph.nodes())
            if node_set not in seen_node_sets:
                seen_node_sets.add(node_set)
                unique_subgraphs.append(subgraph)
        
        all_subgraphs = unique_subgraphs
        print('number of subgraphs after deduplication: ', len(all_subgraphs))

        # sort the subgraphs by the size
        all_subgraphs = sorted(all_subgraphs, key=lambda x: len(x), reverse=True)

        # create the seg matrix as a sparse matrix
        # Prepare data for sparse matrix construction
        row_indices = []
        col_indices = []
        data = []
        
        for i in range(len(all_subgraphs)):
            subgraph_size = len(all_subgraphs[i])
            value = 1 / subgraph_size
            for node in all_subgraphs[i]:
                row_indices.append(i)
                col_indices.append(node)
                data.append(value)
        
        # Create sparse matrix in CSR format
        seg_matrix = csr_matrix((data, (row_indices, col_indices)), 
                            shape=(len(all_subgraphs), len(coords)))
    
    else:
        seg_matrix = np.ones((1, 1, 1))

    return seg_matrix, node_cluster_flags
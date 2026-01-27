import vtk
import numpy as np
from vtk.util import numpy_support
import networkx as nx
from scipy.sparse import csr_matrix

def merge_small_components(graph_to_merge, all_graphs, original_graph, max_expansion_hops=10):
    """
    Expand a small graph by neighbor search until it touches another graph, then merge it.
    
    Args:
        graph_to_merge: A small graph (NetworkX Graph) that needs to be merged
        all_graphs: List of graphs that graph_to_merge should potentially merge into
        original_graph: The original full graph to perform neighbor searches on
        max_expansion_hops: Maximum number of hops to expand (to avoid infinite loops)
    
    Returns:
        Index of the graph in all_graphs that graph_to_merge was merged into, or None if no merge occurred
    """
    nodes_to_merge = set(graph_to_merge.nodes())
    current_expansion = nodes_to_merge.copy()
    
    # Create a mapping of node to graph index for fast lookup
    node_to_graph_idx = {}
    for idx, g in enumerate(all_graphs):
        for node in g.nodes():
            node_to_graph_idx[node] = idx
    
    # Expand by neighbor search
    for hop in range(max_expansion_hops):
        # Find neighbors of current expansion set
        neighbors = set()
        for node in current_expansion:
            neighbors.update(original_graph.neighbors(node))
        
        # Remove nodes we've already explored
        new_neighbors = neighbors - nodes_to_merge
        
        # Check if any new neighbors belong to graphs in all_graphs
        for neighbor in new_neighbors:
            if neighbor in node_to_graph_idx:
                # Found a connection! Merge into this graph
                target_idx = node_to_graph_idx[neighbor]
                # Merge graph_to_merge into all_graphs[target_idx]
                target_graph = all_graphs[target_idx]
                # Combine nodes
                merged_nodes = set(target_graph.nodes()) | nodes_to_merge
                # Create new merged graph
                merged_graph = original_graph.subgraph(merged_nodes).copy()
                all_graphs[target_idx] = merged_graph
                return target_idx
        
        # If no connection found, expand further
        if len(new_neighbors) == 0:
            # No more neighbors to explore
            break
        
        nodes_to_merge.update(new_neighbors)
        current_expansion = new_neighbors
    
    # No merge occurred
    return None

def split_mesh_by_sharp_edges(graph, flags, min_graph_size):

    # ensure the "graph" contains only one connected component
    assert len(list(nx.connected_components(graph))) == 1, 'the graph should contain only one connected component'

    # remove the nodes whose flags are 1
    nodes_to_remove = []
    for node in graph.nodes():
        if flags[node] == 1:
            nodes_to_remove.append(node)
    new_graph = graph.copy()
    nodes_removed_in_this_subgraph = list(set(new_graph.nodes()) & set(nodes_to_remove))
    new_graph.remove_nodes_from(nodes_removed_in_this_subgraph)

    # find the seperated components and keep the large ones
    # return a list of sets of nodes
    splited_graph_nodes = list(nx.connected_components(new_graph))
    
    # Get connected components from the removed nodes (intersection between nodes_to_remove and original graph)
    removed_nodes_subgraph = graph.subgraph(nodes_removed_in_this_subgraph)
    removed_components = list(nx.connected_components(removed_nodes_subgraph))

    # sort the splited_graph_nodes by the size from largest to smallest
    splited_graph_nodes = sorted(splited_graph_nodes, key=lambda x: len(x), reverse=True)
            
    all_graph_nodes = splited_graph_nodes + removed_components

    # create the graphs
    all_large_graphs = []
    for i in range(len(all_graph_nodes)):
        nodes = all_graph_nodes[i]
        if len(nodes) > min_graph_size:
            splited_subgraph = graph.subgraph(nodes).copy()
            all_large_graphs.append(splited_subgraph)
    
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

    # print(f"Feature edges detected: {np.sum(flags)} points marked as sharp edges")
    # print(f"Sharp edge ratio: {np.sum(flags) / N * 100:.2f}%")

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
    
    # print('max node id: ', max_node_id)

    return nearest_neighbors


def print_node_assignment_statistics(all_subgraphs, num_nodes):
    """
    Calculate and print statistics about node assignments to clusters.
    
    Args:
        all_subgraphs: List of NetworkX Graph objects representing clusters
        num_nodes: Total number of nodes in the graph
    """
    # Check for no-cluster nodes (nodes that don't belong to any cluster)
    assigned_nodes = set()
    for subgraph in all_subgraphs:
        assigned_nodes.update(subgraph.nodes())
    
    all_nodes = set(range(num_nodes))
    no_cluster_nodes = all_nodes - assigned_nodes
    no_cluster_percentage = (len(no_cluster_nodes) / num_nodes) * 100 if num_nodes > 0 else 0
    
    print(f'No-cluster nodes: {len(no_cluster_nodes)} / {num_nodes} ({no_cluster_percentage:.2f}%)')
    
    # Check for multi-cluster nodes (nodes that are assigned to multiple clusters)
    node_cluster_count = {}
    for subgraph in all_subgraphs:
        for node in subgraph.nodes():
            node_cluster_count[node] = node_cluster_count.get(node, 0) + 1
    
    multi_cluster_nodes = [node for node, count in node_cluster_count.items() if count > 1]
    multi_cluster_percentage = (len(multi_cluster_nodes) / num_nodes) * 100 if num_nodes > 0 else 0
    
    print(f'Multi-cluster nodes: {len(multi_cluster_nodes)} / {num_nodes} ({multi_cluster_percentage:.2f}%)')


def create_seg_matrix(coords, polydata, threshold_angles, 
    min_graph_size_ratio = 0.0001, FIND_SEG = True, 
    MERGE_SMALL_GRAPHS = False,
    INCLUDE_NO_CLUSTER_NODES = False):

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

    if FIND_SEG:
        '''
        The graph is already a combination of connected components
        Only the main surface is changing, so we only need to extract physics tokens from it
        '''
        graph_nodes = list(nx.connected_components(org_graph))  # Convert generator to list
        # print(f"Number of connected components: {len(graph_nodes)}")
        # print(f"Component sizes: {[len(comp) for comp in graph_nodes]}")

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
        for threshold_angle in threshold_angles:
            # print(f"Processing threshold angle: {threshold_angle}")
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
            identified_sharp_edges_last_iteration = sharp_edges_this_threshold
            
            # seperate the mesh by the sharp edges
            graph_still_need_split_this_iteration = []
            for graph in graph_still_need_split:
                result = split_mesh_by_sharp_edges(graph, flags, min_graph_size=int(min_graph_size_ratio * num_nodes))
                graph_still_need_split_this_iteration.extend(result)

            # update the graph_still_need_split
            graph_still_need_split = graph_still_need_split_this_iteration
            # print('current number of subgraphs: ', len(graph_still_need_split))
            if len(graph_still_need_split) == 0:
                break

        # combine the subgraphs 
        all_subgraphs = graph_still_need_split
        # print('number of subgraphs before physics token purning: ', len(all_subgraphs))

        # sort the subgraphs by the size
        all_subgraphs = sorted(all_subgraphs, key=lambda x: len(x), reverse=True)

        if INCLUDE_NO_CLUSTER_NODES:
            # Find nodes not assigned to any subgraph and create subgraphs from them
            assigned_nodes = set()
            for subgraph in all_subgraphs:
                assigned_nodes.update(subgraph.nodes())
            
            all_nodes = set(range(len(coords)))
            unassigned_nodes = all_nodes - assigned_nodes
            
            # Create subgraphs from unassigned nodes using connected components
            if len(unassigned_nodes) > 0:
                unassigned_subgraph = org_graph.subgraph(unassigned_nodes)
                unassigned_components = list(nx.connected_components(unassigned_subgraph))
                for comp in unassigned_components:
                    comp_subgraph = org_graph.subgraph(comp).copy()
                    all_subgraphs.append(comp_subgraph)
                # print(f'Added {len(unassigned_components)} subgraphs from {len(unassigned_nodes)} unassigned nodes')

        if MERGE_SMALL_GRAPHS:
            # Keep the 256 largest graphs, merge smaller ones back to big graphs
            # Sort again after adding unassigned nodes
            all_subgraphs = sorted(all_subgraphs, key=lambda x: len(x), reverse=True)
            num_large_graphs = min(256, len(all_subgraphs))
            large_graphs = all_subgraphs[:num_large_graphs]
            small_graphs = all_subgraphs[num_large_graphs:]
            
            # Merge small graphs into large graphs using neighbor search
            # Start with the smallest graphs first (reverse order)
            if len(small_graphs) > 0:
                merged_count = 0
                unmerged_small_graphs = []
                # Sort small graphs by size ascending (smallest first)
                small_graphs_sorted = sorted(small_graphs, key=lambda x: len(x))
                for small_graph in small_graphs_sorted:
                    merge_result = merge_small_components(small_graph, large_graphs, org_graph)
                    assert merge_result is not None, 'the small graph should be merged into a large graph'
                    if merge_result is not None:
                        merged_count += 1
                    else:
                        # Couldn't be merged, keep it separate
                        unmerged_small_graphs.append(small_graph)
                
                # print(f"Merged {merged_count} small graphs into large graphs")
                if len(unmerged_small_graphs) > 0:
                    print(f"Could not merge {len(unmerged_small_graphs)} small graphs")
                
                # Combine large graphs with unmerged small graphs
                all_subgraphs = large_graphs + unmerged_small_graphs
            else:
                all_subgraphs = large_graphs
        else:
            all_subgraphs = sorted(all_subgraphs, key=lambda x: len(x), reverse=True)
            num_large_graphs = min(256, len(all_subgraphs))
            all_subgraphs = all_subgraphs[:num_large_graphs]


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
        
        # After all operations, check for node assignment statistics
        print_node_assignment_statistics(all_subgraphs, len(coords))
    
    else:
        seg_matrix = np.ones((1, 1, 1))
        all_subgraphs = []

    # create the node_cluster_flags
    # Assign one label per cluster, with multi-cluster nodes getting a separate label
    node_cluster_flags = np.zeros((len(coords),), dtype=np.int32)
    
    if FIND_SEG and len(all_subgraphs) > 0:
        # Track which nodes appear in which clusters
        node_to_clusters = {}
        for cluster_idx, subgraph in enumerate(all_subgraphs):
            for node in subgraph.nodes():
                if node not in node_to_clusters:
                    node_to_clusters[node] = []
                node_to_clusters[node].append(cluster_idx)
        
        # Identify nodes that appear in multiple clusters
        multi_cluster_nodes = {node: clusters for node, clusters in node_to_clusters.items() 
                               if len(clusters) > 1}
        
        # Assign labels: regular clusters get labels 0, 1, 2, ...
        # Multi-cluster nodes get a special label (separate cluster)
        special_cluster_label = len(all_subgraphs)
        graph_indexes = np.arange(len(all_subgraphs))
        np.random.shuffle(graph_indexes)
        
        for id, subgraph in enumerate(all_subgraphs):
            cluster_idx = graph_indexes[id]
            for node in subgraph.nodes():
                # Only assign if node is not a multi-cluster node
                if node not in multi_cluster_nodes:
                    node_cluster_flags[node] = cluster_idx
        
        # Assign special label to all multi-cluster nodes
        for node in multi_cluster_nodes:
            node_cluster_flags[node] = special_cluster_label
    
    # Reshape to (N, 1) format
    node_cluster_flags = node_cluster_flags.reshape(-1, 1)
    
    return seg_matrix, node_cluster_flags
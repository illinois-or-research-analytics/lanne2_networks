import pandas as pd
import numpy as np
import graph_tool.all as gt
from graph_tool.all import *
from matplotlib.pylab import poisson
import argparse
import typer
import os
import time
import logging
import matplotlib.pyplot as plt
import stats
import networkit as nk
import ng_eds as ng_eds

def read_graph(filepath):
    # graph = gt.load_graph_from_csv(filepath, directed=False, csv_options={'delimiter': '\t'})
    # return graph
    edgelist_reader = nk.graphio.EdgeListReader("\t", 0, directed=False, continuous=False)
    nk_graph = edgelist_reader.read(filepath)
    node_mapping = edgelist_reader.getNodeMap()
    # numerical_to_string_mapping = {v: k for k, v in node_mapping.items()}
    return nk_graph, node_mapping

def read_clustering(filepath):
    cluster_df = pd.read_csv(filepath, sep="\t", header=None, names=["node_id", "cluster_name"])
    unique_values = cluster_df["cluster_name"].unique()
    value_map = {value: idx for idx, value in enumerate(unique_values)}
    cluster_df['cluster_id'] = cluster_df['cluster_name'].map(value_map)
    return cluster_df[['node_id', 'cluster_id']]


def get_graph_stats(graph):
    num_vertices = graph.numberOfNodes()
    num_edges = graph.numberOfEdges()
    print("Number of vertices : ", num_vertices)
    print("Num edges : ", num_edges)
    return num_vertices, num_edges

def remove_edges(G, G_c):
    clustered_edges = []
    for edge in G_c.iterEdges():
        clustered_edges.append(edge)
    G_star = nk.Graph(G)
    for edge in clustered_edges:
        G_star.removeEdge(edge[0], edge[1])
    return G_star

# def get_probs(G_c, node_mapping, cluster_df):
#     numerical_to_string_mapping = {v: k for k, v in node_mapping.items()}
#     cluster_ids = cluster_df['cluster_id'].unique()
#     num_clusters = len(cluster_ids)
#     probs = np.zeros((num_clusters, num_clusters))
#     cluster_id_to_idx = {cluster_id: idx for idx, cluster_id in enumerate(cluster_ids)}
#     for cluster_id in cluster_ids:
#         cluster_nodes = cluster_df[cluster_df['cluster_id'] == cluster_id]['node_id'].values
#         probs_row = np.zeros(num_clusters)
#         for node in cluster_nodes:
#             neighbors = [v for v in G_c.iterNeighbors(node_mapping.get(str(node)))]
#             for neighbor in neighbors:
#                 neighbor_cluster_id = cluster_df[cluster_df['node_id'] == int(numerical_to_string_mapping.get(neighbor))]['cluster_id'].values[0]
#                 neighbor_cluster_idx = cluster_id_to_idx[neighbor_cluster_id]
#                 probs_row[neighbor_cluster_idx] += 1
#         cluster_idx = cluster_id_to_idx[cluster_id]
#         probs[cluster_idx, :] = probs_row
#     return probs

def get_probs(G_c, node_mapping, cluster_df):
    # Create a mapping from numerical node IDs to string IDs
    numerical_to_string_mapping = {v: str(k) for k, v in node_mapping.items()}
    
    # Get unique cluster IDs and their counts
    cluster_ids, counts = np.unique(cluster_df['cluster_id'], return_counts=True)
    num_clusters = len(cluster_ids)
    
    # Create a mapping from cluster ID to its index
    cluster_id_to_idx = {cluster_id: idx for idx, cluster_id in enumerate(cluster_ids)}
    
    # Precompute the cluster nodes dictionary for faster lookup
    cluster_nodes_dict = {}
    for cluster_id, count in zip(cluster_ids, counts):
        cluster_nodes_dict[cluster_id] = cluster_df[cluster_df['cluster_id'] == cluster_id]['node_id'].values
    # Initialize the probabilities matrix
    probs = np.zeros((num_clusters, num_clusters))
    
    # Iterate over cluster IDs
    for cluster_id in cluster_ids:
        # Get the index of the current cluster
        cluster_idx = cluster_id_to_idx[cluster_id]
        
        # Get the nodes belonging to the current cluster
        cluster_nodes = cluster_nodes_dict[cluster_id]
        
        
        # Iterate over nodes in the current cluster
        for node in cluster_nodes:
            # Get the neighbors of the current node
            neighbors = G_c.iterNeighbors(node_mapping[str(node)])
            # print("Node : ", node, "Neighbors : ", neighbors)
            
            # Iterate over neighbors
            for neighbor in neighbors:
                # Get the cluster ID of the neighbor
                neighbor_cluster_id = cluster_df[cluster_df['node_id'] == int(numerical_to_string_mapping.get(neighbor))]['cluster_id'].values[0]
                # neighbor_cluster_id = cluster_df.at[int(numerical_to_string_mapping[neighbor]), 'cluster_id']
                # print("Node : ", node, "Neighbor : ", neighbor, "neighbor cluster_id" , neighbor_cluster_id)
                # Get the index of the neighbor's cluster
                neighbor_cluster_idx = cluster_id_to_idx[neighbor_cluster_id]
                # print("Node : ", node, "Neighbor : ", neighbor, "neighbor cluster_id" , neighbor_cluster_id, "neighbor_cluster_idx", neighbor_cluster_idx)
                # Increment the corresponding entry in the probabilities matrix
                probs[cluster_idx, neighbor_cluster_idx] += 1
                # print("Node : ", node, "Neighbor : ", neighbor, "neighbor cluster_id" , neighbor_cluster_id, "neighbor_cluster_idx", neighbor_cluster_idx)
    print(probs.trace()//2 + (probs.sum() - probs.trace())//2)
    return probs

def get_degree_sequence(cluster_df, G_c, node_mapping,step, non_singleton_components):
    deg_seq = []
    if step ==3:
        for idx, row in cluster_df.iterrows():
            deg_seq.append(G_c.degree(node_mapping.get(str(row['node_id']))))
    elif step ==4:
        for component in non_singleton_components:
            deg_seq_component = []
            for node in component:
                deg_seq_component.append(G_c.degree(node))
            deg_seq.append(deg_seq_component)
    return deg_seq

def get_connected_components(G_star):
    cc = nk.components.ConnectedComponents(G_star)
    cc.run()
    connected_components = cc.getComponents()
    non_singleton_components = [component for component in connected_components if len(component) > 1]
    # print(cc.numberOfComponents(),len(connected_components))
    component_sizes = cc.getComponentSizes()
    return non_singleton_components

def rewire_non_singleton_components(non_singleton_components, deg_sequences, node_mapping):
    numerical_to_string_mapping = {v: k for k, v in node_mapping.items()}
    edge_lists = []
    for idx,component in enumerate(non_singleton_components):
        generated_graph = ng_eds.generate_graph(deg_sequences[idx])
        # print(generated_graph)
        vertices = generated_graph.get_vertices()
        edge_list = []
        edges = generated_graph.get_edges()
        for edge in edges:
            # print(edge, [component[edge[0]], component[edge[1]]])
            edge_list.append([int(numerical_to_string_mapping.get(component[edge[0]])), int(numerical_to_string_mapping.get(component[edge[1]]))])
        edge_lists.append(edge_list)
    return edge_lists

# def get_N_c_edge_list(N_c, cluster_df, node_mapping):
#     edge_list = []
#     for edge in N_c.get_edges():
#         edge_list.append([cluster_df.iloc[edge[0]]['node_id'], cluster_df.iloc[edge[1]]['node_id']])
#     return edge_list

def save_generated_graph(edges_list, out_edge_file):
    edge_df = pd.DataFrame(edges_list, columns=['source', 'target'])
    edge_df.to_csv(out_edge_file, sep='\t', index=False, header=None)

def main(edge_input: str = typer.Option(..., "--filepath", "-f"),
    cluster_input: str = typer.Option(..., "--cluster_filepath", "-c"),
    net_name: str = typer.Option(..., "--net_name", "-m"),
    out_edge_file: str = typer.Option("", "--out_edge_file", "-oe"),
    out_node_file: str = typer.Option("", "--out_node_file", "-on"),
    out_edge_file_Gstar: str = typer.Option("", "--out_edge_file_Gstar", "-gs"),
    out_edge_file_NGstar: str = typer.Option("", "--out_edge_file_Gstar", "-ngs"),
    out_edge_file_G_c: str = typer.Option("", "--out_edge_file_G_c", "-gc"),
    out_edge_file_N_c: str = typer.Option("", "--out_edge_file_N_c", "-nc")):

    if out_edge_file == "":
        out_edge_file = f'NG_rewiring_samples/{net_name}/Nstar_graph_edge_list.tsv'
    else:
        out_edge_file = out_edge_file

    if out_edge_file_N_c == "":
        out_edge_file_N_c = f'NG_rewiring_samples/{net_name}/N_c_graph_edge_list.tsv'
    else:
        out_edge_file_N_c = out_edge_file_N_c

    if out_edge_file_Gstar == "":
        out_edge_file_Gstar = f'NG_rewiring_samples/{net_name}/Gstar_graph_edge_list.tsv'
    else:
        out_edge_file_Gstar = out_edge_file_Gstar
    
    if out_edge_file_NGstar == "":
        out_edge_file_NGstar = f'NG_rewiring_samples/{net_name}/NGstar_graph_edge_list.tsv'
    else:
        out_edge_file_NGstar = out_edge_file_NGstar

    if out_edge_file_G_c == "":
        out_edge_file_G_c = f'NG_rewiring_samples/{net_name}/G_c_graph_edge_list.tsv'
    else:
        out_edge_file_G_c = out_edge_file_G_c
    
    if out_node_file == "":
        out_node_file = f'NG_rewiring_samples/{net_name}/Nstar_graph_node_list.tsv'
    else:
        out_node_file = out_node_file
    
    output_dir = os.path.dirname(out_edge_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    logging.basicConfig(filename=f'NG_rewiring_samples/{net_name}/NG_Rewiring_output.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    rewiring_start_time = time.time()

    try:
        logging.info("Reading generated graph...")
        start_time = time.time()
        G,node_mapping = read_graph(edge_input)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        logging.info("Statistics of read graph:")
        # num_vertices, num_edges = get_graph_stats(G)
        stats_df_G, fig = stats.main(edge_input, [], 'original_input_graph')
        print(stats_df_G)
        fig.savefig(output_dir+f"/{net_name}_original_degree_distribution.png")
        logging.info("Reading graph clustering:")
        start_time = time.time()
        cluster_df = read_clustering(cluster_input)
        # print(cluster_df.describe())
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        logging.info("Getting subgraph G_c")
        start_time = time.time()
        clustered_nodes = cluster_df['node_id'].unique()
        clustered_nodes_mapped = [node_mapping.get(str(v)) for v in clustered_nodes]
        G_c = nk.graphtools.subgraphFromNodes(G, clustered_nodes_mapped)
        num_clustered_nodes, num_cluster_edges = get_graph_stats(G_c)
        G_c_edge_list = []
        for edge in G_c.iterEdges():
            G_c_edge_list.append(edge)
        save_generated_graph(G_c_edge_list, out_edge_file_G_c)
        print("Number of clustered nodes : ", num_clustered_nodes , "\t Number of clustered edges : ", num_cluster_edges)
        logging.info("Getting graph stats for G_c")
        start_time = time.time()
        stats_df_G_c, fig = stats.main(out_edge_file_G_c, [], 'G_c_step1')
        print(stats_df_G_c)
        fig.savefig(output_dir+f"/{net_name}_G_c_degree_distribution.png")
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        logging.info("Removing cluster edges from G to get G* or G_star")
        start_time = time.time()
        Gstar = remove_edges(G, G_c)
        Gstar_edge_list = []
        for edge in Gstar.iterEdges():
            Gstar_edge_list.append(edge)
        save_generated_graph(Gstar_edge_list, out_edge_file_Gstar)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        logging.info("Getting graph stats for G_star")
        start_time = time.time()
        num_nodes_Gstar, num_edges_Gstar = get_graph_stats(Gstar)
        stats_df_Gstar, fig = stats.main(out_edge_file_Gstar, [], 'Gstar_step2')
        print(stats_df_Gstar)
        fig.savefig(output_dir+f"/{net_name}_Gstar_degree_distribution.png")
        print("Number of nodes in G*: ", num_nodes_Gstar , "\t Number of edges in G*: ", num_edges_Gstar)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        logging.info("Getting edge probs matrix for G_c ")
        start_time = time.time()
        probs = get_probs(G_c, node_mapping, cluster_df)
        print("Intra cluster edges : ", (np.sum(probs) - np.trace(probs))//2)
        print("Inter cluster edges : ",np.trace(probs) // 2)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        logging.info("Generating Synthetic graph N_c with probs")
        start_time = time.time()
        b = cluster_df['cluster_id'].to_numpy()
        out_deg_seq = get_degree_sequence(cluster_df, G_c, node_mapping, 3, [])
        # print(len(out_deg_seq), type(out_deg_seq))
        N_c = gt.generate_sbm(b, probs, out_degs=out_deg_seq, micro_ers=True, micro_degs=True)
        print("N_c graph statistics : ", N_c.num_vertices(), N_c.num_edges())
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        logging.info("Generating non-singleton components of G* ")
        start_time = time.time()
        non_singleton_components = get_connected_components(Gstar)
        print("Number of non singleton components : ", len(non_singleton_components))
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        logging.info("Compute degree sequence for each of the non-singleton components!")
        start_time = time.time()
        deg_sequences = get_degree_sequence(cluster_df, Gstar, node_mapping,4, non_singleton_components)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        logging.info("Performing ng-eds on each non-singleton component")
        start_time = time.time()
        N_i_edge_lists = rewire_non_singleton_components(non_singleton_components, deg_sequences, node_mapping)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        logging.info("Combining N_is and N_c edges")
        start_time = time.time()
        N_c_edge_list = []
        N_c_nodes_list = set()
        for edge in N_c.get_edges():
            source = cluster_df.iloc[edge[0]]['node_id']
            target = cluster_df.iloc[edge[1]]['node_id']
            N_c_edge_list.append([source, target])
            N_c_nodes_list.add(source)
            N_c_nodes_list.add(target)
        print(len(N_c_nodes_list), len(N_c_edge_list))
        save_generated_graph(N_c_edge_list, out_edge_file_N_c)
        logging.info("Statistics of generated graph N_c:")
        start_time = time.time()
        stats_df_N_c, fig = stats.main(out_edge_file_N_c, [], 'N_c_step1')
        print(stats_df_N_c)
        fig.savefig(output_dir+f"/{net_name}_N_c_step1_degree_distribution.png")
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        N_star_edges = []
        for Ni_edges in N_i_edge_lists:
            N_star_edges = N_star_edges + Ni_edges
        #Save N(G*) till here
        save_generated_graph(N_star_edges, out_edge_file_NGstar)
        logging.info("Statistics of generated graph NGstar:")
        start_time = time.time()
        stats_df_NGstar, fig = stats.main(out_edge_file_NGstar, [], 'NGstar_step2')
        print(stats_df_NGstar)
        fig.savefig(output_dir+f"/{net_name}_NGstar_step2_degree_distribution.png")
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")

        N_star_edges = N_star_edges + N_c_edge_list
        print("N_star_edges : " ,len(N_star_edges))
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        logging.info("Saving generated graph edge list! ")
        start_time = time.time()
        save_generated_graph(N_star_edges, out_edge_file)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        # logging.info("Statistics of generated graph N_c:")
        # start_time = time.time()
        # stats_df_N_c, fig = stats.main(out_edge_file_N_c, [], 'N_c_step1')
        # print(stats_df_N_c)
        # fig.savefig(output_dir+f"/{net_name}_N_c_step1_degree_distribution.png")
        # logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        logging.info("Statistics of generated graph N_star:")
        start_time = time.time()
        # num_vertices, num_edges = get_graph_stats()
        stats_df_Nstar, fig = stats.main(out_edge_file, [], 'N_star')
        print(stats_df_Nstar)
        fig.savefig(output_dir+f"/{net_name}_N_star_step3_degree_distribution.png")
        combined_df = pd.concat([stats_df_G_c, stats_df_N_c])
        combined_df = pd.concat([combined_df, stats_df_Gstar])
        combined_df = pd.concat([combined_df, stats_df_NGstar])
        combined_df = pd.concat([combined_df, stats_df_G])
        combined_df = pd.concat([combined_df, stats_df_Nstar])
        print(combined_df)
        combined_df.to_csv(f'{output_dir}/{net_name}_stats.csv')
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        logging.info(f"Total Time taken: {round(time.time() - rewiring_start_time, 3)} seconds")

    except Exception as e:
        print(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate graph ')
    parser.add_argument(
        '-f', metavar='edge_list_filepath', type=str, required=True,
        help='network edge_list filepath'
        )
    parser.add_argument(
        '-c', metavar='clustering_filepath', type=str, required=True,
        help='network clustering filepath'
        )
    parser.add_argument(
        '-m', metavar='network_name', type=str, required=True,
        help='name of the network'
        )
    parser.add_argument(
        '-oe', metavar='out_edge_file', type=str, required=False,
        help='output edgelist path'
        )
    parser.add_argument(
        '-on', metavar='out_node_file', type=str, required=False,
        help='output nodelist path'
        )
    parser.add_argument(
        '-nc', metavar='out_edge_file_N_c', type=str, required=False,
        help='output nodelist path for N_c'
        )
    parser.add_argument(
        '-gs', metavar='out_edge_file_Gstar', type=str, required=False,
        help='output nodelist path for Gstar'
        )
    parser.add_argument(
        '-ngs', metavar='out_edge_file_NGstar', type=str, required=False,
        help='output nodelist path for NGstar'
        )
    parser.add_argument(
        '-gc', metavar='out_edge_file_G_c', type=str, required=False,
        help='output nodelist path for G_c'
        )
    args = parser.parse_args()

    typer.run(main)
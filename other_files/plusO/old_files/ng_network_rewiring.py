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
import lanne2.lanne2_networks.plusO.old_files.ng_eds as ng_eds
from scipy.sparse import dok_matrix
import psutil

def read_graph(filepath):
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

def get_probs(G_c, node_mapping, cluster_df):
    numerical_to_string_mapping = {v: int(k) for k, v in node_mapping.items()}
    cluster_ids, counts = np.unique(cluster_df['cluster_id'], return_counts=True)
    num_clusters = len(cluster_ids)
    
    node_to_cluster_dict = cluster_df.set_index('node_id')['cluster_id'].to_dict()

    probs = dok_matrix((num_clusters, num_clusters), dtype=int)
    count = 0
    for edge in G_c.iterEdges():
        source = numerical_to_string_mapping.get(edge[0])
        target = numerical_to_string_mapping.get(edge[1])
        source_cluster_idx = node_to_cluster_dict.get(source)
        target_cluster_idx = node_to_cluster_dict.get(target)

        probs[source_cluster_idx, target_cluster_idx] += 1
        probs[target_cluster_idx, source_cluster_idx ] += 1
        
    probs = probs.tocsr(copy=False)
    print("Number of edges as per probs matrix: " , probs.trace()//2 + (probs.sum() - probs.trace())//2)
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
            edge_list.append((int(numerical_to_string_mapping.get(component[edge[0]])), int(numerical_to_string_mapping.get(component[edge[1]]))))
        edge_lists.append(edge_list)
    return edge_lists

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

    def log_cpu_ram_usage(step_name):
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage('/').percent
        logging.info(f"Step: {step_name} | CPU Usage: {cpu_percent}% | RAM Usage: {ram_percent}% | Disk Usage: {disk_percent}")


    try:
        log_cpu_ram_usage("Start")
        logging.info("Reading generated graph...")
        start_time = time.time()
        G,node_mapping = read_graph(edge_input)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("After reading graph")
        logging.info("Statistics of read graph:")
        stats_df_G, fig = stats.main(edge_input, [], 'original_input_graph')
        print(stats_df_G)
        fig.savefig(output_dir+f"/{net_name}_original_degree_distribution.png")
        logging.info("Reading graph clustering:")
        start_time = time.time()
        cluster_df = read_clustering(cluster_input)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("After getting probs matrix")
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
        log_cpu_ram_usage("")
        logging.info("Removing cluster edges from G to get G* or G_star")
        start_time = time.time()
        Gstar = remove_edges(G, G_c)
        Gstar_edge_list = []
        for edge in Gstar.iterEdges():
            Gstar_edge_list.append(edge)
        save_generated_graph(Gstar_edge_list, out_edge_file_Gstar)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("")
        logging.info("Getting graph stats for G_star")
        start_time = time.time()
        num_nodes_Gstar, num_edges_Gstar = get_graph_stats(Gstar)
        stats_df_Gstar, fig = stats.main(out_edge_file_Gstar, [], 'Gstar_step2')
        print(stats_df_Gstar)
        fig.savefig(output_dir+f"/{net_name}_Gstar_degree_distribution.png")
        print("Number of nodes in G*: ", num_nodes_Gstar , "\t Number of edges in G*: ", num_edges_Gstar)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("")
        logging.info("Getting edge probs matrix for G_c ")
        start_time = time.time()
        probs = get_probs(G_c, node_mapping, cluster_df)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("After getting probs matrix")
        logging.info("Generating Synthetic graph N_c with probs")
        start_time = time.time()
        b = cluster_df['cluster_id'].to_numpy()
        out_deg_seq = get_degree_sequence(cluster_df, G_c, node_mapping, 3, [])
        N_c = gt.generate_sbm(b, probs, out_degs=out_deg_seq, micro_ers=True, micro_degs=True)
        print("N_c graph statistics : ", N_c.num_vertices(), N_c.num_edges())
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("")
        logging.info("Generating non-singleton components of G* ")
        start_time = time.time()
        non_singleton_components = get_connected_components(Gstar)
        print("Number of non singleton components : ", len(non_singleton_components))
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("")
        logging.info("Compute degree sequence for each of the non-singleton components!")
        start_time = time.time()
        deg_sequences = get_degree_sequence(cluster_df, Gstar, node_mapping,4, non_singleton_components)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("")
        logging.info("Performing ng-eds on each non-singleton component")
        start_time = time.time()
        N_i_edge_lists = rewire_non_singleton_components(non_singleton_components, deg_sequences, node_mapping)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("After computing ng-eds for all singleton components")
        logging.info("Combining N_is and N_c edges")
        start_time = time.time()
        node_idx_set = cluster_df['node_id'].tolist()
        N_c_edge_list = set()
        N_c_nodes_list = set()
        for edge in N_c.get_edges():
            source = node_idx_set[edge[0]]
            target = node_idx_set[edge[1]]
            N_c_edge_list.add((source, target))
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
        log_cpu_ram_usage("After N_c graph generation and stats calculation")
        N_star_edges = set()
        for Ni_edges in N_i_edge_lists:
            N_star_edges.update(Ni_edges)
        #Save N(G*) till here
        save_generated_graph(N_star_edges, out_edge_file_NGstar)
        logging.info("Statistics of generated graph NGstar:")
        start_time = time.time()
        stats_df_NGstar, fig = stats.main(out_edge_file_NGstar, [], 'NGstar_step2')
        print(stats_df_NGstar)
        fig.savefig(output_dir+f"/{net_name}_NGstar_step2_degree_distribution.png")
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("After NGstar graph generation and stats calculation!")

        N_star_edges.update(N_c_edge_list)
        print("N_star_edges : " ,len(N_star_edges))
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        logging.info("Saving generated graph edge list! ")
        start_time = time.time()
        save_generated_graph(N_star_edges, out_edge_file)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("After saving N_star edges")
        logging.info("Statistics of generated graph N_star:")
        start_time = time.time()
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
        log_cpu_ram_usage("After N_star computation!!")
        logging.info(f"Total Time taken: {round(time.time() - rewiring_start_time, 3)} seconds")
        log_cpu_ram_usage("Usage statistics after job completion!")

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
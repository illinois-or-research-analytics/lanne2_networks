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
from scipy.sparse import dok_matrix
import psutil
import traceback

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
    out_node_file: str = typer.Option("", "--out_node_file", "-on")):

    if out_edge_file == "":
        out_edge_file = f'SBM_unmodified_samples/{net_name}/N_graph_edge_list.tsv'
    else:
        out_edge_file = out_edge_file
    
    if out_node_file == "":
        out_node_file = f'SBM_unmodified_samples/{net_name}/N_graph_node_list.tsv'
    else:
        out_node_file = out_node_file
    
    output_dir = os.path.dirname(out_edge_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    logging.basicConfig(filename=f'SBM_unmodified_samples/{net_name}/SBM_unmodified_samples.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    job_start_time = time.time()

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
        # logging.info("Statistics of read graph:")
        # start_time = time.time()
        # stats_df_G, fig = stats.main(edge_input, [], 'original_input_graph')
        # print(stats_df_G)
        # fig.savefig(output_dir+f"/{net_name}_original_degree_distribution.png")
        # logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        logging.info("Reading graph clustering:")
        start_time = time.time()
        cluster_df = read_clustering(cluster_input)
        clustering_dict = dict(zip(cluster_df['node_id'], cluster_df['cluster_id']))
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("")
        logging.info("Assigning all unclustered nodes to a new cluster")
        start_time = time.time()
        new_cluster_id = np.max(cluster_df['cluster_id']) + 1
        node_mapping_reversed = {v: int(k) for k, v in node_mapping.items()}
        clustered_nodes_id_org = cluster_df['node_id'].to_numpy()
        nodes_set = set()
        for u in G.iterNodes():
            nodes_set.add(node_mapping_reversed.get(u))
        # clustered_nodes = cluster_df['node_id'].unique()
        # clustered_nodes_mapped = set([node_mapping.get(str(v)) for v in clustered_nodes])
        unclustered_nodes = nodes_set.difference(clustered_nodes_id_org)
        
        unclustered_node_cluster_mapping = []
        for v in unclustered_nodes:
            row = {'node_id': v, 'cluster_id' : new_cluster_id}
            unclustered_node_cluster_mapping.append(row)
            new_cluster_id += 1
        
        # cluster_df = pd.concat([cluster_df, pd.DataFrame([row])], ignore_index=True)
        cluster_df = cluster_df._append(unclustered_node_cluster_mapping, ignore_index=True)
        cluster_df = cluster_df.reset_index()
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("")
        # logging.info("Statistics of read graph:")
        # start_time = time.time()
        # stats_df_G, fig, participation_coeffs_G ,participation_dict_G = stats.main(edge_input, unclustered_nodes, 'original_input_graph', "", clustering_dict)
        # print(stats_df_G)
        # fig.savefig(output_dir+f"/{net_name}_original_degree_distribution.png")
        # logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        logging.info("Edges calculation of read graph:")
        clustered_nodes_id_org_set = set(clustered_nodes_id_org)
        start_time = time.time()
        clustered_edges_org = 0
        outlier_cluster_edges_org = 0
        outlier_edges_org = 0
        for edge in G.iterEdges():
            source = node_mapping_reversed.get(min(edge[0],edge[1]))
            target = node_mapping_reversed.get(max(edge[0],edge[1]))
            if source in clustered_nodes_id_org_set and target in clustered_nodes_id_org_set:
                clustered_edges_org += 1
            elif source in clustered_nodes_id_org_set or target in clustered_nodes_id_org_set:
                outlier_cluster_edges_org += 1
            else:
                outlier_edges_org += 1
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        logging.info("Getting edge probs matrix for G")
        start_time = time.time()
        probs = get_probs(G, node_mapping, cluster_df)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("After getting probs matrix")
        logging.info("Generating Synthetic graph N_c with probs")
        start_time = time.time()
        cluster_assignment = cluster_df['cluster_id'].to_numpy()
        out_deg_seq = get_degree_sequence(cluster_df, G, node_mapping, 3, [])
        N = gt.generate_sbm(cluster_assignment, probs, out_degs=out_deg_seq, micro_ers=True, micro_degs=True)
        print("N graph statistics : ", N.num_vertices(), N.num_edges())
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("")
        logging.info("Saving generated graph edge list! ")
        start_time = time.time()
        N_edge_list = set()
        node_idx_set = cluster_df['node_id'].tolist()
        N_self_loops = set()
        cluster_edges = 0
        outlier_edges = 0
        outlier_cluster_edges = 0
        for edge in N.get_edges():
            source = node_idx_set[min(edge[0],edge[1])]
            target = node_idx_set[max(edge[1], edge[0])]
            if (source==target):
                N_self_loops.add((source, target))
            N_edge_list.add((source, target))
            if(source != target):
                if source in unclustered_nodes and target in unclustered_nodes:
                    outlier_edges += 1
                elif source in unclustered_nodes or target in unclustered_nodes:
                    outlier_cluster_edges += 1
                else:
                    cluster_edges += 1

        save_generated_graph(N_edge_list, out_edge_file)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("After saving N graph edges")
        # logging.info("Statistics of generated graph N:")
        # start_time = time.time()
        # stats_df_N, fig,participation_coeffs_N,participation_dict_N = stats.main(out_edge_file, unclustered_nodes, 'N',"",clustering_dict)
        # print(stats_df_N)
        # fig.savefig(output_dir+f"/{net_name}_N_degree_distribution.png")
        # combined_df = pd.concat([stats_df_G, stats_df_N])
        # combined_df = combined_df.reset_index()
        # combined_df.columns = ['Metric','Stat']
        # combined_df.loc[len(combined_df.index)] = ['Num_of_clustered_nodes', len(clustering_dict.keys())]
        # combined_df.loc[len(combined_df.index)] = ['Num_of_clustered_edges', clustered_edges_org]
        # combined_df.loc[len(combined_df.index)] = ['Num_of_outlier_non_outlier_edges', outlier_cluster_edges_org]
        # combined_df.loc[len(combined_df.index)] = ['generated_Num_of_clustered_nodes', len(clustering_dict.keys())]
        # combined_df.loc[len(combined_df.index)] = ['Generated_clustered_num_edges', cluster_edges]
        # combined_df.loc[len(combined_df.index)] = ['Generated_outlier_non_outlier_edges', outlier_cluster_edges]
        # combined_df.loc[len(combined_df.index)] = ['Generated_outlier_outlier_edges', outlier_edges]
        # print(combined_df)
        # combined_df.to_csv(f'{output_dir}/{net_name}_stats.csv')
        # logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        # log_cpu_ram_usage("After N graph computation!!")
        logging.info(f"Total Time taken: {round(time.time() - job_start_time, 3)} seconds")
        log_cpu_ram_usage("Usage statistics after job completion!")

    except Exception as e:
        print(e)
        traceback.print_exc()
        logging.error("Exception occurred", exc_info=True)

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
    
    args = parser.parse_args()

    typer.run(main)
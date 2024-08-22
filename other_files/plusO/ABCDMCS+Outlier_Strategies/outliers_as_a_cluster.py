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
import networkit as nk
from scipy.sparse import dok_matrix
import psutil
import traceback
import shutil

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

def deg_sampler(index):
    global degree_seq
    return degree_seq[index]

def generate_graph(deg_seq):
    global degree_seq
    degree_seq = deg_seq
    N = len(deg_seq)
    generated_graph = gt.random_graph(N, deg_sampler=deg_sampler, directed=False, parallel_edges=False, self_loops=False)
    return generated_graph

def rewire_non_singleton_components(non_singleton_components, deg_sequences, node_mapping):
    numerical_to_string_mapping = {v: k for k, v in node_mapping.items()}
    edge_lists = []
    for idx,component in enumerate(non_singleton_components):
        generated_graph = generate_graph(deg_sequences[idx])
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

def copy_and_append(src_file, dest_file, new_edges):
    shutil.copy(src_file, dest_file)
    print(f"File copied from {src_file} to {dest_file}")

    with open(dest_file, 'a') as f:
        for edge in new_edges:
            f.write(str(edge[0])+"\t"+str(edge[1])+"\n")
        print(f"Appended new edges to {dest_file}")

def main(edge_input: str = typer.Option(..., "--filepath", "-f"),
    cluster_input: str = typer.Option(..., "--cluster_filepath", "-c"),
    net_name: str = typer.Option(..., "--net_name", "-m"),
    out_edge_file: str = typer.Option("", "--out_edge_file", "-oe"),
    subnet_file: str = typer.Option("", "--subnet_file", "-s")):

    if out_edge_file == "":
        out_edge_file = f'ABCDMCS_outlier_strategy_OC_samples/{net_name}/0.001/rep_0.tsv'
    else:
        out_edge_file = out_edge_file
    
    output_dir = os.path.dirname(out_edge_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    log_path = os.path.join(output_dir, f'{net_name}.log')
    log_dir = os.path.dirname(log_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(filename=log_path, filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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

        logging.info("Reading graph clustering:")
        start_time = time.time()
        cluster_df = read_clustering(cluster_input)
        clustering_dict = dict(zip(cluster_df['node_id'], cluster_df['cluster_id']))
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("")

        logging.info("Getting subgraph G_c")
        start_time = time.time()
        clustered_nodes = cluster_df['node_id'].unique()
        clustered_nodes_mapped = [node_mapping.get(str(v)) for v in clustered_nodes]
        G_c = nk.graphtools.subgraphFromNodes(G, clustered_nodes_mapped)
        num_clustered_nodes, num_cluster_edges = get_graph_stats(G_c)
        print("Number of clustered nodes : ", num_clustered_nodes , "\t Number of clustered edges : ", num_cluster_edges)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("")

        logging.info("Getting subgraph G_star :  G - G_c")
        start_time = time.time()
        G_star = remove_edges(G,G_c)
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
        
        # cluster_df = pd.concat([cluster_df, pd.DataFrame([row])], ignore_index=True)
        cluster_df = cluster_df._append(unclustered_node_cluster_mapping, ignore_index=True)
        cluster_df = cluster_df.reset_index()
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("")
        
        N_edge_list = set()
        if len(unclustered_nodes)>0:
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

            

            logging.info("Getting edge probs matrix for G_star")
            start_time = time.time()
            probs = get_probs(G_star, node_mapping, cluster_df)
            logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
            log_cpu_ram_usage("After getting probs matrix")
            logging.info("Generating Synthetic graph N_star with probs")
            start_time = time.time()
            cluster_assignment = cluster_df['cluster_id'].to_numpy()
            out_deg_seq = get_degree_sequence(cluster_df, G_star, node_mapping, 3, [])
            N = gt.generate_sbm(cluster_assignment, probs, out_degs=out_deg_seq, micro_ers=True, micro_degs=True)
            print("N graph statistics : ", N.num_vertices(), N.num_edges())
            logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
            log_cpu_ram_usage("")
            logging.info("Saving generated graph edge list! ")
            start_time = time.time()
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

        # save_generated_graph(N_edge_list, out_edge_file)
        copy_and_append(subnet_file, out_edge_file, N_edge_list)
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
        log_cpu_ram_usage("After N graph computation!!")
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
        '-s', metavar='subnet_file', type=str, required=True,
        help='clustered component edge list'
        )
    
    args = parser.parse_args()

    typer.run(main)
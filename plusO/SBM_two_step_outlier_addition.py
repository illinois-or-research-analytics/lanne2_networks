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


def save_generated_graph(edges_list, out_edge_file):
    edge_df = pd.DataFrame(edges_list, columns=['source', 'target'])
    edge_df.to_csv(out_edge_file, sep='\t', index=False, header=None)

def main(edge_input: str = typer.Option(..., "--filepath", "-f"),
    cluster_input: str = typer.Option(..., "--cluster_filepath", "-c"),
    net_name: str = typer.Option(..., "--net_name", "-m"),
    out_edge_file: str = typer.Option("", "--out_edge_file", "-oe"),
    out_node_file: str = typer.Option("", "--out_node_file", "-on")):

    if out_edge_file == "":
        out_edge_file = f'SBM_twostep_outlier_addition_samples/{net_name}/N_graph_edge_list.tsv'
    else:
        out_edge_file = out_edge_file
    
    if out_node_file == "":
        out_node_file = f'SBM_twostep_outlier_addition_samples/{net_name}/N_graph_node_list.tsv'
    else:
        out_node_file = out_node_file
    
    output_dir = os.path.dirname(out_edge_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    logging.basicConfig(filename=f'SBM_twostep_outlier_addition_samples/{net_name}/SBM_twostep_outlier_addition_samples.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("")
        
        logging.info("Compute edges for each unclustered node")
        start_time = time.time()
        clustering_dict = dict(zip(cluster_df['node_id'], cluster_df['cluster_id']))
        outlier_cluster_edge_counts = {}
        node_mapping_reversed = {v: int(k) for k, v in node_mapping.items()}
        clustered_nodes_id_org = cluster_df['node_id'].to_numpy()
        nodes_set = set()
        for u in G.iterNodes():
            nodes_set.add(node_mapping_reversed.get(u))
        unclustered_nodes = nodes_set.difference(clustered_nodes_id_org)

        for node in unclustered_nodes:
            clustering_dict[node] = -1

        for edge in G.iterEdges():
            source = node_mapping_reversed.get(edge[0])
            target = node_mapping_reversed.get(edge[1])
            if (source in unclustered_nodes):
                cluster_id_of_target = clustering_dict.get(target)
                if source in outlier_cluster_edge_counts.keys():
                    cluster_edge_counts = outlier_cluster_edge_counts.get(source)
                    if cluster_id_of_target in cluster_edge_counts.keys():
                        cluster_edge_counts[cluster_id_of_target] = cluster_edge_counts.get(cluster_id_of_target)+1
                    else:
                        cluster_edge_counts[cluster_id_of_target] = 1
                else:
                    outlier_cluster_edge_counts[source] = {cluster_id_of_target : 1}
                
            if (target in unclustered_nodes):
                cluster_id_of_target = clustering_dict.get(source)
                if target in outlier_cluster_edge_counts.keys():
                    cluster_edge_counts = outlier_cluster_edge_counts.get(target)
                    if cluster_id_of_target in cluster_edge_counts.keys():
                        cluster_edge_counts[cluster_id_of_target] = cluster_edge_counts.get(cluster_id_of_target)+1
                    else:
                        cluster_edge_counts[cluster_id_of_target] = 1
                    cluster_edge_counts[cluster_id_of_target] = cluster_edge_counts.get(cluster_id_of_target)+1
                else:
                    outlier_cluster_edge_counts[target] = {cluster_id_of_target : 1}

        print("len(unclustered_nodes), len(outlier_cluster_edge_count.keys()) : ", len(unclustered_nodes), len(outlier_cluster_edge_counts.keys()))

        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("")

        logging.info("Statistics of read graph:")
        start_time = time.time()
        stats_df_G, fig = stats.main(edge_input, unclustered_nodes, 'original_input_graph')
        print(stats_df_G)
        fig.savefig(output_dir+f"/{net_name}_original_degree_distribution.png")
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")

        logging.info("Getting subgraph G_c")
        start_time = time.time()
        clustered_nodes = cluster_df['node_id'].unique()
        clustered_nodes_mapped = [node_mapping.get(str(v)) for v in clustered_nodes]
        G_c = nk.graphtools.subgraphFromNodes(G, clustered_nodes_mapped)
        num_clustered_nodes, num_cluster_edges = get_graph_stats(G_c)
        print("Number of clustered nodes : ", num_clustered_nodes , "\t Number of clustered edges : ", num_cluster_edges)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("")

        logging.info("Getting edge probs matrix for G_c")
        start_time = time.time()
        probs = get_probs(G_c, node_mapping, cluster_df)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("After getting probs matrix")

        logging.info("Generating Synthetic graph N_c with probs")
        start_time = time.time()
        cluster_assignment = cluster_df['cluster_id'].to_numpy()
        out_deg_seq = get_degree_sequence(cluster_df, G_c, node_mapping, 3, [])
        N_c = gt.generate_sbm(cluster_assignment, probs, out_degs=out_deg_seq, micro_ers=True, micro_degs=True)
        print("N_c graph statistics : Vertices , Edges : ", N_c.num_vertices(), N_c.num_edges())
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("")

        logging.info("Randomly assigning edges to unclustered nodes and Saving generated graph edge list!")
        start_time = time.time()
        N_edge_list = set()

        unclustered_node_cluster_mapping = []
        for v in unclustered_nodes:
            row = {'node_id': v, 'cluster_id' : -1}
            unclustered_node_cluster_mapping.append(row) 
        cluster_df = cluster_df._append(unclustered_node_cluster_mapping, ignore_index=True)
        cluster_df = cluster_df.reset_index()    

        node_idx_set = cluster_df['node_id'].tolist()
        for edge in N_c.get_edges():
            source = node_idx_set[edge[0]]
            target = node_idx_set[edge[1]]
            N_edge_list.add((source, target))
        
        cluster_id_idx_dict = {}
        for index,row in cluster_df.iterrows():
            cluster_id = row['cluster_id']
            if cluster_id in cluster_id_idx_dict:
                cluster_id_idx_dict[cluster_id].append(index)
            else:
                cluster_id_idx_dict[cluster_id] = [index]

        # # Checking the outlier edge count:
        # count = 0
        # for v in unclustered_nodes:
        #     cluster_edge_counts = outlier_cluster_edge_counts.get(v)
        #     for cluster_id in cluster_edge_counts.keys():
        #         if cluster_id == -1:
        #             count += cluster_edge_counts.get(cluster_id)/2
        #         else:
        #             count += cluster_edge_counts.get(cluster_id)
        # print("Total num of outlier edges as per cluster level dict : " , count)

        
        count = 0
        for source in unclustered_nodes:
            cluster_edge_counts = outlier_cluster_edge_counts.get(source)
            for cluster_id in cluster_edge_counts.keys():
                num_nodes_in_cluster = len(cluster_id_idx_dict.get(cluster_id))
                num_edges_in_cluster = cluster_edge_counts.get(cluster_id)
                random_indices = np.random.choice(num_nodes_in_cluster, num_edges_in_cluster)
                node_idxs_in_cluster = cluster_id_idx_dict.get(cluster_id)
                for random_index in random_indices:
                    target = node_idx_set[node_idxs_in_cluster[random_index]]
                    N_edge_list.add((source, target))
                    count += 1
        print("Number of outlier edges added : ", count)
        print("Number of outlier edges to be added : ", (G.numberOfEdges() - G_c.numberOfEdges()))

        save_generated_graph(N_edge_list, out_edge_file)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("After saving N graph edges")
        logging.info("Statistics of generated graph N:")
        start_time = time.time()
        stats_df_N, fig = stats.main(out_edge_file, unclustered_nodes, 'N')
        print(stats_df_N)
        fig.savefig(output_dir+f"/{net_name}_N_degree_distribution.png")
        combined_df = pd.concat([stats_df_G, stats_df_N])
        print(combined_df)
        combined_df.to_csv(f'{output_dir}/{net_name}_stats.csv')
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
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
        '-on', metavar='out_node_file', type=str, required=False,
        help='output nodelist path'
        )
    
    args = parser.parse_args()

    typer.run(main)
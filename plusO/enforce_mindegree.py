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
from scipy.sparse import dok_matrix
from pymincut.pygraph import PyGraph
import shutil

import warnings
warnings.filterwarnings("ignore")

from collections import defaultdict


def read_graph(input_network):
    # Read the network
    elr = nk.graphio.EdgeListReader('\t', 0, continuous=False, directed=False)
    graph = elr.read(input_network)
    graph.removeMultiEdges()
    graph.removeSelfLoops()
    node_mapping_dict = elr.getNodeMap()
    return graph, node_mapping_dict

def read_clustering(input_clustering, required_cluster_stats):
    # Read the clustering
    cluster_df = pd.read_csv(input_clustering, sep="\t", header=None, names=[
                             "node_id", "cluster_name"], dtype=str)
    
    # Read required cluster stats
    cluster_stats_df = pd.read_csv(required_cluster_stats)
    cluster_stats_df['cluster'] = cluster_stats_df['cluster'].str.strip('"')
    cluster_stats_df = cluster_stats_df[cluster_stats_df['cluster'] != 'Overall']
    cluster_stats_df = cluster_stats_df[['cluster', 'connectivity']]

    return cluster_df, cluster_stats_df

def remove_edges(G_c, edges_to_remove):
    for edge in edges_to_remove:
        G_c.removeEdge(edge[0], edge[1])

def save_generated_graph(edges_list, out_edge_file):
    edge_df = pd.DataFrame(edges_list, columns=['source', 'target'])
    edge_df.to_csv(out_edge_file, sep='\t', index=False, header=None)

def copy_and_append(src_file, dest_file, new_edges):
    shutil.copy(src_file, dest_file)
    print(f"File copied from {src_file} to {dest_file}")

    with open(dest_file, 'a') as f:
        for edge in new_edges:
            f.write(edge[0]+"\t"+edge[1]+"\n")
        print(f"Appended new edges to {dest_file}")

def main(edge_input: str = typer.Option(..., "--filepath", "-f"),
    cluster_input: str = typer.Option(..., "--cluster_filepath", "-c"),
    net_name: str = typer.Option(..., "--net_name", "-m"),
    cluster_stats: str = typer.Option(..., "--cluster_stats", "-cs"),
    out_edge_file: str = typer.Option("", "--out_edge_file", "-oe")):

    out_edge_file2 = f'SBM_enforce_mincut_samples/{net_name}_graph_edge_list_copied.tsv'
    if out_edge_file == "":
        out_edge_file = f'SBM_enforce_mincut_samples/{net_name}_graph_edge_list.tsv'
    else:
        out_edge_file = out_edge_file
        
    
    output_dir = os.path.dirname(out_edge_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    logging.basicConfig(filename=f'SBM_enforce_mincut_samples/logs/SBM_enforce_mincut_{net_name}.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
        logging.info("Reading input graph...")
        start_time = time.time()
        graph,node_mapping = read_graph(edge_input)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        print("Number of nodes in the network : ", graph.numberOfNodes())
        print("Number of edges in the network : ",graph.numberOfEdges())
        
        log_cpu_ram_usage("After reading graph!")
        
        logging.info("Reading graph clustering:")
        start_time = time.time()
        cluster_df,cluster_stats_df = read_clustering(cluster_input, cluster_stats)
        ## -- clustering_dict  = {node_id : cluster_id} --
        clustering_dict = dict(
                zip(
                    cluster_df['node_id'],
                    cluster_df['cluster_name'],
                )
            )
        ## -- cluster_node_mapping  = {cluster_id : set(node_iid)} --
        cluster_node_mapping = defaultdict(set)
        for node, cluster in clustering_dict.items():
            cluster_node_mapping[cluster].add(node_mapping[node])

        
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("After reading clustering data!")

        logging.info("Getting clustered component of the network!")
        start_time = time.time()
        clustered_nodes = [node_mapping[v] for v in clustering_dict.keys()]
        G_c = nk.graphtools.subgraphFromNodes(graph, clustered_nodes)

        edges_to_remove = []
        clustered_nodes_set = set(clustered_nodes)
        ## node_mapping = {node_id : node_iid}
        ## node_mapping_reversed = {node_iid : node_id}
        node_mapping_reversed = {u:str(v) for v,u in node_mapping.items()}
        for edge in G_c.iterEdges():
            if clustering_dict[node_mapping_reversed[edge[0]]] != clustering_dict[node_mapping_reversed[edge[1]]]:
                edges_to_remove.append(edge)

        print("Number of edges in G_c : ",G_c.numberOfEdges())
        remove_edges(G_c, edges_to_remove)
        print("Number of edges in G_c after removing inter cluster edges : ",G_c.numberOfEdges())
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("After computing G_c!")
        
        # logging.info("Ensuring connected clusters!")
        # start_time = time.time()
        # disconnected_clusters = 0
        # num_connected_components = []
        # processed_clusters = 0
        # for cluster_id in cluster_node_mapping.keys():
        #     nodes_c_id = list(cluster_node_mapping[cluster_id])
        #     G_c_id = nk.graphtools.subgraphFromNodes(G_c, nodes_c_id)
        #     cc = nk.components.ConnectedComponents(G_c_id)
        #     cc.run()
        #     num_cc_c_id = cc.numberOfComponents()
        #     if num_cc_c_id > 1:
        #         num_connected_components.append(num_cc_c_id)
        #         disconnected_clusters += 1
        #     processed_clusters += 1
        #     if(processed_clusters%1000==0):
        #         print(processed_clusters)
        # print(disconnected_clusters, (disconnected_clusters/len(cluster_node_mapping.keys())))
        # print(min(num_connected_components), max(num_connected_components), np.mean(num_cc_c_id))
        # logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")


        logging.info("Ensuring minimum degree")
        start_time = time.time()
        cluster_count = 0
        total_edges = 0
        new_edges = set()
        for cluster_id in cluster_node_mapping.keys():
            nodes = list(cluster_node_mapping[cluster_id])
            min_cut_required = int(cluster_stats_df[cluster_stats_df['cluster']==cluster_id]['connectivity'])
            for node in nodes:
                node_deg = G_c.degree(node)
                if G_c.degree(node)<min_cut_required:
                    deg_diff  = min_cut_required - node_deg
                    # total_edges += deg_diff
                    selected = set()
                    selected.add(node)
                    while deg_diff>0:
                        idx = np.random.choice(len(nodes), 1)
                        edge_end = nodes[idx[0]]
                        if edge_end not in selected and edge_end not in set(G_c.iterNeighbors(node)):
                            G_c.addEdge(node, edge_end)
                            selected.add(edge_end)
                            deg_diff -= 1
                            total_edges += 1
                            new_edges.add((node,edge_end))

            cluster_count += 1
            if cluster_count%1000 ==0:
                print("Number of clusters processed : ", cluster_count)
        print("Number of edges added : ", total_edges, len(new_edges))
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        logging.info("Saving new graph!")
        start_time = time.time()
        new_edges_ids = [(node_mapping_reversed[u], node_mapping_reversed[v]) for (u,v) in new_edges]
        copy_and_append(edge_input, out_edge_file, new_edges_ids)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")


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
        '-cs', metavar='cluster_stats', type=str, required=True,
        help='stats of the clusters of network'
        )
    parser.add_argument(
        '-oe', metavar='out_edge_file', type=str, required=False,
        help='output edgelist path'
        )
    
    
    args = parser.parse_args()

    typer.run(main)
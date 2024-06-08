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

from typing import Dict, List
from hm01.graph import Graph, IntangibleSubgraph
from hm01.mincut import viecut
import csv

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
    unique_values = cluster_df['cluster_name'].unique()
    value_map = {
        value: idx
        for idx, value in enumerate(unique_values)
    }
    cluster_df['cluster_id'] = cluster_df['cluster_name'].map(value_map)
    
    
    # Read required cluster stats
    cluster_stats_df = pd.read_csv(required_cluster_stats)
    cluster_stats_df['cluster'] = cluster_stats_df['cluster'].str.strip('"')
    cluster_stats_df = cluster_stats_df[cluster_stats_df['cluster'] != 'Overall']
    cluster_stats_df = cluster_stats_df[['cluster', 'connectivity']]

    return cluster_df, cluster_stats_df, value_map

def remove_edges(G_c, edges_to_remove):
    for edge in edges_to_remove:
        G_c.removeEdge(edge[0], edge[1])
def add_edges(graph, new_edges):
    for edge in new_edges:
        graph.addEdge(edge[0],edge[1])
    return graph

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

def compute_cluster_stats(network_fp, clustering_fp, cluster_iid2id, cluster_order):
    clusters = \
        load_clusters(
            clustering_fp,
            cluster_iid2id,
            cluster_order,
        )
    ids = [
        cluster.index
        for cluster in clusters
    ]
    ns = [
        cluster.n()
        for cluster in clusters
    ]

    # TODO: check this reader
    edgelist_reader = nk.graphio.EdgeListReader("\t", 0)
    nk_graph = edgelist_reader.read(network_fp)

    global_graph = Graph(nk_graph, "")
    ms = [
        cluster.count_edges(global_graph)
        for cluster in clusters
    ]

    clusters = [
        cluster.realize(global_graph)
        for cluster in clusters
    ]

    mincuts = [
        viecut(cluster)[-1]
        for cluster in clusters
    ]
    mincuts_normalized = [
        mincut / np.log10(ns[i])
        for i, mincut in enumerate(mincuts)
    ]

    df = pd.DataFrame(
        list(
            zip(
                ids,
                ns,
                ms,
                mincuts,
                mincuts_normalized,
            )
        ),
        columns=[
            'cluster',
            'n',
            'm',
            'connectivity',
            'connectivity_normalized_log10(n)',
        ]
    )

    return df

def load_clusters(filepath, cluster_iid2id, cluster_order) -> List[IntangibleSubgraph]:
    clusters: Dict[str, IntangibleSubgraph] = {}
    with open(filepath) as f:
        csv_reader = csv.reader(f, delimiter='\t')
        for line in csv_reader:
            node_id, cluster_id = line
            clusters.setdefault(
                cluster_id, IntangibleSubgraph([], cluster_id)
            ).subset.append(int(node_id))
    return [
        clusters[cluster_iid2id[cluster_iid]]
        for cluster_iid in cluster_order
    ]

def main(edge_input: str = typer.Option(..., "--filepath", "-f"),
    cluster_input: str = typer.Option(..., "--cluster_filepath", "-c"),
    net_name: str = typer.Option(..., "--net_name", "-m"),
    cluster_stats: str = typer.Option(..., "--cluster_stats", "-cs"),
    out_edge_file: str = typer.Option("", "--out_edge_file", "-oe")):

    if out_edge_file == "":
        out_edge_file = f'SBM_enforce_minconnectivity_samples/{net_name}_mindegree_graph_edge_list.tsv'
    else:
        out_edge_file = out_edge_file

    out_edge_file2 = f'SBM_enforce_minconnectivity_samples/{net_name}_connected_graph_edge_list.tsv'
    out_edge_file3 = f'SBM_enforce_minconnectivity_samples/{net_name}_wellconnected_graph_edge_list.tsv'
  
    
    output_dir = os.path.dirname(out_edge_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    logging.basicConfig(filename=f'SBM_enforce_minconnectivity_samples/logs/SBM_enforce_minconnectivity_{net_name}.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
        cluster_df,cluster_stats_df,cluster_mapping_dict = read_clustering(cluster_input, cluster_stats)
        ## -- clustering_dict  = {node_id : cluster_iid} --
        clustering_dict = dict(
            zip(
                cluster_df['node_id'],
                cluster_df['cluster_id'],
            )
        )
        ## -- cluster_node_mapping  = {cluster_iid : set(node_iid)} --
        cluster_node_mapping = defaultdict(set)
        for node, cluster in clustering_dict.items():
            cluster_node_mapping[cluster].add(node_mapping[node])

        cluster_mapping_dict_reversed = {
            v: k
            for k, v in cluster_mapping_dict.items()
        }
        # cluster_mapping_dict: {cluster_id: cluster_iid}
        # cluster_mapping_dict_reversed: {cluster_iid: cluster_id}
        cluster_order = list(cluster_mapping_dict.values())  # list of cluster_iids
            
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
        
        logging.info("Ensuring minimum degree")
        start_time = time.time()
        cluster_count = 0
        total_edges = 0
        new_edges = set()
        for cluster_id in cluster_order:
            nodes = list(cluster_node_mapping[cluster_id])
            min_cut_required = int(cluster_stats_df[cluster_stats_df['cluster']==cluster_mapping_dict_reversed[cluster_id]]['connectivity'])
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
            if cluster_count%10000 ==0:
                print("Number of clusters processed : ", cluster_count)
        print("Number of edges added : ", total_edges, len(new_edges))
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        logging.info("Saving new graph! - G1")
        start_time = time.time()
        new_edges_ids = [(node_mapping_reversed[u], node_mapping_reversed[v]) for (u,v) in new_edges]
        copy_and_append(edge_input, out_edge_file, new_edges_ids)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")

        logging.info("Adding new edges to graph!")
        start_time = time.time()
        G1 = add_edges(graph,new_edges)
        print(graph.numberOfEdges(), G1.numberOfEdges())
        logging.info(f"Total time taken: {round(time.time()-start_time,3)} seconds")

        logging.info("Compute cluster stats for new graph! - G1")
        start_time = time.time()

        cluster_stats_G1 = \
            compute_cluster_stats(
                out_edge_file,
                cluster_input,
                cluster_mapping_dict_reversed,
                cluster_order,
            )
        
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("After cluster stats!")

        logging.info("Enforce min connectivity in disconnected clusters!")
        start_time = time.time()
        conn_new_edges = set()
        total_edges = 0
        disconnected_clusters = cluster_stats_G1[cluster_stats_G1['connectivity']==0]['cluster']

        print("Number of disconnected clusters after step - 1 of post processing:  ", len(disconnected_clusters))

        for cluster_id in disconnected_clusters:
            cluster_new_edges = []
            nodes = list(cluster_node_mapping[cluster_mapping_dict[cluster_id]])
            sub_graph = nk.graphtools.subgraphFromNodes(G1,nodes)
            cluster_nodes = list(sub_graph.iterNodes())
            min_cut_required = int(cluster_stats_df[cluster_stats_df['cluster']==cluster_id]['connectivity'])
            cluster_edges = list(sub_graph.iterEdges())
            sub_G = PyGraph(cluster_nodes, cluster_edges)
            mincut_result = sub_G.mincut("cactus", "bqueue", True)

            partitions = mincut_result[:-1]
            cut_size = mincut_result[-1]
            # print("Partition lengths : ", [len(p) for p in partitions])
            req_edges = min_cut_required-cut_size
            largest_index, large_partition = max(enumerate(partitions), key=lambda x: len(x[1]))
            for i in range(len(partitions)):
                if i != largest_index:
                    part_nodes_list = list(partitions[i])
                    node_idxs = np.random.choice(len(large_partition), min_cut_required, replace=False)
                    part_node_idx = np.random.choice(len(part_nodes_list), min_cut_required)
                    for j in range(min_cut_required):
                        n1 = int(part_nodes_list[part_node_idx[j]])
                        n2 = int(large_partition[node_idxs[j]])
                        sub_graph.addEdge(n1,n2)
                        conn_new_edges.add((n1,n2))
                        cluster_new_edges.append((n1,n2))
                        total_edges += 1

        print("Number of edges added : ", total_edges, len(conn_new_edges))

        logging.info("Adding new edges to graph!")
        start_time = time.time()
        G2 = add_edges(G1,conn_new_edges)
        print(G1.numberOfEdges(), G2.numberOfEdges())
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")

        
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")

        logging.info("Saving new graph! G2")
        start_time = time.time()
        conn_new_edges_ids = [(node_mapping_reversed[u], node_mapping_reversed[v]) for (u,v) in conn_new_edges]
        copy_and_append(out_edge_file, out_edge_file2, conn_new_edges_ids)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")

        logging.info("Compute cluster stats for new graph - G2!")
        start_time = time.time()

        cluster_stats_G2 = \
            compute_cluster_stats(
                out_edge_file2,
                cluster_input,
                cluster_mapping_dict_reversed,
                cluster_order,
            )

        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("After cluster stats of G2!")

        logging.info("Enforcing minconnectivity in connected clusters of G2!")
        start_time = time.time()
        poor_connected_clusters_G2 = cluster_stats_G2[cluster_stats_G2['connectivity_normalized_log10(n)']<1]['cluster']
        print("Number of poorly connected clusters after step-2 of processing:  ", len(poor_connected_clusters_G2))
        mincut_edges = []
        clusters_processed = 0
        for cluster_id in poor_connected_clusters_G2:
            all_partitions = []
            nodes = list(cluster_node_mapping[cluster_mapping_dict[cluster_id]])
            sub_graph = nk.graphtools.subgraphFromNodes(G2,nodes)
            cluster_nodes = list(sub_graph.iterNodes())
            min_cut_required = int(cluster_stats_df[cluster_stats_df['cluster']==cluster_id]['connectivity'])
            cluster_edges = list(sub_graph.iterEdges())
            sub_G = PyGraph(cluster_nodes, cluster_edges)
            mincut_result = sub_G.mincut("cactus", "bqueue", True)

            partitions = list(mincut_result[:-1])
            cut_size = mincut_result[-1]
            # print("Partition lengths : ", [len(p) for p in partitions])
            # req_edges = min_cut_required-cut_size
            # partitions = sorted(partitions, key=len, reverse=True)
            while len(partitions)>=1:
                # print(f"{cluster_id} length of partitions" , len(partitions))
                # print(f"{cluster_id} - partitions : ",partitions)
                partition = partitions[0]
                if len(partition)>1:
                    partition_sub_graph = nk.graphtools.subgraphFromNodes(G2,partition)
                    part_sub_G = PyGraph(partition, list(partition_sub_graph.iterEdges()))
                    part_mincut_result = part_sub_G.mincut("cactus", "bqueue", True)
                    part_cut_size = part_mincut_result[-1]
                    if part_cut_size>= min_cut_required:
                        all_partitions.append(partition)
                    else:
                        partitions.extend(part_mincut_result[:-1])
                else:
                    all_partitions.append(partition)
                # print(f"{cluster_id} - All partitions : ",all_partitions)
                partitions.remove(partition)
                # print(f"{cluster_id} length of partitions" , len(partitions))
            # print(f"{cluster_id}  processed, ", len(all_partitions))

            all_partitions = sorted(all_partitions, key=len, reverse=True)
            largest_part = all_partitions[0]
            for j in range(1, len(all_partitions)):
                node_idx = np.random.choice(len(largest_part), min_cut_required, replace=False)
                part_idx = np.random.choice(len(all_partitions[j]), min_cut_required)
                for k in range(min_cut_required):
                    n1 = largest_part[node_idx[k]]
                    n2 = all_partitions[j][part_idx[k]]
                    G2.addEdge(n1,n2)
                    mincut_edges.append((n1,n2))
            
            clusters_processed += 1
            if(clusters_processed%100)==0:
                print("Clusters processed : ",clusters_processed)

        print("Total number of mincut edges added : ", len(mincut_edges))
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")

        logging.info("Adding new edges to graph - G3!")
        start_time = time.time()
        G3 = add_edges(G2,mincut_edges)
        print(G2.numberOfEdges(), G3.numberOfEdges())
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")

        logging.info("Saving new graph! G3")
        start_time = time.time()
        mincut_new_edges_ids = [(node_mapping_reversed[u], node_mapping_reversed[v]) for (u,v) in mincut_edges]
        copy_and_append(out_edge_file2, out_edge_file3, mincut_new_edges_ids)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")

        logging.info("Compute cluster stats for new graph - G3!")
        start_time = time.time()
        cluster_stats_G3 = \
            compute_cluster_stats(
                out_edge_file3,
                cluster_input,
                cluster_mapping_dict_reversed,
                cluster_order,
            )
        poor_connected_clusters_G3 = cluster_stats_G3[cluster_stats_G3['connectivity_normalized_log10(n)']<1]
        print("Number of poor connected clusters after Step-3 of processing : ", len(poor_connected_clusters_G3))
        print(poor_connected_clusters_G3.to_string(index=False))
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("After cluster stats of G2!")

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
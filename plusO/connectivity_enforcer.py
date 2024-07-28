import pandas as pd
import numpy as np
import argparse
import typer
import os
import time
import logging
import networkit as nk
import psutil
import traceback
from pymincut.pygraph import PyGraph
import shutil

from typing import Dict, List
from hm01.graph import Graph, IntangibleSubgraph
from hm01.mincut import viecut
import csv
import heapq

import plusedges_v2 as pe_v2


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

def read_clustering(input_clustering):
    # Read the clustering
    cluster_df = pd.read_csv(input_clustering, sep="\t", header=None, names=[
                             "node_id", "cluster_name"], dtype=str)
    unique_values = cluster_df['cluster_name'].unique()
    value_map = {
        value: idx
        for idx, value in enumerate(unique_values)
    }
    cluster_df['cluster_id'] = cluster_df['cluster_name'].map(value_map)

    return cluster_df, value_map

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
    output_dir: str = typer.Option("", "--output_dir", "-o"),
    emp_edge_input: str = typer.Option(..., "--empirical_filepath", "-ef")):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    out_edge_file = os.path.join(output_dir, f'ce.tsv')

    out_edge_file1 = out_edge_file.replace('ce.tsv','ce_mindeg.tsv')
    out_edge_file2 = out_edge_file.replace('ce.tsv','ce_connected.tsv')
    out_edge_file3 = out_edge_file.replace('ce.tsv','ce_wellconnected.tsv')
    out_edge_file4 = out_edge_file.replace('ce.tsv','ce_plusedges_v1.tsv')
    out_edge_file5 = out_edge_file.replace('ce.tsv','ce_plusedges_v2.tsv')
    
    
    
    log_path = os.path.join(output_dir, f'ce.log')
    # log_dir = os.path.dirname(log_path)
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)
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
        logging.info("Reading input graph G!")
        start_time = time.time()
        graph,node_mapping = read_graph(edge_input)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        print("Number of nodes in the network : ", graph.numberOfNodes())
        print("Number of edges in the network : ",graph.numberOfEdges())
        
        log_cpu_ram_usage("After reading graph!")
        
        logging.info("Reading graph clustering:")
        start_time = time.time()
        cluster_df,cluster_mapping_dict = read_clustering(cluster_input)
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

        logging.info("Compute cluster stats for empirical graph! - emp_G")
        start_time = time.time()

        cluster_stats_df = \
            compute_cluster_stats(
                emp_edge_input,
                cluster_input,
                cluster_mapping_dict_reversed,
                cluster_order,
            )
        
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("After cluster stats!")

        logging.info("Reading empirical graph emp_G!")
        start_time = time.time()
        emp_graph,emp_node_mapping = read_graph(emp_edge_input)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        print("Number of nodes in the empirical network : ", emp_graph.numberOfNodes())
        print("Number of edges in the empirical network : ",emp_graph.numberOfEdges())
        
        log_cpu_ram_usage("After reading empirical graph!")


        logging.info("Getting clustered component (without inter cluster edges) of the network G!")
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
        
        logging.info("Finding nodes with available degrees")
        start_time = time.time()
        ## Find clustered nodes with degree less than expected
        emp_degrees = dict()
        sbm_degrees = dict()
        available_node_degrees = dict()
    
        ## Correction
        for c_node in clustering_dict.keys(): 
            emp_node = emp_node_mapping[c_node]
            syn_node = node_mapping[c_node]
            emp_degrees[c_node] = emp_graph.degree(emp_node)
            sbm_degrees[c_node] = graph.degree(syn_node)
            deg_diff = emp_degrees[c_node]-sbm_degrees[c_node]
            if(deg_diff>0):
                available_node_degrees[syn_node] = deg_diff

        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")

        ## Get all the nodes with available degrees
        available_node_set = set(available_node_degrees.keys())
        
        logging.info("Stage 1 : Ensuring minimum degree")
        start_time = time.time()
        cluster_count = 0
        total_edges = 0
        degree_corrected = 0
        new_edges = set()
       
        for cluster_id in cluster_order:
            nodes = set(cluster_node_mapping[cluster_id])
            available_c_nodes = available_node_set.intersection(nodes)
            min_cut_required = int(cluster_stats_df[cluster_stats_df['cluster'] == cluster_mapping_dict_reversed[cluster_id]]['connectivity'])
            
            for node in nodes:
                neighbors = set(G_c.iterNeighbors(node))
                neighbors.add(node)
                available_non_neighbors = available_c_nodes - neighbors
                node_deg = G_c.degree(node)
                
                if node_deg < min_cut_required:
                    deg_diff = min_cut_required - node_deg
                    selected = set()
                    selected.add(node)
                    
                    while deg_diff > 0:
                        edge_end = None
                        idx_available = False
                        if len(available_non_neighbors)>0:
                            edge_end = np.random.choice(list(available_non_neighbors))
                            idx_available = True
                        else:
                            edge_end = np.random.choice(list(nodes))
                        
                        if edge_end not in selected and edge_end not in neighbors:
                            G_c.addEdge(node, edge_end)
                            selected.add(edge_end)
                            deg_diff -= 1
                            total_edges += 1
                            new_edges.add((node, edge_end))
                            neighbors.add(edge_end)
                            
                            ## reduce available degree and remove if node is exhausted
                            if idx_available:
                                degree_corrected += 1
                                available_node_degrees[edge_end] -= 1
                                available_non_neighbors.remove(edge_end)
                                if available_node_degrees[edge_end] == 0:
                                    available_c_nodes.remove(edge_end)
                                    available_node_set.remove(edge_end)
                                    del available_node_degrees[edge_end]

                    if node in available_node_degrees.keys():
                        available_node_degrees[node] -= deg_diff
                        if available_node_degrees[node] <= 0:
                                    available_c_nodes.discard(node)
                                    available_node_set.discard(node)
                                    del available_node_degrees[node]

            cluster_count += 1
            if cluster_count % 10000 == 0:
                print("Number of clusters processed: ", cluster_count)
        print("Number of edges added : ", total_edges, len(new_edges))
        if total_edges > 0:
            print("Number and percentage of degree correction edges added out of total edges added : ", degree_corrected, (degree_corrected/total_edges * 100))
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")

        logging.info("Saving new graph! - G1")
        start_time = time.time()
        new_edges_ids = [(node_mapping_reversed[u], node_mapping_reversed[v]) for (u,v) in new_edges]
        copy_and_append(edge_input, out_edge_file1, new_edges_ids)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")

        logging.info("Adding new edges to graph!")
        start_time = time.time()
        G1 = add_edges(graph,new_edges)
        print(graph.numberOfEdges(), G1.numberOfEdges())
        logging.info(f"Time taken: {round(time.time()-start_time,3)} seconds")

        logging.info("Compute cluster stats for new graph! - G1")
        start_time = time.time()

        cluster_stats_G1 = \
            compute_cluster_stats(
                out_edge_file1,
                cluster_input,
                cluster_mapping_dict_reversed,
                cluster_order,
            )
        
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("After cluster stats!")

        logging.info("Stage 2 : Enforce connectivity in disconnected clusters!")
        start_time = time.time()
        conn_new_edges = set()
        total_edges = 0
        degree_corrected = 0
        disconnected_clusters = cluster_stats_G1[cluster_stats_G1['connectivity']==0]['cluster']

        print("Number of disconnected clusters after step - 1 of post processing:  ", len(disconnected_clusters))

        for cluster_id in disconnected_clusters:
            # cluster_new_edges = []
            nodes = set(cluster_node_mapping[cluster_mapping_dict[cluster_id]])
            available_c_nodes = available_node_set.intersection(nodes)
            # nodes = list(cluster_node_mapping[cluster_mapping_dict[cluster_id]])
            sub_graph = nk.graphtools.subgraphFromNodes(G1,list(nodes))
            cluster_nodes = list(sub_graph.iterNodes())
            min_cut_required = int(cluster_stats_df[cluster_stats_df['cluster']==cluster_id]['connectivity'])
            cluster_edges = list(sub_graph.iterEdges())
            sub_G = PyGraph(cluster_nodes, cluster_edges)
            mincut_result = sub_G.mincut("cactus", "bqueue", True)

            partitions = mincut_result[:-1]
            cut_size = mincut_result[-1]
            largest_index, large_partition = max(enumerate(partitions), key=lambda x: len(x[1]))
            for i in range(len(partitions)):
                if i != largest_index:
                    part_nodes_list = list(partitions[i])
                    k = min_cut_required
                    while k>0:
                        available_part_nodes = available_c_nodes.intersection(set(part_nodes_list))
                        available_large_part_nodes = available_c_nodes.intersection(set(large_partition))
                        edge_end = None
                        part_edge_end = None
                        large_idx_available = False
                        part_idx_available = False
                        if len(available_part_nodes)>0:
                            part_edge_end = np.random.choice(list(available_part_nodes))
                            part_idx_available = True                
                        else:
                            part_edge_end = np.random.choice(part_nodes_list)

                        # remove neighbors of part_edge_end from available_large_part_nodes
                        for neighbor in set(sub_graph.iterNeighbors(part_edge_end)):
                            available_large_part_nodes.discard(neighbor) 

                        if len(available_large_part_nodes)>0:
                            edge_end = np.random.choice(list(available_large_part_nodes))
                            large_idx_available = True 
                        else:
                            edge_end = np.random.choice(large_partition)
                
                        if edge_end not in set(sub_graph.iterNeighbors(part_edge_end)):
                            sub_graph.addEdge(part_edge_end,edge_end)
                            conn_new_edges.add((part_edge_end,edge_end))
                            total_edges += 1

                            if large_idx_available:
                                available_node_degrees[edge_end] -= 1
                                if available_node_degrees[edge_end] == 0:
                                    available_c_nodes.remove(edge_end)
                                    available_node_set.remove(edge_end)
                                    del available_node_degrees[edge_end]

                            if part_idx_available:
                                available_node_degrees[part_edge_end] -= 1
                                if available_node_degrees[part_edge_end] == 0:
                                    available_c_nodes.remove(part_edge_end)
                                    available_node_set.remove(part_edge_end)
                                    del available_node_degrees[part_edge_end]
                            
                            if part_idx_available or large_idx_available:
                                degree_corrected += 1
                            
                            k -= 1
                        # else:
                        #     print(cluster_id, edge_end, part_edge_end, available_part_nodes, available_large_part_nodes, part_idx_available, large_idx_available, set(sub_graph.iterNeighbors(part_edge_end)), large_partition, part_nodes_list, partitions)

                    large_partition.extend(part_nodes_list)

        print("Number of edges added : ", total_edges, len(conn_new_edges))
        if total_edges > 0:
            print("Number of degree corrected edges added out of total edges : ", degree_corrected, (degree_corrected/total_edges * 100))
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        logging.info("Adding new edges to graph!")
        start_time = time.time()
        G2 = add_edges(G1,conn_new_edges)
        print(G1.numberOfEdges(), G2.numberOfEdges())
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")

        
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")

        logging.info("Saving new graph! G2")
        start_time = time.time()
        conn_new_edges_ids = [(node_mapping_reversed[u], node_mapping_reversed[v]) for (u,v) in conn_new_edges]
        copy_and_append(out_edge_file1, out_edge_file2, conn_new_edges_ids)
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

        logging.info("Stage 3 : Enforcing required connectivity in connected clusters of G2!")
        start_time = time.time()
        # Convert the 'connectivity' columns to integers
        cluster_stats_G2['connectivity'] = cluster_stats_G2['connectivity'].astype(int)
        cluster_stats_df['connectivity'] = cluster_stats_df['connectivity'].astype(int)
        merged_cluster_stats = pd.merge(
            cluster_stats_G2[['cluster', 'connectivity']], 
            cluster_stats_df[['cluster', 'connectivity']], 
            on='cluster', 
            suffixes=('_G2', '_df')
        )

        # Select clusters where connectivity in G2 is less than in cluster_stats_df
        poor_connected_clusters_G2 = merged_cluster_stats[
            merged_cluster_stats['connectivity_G2'] < merged_cluster_stats['connectivity_df']]['cluster']
        # poor_connected_clusters_G2 = cluster_stats_G2[cluster_stats_G2['connectivity_normalized_log10(n)']<=1]['cluster']
        print("Number of poorly connected clusters after step-2 of processing:  ", len(poor_connected_clusters_G2))
        mincut_edges = []
        clusters_processed = 0
        degree_corrected = 0
        for cluster_id in poor_connected_clusters_G2:
            is_mincut_statisfied = False
            min_cut_required = int(cluster_stats_df[cluster_stats_df['cluster']==cluster_id]['connectivity'])
            nodes = list(cluster_node_mapping[cluster_mapping_dict[cluster_id]])
            available_c_nodes = available_node_set.intersection(set(nodes))
            while is_mincut_statisfied==False:
                sub_graph = nk.graphtools.subgraphFromNodes(G2,nodes)
                cluster_nodes = list(sub_graph.iterNodes())
                cluster_edges = list(sub_graph.iterEdges())
                sub_G = PyGraph(cluster_nodes, cluster_edges)
                mincut_result = sub_G.mincut("cactus", "bqueue", True)

                heavy_part = mincut_result[0]
                light_part = mincut_result[1]
                cut_size = mincut_result[2]
                if cut_size >= min_cut_required:
                    is_mincut_statisfied = True
                else:
                    req_edges = min_cut_required - cut_size
                    k = req_edges
                    while k>0:
                        available_heavy_part_nodes = available_c_nodes.intersection(set(heavy_part))
                        available_light_part_nodes = available_c_nodes.intersection(set(light_part))
                        edge_end_1 = None
                        edge_end_2 = None
                        heavy_idx_available = False
                        light_idx_available = False

                        if len(available_light_part_nodes)>0:
                            edge_end_2 = np.random.choice(list(available_light_part_nodes))
                            light_idx_available = True
                        else:
                            edge_end_2 = np.random.choice(light_part)
                        
                        #Remove neighbors of edge_end_2 from available_heavy_part_nodes
                        for neighbor in set(sub_graph.iterNeighbors(edge_end_2)):
                            available_heavy_part_nodes.discard(neighbor)

                        if len(available_heavy_part_nodes)>0:
                            edge_end_1 = np.random.choice(list(available_heavy_part_nodes))
                            heavy_idx_available = True
                        else:
                            edge_end_1 = np.random.choice(heavy_part)

                        if edge_end_1 not in  set(sub_graph.iterNeighbors(edge_end_2)):
                            sub_graph.addEdge(edge_end_1, edge_end_2)
                            G2.addEdge(edge_end_1, edge_end_2)
                            mincut_edges.append((edge_end_1,edge_end_2))

                            if heavy_idx_available:
                                available_node_degrees[edge_end_1] -= 1
                                if available_node_degrees[edge_end_1] == 0:
                                    available_c_nodes.remove(edge_end_1)
                                    available_node_set.remove(edge_end_1)
                                    del available_node_degrees[edge_end_1]

                            if light_idx_available:
                                available_node_degrees[edge_end_2] -= 1
                                if available_node_degrees[edge_end_2] == 0:
                                    available_c_nodes.remove(edge_end_2)
                                    available_node_set.remove(edge_end_2)
                                    del available_node_degrees[edge_end_2]
                            
                            if light_idx_available or heavy_idx_available:
                                degree_corrected += 1

                            k -= 1
            
            clusters_processed += 1
            if(clusters_processed%100)==0:
                print("Clusters processed : ",clusters_processed)

        print("Total number of mincut edges added : ", len(mincut_edges))
        if len(mincut_edges)>0:
            print("Total number of degree corrected edges added out of total edges : ", degree_corrected, (degree_corrected/len(mincut_edges) * 100))
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")

        # logging.info("Adding new edges to graph - G3!")
        # start_time = time.time()
        # G3 = add_edges(G2,mincut_edges)
        # print(G2.numberOfEdges(), G3.numberOfEdges())
        # logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")

        logging.info("Saving new graph! G3")
        start_time = time.time()
        # mincut_edges.extend(degree_edges)
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
        poor_connected_clusters_G3 = cluster_stats_G3[cluster_stats_G3['connectivity_normalized_log10(n)']<=1]
        print("Number of poor connected clusters after Step-3 of processing : ", len(poor_connected_clusters_G3))
        print(poor_connected_clusters_G3.to_string(index=False))
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("After cluster stats of G3!")
        
        ## Testing Stage 4 - trial 1
        logging.info("PlusEdges V1 : Adding edges to final graph to satisfy degree sequence!")
        print("Current number of edges in graph : ", G2.numberOfEdges())
        start_time = time.time()
        degree_edges = set()
        nodes_processed = 0
        degree_corrected = 0
        print("Number of available nodes : ", len(available_node_degrees.keys()))
        
        # Convert available_node_degrees to a max-heap
        available_node_degrees = {node: degree for node, degree in available_node_degrees.items() if degree > 0}
        max_heap = [(-degree, node) for node, degree in available_node_degrees.items()]
        heapq.heapify(max_heap)

        while max_heap:
            _, available_c_node = heapq.heappop(max_heap)
            if available_c_node not in available_node_degrees:
                continue

            neighbors = set(G2.iterNeighbors(available_c_node))
            neighbors.add(available_c_node)

            # available_non_neighbors =  available_node_set - neighbors
            available_non_neighbors =  available_node_set.copy()
            
            for neighbor in neighbors:
                available_non_neighbors.discard(neighbor)

            avail_degree = available_node_degrees[available_c_node]
            k = avail_degree
            avail_k = min(k, len(available_non_neighbors))
            for i in range(avail_k):
                edge_end = available_non_neighbors.pop()
                G2.addEdge(available_c_node, edge_end)
                degree_edges.add((available_c_node, edge_end))
                available_node_degrees[edge_end] -= 1
                if available_node_degrees[edge_end] == 0:
                    available_node_set.remove(edge_end)
                    del available_node_degrees[edge_end]
                degree_corrected += 1
                    # avail_k -= 1

            del available_node_degrees[available_c_node]
            available_node_set.remove(available_c_node)
            nodes_processed += 1
            if nodes_processed % 1000 == 0:
                print("Nodes processed and available nodes : ", nodes_processed,len(available_node_degrees.keys()))

        print("Total number of edges after addition: ", G2.numberOfEdges())
        print("Total number of edges in empirical network: ", emp_graph.numberOfEdges())
        print("Total number of degree edges added: ", len(degree_edges))
        print("Number of available node degrees: ", available_node_degrees)
        if degree_edges:
            print("Total number of degree corrected edges added out of total edges: ", degree_corrected, (degree_corrected / len(degree_edges) * 100))
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")

        logging.info("Saving new graph! G4")
        start_time = time.time()
        degree_new_edges_ids = [(node_mapping_reversed[u], node_mapping_reversed[v]) for (u,v) in degree_edges]
        copy_and_append(out_edge_file3, out_edge_file4, degree_new_edges_ids)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        
        ## Testing Stage 4 - trial 2
        logging.info("PlusEdges V2 : Adding edges to final graph to satisfy degree sequence!")
        start_time = time.time()
        pe_v2.main(edge_input=out_edge_file3, cluster_input=cluster_input,out_edge_file=out_edge_file5,emp_edge_input=emp_edge_input)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")

        # logging.info("Deleting Stage 1 and 2 files!")
        # start_time = time.time()
        # if os.path.isfile(out_edge_file1):
        #         os.remove(out_edge_file1)
        #         print(f"Deleted: {out_edge_file1}")
        # else:
        #     print(f"Skipped (not a file): {out_edge_file1}")

        # if os.path.isfile(out_edge_file2):
        #         os.remove(out_edge_file2)
        #         print(f"Deleted: {out_edge_file2}")
        # else:
        #     print(f"Skipped (not a file): {out_edge_file2}")
        # logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        # log_cpu_ram_usage("After deleting files!")

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
    # parser.add_argument(
    #     '-m', metavar='network_name', type=str, required=False,
    #     help='name of the network'
    #     )
    parser.add_argument(
        '-o', metavar='output_dir', type=str, required=True,
        help='output directory'
        )
    parser.add_argument(
        '-ef', metavar='emp_edge_input', type=str, required=True,
        help='empirical network edgelist path'
        )
    
    
    args = parser.parse_args()

    typer.run(main)
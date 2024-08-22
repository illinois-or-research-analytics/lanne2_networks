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

def get_degree_sequence(cluster_df, G_c, node_mapping):
    deg_seq = []
    for idx, row in cluster_df.iterrows():
        deg_seq.append(G_c.degree(node_mapping.get(str(row['node_id']))))
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

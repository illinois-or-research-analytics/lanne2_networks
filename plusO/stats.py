import typer
import networkit as nk
import pandas as pd
import os
import json
import logging
import matplotlib.pyplot as plt
import argparse
import numpy as np 
import pandas as pd

def get_isolated_vertices(graph):
    isolated_vertices = []
    degrees_list = []
    for v in graph.iterNodes():
        deg = graph.degree(v)
        if graph.isIsolated(v):
            isolated_vertices.append(v)
        degrees_list.append(deg)
    # degree_stats = [np.median(degrees_list), np.average(degrees_list),np.percentile(degrees_list, 25),np.percentile(degrees_list, 75),np.min(degrees_list),np.max(degrees_list)]
    # print("degree stats from isolated vertices function ", degree_stats)
    return isolated_vertices, degrees_list

def get_graph_stats(graph, outlier_nodes, stage, node_mapping):
    print(graph.numberOfSelfLoops())
    
    stats = {}
    num_vertices = graph.numberOfNodes()
    num_edges = graph.numberOfEdges()
    print("Number of vertices : ", num_vertices)
    print("Num edges : ", num_edges)
    stats[f'num_vertices_{stage}'] = num_vertices
    stats[f'num_edges_{stage}'] = num_edges

    graph.removeMultiEdges()
    graph.removeSelfLoops()
    num_vertices = graph.numberOfNodes()
    num_edges = graph.numberOfEdges()
    print("Number of vertices : ", num_vertices)
    print("Num edges after removing self-loops and duplicate parallel edges: ", num_edges)
    stats[f'num_vertices_cleaned_{stage}'] = num_vertices
    stats[f'num_edges_cleaned_{stage}'] = num_edges
    isolated_vertices, degrees_list = get_isolated_vertices(graph)
    stats[f'num_isolated_{stage}'] = len(isolated_vertices)

    # dd = sorted(nk.centrality.DegreeCentrality(graph).run().scores(), reverse=True)
    # degrees, numberOfNodes = np.unique(dd, return_counts=True)
    degrees_list = np.array(degrees_list)
    degrees, counts = np.unique(degrees_list, return_counts=True)
    if degrees.min()>1:
        degrees = np.append(degrees,1)
        counts = np.append(counts,0)
    fig = plt.figure()
    plt.title(f"Degree distribution - {stage}")
    plt.xscale("log")
    plt.xlabel("degree")
    plt.yscale("log")
    plt.ylabel("number of nodes")
    plt.plot(degrees, counts)

    # degrees_list = []
    # for idx, degree in enumerate(degrees):
    #     if(degree==0):
    #         print("Zero degree found: ",degree, numberOfNodes[idx])
    #     degrees_list.extend([degree] * numberOfNodes[idx])
    # degrees_list = np.array(degrees_list)
    # median, average, q1, q3, min, max
    degree_stats = [np.median(degrees_list), np.average(degrees_list),np.percentile(degrees_list, 25),np.percentile(degrees_list, 75),np.min(degrees_list),np.max(degrees_list)]
    stats[f'degree_dist_{stage}_(# median, average, q1, q3, min, max)'] = np.array(degree_stats)
    outlier_degrees = []
    numerical_to_string_mapping = {v: k for k, v in node_mapping.items()}
    if len(outlier_nodes)>0:
        outlier_edges = 0
        for node in outlier_nodes:
            outlier_degrees.append(graph.degree(int(node_mapping.get(str(node)))))
            for neighbor in graph.iterNeighbors(int(node_mapping.get(str(node)))):
                if int(numerical_to_string_mapping.get(neighbor)) in outlier_nodes:
                    outlier_edges += 1
        
        outlier_degree_stats = [np.median(outlier_degrees), np.average(outlier_degrees),np.percentile(outlier_degrees, 25),np.percentile(outlier_degrees, 75),np.min(outlier_degrees),np.max(outlier_degrees)]
        stats[f'outlier_degree_dist_{stage}_(# median, average, q1, q3, min, max)'] = outlier_degree_stats
        stats[f'outlier_edges_{stage}'] = outlier_edges//2
    # else:
        # stats[f'outlier_degree_dist_{stage}_(# median, average, q1, q3, min, max)'] = []
    cc = nk.components.ConnectedComponents(graph)
    cc.run()
    stats[f'Num_connected_components_{stage}'] = cc.numberOfComponents()
    stats[f'Size_connected_components_{stage}'] = cc.getComponentSizes()
    stats_df = pd.DataFrame.from_dict(stats, orient='index')
    return stats_df, fig

def read_graph(filepath):
    edgelist_reader = nk.graphio.EdgeListReader("\t", 0, directed=False, continuous=False)
    nk_graph = edgelist_reader.read(filepath)
    node_mapping = edgelist_reader.getNodeMap()
    return nk_graph, node_mapping

def main(filepath, outlier_nodes, stage):
    graph, node_mapping = read_graph(filepath)
    stats_df, fig = get_graph_stats(graph, outlier_nodes, stage, node_mapping)
    return stats_df, fig

if __name__ == "__main__":
    main()

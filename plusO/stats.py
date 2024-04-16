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
    degrees_dict = {}
    degrees_list = []
    for v in graph.iterNodes():
        deg = graph.degree(v)
        if graph.isIsolated(v):
            isolated_vertices.append(v)
        degrees_list.append(deg)
        degrees_dict[v] = deg
    return isolated_vertices, degrees_list, degrees_dict

def get_graph_stats(graph, outlier_nodes, stage, node_mapping):
    print("Number of self loops : ",graph.numberOfSelfLoops())
    
    stats = {}
    num_vertices = graph.numberOfNodes()
    num_edges_read = graph.numberOfEdges()
    print("Number of vertices : ", num_vertices)
    print("Num edges : ", num_edges_read)
    stats[f'num_vertices_{stage}'] = num_vertices
    stats[f'num_edges_{stage}'] = num_edges_read

    graph.removeMultiEdges()
    graph.removeSelfLoops()
    num_vertices = graph.numberOfNodes()
    num_edges = graph.numberOfEdges()
    print("Number of vertices : ", num_vertices)
    print("Number of parallel/multiedges : " , (num_edges_read-num_edges))
    print("Num edges after removing self-loops and duplicate parallel edges: ", num_edges)
    stats[f'num_vertices_cleaned_{stage}'] = num_vertices
    stats[f'num_edges_cleaned_{stage}'] = num_edges
    isolated_vertices, degrees_list, degrees_dict = get_isolated_vertices(graph)
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

    degree_stats = [np.min(degrees_list),np.percentile(degrees_list, 25),np.median(degrees_list), np.percentile(degrees_list, 75),np.max(degrees_list),np.average(degrees_list)]
    stats[f'degree_dist_{stage}_(#min,q1,median,q3,max,average)'] = [round(num, 2) for num in degree_stats]
    outlier_degrees = []
    print("Started outlier degree statistics!")
    outlier_degrees_dict = {}
    mapped_outlier_nodes = set()
    for node in outlier_nodes:
        mapped_outlier_nodes.add(node_mapping.get(str(node)))
    print("Got mapped outlier nodes!")
    if len(outlier_nodes)>0:
        outlier_edges = 0
        for edge in graph.iterEdges():
            if edge[0] in mapped_outlier_nodes:
                deg = degrees_dict.get(edge[0])
                outlier_degrees_dict[edge[0]] = deg
                if edge[1] in mapped_outlier_nodes:
                    outlier_edges += 1
        print("Processed all edges")
        stats[f'outlier_edges_{stage}'] = outlier_edges
        outlier_degrees = list(outlier_degrees_dict.values())
        outlier_degree_stats = [np.min(outlier_degrees),np.percentile(outlier_degrees, 25),np.median(outlier_degrees),np.percentile(outlier_degrees, 75),np.max(outlier_degrees), np.average(outlier_degrees)]
        stats[f'outlier_degree_dist_{stage}_(#min,q1,median,q3,max,average)'] = [round(num, 2) for num in outlier_degree_stats]
        isolated_outlier_vertices = set(isolated_vertices) & mapped_outlier_nodes
        stats[f'isolated_outlier_nodes_{stage}'] = len(isolated_outlier_vertices)
    print("Outlier degree statistics done")

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

import pandas as pd
import numpy as np
import graph_tool.all as gt
from graph_tool.all import *
from matplotlib.pylab import poisson
import networkit as nk
import argparse
import typer
import os
import time
import logging
import stats
import matplotlib.pyplot as plt

def add_outlier_nodes(N, p, graph,node_mapping ):
    modified_graph = graph
    max_node_id = int(max(node_mapping.values(), key=int))
    outlier_nodes = []
    
    vertices = [v for v in modified_graph.iterNodes()]

    for node_id in range(max_node_id+1, max_node_id+N+1):
        # nodes_in_block = block_assignment_df[block_assignment_df['block']==block_id]['node_id'].to_numpy()
        # vertices = graph.get_vertices()
        # node_choices = np.concatenate([vertices, np.arange((max_node_id+1), node_id)])
        node_choices = np.concatenate([vertices, outlier_nodes])
        num_new_edges = int(p * len(node_choices))
        selected_nodes = np.random.choice(node_choices, size=num_new_edges, replace=False)
        for target in selected_nodes:
            modified_graph.addEdge(node_id, target, addMissing = True)
        outlier_nodes.append(node_id)
        if (len(outlier_nodes) % 1000 == 0):
            print("Number of nodes added : " ,len(outlier_nodes))
        node_mapping[node_id] = str(node_id)
    return modified_graph, outlier_nodes, node_mapping

def save_generated_graph(graph, node_mapping,out_edge_file, out_node_file):
    edges = []
    for edge in graph.iterEdges():
        edges.append([node_mapping.get(edge[0]), node_mapping.get(edge[1])])
    edge_df = pd.DataFrame(edges, columns=['source', 'target'])
    edge_df.to_csv(out_edge_file, sep='\t', index=False, header=None)

    # nodes = [node_mapping.get(v) for v in graph.iterNodes()]
    nodes = [v for v in node_mapping.keys()]
    nodes_df = pd.DataFrame(nodes, columns=['node'])
    nodes_df.to_csv(out_node_file, index=False)

def main(edge_input: str = typer.Option(..., "--filepath", "-f"),
    num_nodes: int = typer.Option(..., "--num_nodes", "-n"),
    probability: float = typer.Option(..., "--probability", "-p"),
    net_name: str = typer.Option(..., "--net_name", "-m"),
    out_edge_file: str = typer.Option("", "--out_edge_file", "-oe"),
    out_node_file: str = typer.Option("", "--out_node_file", "-on")):

    if out_edge_file == "":
        out_edge_file = f'SBM_samples/{net_name}/modified_graph_edge_list_{num_nodes}_{probability}.tsv'
    else:
        out_edge_file = out_edge_file
    
    if out_node_file == "":
        out_node_file = f'SBM_samples/{net_name}/modified_graph_node_list_{num_nodes}_{probability}.tsv'
    else:
        out_node_file = out_node_file

    output_dir = os.path.dirname(out_edge_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    logging.basicConfig(filename=f'SBM_samples/{net_name}/SBM_plusO_{num_nodes}_{probability}_output.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    plusO_start_time = time.time()

    try:
        logging.info("Reading generated graph...")
        graph,node_mapping = read_graph(edge_input)
        logging.info("Statistics of read graph:")
        num_vertices, num_edges = get_graph_stats(graph)
        stats_df_before, fig = stats.main(edge_input, [], 'beforeplusO')
        print(stats_df_before)
        # fig.title(f'Degree distribution - beforeplusO N= {num_nodes} p = {probability}')
        fig.savefig(output_dir+f"/{net_name}_{num_nodes}_{probability}_beforeplusO_degree_distribution.png")
        new_num_nodes = int(num_nodes/100  * num_vertices)
        logging.info(f"Adding Outlier nodes {new_num_nodes} with probability {probability}")
        modified_graph,outlier_nodes, node_mapping = add_outlier_nodes(new_num_nodes, probability, graph,node_mapping)
        # get_graph_stats(modified_graph)
        
        logging.info("Saving Modified graph edgelist and node list!")
        save_generated_graph(modified_graph, node_mapping, out_edge_file, out_node_file)
        logging.info("Statistics of modified graph:")
        stats_df_after, fig = stats.main(out_edge_file, outlier_nodes, 'afterplusO')
        print(stats_df_after)
        combined_df = pd.concat([stats_df_before, stats_df_after])
        # fig.title(f'Degree distribution - afterplusO N= {num_nodes} p = {probability}')
        fig.savefig(output_dir+f"/{net_name}_{num_nodes}_{probability}_afterplusO_degree_distribution.png")
        print(combined_df)
        combined_df.to_csv(f'{output_dir}/{net_name}_{num_nodes}_{probability}_stats.csv')
        logging.info(f"Total Time taken: {round(time.time() - plusO_start_time, 3)} seconds")
    except Exception as e:
        print(e)

def get_graph_stats(graph):
    num_vertices = graph.numberOfNodes()
    num_edges = graph.numberOfEdges()
    print("Number of vertices : ", num_vertices)
    print("Num edges : ", num_edges)
    return num_vertices, num_edges

def read_graph(filepath):
    # graph = gt.load_graph_from_csv(filepath, directed=False, csv_options={'delimiter': '\t'})
    # return graph
    edgelist_reader = nk.graphio.EdgeListReader("\t", 0, directed=False, continuous=False)
    nk_graph = edgelist_reader.read(filepath)
    node_mapping = edgelist_reader.getNodeMap()
    numerical_to_string_mapping = {v: k for k, v in node_mapping.items()}
    # print(nk_graph.numberOfNodes(), len(numerical_to_string_mapping))
    return nk_graph, numerical_to_string_mapping

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate graph ')
    parser.add_argument(
        '-f', metavar='graph_filepath', type=str, required=True,
        help='edgelist tsv file for SBM generated graph'
        )
    parser.add_argument(
        '-n', metavar='num_nodes', type=int, required=True,
        help='number of nodes to add as outliers - percentage'
        )
    parser.add_argument(
        '-p', metavar='probability', type=float, required=True,
        help='probability of outlier nodes having an edge with existing nodes'
        )
    parser.add_argument(
        '-m', metavar='net_name', type=str, required=True,
        help='network name'
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
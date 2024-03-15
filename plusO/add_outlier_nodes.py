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

def add_outlier_nodes(N, p, graph, block_assignment_df):
    modified_graph = graph.copy()
    max_node_id = max(graph.get_vertices())
    block_summary =  block_assignment_df.groupby('block').agg(num_nodes=('node_id', 'nunique'))
    block_summary = block_summary[block_summary['num_nodes'] > 1].reset_index()
    block_ids = block_summary['block'].to_numpy()
    for node_id in range(max_node_id+1, max_node_id+N+1):
        for block_id in block_ids:
            nodes_in_block = block_assignment_df[block_assignment_df['block']==block_id]['node_id'].to_numpy()
            node_choices = np.concatenate([nodes_in_block, np.arange((max_node_id+1), node_id)])
            num_new_edges = int(p * len(node_choices))
            selected_nodes = np.random.choice(node_choices, size=num_new_edges, replace=False)
            for target in selected_nodes:
                modified_graph.add_edge(node_id, target, add_missing = True)
    return modified_graph

def save_generated_graph(graph, out_edge_file, out_node_file):
    edges = graph.get_edges()
    edge_df = pd.DataFrame(edges, columns=['source', 'target'])
    edge_df.to_csv(out_edge_file, sep='\t', index=False, header=None)
    nodes = graph.get_vertices()
    nodes_df = pd.DataFrame(nodes, columns=['node'])
    nodes_df.to_csv(out_node_file, index=False)

def main(edge_input: str = typer.Option(..., "--filepath", "-f"),
    block_input: str = typer.Option(..., "--block_filepath", "-b"),
    num_nodes: int = typer.Option(..., "--num_nodes", "-n"),
    probability: float = typer.Option(..., "--probability", "-p"),
    out_edge_file: str = typer.Option("", "--out_edge_file", "-oe"),
    out_node_file: str = typer.Option("", "--out_node_file", "-on")):
    
    logging.basicConfig(filename='SBM_plusO_output.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    generation_start_time = time.time()

    if out_edge_file == "":
        out_edge_file = f'SBM_samples/modified_graph_edge_list_{num_nodes}_{probability}.tsv'
    else:
        out_edge_file = out_edge_file
    
    if out_node_file == "":
        out_node_file = f'SBM_samples/modified_graph_node_list_{num_nodes}_{probability}.tsv'
    else:
        out_node_file = out_node_file

    output_dir = os.path.dirname(out_edge_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        logging.info("Reading generated graph...")
        graph, block_assignment_df = read_graph(edge_input, block_input)
        logging.info("Statistics of read graph:")
        get_graph_stats(graph)
        logging.info(f"Adding Outlier nodes {num_nodes} with probability {probability}")
        modified_graph = add_outlier_nodes(num_nodes, probability, graph, block_assignment_df)
        logging.info("Statistics of generated graph:")
        get_graph_stats(modified_graph)
        logging.info("Saving Modified graph edgelist and node list!")
        save_generated_graph(modified_graph,  out_edge_file, out_node_file)
        
    except Exception as e:
        print(e)

def get_graph_stats(graph):
    num_vertices = graph.num_vertices()
    num_edges = graph.num_edges()
    print("Number of vertices : ", num_vertices)
    print("Num edges : ", num_edges)

def read_graph(filepath, block_filepath):
    graph = gt.load_graph_from_csv(filepath, directed=False, csv_options={'delimiter': '\t'})
    block_assignment_df = pd.read_csv(block_filepath)
    return graph, block_assignment_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate graph ')
    parser.add_argument(
        '-f', metavar='graph_filepath', type=str, required=True,
        help='edgelist tsv file for SBM generated graph'
        )
    parser.add_argument(
        '-b', metavar='graph_filepath', type=str, required=True,
        help='block membership csv file for SBM generated graph'
        )
    parser.add_argument(
        '-n', metavar='num_nodes', type=int, required=True,
        help='number of nodes to add as outliers'
        )
    parser.add_argument(
        '-p', metavar='probability', type=float, required=True,
        help='probability of outlier nodes having an edge with existing nodes'
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
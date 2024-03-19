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

def deg_sampler(index):
    global degree_seq
    return degree_seq[index]

def generate_graph(deg_seq):
    N = len(deg_seq)
    generated_graph = gt.random_graph(N, deg_sampler=deg_sampler, directed=False, parallel_edges=False, self_loops=False)
    return generated_graph

def read_file(degree_sequence_filepath):
    global degree_seq
    degree_seq = pd.read_csv(degree_sequence_filepath, header=None)[0].to_numpy()
    print(len(degree_seq))
    return degree_seq

def save_generated_graph(graph, out_edge_file, out_node_file):
    edges = graph.get_edges()
    edge_df = pd.DataFrame(edges, columns=['source', 'target'])
    edge_df.to_csv(out_edge_file, sep='\t', index=False, header=None)
    vertices = graph.get_vertices()
    vertices_df = pd.DataFrame(vertices, columns=['Node'])
    vertices_df.to_csv(out_node_file, index=False, header=None)

def main(degree_sequence_filepath: str = typer.Option(..., "--degree_sequence_filepath", "-f"),
    net_name: str = typer.Option(..., "--net_name", "-m"),
    out_edge_file: str = typer.Option("", "--out_edge_file", "-oe"),
    out_node_file: str = typer.Option("", "--out_node_file", "-on")):

    if out_edge_file == "":
        out_edge_file = f'NG_EDS_samples/{net_name}/generated_graph_edge_list.tsv'
    else:
        out_edge_file = out_edge_file
    if out_node_file == "":
        out_node_file = f'NG_EDS_samples/{net_name}/generated_graph_node_list.tsv'
    else:
        out_node_file = out_node_file

    output_dir = os.path.dirname(out_edge_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logging.basicConfig(filename=f'NG_EDS_samples/{net_name}/NG_EDS_output.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    generation_start_time = time.time()
    
    try:
        
        logging.info("reading degree sequence...")
        start_time = time.time()
        degree_seq = read_file(degree_sequence_filepath)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        logging.info(f'Generating graph with given degree sequence')
        start_time = time.time()
        generated_graph = generate_graph(degree_seq)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        logging.info("Saving generated graph...")
        save_generated_graph(generated_graph, out_edge_file, out_node_file)
        logging.info("Statistics of generated graph:")
        start_time = time.time()
        stats_df, fig = stats.main(out_edge_file, [], 'beforeplusO')
        fig.savefig(output_dir+f"/{net_name}_beforeplusO_degree_distribution.png")
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        logging.info("Saving statistics of generated graph!")
        start_time = time.time()
        stats_df.to_csv(f'{output_dir}/{net_name}_stats.csv')
        logging.info("Saved generated graph and Stats!")
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")

    except Exception as e:
        logging.error(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate graph ')
    parser.add_argument(
        '-f', metavar='degree_sequence_filepath', type=str, required=True,
        help='number of blocks'
        )
    parser.add_argument(
        '-m', metavar='network_name', type=str, required=True,
        help='name of the network'
        )
    args = parser.parse_args()

    typer.run(main)
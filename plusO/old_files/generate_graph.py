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


def prob(a, b):
   if a == b:
       return 0.95
   else:
       return 0.05

def random_block_assignment(num_nodes, num_blocks):
    cluster_sizes = np.linspace(10, np.log10(num_nodes), num_blocks)
    cluster_probabilities = cluster_sizes ** 2 / np.sum(cluster_sizes ** 2)
    np.random.shuffle(cluster_probabilities)
    block_assignment = np.random.choice(np.arange(num_blocks), size=num_nodes, p=cluster_probabilities)
    return block_assignment


def generate_new_graph(num_nodes, block_assignment):
    #using poisson(10) for degree sampler. Using random assignment of block.
    random_graph, bm = gt.random_graph(num_nodes, lambda: poisson(10)+1, directed=False,
                        model="blockmodel",
                        block_membership=block_assignment,
                        edge_probs=prob, self_loops=False, parallel_edges=False)
    block_assignment_dict = {}
    for idx, block in enumerate(bm.get_array()):
        block_assignment_dict[idx] = block
    block_assignment_df = pd.DataFrame(list(block_assignment_dict.items()), columns=['node_id', 'block'])
    return random_graph, block_assignment_df

def get_graph_stats(graph):
    stats = {}
    num_vertices = graph.num_vertices()
    num_edges = graph.num_edges()
    print("Number of vertices : ", num_vertices)
    print("Num edges : ", num_edges)
    stats['num_vertices'] = num_vertices
    stats['num_edges'] = num_edges
    out_degrees = graph.get_out_degrees(range(graph.num_vertices()))
    num_vertices_with_zero_edges = sum(1 for degree in out_degrees if degree == 0)
    stats['num_isolated'] = num_vertices_with_zero_edges
    # Compute the degree distribution
    degree_distribution = graph_tool.stats.vertex_hist(graph, "total")
    # Plot the degree distribution
    bins = degree_distribution[1]
    histogram = degree_distribution[0]
    plt.bar(bins[:-1], histogram, width=np.diff(bins), align="edge")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.title(f"Degree Distribution")
    # plt.savefig(output_dir+f"/{net_name}_degree_distribution.png")
    degrees = []
    for idx, count in enumerate(histogram):
        degrees.extend([bins[idx]] * int(count))
    degrees = np.array(degrees)
    # median, average, q1, q3, min, max
    degree_stats = [np.median(degrees), np.average(degrees),np.percentile(degrees, 25),np.percentile(degrees, 75),np.min(degrees),np.max(degrees)]
    stats['degree_dist'] = np.array(degree_stats)

    print(stats)
    return stats



def save_generated_graph(graph, block_assignment_df, out_edge_file, out_block_file):
    edges = graph.get_edges()
    edge_df = pd.DataFrame(edges, columns=['source', 'target'])
    edge_df.to_csv(out_edge_file, sep='\t', index=False, header=None)
    block_assignment_df.to_csv(out_block_file, index=False)

def main(num_nodes: int = typer.Option(..., "--num_nodes", "-n"),
    num_blocks: int = typer.Option(..., "--num_blocks", "-b"),
    net_name: str = typer.Option(..., "--net_name", "-m"),
    out_edge_file: str = typer.Option("", "--out_edge_file", "-oe"),
    out_block_file: str = typer.Option("", "--out_block_file", "-ob")):
    
    logging.basicConfig(filename='SBM_NG_output.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    generation_start_time = time.time()

    if out_edge_file == "":
        out_edge_file = f'SBM_samples/{net_name}/generated_graph_edge_list_{num_nodes}_{num_blocks}.tsv'
    else:
        out_edge_file = out_edge_file

    if out_block_file == "":
        out_block_file = f'SBM_samples/{net_name}/generated_graph_block_assignment_{num_nodes}_{num_blocks}.csv'
    else:
        out_block_file = out_block_file
    output_dir = os.path.dirname(out_edge_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        
        logging.info("Generating random block assignment...")
        start_time = time.time()
        block_assignment = random_block_assignment(num_nodes, num_blocks)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")

        logging.info(f"Creating new graph with {num_nodes} nodes and {num_blocks} clusters...")
        start_time = time.time()
        random_graph = None
        block_assignment_df = None
        random_graph, block_assignment_df = generate_new_graph(num_nodes, block_assignment)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")

        logging.info("Statistics of generated graph:")
        start_time = time.time()
        get_graph_stats(random_graph)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")

        logging.info("Saving generated graph...")
        start_time = time.time()
        save_generated_graph(random_graph, block_assignment_df, out_edge_file, out_block_file)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")

        total_time = time.time() - generation_start_time
        logging.info(f"Total time taken: {round(total_time, 3)} seconds")

    except Exception as e:
        logging.error(e)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate graph ')
    parser.add_argument(
        '-n', metavar='num_nodes', type=int, required=True,
        help='number of nodes'
        )
    parser.add_argument(
        '-b', metavar='num_blocks', type=int, required=True,
        help='number of blocks'
        ),
    parser.add_argument(
        '-m', metavar='network_name', type=str, required=True,
        help='name of the network'
        )
    parser.add_argument(
        '-oe', metavar='out_edge_file', type=str, required=False,
        help='output edgelist path'
        )
    parser.add_argument(
        '-ob', metavar='out_block_file', type=str, required=False,
        help='output membership path'
        )
    args = parser.parse_args()

    typer.run(main)


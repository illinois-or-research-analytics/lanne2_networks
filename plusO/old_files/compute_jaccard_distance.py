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
from scipy import stats
import psutil
import time

def calculate_jaccard_distance(G, node_mapping, network):
    G.removeMultiEdges()
    G.removeSelfLoops()
    jaccard_dists = []
    nodes_set = set(node_mapping.values())
    neighbor_map = {}
    for u in nodes_set:
        neighbors = set()
        for v in G.iterNeighbors(u):
            neighbors.add(v)
        neighbor_map[u] = neighbors
    
    nodes_set_copy = nodes_set.copy()

    count = 0
    for u in nodes_set_copy:
        for v in nodes_set:
            u_neighbors = neighbor_map.get(u)
            v_neighbors = neighbor_map.get(v)
            intersect = len(u_neighbors & v_neighbors)
            dist = 1 - (intersect)/(len(u_neighbors) + len(v_neighbors) - intersect)
            jaccard_dists.append(dist)
        count += 1
        nodes_set.remove(u)
        if count%10000 == 0:
            print("processed till now : ", count)

    # distances = np.array(jaccard_dists)
    # dists, counts = np.unique(distances, return_counts=True)
    # fig = plt.figure()
    # plt.title(f"Jaccard distance distribution - {network}")
    # plt.xscale("log")
    # plt.xlabel("degree")
    # plt.yscale("log")
    # plt.ylabel("number of nodes")
    # plt.plot(dists, counts)
    # # plt.show()
    # fig.savefig(f"{network}_jaccard_distribution.png")

    return jaccard_dists

def read_graph(filepath):
    edgelist_reader = nk.graphio.EdgeListReader("\t", 0, directed=False, continuous=False)
    nk_graph = edgelist_reader.read(filepath)
    node_mapping = edgelist_reader.getNodeMap()
    return nk_graph, node_mapping

def main(filepath1: str = typer.Option(..., "--filepath", "-f"),
        filepath2: str = typer.Option(..., "--filepath", "-e"),
        network_name1: str = typer.Option(..., "--net_name1", "-n1"),
        network_name2: str = typer.Option(..., "--net_name2", "-n2")):
    
    output_dir = "Jaccard_Evaluation"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    logging.basicConfig(filename=f'Jaccard_Evaluation/Jaccard_Evaluation_output_{network_name1}_{network_name2}.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    rewiring_start_time = time.time()

    def log_cpu_ram_usage(step_name):
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage('/').percent
        logging.info(f"Step: {step_name} | CPU Usage: {cpu_percent}% | RAM Usage: {ram_percent}% | Disk Usage: {disk_percent}")

    try:
        log_cpu_ram_usage("Start")
        logging.info("Reading generated graph...")
        start_time = time.time()
        graph1, node_mapping1 = read_graph(filepath1)
        graph2, node_mapping2 = read_graph(filepath2)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("After reading graph")
        logging.info("Getting Jaccard distances for graph 1:")
        start_time = time.time()
        jaccard_dist1 = calculate_jaccard_distance(graph1, node_mapping1, network_name1)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("After calculating Jaccard distances for graph1")
        logging.info("Getting Jaccard distances for graph 2:")
        start_time = time.time()
        jaccard_dist2 = calculate_jaccard_distance(graph2, node_mapping2, network_name2)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("After calculating Jaccard distances for graph2")
        logging.info("Getting K-s metric:")
        start_time = time.time()
        D, p_value = stats.ks_2samp(jaccard_dist1, jaccard_dist2)
        print(D, p_value)
        logging.info(f"The distance test statistic D, p_value: {D}, {p_value}")
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        logging.info(f"Total Time taken: {round(time.time() - rewiring_start_time, 3)} seconds")
        log_cpu_ram_usage("After calculating test statistic!")


    except Exception as e:
        print(e)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute Graph distance : ')
    parser.add_argument(
        '-f', metavar='edge_list_filepath1', type=str, required=True,
        help='network edge_list filepath1'
        )
    parser.add_argument(
        '-e', metavar='edge_list_filepath2', type=str, required=True,
        help='network edge_list filepath2'
        )
    parser.add_argument(
        '-n1', metavar='network_name1', type=str, required=True,
        help='network name'
        )
    parser.add_argument(
        '-n2', metavar='network_name2', type=str, required=True,
        help='network name'
        )
    args = parser.parse_args()

    typer.run(main)

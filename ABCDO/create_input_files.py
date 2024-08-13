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
import shutil
from collections import defaultdict
from typing import Dict, List
import json
import csv

EDGE = 'edge.tsv'
PARAMS = 'params.json'
COM_SIZES = 'cs.tsv'
CLUSTERING = 'clustering.tsv'
DEG_SEQ = 'deg_seq.tsv'


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

def get_degree_sequence(G, outliers, clustered_nodes):
    deg_seq = []
    for u in outliers:
        deg_seq.append(G.degree(u))
    for v in clustered_nodes:
        deg_seq.append(G.degree(v))
    return deg_seq

def compute_xi(G, clustering_dict):
    in_degree = defaultdict(int)
    out_degree = defaultdict(int)
    for n1, n2 in G.iterEdges():
        # TODO: what to do with outliers' connections?
        # if n1 not in node2com or n2 not in node2com:
        #     continue
        if n1 not in clustering_dict and n2 not in clustering_dict:
            # out_degree[n1] += 1
            # out_degree[n2] += 1
            continue
        if n1 not in clustering_dict or n2 not in clustering_dict:
            out_degree[n1] += 1
            out_degree[n2] += 1
            continue

        if clustering_dict[n1] == clustering_dict[n2]:  # nodes are co-clustered
            in_degree[n1] += 1
            in_degree[n2] += 1
        else:
            out_degree[n1] += 1
            out_degree[n2] += 1
    outs = [out_degree[i] for i in G.iterNodes()]
    total = [in_degree[i] + out_degree[i] for i in G.iterNodes()]
    xi = np.sum(outs) / sum(total)
    return xi

def generate_params_file(xi, seed, num_outliers, output_dir,network):
    params = {'seed': seed}
    params['xi'] = xi
    params['n_outliers'] = num_outliers

    with open(f'{output_dir}/{PARAMS}', 'w') as f:
        json.dump(
            params,
            f,
        )

    print(f'[INFO] {PARAMS} file is created.')

def generate_degree_sequence_file(deg_seq, output_dir, network):
    with open(f'{output_dir}/{DEG_SEQ}', 'w') as f:
        csv_writer = csv.writer(f, delimiter='\t')
        csv_writer.writerows([
            [x]
            for  x in deg_seq
        ])
        f.close()

    print(f'[INFO] {DEG_SEQ} file is created.')

def generate_cluster_sizes_file(com_sizes, output_dir,network):
    with open(f'{output_dir}/{COM_SIZES}', 'w') as f:
        csv_writer = csv.writer(f, delimiter='\t')
        csv_writer.writerows([
            [x]
            for x in com_sizes
        ])
        f.close()

    print(f'[INFO] {COM_SIZES} file is created.')

def main(edge_input: str = typer.Option(..., "--filepath", "-f"),
    cluster_input: str = typer.Option(..., "--cluster_filepath", "-c"),
    net_name: str = typer.Option(..., "--net_name", "-m"),
    output_dir: str = typer.Option("", "--output_dir", "-o")):


    if output_dir == "":
        output_dir = f'ABCDO_samples/{net_name}/0.001/'
    else:
        output_dir = os.path.dirname(output_dir+f'/{net_name}/0.001/')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    log_path = os.path.join(output_dir, f'{net_name}.log')
    log_dir = os.path.dirname(log_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

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
        logging.info(f"Reading generated graph...{net_name}")
        start_time = time.time()
        G,node_mapping = read_graph(edge_input)
        G.removeMultiEdges()
        G.removeSelfLoops()
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("After reading graph")
        logging.info("Reading graph clustering:")
        start_time = time.time()
        cluster_df = read_clustering(cluster_input)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("")

        logging.info("Compute params!")
        start_time = time.time()
        node_mapping_reversed = {v: int(k) for k, v in node_mapping.items()}
        clustering_dict = {node_mapping[str(node_id)]: cluster_id for node_id, cluster_id in zip(cluster_df['node_id'], cluster_df['cluster_id'])}
        clustered_nodes = set(clustering_dict.keys())
        nodes_set = set()
        for u in G.iterNodes():
            nodes_set.add(u)
        outliers = nodes_set.difference(clustered_nodes)

        deg_seq = get_degree_sequence(G, outliers, clustered_nodes)
        deg_seq = sorted(deg_seq, reverse=True)
        xi = compute_xi(G, clustering_dict)
        com_sizes = cluster_df.groupby(['cluster_id']).agg({'node_id':'count'})
        com_sizes = com_sizes.rename(columns={'node_id':'size'}).reset_index()
        com_sizes = com_sizes['size'].to_numpy()
        com_sizes = sorted(com_sizes, reverse=True)
        com_sizes = np.insert(com_sizes, 0, len(outliers))
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("")
        logging.info("Writing to files!")
        start_time = time.time()
        generate_params_file(xi=xi,seed=0,num_outliers=len(outliers),output_dir=output_dir, network=net_name)
        generate_degree_sequence_file(deg_seq=deg_seq, output_dir=output_dir,network =net_name)
        generate_cluster_sizes_file(com_sizes=com_sizes, output_dir=output_dir, network=net_name)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        logging.info(f"Total Time taken: {round(time.time() - job_start_time, 3)} seconds")
        log_cpu_ram_usage("Usage statistics after job completion!")




    except Exception as e:
        print(e)
        traceback.print_exc()
        logging.error("Exception occurred", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate graph ')
    parser.add_argument(
        '-f', metavar='emp_list_filepath', type=str, required=True,
        help='network empirical edge list filepath'
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
        '-o', metavar='output_dir', type=str, required=False,
        help='output directory'
        )
    
    args = parser.parse_args()

    typer.run(main)
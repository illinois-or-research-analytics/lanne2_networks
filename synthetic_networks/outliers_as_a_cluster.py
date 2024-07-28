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
from utils import *

def main(edge_input: str = typer.Option(..., "--filepath", "-f"),
    cluster_input: str = typer.Option(..., "--cluster_filepath", "-c"),
    output_dir: str = typer.Option("", "--output_dir", "-o"),
    subnet_file: str = typer.Option("", "--subnet_file", "-s")):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    out_edge_file = os.path.join(output_dir, f'syn_o_oc.tsv')

    log_path = os.path.join(output_dir, f'outliers_oc.log')

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
        logging.info("Reading generated graph...")
        start_time = time.time()
        G,node_mapping = read_graph(edge_input)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("After reading graph")

        logging.info("Reading graph clustering:")
        start_time = time.time()
        cluster_df = read_clustering(cluster_input)
        clustering_dict = dict(zip(cluster_df['node_id'], cluster_df['cluster_id']))
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("")

        logging.info("Getting subgraph G_c")
        start_time = time.time()
        clustered_nodes = cluster_df['node_id'].unique()
        clustered_nodes_mapped = [node_mapping.get(str(v)) for v in clustered_nodes]
        G_c = nk.graphtools.subgraphFromNodes(G, clustered_nodes_mapped)
        num_clustered_nodes, num_cluster_edges = get_graph_stats(G_c)
        print("Number of clustered nodes : ", num_clustered_nodes , "\t Number of clustered edges : ", num_cluster_edges)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("")

        logging.info("Getting subgraph G_star :  G - G_c")
        start_time = time.time()
        G_star = remove_edges(G,G_c)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("")

        logging.info("Assigning all unclustered nodes to a new cluster")
        start_time = time.time()
        new_cluster_id = np.max(cluster_df['cluster_id']) + 1
        node_mapping_reversed = {v: int(k) for k, v in node_mapping.items()}
        clustered_nodes_id_org = cluster_df['node_id'].to_numpy()
        nodes_set = set()
        for u in G.iterNodes():
            nodes_set.add(node_mapping_reversed.get(u))
        # clustered_nodes = cluster_df['node_id'].unique()
        # clustered_nodes_mapped = set([node_mapping.get(str(v)) for v in clustered_nodes])
        unclustered_nodes = nodes_set.difference(clustered_nodes_id_org)
        
        unclustered_node_cluster_mapping = []
        for v in unclustered_nodes:
            row = {'node_id': v, 'cluster_id' : new_cluster_id}
            unclustered_node_cluster_mapping.append(row)
        
        # cluster_df = pd.concat([cluster_df, pd.DataFrame([row])], ignore_index=True)
        cluster_df = cluster_df._append(unclustered_node_cluster_mapping, ignore_index=True)
        cluster_df = cluster_df.reset_index()
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("")
        
        N_edge_list = set()
        if len(unclustered_nodes)>0:
            logging.info("Edges calculation of read graph:")
            clustered_nodes_id_org_set = set(clustered_nodes_id_org)
            start_time = time.time()
            clustered_edges_org = 0
            outlier_cluster_edges_org = 0
            outlier_edges_org = 0
            for edge in G.iterEdges():
                source = node_mapping_reversed.get(min(edge[0],edge[1]))
                target = node_mapping_reversed.get(max(edge[0],edge[1]))
                if source in clustered_nodes_id_org_set and target in clustered_nodes_id_org_set:
                    clustered_edges_org += 1
                elif source in clustered_nodes_id_org_set or target in clustered_nodes_id_org_set:
                    outlier_cluster_edges_org += 1
                else:
                    outlier_edges_org += 1
            logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")

            

            logging.info("Getting edge probs matrix for G_star")
            start_time = time.time()
            probs = get_probs(G_star, node_mapping, cluster_df)
            logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
            log_cpu_ram_usage("After getting probs matrix")
            logging.info("Generating Synthetic graph N_star with probs")
            start_time = time.time()
            cluster_assignment = cluster_df['cluster_id'].to_numpy()
            out_deg_seq = get_degree_sequence(cluster_df, G_star, node_mapping)
            N = gt.generate_sbm(cluster_assignment, probs, out_degs=out_deg_seq, micro_ers=True, micro_degs=True)
            print("N graph statistics : ", N.num_vertices(), N.num_edges())
            logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
            log_cpu_ram_usage("")
            logging.info("Saving generated graph edge list! ")
            start_time = time.time()
            node_idx_set = cluster_df['node_id'].tolist()
            N_self_loops = set()
            cluster_edges = 0
            outlier_edges = 0
            outlier_cluster_edges = 0
            for edge in N.get_edges():
                source = node_idx_set[min(edge[0],edge[1])]
                target = node_idx_set[max(edge[1], edge[0])]
                if (source==target):
                    N_self_loops.add((source, target))
                N_edge_list.add((source, target))
                if(source != target):
                    if source in unclustered_nodes and target in unclustered_nodes:
                        outlier_edges += 1
                    elif source in unclustered_nodes or target in unclustered_nodes:
                        outlier_cluster_edges += 1
                    else:
                        cluster_edges += 1

        # save_generated_graph(N_edge_list, out_edge_file)
        copy_and_append(subnet_file, out_edge_file, N_edge_list)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("After N graph computation!!")
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
    parser.add_argument(
        '-o', metavar='output_dir', type=str, required=True,
        help='output directory'
        )
    parser.add_argument(
        '-s', metavar='subnet_file', type=str, required=True,
        help='clustered component edge list'
        )
    
    args = parser.parse_args()

    typer.run(main)
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
    
    out_edge_file = os.path.join(output_dir, f'syn_o_rg.tsv')

    log_path = os.path.join(output_dir, f'outliers_rg.log')

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
        logging.info(f"Reading empirical graph...")
        start_time = time.time()
        G,node_mapping = read_graph(edge_input)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("After reading graph")
        logging.info("Reading graph clustering:")
        start_time = time.time()
        cluster_df = read_clustering(cluster_input)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("")
        
        logging.info("Compute edges for each unclustered node")
        start_time = time.time()
        clustering_dict = dict(zip(cluster_df['node_id'], cluster_df['cluster_id']))
        outlier_cluster_edge_counts = {}
        node_mapping_reversed = {v: int(k) for k, v in node_mapping.items()}
        clustered_nodes_id_org = cluster_df['node_id'].to_numpy()
        nodes_set = set()
        for u in G.iterNodes():
            nodes_set.add(node_mapping_reversed.get(u))
        unclustered_nodes = nodes_set.difference(clustered_nodes_id_org)

        for node in unclustered_nodes:
            clustering_dict[node] = -1

        for edge in G.iterEdges():
            source = node_mapping_reversed.get(edge[0])
            target = node_mapping_reversed.get(edge[1])
            outlier_nodes_list = []
            dest_cluster_id = None
            if(source in unclustered_nodes):
                dest_cluster_id = clustering_dict.get(target)
                outlier_nodes_list.append(source)
            if(target in unclustered_nodes):
                dest_cluster_id = clustering_dict.get(source)
                outlier_nodes_list.append(target)
            
            for n in outlier_nodes_list:
                if n in outlier_cluster_edge_counts.keys():
                    cluster_edge_counts = outlier_cluster_edge_counts.get(n)
                    if dest_cluster_id in cluster_edge_counts.keys():
                        cluster_edge_counts[dest_cluster_id] = cluster_edge_counts.get(dest_cluster_id)+1
                    else:
                        cluster_edge_counts[dest_cluster_id] = 1
                else:
                    outlier_cluster_edge_counts[n] = {dest_cluster_id : 1}

        print("len(unclustered_nodes), len(outlier_cluster_edge_count.keys()) : ", len(unclustered_nodes), len(outlier_cluster_edge_counts.keys()))
        logging.info(f"len(unclustered_nodes), len(outlier_cluster_edge_count.keys()) : , {len(unclustered_nodes)}, {len(outlier_cluster_edge_counts.keys())}")
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("")

        logging.info("Randomly assigning edges to unclustered nodes!")
        start_time = time.time()
        N_edge_list = set()

        unclustered_node_cluster_mapping = []
        for v in unclustered_nodes:
            row = {'node_id': v, 'cluster_id' : -1}
            unclustered_node_cluster_mapping.append(row) 
        cluster_df = cluster_df._append(unclustered_node_cluster_mapping, ignore_index=True)
        cluster_df = cluster_df.reset_index()    

        node_idx_set = cluster_df['node_id'].tolist()
        # self_loops = set()
        # # for edge in N_c.get_edges():
        # #     source = node_idx_set[min(edge[0], edge[1])]
        # #     target = node_idx_set[max(edge[1],edge[0])]
        # #     if(source==target):
        # #         self_loops.add((source, target))
        # #     N_edge_list.add((source, target))
        # N_c_num_edges = len(N_edge_list) - len(self_loops)
        # print("len(N_edge_list), N_c_num_edges, self_loops : ",len(N_edge_list), N_c_num_edges, len(self_loops))
        # print("G_c and N_c number of edges after removing duplicate and parallel edges : ",num_cluster_edges ,N_c_num_edges)

        cluster_id_idx_dict = {}
        for index,row in cluster_df.iterrows():
            cluster_id = row['cluster_id']
            if cluster_id in cluster_id_idx_dict:
                cluster_id_idx_dict[cluster_id].append(index)
            else:
                cluster_id_idx_dict[cluster_id] = [index]

        # # # Checking the outlier edge count:
        # count = 0
        # for v in unclustered_nodes:
        #     cluster_edge_counts = outlier_cluster_edge_counts.get(v)
        #     for cluster_id in cluster_edge_counts.keys():
        #         if cluster_id != -1:
        #             count += cluster_edge_counts.get(cluster_id)
        #         # else:
        #         #     count += cluster_edge_counts.get(cluster_id)
        # print("Total num of outlier edges as per cluster level dict : " , count)

        outlier_non_outlier_edges_count = 0
        outlier_non_outlier_edges = set()
        outlier_cluster_degree_seq = []
        unclustered_nodes_list = list(unclustered_nodes)
        for source in unclustered_nodes_list:
            cluster_edge_counts = outlier_cluster_edge_counts.get(source)
            for cluster_id in cluster_edge_counts.keys():
                if(cluster_id != -1):
                    num_nodes_in_cluster = len(cluster_id_idx_dict.get(cluster_id))
                    num_edges_in_cluster = cluster_edge_counts.get(cluster_id)
                    random_indices = np.random.choice(num_nodes_in_cluster, num_edges_in_cluster)
                    node_idxs_in_cluster = cluster_id_idx_dict.get(cluster_id)
                    for random_index in random_indices:
                        target = node_idx_set[node_idxs_in_cluster[random_index]]
                        N_edge_list.add((source, target))
                        outlier_non_outlier_edges_count += 1
                        outlier_non_outlier_edges.add((source, target))
            if(-1 in cluster_edge_counts.keys()):
                outlier_cluster_degree_seq.append(cluster_edge_counts.get(-1))
            else:
                outlier_cluster_degree_seq.append(0)

        print(len(unclustered_nodes_list), len(np.array(outlier_cluster_degree_seq)))
        # print("Total number of outlier edges to be added in theory : ", (G.numberOfEdges() - G_c.numberOfEdges()))
        print("Total number of outlier edges to be added in clusters in theory : ", outlier_non_outlier_edges_count )
        print("Number of outlier edges added into clusters: ", len(outlier_non_outlier_edges))
        logging.info(f"Total number of outlier edges to be added in clusters in theory : , {outlier_non_outlier_edges_count} ")
        logging.info(f"Number of outlier edges added into clusters: , {len(outlier_non_outlier_edges)}")
        # print("Number of outlier edges to be added within outliers: ", (G.numberOfEdges() - G_c.numberOfEdges() - outlier_non_outlier_edges_count))
        """Calling generate_graph function to rewire edges within the outlier nodes"""
        N_o = generate_graph(np.array(outlier_cluster_degree_seq))
        N_o_num_nodes = N_o.num_vertices()
        N_o_num_edges = N_o.num_edges()
        print("Number of vertices in N_o : " , N_o_num_nodes, "Number of edges in N_o :" , N_o_num_edges)
        logging.info(f"Number of vertices in N_o : , {N_o_num_nodes}, Number of edges in N_o : ,{N_o_num_edges}")
        
        for o_edge in N_o.get_edges():
            N_edge_list.add((unclustered_nodes_list[o_edge[0]], unclustered_nodes_list[o_edge[1]]))

        print("Total number of outlier outlier edges added : ", N_o_num_edges)
        print("Final number of edges in output graph : ",len(N_edge_list))
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        logging.info("Saving generated graph edge list!")
        start_time = time.time()
        # save_generated_graph(N_edge_list, out_edge_file)
        copy_and_append(subnet_file, out_edge_file, N_edge_list)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("After saving N graph edges")
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
        '-o', metavar='output_dir', type=str, required=True,
        help='output directory'
        )
    parser.add_argument(
        '-s', metavar='subnet_file', type=str, required=True,
        help='clustered component edge list'
        )
    
    args = parser.parse_args()

    typer.run(main)
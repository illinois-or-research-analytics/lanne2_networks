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
import networkit as nk
import lanne2.lanne2_networks.plusO.old_files.ng_eds as ng_eds
from scipy.sparse import dok_matrix
import psutil
import traceback
from lanne2.lanne2_networks.plusO.old_files.ng_eds import generate_graph

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


def get_graph_stats(graph):
    num_vertices = graph.numberOfNodes()
    num_edges = graph.numberOfEdges()
    print("Number of vertices : ", num_vertices)
    print("Num edges : ", num_edges)
    return num_vertices, num_edges

def get_probs(G_c, node_mapping, cluster_df):
    numerical_to_string_mapping = {v: int(k) for k, v in node_mapping.items()}
    cluster_ids, counts = np.unique(cluster_df['cluster_id'], return_counts=True)
    num_clusters = len(cluster_ids)
    
    node_to_cluster_dict = cluster_df.set_index('node_id')['cluster_id'].to_dict()

    probs = dok_matrix((num_clusters, num_clusters), dtype=int)
    count = 0
    for edge in G_c.iterEdges():
        source = numerical_to_string_mapping.get(edge[0])
        target = numerical_to_string_mapping.get(edge[1])
        source_cluster_idx = node_to_cluster_dict.get(source)
        target_cluster_idx = node_to_cluster_dict.get(target)
        probs[source_cluster_idx, target_cluster_idx] += 1
        probs[target_cluster_idx, source_cluster_idx ] += 1
        
    probs = probs.tocsr(copy=False)
    print("Number of edges as per probs matrix: " , probs.trace()//2 + (probs.sum() - probs.trace())//2)
    return probs

def get_degree_sequence(cluster_df, G_c, node_mapping,step, non_singleton_components):
    deg_seq = []
    if step ==3:
        for idx, row in cluster_df.iterrows():
            deg_seq.append(G_c.degree(node_mapping.get(str(row['node_id']))))
    elif step ==4:
        for component in non_singleton_components:
            deg_seq_component = []
            for node in component:
                deg_seq_component.append(G_c.degree(node))
            deg_seq.append(deg_seq_component)
    return deg_seq


def save_generated_graph(edges_list, out_edge_file):
    edge_df = pd.DataFrame(edges_list, columns=['source', 'target'])
    edge_df.to_csv(out_edge_file, sep='\t', index=False, header=None)

def main(edge_input: str = typer.Option(..., "--filepath", "-f"),
    cluster_input: str = typer.Option(..., "--cluster_filepath", "-c"),
    net_name: str = typer.Option(..., "--net_name", "-m"),
    out_edge_file: str = typer.Option("", "--out_edge_file", "-oe"),
    out_node_file: str = typer.Option("", "--out_node_file", "-on")):

    if out_edge_file == "":
        out_edge_file = f'SBM_twostep_outlier_addition_V1_samples/{net_name}/N_graph_edge_list.tsv'
    else:
        out_edge_file = out_edge_file
    
    if out_node_file == "":
        out_node_file = f'SBM_twostep_outlier_addition_V1_samples/{net_name}/N_graph_node_list.tsv'
    else:
        out_node_file = out_node_file
    
    output_dir = os.path.dirname(out_edge_file)
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
        logging.info("Reading generated graph...")
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

        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("")

        # logging.info("Statistics of read graph:")
        # start_time = time.time()
        # stats_df_G, fig,participation_coeffs_G ,participation_dict_G = stats.main(edge_input, unclustered_nodes, 'original_input_graph',"",clustering_dict)
        # print(stats_df_G)
        # fig.savefig(output_dir+f"/{net_name}_original_degree_distribution.png")
        # logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")

        logging.info("Getting subgraph G_c")
        start_time = time.time()
        clustered_nodes = cluster_df['node_id'].unique()
        clustered_nodes_mapped = [node_mapping.get(str(v)) for v in clustered_nodes]
        G_c = nk.graphtools.subgraphFromNodes(G, clustered_nodes_mapped)
        num_clustered_nodes, num_cluster_edges = get_graph_stats(G_c)
        print("Number of clustered nodes : ", num_clustered_nodes , "\t Number of clustered edges : ", num_cluster_edges)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("")

        logging.info("Getting edge probs matrix for G_c")
        start_time = time.time()
        probs = get_probs(G_c, node_mapping, cluster_df)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("After getting probs matrix")

        logging.info("Generating Synthetic graph N_c with probs")
        start_time = time.time()
        cluster_assignment = cluster_df['cluster_id'].to_numpy()
        out_deg_seq = get_degree_sequence(cluster_df, G_c, node_mapping, 3, [])
        N_c = gt.generate_sbm(cluster_assignment, probs, out_degs=out_deg_seq, micro_ers=True, micro_degs=True)
        print("N_c graph statistics : Vertices , Edges : ", N_c.num_vertices(), N_c.num_edges())
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
        self_loops = set()
        for edge in N_c.get_edges():
            source = node_idx_set[min(edge[0], edge[1])]
            target = node_idx_set[max(edge[1],edge[0])]
            if(source==target):
                self_loops.add((source, target))
            N_edge_list.add((source, target))
        N_c_num_edges = len(N_edge_list) - len(self_loops)
        print("len(N_edge_list), N_c_num_edges, self_loops : ",len(N_edge_list), N_c_num_edges, len(self_loops))
        print("G_c and N_c number of edges after removing duplicate and parallel edges : ",num_cluster_edges ,N_c_num_edges)

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
        print("Total number of outlier edges to be added in theory : ", (G.numberOfEdges() - G_c.numberOfEdges()))
        print("Total number of outlier edges to be added in clusters in theory : ", outlier_non_outlier_edges_count )
        print("Number of outlier edges added into clusters: ", len(outlier_non_outlier_edges))
        print("Number of outlier edges to be added within outliers: ", (G.numberOfEdges() - G_c.numberOfEdges() - outlier_non_outlier_edges_count))
        """Calling ng_eds function to rewire edges within the outlier nodes"""
        N_o = generate_graph(np.array(outlier_cluster_degree_seq))
        N_o_num_nodes = N_o.num_vertices()
        N_o_num_edges = N_o.num_edges()
        print("Number of vertices in N_o : " , N_o_num_nodes, "Number of edges in N_o :" , N_o_num_edges)
        
        for o_edge in N_o.get_edges():
            N_edge_list.add((unclustered_nodes_list[o_edge[0]], unclustered_nodes_list[o_edge[1]]))

        print("Total number of outlier outlier edges added : ", N_o_num_edges)
        print("Final number of edges in output graph : ",len(N_edge_list))
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        logging.info("Saving generated graph edge list!")
        start_time = time.time()
        save_generated_graph(N_edge_list, out_edge_file)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("After saving N graph edges")
        # logging.info("Statistics of generated graph N:")
        # start_time = time.time()
        # stats_df_N, fig,participation_coeffs_N,participation_dict_N = stats.main(out_edge_file, unclustered_nodes, 'N',"",clustering_dict)
        # print(stats_df_N)
        # fig.savefig(output_dir+f"/{net_name}_N_degree_distribution.png")
        # combined_df = pd.concat([stats_df_G, stats_df_N])
        # combined_df = combined_df.reset_index()
        # combined_df.columns = ['Metric','Stat']
        # combined_df.loc[len(combined_df.index)] = ['Num_of_clustered_nodes', num_clustered_nodes]
        # combined_df.loc[len(combined_df.index)] = ['Num_of_clustered_edges', num_cluster_edges]
        # combined_df.loc[len(combined_df.index)] = ['Num_of_outlier_non_outlier_edges', (G.numberOfEdges() - G_c.numberOfEdges() - combined_df[combined_df['Metric']=='outlier_edges_original_input_graph']['Stat'].values[0])]
        # combined_df.loc[len(combined_df.index)] = ['generated_Num_of_clustered_nodes', N_c.num_vertices()]
        # combined_df.loc[len(combined_df.index)] = ['Generated_clustered_num_edges', N_c_num_edges]
        # combined_df.loc[len(combined_df.index)] = ['Generated_outlier_non_outlier_edges', len(outlier_non_outlier_edges)]
        # combined_df.loc[len(combined_df.index)] = ['Generated_outlier_outlier_edges', N_o_num_edges]
        # print(combined_df)
        # combined_df.to_csv(f'{output_dir}/{net_name}_stats.csv',index=False)
        # logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
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
        '-m', metavar='network_name', type=str, required=True,
        help='name of the network'
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
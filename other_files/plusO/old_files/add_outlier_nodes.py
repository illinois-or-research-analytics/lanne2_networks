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
import psutil

def add_outlier_nodes(N, p, graph,node_mapping, edge_list_filepath,output_graph_file_path, out_node_file):
    # output_graph_file_path = "./new_cit_patents.tsv"
    node_set = set()
    with open(edge_list_filepath, "r") as f:
        for line in f:
            u,v = line.strip().split()
            node_set.add(int(u))
            node_set.add(int(v))
    max_node_id = max(node_set)
    new_edges = []
    outlier_nodes = []

    for new_node_id in range(max_node_id + 1, max_node_id + 1 + N):
        current_num_new_edges = int(p * len(node_set))
        edge_end_point_arr = np.random.choice(len(node_set), current_num_new_edges)
        for new_end_point in edge_end_point_arr:
            new_edges.append((new_node_id, new_end_point))
        node_set.add(new_node_id)
        outlier_nodes.append(new_node_id)
        node_set.add(new_node_id)
        node_mapping[new_node_id] = str(new_node_id)

    new_edges_modified = []
    keys_list = list(node_mapping.keys())
    for new_edge in new_edges:
        new_edges_modified.append((new_edge[0],int(node_mapping.get(keys_list[new_edge[1]]))))
    print("Saving modified graph edge_list! ")
    with open(output_graph_file_path, "w") as fw:
        for new_u,new_v in new_edges_modified:
            fw.write(f"{new_u}\t{new_v}\n")
        with open(edge_list_filepath, "r") as fr:
            for line in fr:
                fw.write(line)
    print("Saving modified graph node_list!")
    nodes = [v for v in node_mapping.keys()]
    nodes_df = pd.DataFrame(nodes, columns=['node'])
    nodes_df.to_csv(out_node_file, index=False)
    return new_edges_modified, outlier_nodes, node_mapping

def save_generated_graph(graph,added_edges, node_mapping,out_edge_file, out_node_file):
    edges = []
    for edge in graph.iterEdges():
        edges.append([node_mapping.get(edge[0]), node_mapping.get(edge[1])])
    for edge in added_edges:
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
        out_edge_file = f'Add_Outliers_samples/{net_name}/modified_graph_edge_list_{num_nodes}_{probability}.tsv'
    else:
        out_edge_file = out_edge_file
    
    if out_node_file == "":
        out_node_file = f'Add_Outliers_samples/{net_name}/modified_graph_node_list_{num_nodes}_{probability}.tsv'
    else:
        out_node_file = out_node_file

    output_dir = os.path.dirname(out_edge_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    logging.basicConfig(filename=f'Add_Outliers_samples/{net_name}/plusO_{num_nodes}_{probability}_output.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    plusO_start_time = time.time()
    def log_cpu_ram_usage(step_name):
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage('/').percent
        logging.info(f"Step: {step_name} | CPU Usage: {cpu_percent}% | RAM Usage: {ram_percent}% | Disk Usage: {disk_percent}")



    try:
        # Log CPU and RAM usage at the beginning
        log_cpu_ram_usage("Start")
        logging.info("Reading generated graph...")
        start_time = time.time()
        graph,node_mapping = read_graph(edge_input)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("")
        logging.info("Statistics of read graph:")
        start_time = time.time()
        num_vertices, num_edges = get_graph_stats(graph)
        stats_df_before, fig = stats.main(edge_input, [], 'beforeplusO')
        print(stats_df_before)
        # fig.title(f'Degree distribution - beforeplusO N= {num_nodes} p = {probability}')
        fig.savefig(output_dir+f"/{net_name}_{num_nodes}_{probability}_beforeplusO_degree_distribution.png")
        new_num_nodes = int(num_nodes/100  * num_vertices)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("")
        logging.info(f"Adding Outlier nodes {new_num_nodes} with probability {probability}")
        start_time = time.time()
        added_edges,outlier_nodes, node_mapping = add_outlier_nodes(new_num_nodes, probability, graph,node_mapping,edge_input, out_edge_file, out_node_file)
        # get_graph_stats(modified_graph)
        # logging.info("Saving Modified graph edgelist and node list!")
        # save_generated_graph(graph, added_edges,node_mapping, out_edge_file, out_node_file)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("")
        logging.info("Statistics of modified graph:")
        start_time = time.time()
        stats_df_after, fig = stats.main(out_edge_file, outlier_nodes, 'afterplusO')
        print(stats_df_after)
        combined_df = pd.concat([stats_df_before, stats_df_after])
        fig.savefig(output_dir+f"/{net_name}_{num_nodes}_{probability}_afterplusO_degree_distribution.png")
        print(combined_df)
        combined_df.to_csv(f'{output_dir}/{net_name}_{num_nodes}_{probability}_stats.csv')
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        log_cpu_ram_usage("")
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
        help='edgelist tsv file for input graph'
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
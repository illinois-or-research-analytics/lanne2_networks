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
CLUSTERING = 'abcd_clustering.tsv'
CLUSTERING_WITHOUT_OUTLIERS = 'clustering_without_outliers.tsv'
DEG_SEQ = 'deg_seq.tsv'
DEG = 'deg_seq_generated.tsv'

def main(output_dir: str = typer.Option("", "--output_dir", "-o")):

    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    
    log_path = os.path.join(output_dir, f'gen_abcdo.log')
    log_dir = os.path.dirname(log_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    success_stats_path = os.path.join(output_dir, f'abcdo_successes.log')
    with open(success_stats_path, 'w') as f:
        f.write('Networks\tStatus\n')

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
        logging.info(f"Get all network names in the output directory!")
        start_time = time.time()
        networks = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
        # print(networks)
        logging.info(f"Time taken: {round(time.time() - start_time, 3)} seconds")
        logging.info(f"Call Julia command to generate abcd o network!")
        start_time = time.time()
        for network in networks:
            logging.info(f"Starting network :{network}!")
            net_start_time = time.time()
            output_dir_network = f'{output_dir}/{network}/0.001/'

            with open(f'{output_dir_network}/{PARAMS}', 'r') as f:
                params = json.load(f)
                seed = params['seed']
                xi = params['xi']
                n_outliers = params['n_outliers']
            cmd = f'''julia /projects/illinois/eng/cs/chackoge/lanne2/lanne2_networks/ABCDO/ABCDGraphGenerator.jl/utils/graph_sampler.jl \
                    {output_dir_network}/{EDGE} {output_dir_network}/{CLUSTERING} \
                    {output_dir_network}/{DEG_SEQ} {output_dir_network}/{COM_SIZES} \
                    xi {xi} false false {seed} {n_outliers}''' 
            
            os.system(cmd)

            if os.path.exists(f'{output_dir_network}/{CLUSTERING}') and n_outliers > 0:
                # TODO: is the 1st cluster always the outlier cluster?
                with open(f'{output_dir_network}/{CLUSTERING}', 'r') as f:
                    csv_reader = csv.reader(f, delimiter='\t')
                    rows = []
                    all_vertices = set()
                    for v, c in csv_reader:
                        if c != '1':
                            rows.append([v, c])
                        all_vertices.add(v)
                with open(f'{output_dir_network}/{CLUSTERING_WITHOUT_OUTLIERS}', 'w') as f:
                    csv_writer = csv.writer(f, delimiter='\t')
                    csv_writer.writerows(rows)
            
                # append self loop for all nodes (so the evaluation script can run)
                with open(f'{output_dir_network}/{EDGE}', 'a') as f:
                    for v in all_vertices:
                        f.write(f'{v}\t{v}\n')

            if os.path.exists(f'{output_dir_network}/{CLUSTERING}') and os.path.exists(f'{output_dir_network}/{EDGE}'):
                with open(success_stats_path,'a') as f:
                    f.write(f'{network}\tsuccess\n')
            else:
                with open(success_stats_path,'a') as f:
                    f.write(f'{network}\tfail\n')

            logging.info(f"Time taken for network {network}: {round(time.time() - net_start_time, 3)} seconds")
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
        '-o', metavar='output_dir', type=str, required=True,
        help='output directory'
        )
    
    args = parser.parse_args()

    typer.run(main)
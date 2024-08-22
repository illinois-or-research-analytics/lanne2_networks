# Synthetic Networks That Preserve Edge Connectivity (RECCS)

This repository presents the REalistic Cluster Connectivity Simulator, a
technique that modifies an SBM synthetic network to improve the fit to
a given clustered real-world network with respect to edge-connectivity
within clusters, while maintaining the good fit with respect to other net-
work and cluster statistics.

Firstly, there are two steps in creating a synthetic benchmark network N using the parameters calculated from a real-world network G and it's clustering C.
Given a network G and clustering C, first step involves only the clustered sub network G_c. That is, the sub network involving the clustered nodes and edges between them.
Second step involves adding the outliers back to the output of the first step.
 
**First step again has two steps:**
i) Generate a network N_c using SBM by computing the input parameters from G_c and C .
ii) Use RECCS (either version) to add edges to ensure minimum cut sequence in the synthetic network.
 
Code files:
To get the clustered sub network : https://github.com/vltanh/synnet/blob/main/clean_outlier.py
 
First Step :
                 a) Generate SBM network : https://github.com/illinois-or-research-analytics/lanne2_networks/blob/main/synthetic_networks/SBM_unmodified.py
                                ~ Sample Command : python3 SBM_unmodified.py -f ‘subnetwork edge list path’ -c ‘clustering file path’ -o ‘output directory’
                 b) Connectivity Enhancer (both versions) : https://github.com/illinois-or-research-analytics/lanne2_networks/blob/main/synthetic_networks/connectivity_enforcer.py
                                ~Sample Command : python3 connectivity_enforcer.py -f ‘output edge list from step (a)’ -c ‘clustering file path’ -o ‘output directory’ -ef ‘empirical edge list file path’
               
                                Note : Output directory will have edge lists saved after each stage of Connectivity Enhancer. We only need to consider two files : ce_plusedges_v1.tsv and ce_plusedges_v2.tsv
 
 Second Step: Add outliers back : https://github.com/illinois-or-research-analytics/lanne2_networks/blob/main/synthetic_networks/outliers_in_its_cluster.py
                                ~ Sample Command : python3 outliers_in_its_cluster.py -f ‘empirical edge list file path’ -c ‘clustering file path’ -o ‘output directory’ -s ‘subnetwork edge list (one of the output files from First Step)’
 

**File** : generate_graph.py <br />
**Description** : Function call to create a network with N nodes and b communities generated using graph-tool random-graph generation technique. <br />
**Example command** - python generate_graph.py -n 1000 -b 4 <br />
**Output** : edge_list.tsv, block_membership.csv <br />

**File** : add_outlier_nodes.py <br />
**Description** : Add N outlier nodes, each with probability p to have an edge with nodes within each community and other outlier nodes. <br />
**Example command** : python add_outlier_nodes.py -f 'edge_list_filepath.tsv' -b 'block_membership_filepath.csv' -n 15 -p 0.01 <br />
**Output** : new_edge_list.tsv, new_node_list.tsv <br />



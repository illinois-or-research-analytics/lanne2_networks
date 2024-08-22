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
1) Generate a network N_c using SBM by computing the input parameters from G_c and C.
2) Use RECCS (either version) to add edges to ensure minimum cut sequence in the synthetic network.
 
**CODE :**
<br>

To get the clustered sub network you may use the helper script : generate_synthetic_networks/clean_outliers.py <br>
 
**First Step :**

a) **Generate SBM network :** <br>
        - Code file : generate_synthetic_networks/gen_SBM.py <br>
        - Sample Command : python3 gen_SBM.py -f ‘subnetwork edge list filepath’ -c ‘clustering file path’ -o ‘output directory’ <br><br>
b) **RECCS (both versions) :** <br>
        - Code file : generate_synthetic_networks/reccs.py <br>
        - Sample Command : python3 reccs.py -f ‘output edge list from step (a)’ -c ‘clustering file path’ -o ‘output directory’ -ef ‘empirical edge list file path’ <br>
               
**Note :** Output directory will have edge lists saved after each stage of RECCS. We only need to consider two files : ce_plusedges_v1.tsv (RECCSv1) and ce_plusedges_v2.tsv (RECCSv2)<br>
 
**Second Step :**

**Add outliers back :** <br>
     
- Using recommended Outlier Strategy : Strategy 1 <br>
       - Code file : generate_synthetic_networks/outliers_strategy1.py <br>
       - Sample Command : python3 outliers_in_its_cluster.py -f ‘empirical edge list file path’ -c ‘clustering file path’ -o ‘output directory’ -s ‘subnetwork edge list (one of the output files from First Step)’ <br>





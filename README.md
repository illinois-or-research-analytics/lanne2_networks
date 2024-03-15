# lanne2_networks

Plus O - Adding outlier nodes that are truly unclusterable.

**File** : generate_graph.py <br />
**Description** : Function call to create a network with N nodes and b communities generated using graph-tool random-graph generation technique. <br />
**Example command** - python generate_graph.py -n 1000 -b 4 <br />
**Output** : edge_list.tsv, block_membership.csv <br />

**File** : add_outlier_nodes.py <br />
**Description** : Add N outlier nodes, each with probability p to have an edge with nodes within each community and other outlier nodes. <br />
**Example command** : python add_outlier_nodes.py -f 'edge_list_filepath.tsv' -b 'block_membership_filepath.csv' -n 15 -p 0.01 <br />
**Output** : new_edge_list.tsv, new_node_list.tsv <br />



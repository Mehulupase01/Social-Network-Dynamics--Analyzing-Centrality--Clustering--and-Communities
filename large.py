import networkit as nk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.stats import kendalltau
from cdlib import algorithms, evaluation
from matplotlib.colors import rgb2hex

# Load data
data_df = pd.read_csv("large.in", sep=" ", index_col=None, header=None, names=["source", "target"])
print(data_df.head())

# Create Networkit graph object
networkit_graph = nk.Graph(directed=True)

# Add edges to the graph
for edge in data_df[["source", "target"]].to_records(index=False).tolist():
    networkit_graph.addEdge(edge[0], edge[1], addMissing=True)

# Remove a specific node (Node 0 in this case)
networkit_graph.removeNode(0)

# Question 3.1
print("Number of directed edges: ", networkit_graph.numberOfEdges())

# Question 3.2
print("Number of users (nodes) in the network: ", networkit_graph.numberOfNodes())

# Question 3.3
indegree_counts = np.unique([networkit_graph.degreeIn(node_id) for node_id in networkit_graph.iterNodes()], return_counts=True)
outdegree_counts = np.unique([networkit_graph.degreeOut(node_id) for node_id in networkit_graph.iterNodes()], return_counts=True)

fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True, tight_layout=True)

ax[0].scatter(indegree_counts[0], indegree_counts[1])
ax[1].scatter(outdegree_counts[0], outdegree_counts[1])

ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[1].set_xscale("log")
ax[1].set_yscale("log")

fig.supxlabel("Number of nodes (log-scale)")
fig.supylabel("Frequency (log-scale)")

ax[0].title.set_text("Indegree")
ax[1].title.set_text("Outdegree")

plt.show()

# Question 3.4
strongly_connected_components = nk.components.StronglyConnectedComponents(networkit_graph)
strongly_connected_components.run()

print(f"Number of strongly connected components: ", strongly_connected_components.numberOfComponents())

weakly_connected_components = nk.components.WeaklyConnectedComponents(networkit_graph)
weakly_connected_components.run()

print(f"Number of weakly connected components: ", weakly_connected_components.numberOfComponents())

largest_scc_nodes = strongly_connected_components.getComponents()[tuple(sorted(strongly_connected_components.getComponentSizes().items(), key=lambda item: item[1], reverse=True))[0][0]]
largest_scc = nk.graphtools.subgraphFromNodes(networkit_graph, largest_scc_nodes)

print(f"Number of nodes in largest strongly connected component: ", largest_scc.numberOfNodes())
print(f"Number of edges in largest strongly connected component: ", largest_scc.numberOfEdges())

# Question 3.5
print("Average local clustering coefficient: ", nk.globals.clustering(networkit_graph))

# Question 3.6
largest_wcc_nodes = np.random.choice(weakly_connected_components.getComponents()[tuple(sorted(weakly_connected_components.getComponentSizes().items(), key=lambda item: item[1], reverse=True))[0][0]], size=1000)
largest_wcc = nk.graphtools.subgraphFromNodes(networkit_graph, largest_wcc_nodes)

APSP = nk.distance.APSP(largest_wcc)
APSP.run()
distance_counts = np.unique(APSP.getDistances(), return_counts=True)

plt.figure(figsize=(5, 5))

plt.bar(distance_counts[0][:-1], distance_counts[1][:-1], align="center")

plt.yscale("log")

plt.xlabel("Distance between nodes")
plt.ylabel("Frequency (log-scaled)")

plt.show()

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
data_df = pd.read_csv("medium.in", sep=" ", index_col=None, header=None, names=["source", "target"])
print(data_df.head())

# Create Networkit graph object
networkit_graph = nk.Graph(directed=True)

for row in data_df[["source", "target"]].to_records(index=False).tolist():
    networkit_graph.addEdge(row[0], row[1], addMissing=True)

networkit_graph.removeNode(0)

# Question 3.1
print("Number of directed edges: ", networkit_graph.numberOfEdges())

# Question 3.2
print("Number of users(nodes) in the network: ", networkit_graph.numberOfNodes())

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
scc = nk.components.StronglyConnectedComponents(networkit_graph)
scc.run()

print(f"Number of strongly connected components: ", scc.numberOfComponents())

wcc = nk.components.WeaklyConnectedComponents(networkit_graph)
wcc.run()

print(f"Number of weakly connected components: ", wcc.numberOfComponents())

largest_scc_nodes = scc.getComponents()[tuple(sorted(scc.getComponentSizes().items(), key=lambda item: item[1], reverse=True))[0][0]]
largest_scc = nk.graphtools.subgraphFromNodes(networkit_graph, largest_scc_nodes)

print(f"Number of nodes in largest strongly connected component: ", largest_scc.numberOfNodes())
print(f"Number of edges in largest strongly connected component: ", largest_scc.numberOfEdges())

# Question 3.5
print("Average local clustering coefficient: ", nk.globals.clustering(networkit_graph))

# Question 3.6
largest_wcc_nodes = wcc.getComponents()[tuple(sorted(wcc.getComponentSizes().items(), key=lambda item: item[1], reverse=True))[0][0]]
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

# Question 3.7
centrality_measures = {}

# Degree centrality
centrality_measures["degree"] = nk.centrality.DegreeCentrality(networkit_graph, normalized=True)
centrality_measures["degree"].run()

# Betweenness centrality
centrality_measures["betweenness"] = nk.centrality.Betweenness(networkit_graph, normalized=True)
centrality_measures["betweenness"].run()

# Pagerank centrality
centrality_measures["pagerank"] = nk.centrality.PageRank(networkit_graph)
centrality_measures["pagerank"].run()

rankings_df = pd.DataFrame({
    "degree": [node_id for node_id, score in centrality_measures["degree"].ranking()[:20]],
    "betweenness": [node_id for node_id, score in centrality_measures["betweenness"].ranking()[:20]],
    "pagerank": [node_id for node_id, score in centrality_measures["pagerank"].ranking()[:20]]
})

print(rankings_df.to_latex())

kendalltau_corr = {}

for measure_x in rankings_df.columns:
    kendalltau_corr[measure_x] = {}

    for measure_y in rankings_df.columns:

        y_true = pd.DataFrame({"nodes": [node_id for node_id, score in centrality_measures[measure_x].ranking()[:20]],
                               "ranking_x": [score for node_id, score in centrality_measures[measure_x].ranking()[:20]]})
        y_score = pd.DataFrame({"nodes": [node_id for node_id, score in centrality_measures[measure_y].ranking()[:20]],
                                "ranking_y": [score for node_id, score in centrality_measures[measure_y].ranking()[:20]]})

        y_df = y_true.merge(y_score, how="outer", on="nodes").fillna(0)

        kendalltau_corr[measure_x][measure_y], _ = kendalltau(y_df["ranking_x"], y_df["ranking_y"])

sns.heatmap(pd.DataFrame(kendalltau_corr).T, center=0, square=True, annot=True, annot_kws={"fontsize": 15})

plt.show()

# Question 3.8
networkx_graph = nx.DiGraph()

networkx_graph.add_edges_from(largest_wcc.iterEdges())

communities = algorithms.leiden(networkx_graph)

communities_df = pd.DataFrame(dict(communities.to_node_community_map())).T.reset_index()
communities_df.columns = ["node_id", "community"]

centrality_df = pd.DataFrame(centrality_measures["pagerank"].ranking())
centrality_df.columns = ["node_id", "centrality"]

communities_df = communities_df.merge(centrality_df, how="left", on="node_id")

print(f"Modularity score: ", evaluation.modularity_density(networkx_graph, communities).score)

plt.figure(figsize=(5, 5))

communities_counts = np.unique(communities_df["community"], return_counts=True)

plt.bar(communities_counts[0], communities_counts[1])

plt.yscale("log")
plt.xlabel("Community ID")
plt.xticks(communities_counts[0])
plt.ylabel("Number of nodes (log-scale)")

plt.show()

n_colors = 6
n_shapes = 4
cmap = plt.cm.get_cmap("hsv", n_colors)

pos = nx.spring_layout(networkx_graph)

shapes = ["o", "^", "X", "s"]

plt.figure(figsize=(10, 10))

nodes_to_plot = []

# Plot nodes
for community in communities_df["community"].unique():
    community_members =

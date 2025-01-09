import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from networkx.algorithms import bipartite

"""### Exercise 1.3"""

# Create a bipartite graph
bipartite_graph = nx.Graph()

# Add nodes with bipartite attribute
bipartite_graph.add_nodes_from([1, 2, 3, 4, 5], bipartite=0)
bipartite_graph.add_nodes_from(["1", "2", "3", "4", "5", "6"], bipartite=1)

# Add edges to the bipartite graph
edge_list = [(1, "1"), (1, "2"), (1, "3"), (2, "3"), (2, "5"), (3, "2"), (3, "6"), (4, "3"), (4, "4"), (5, "1"), (5, "5")]
bipartite_graph.add_edges_from(edge_list)

# Draw the bipartite graph
nx.draw_networkx(bipartite_graph, pos=nx.drawing.layout.bipartite_layout(bipartite_graph, [1, 2, 3, 4, 5]),
                 node_color=["grey", "grey", "grey", "grey", "grey", "white", "white", "white", "white", "white", "white"])

# Project the bipartite graph onto the two node sets
projected_graph_1 = bipartite.projected_graph(bipartite_graph, [1, 2, 3, 4, 5])
projected_graph_2 = bipartite.projected_graph(bipartite_graph, ["1", "2", "3", "4", "5", "6"])

# Draw the two projected graphs
nx.draw_networkx(projected_graph_1, pos=nx.spring_layout(projected_graph_1), with_labels=True, node_color="white")
plt.show()

nx.draw_networkx(projected_graph_2, pos=nx.spring_layout(projected_graph_2), with_labels=True, node_color="grey")
plt.show()

"""### Exercise 2"""

# Function to create a sample graph
def create_sample_graph():
    graph = nx.Graph()
    edges = [("A", "B"), ("A", "D"), ("A", "E"), ("B", "C"), ("B", "E"), ("B", "F"), ("C", "F"), ("E", "F"), ("E", "I"),
             ("E", "J"), ("F", "J"), ("F", "H"), ("F", "K"), ("H", "K"), ("J", "K"), ("K", "L")]
    graph.add_edges_from(edges)
    return graph

# Draw the sample graph
sample_graph = create_sample_graph()
nx.draw_networkx(sample_graph, pos=nx.spring_layout(sample_graph), with_labels=True)
plt.show()

# Analyze the graph through iterations
def analyze_graph_iteration(graph):
    data = {}
    graphs = {}
    iteration = 0

    while iteration >= 0:
        data[iteration] = {}
        data[iteration]["degree"] = dict(graph.degree())
        data[iteration]["avg_degree"] = np.mean(list(data[iteration]["degree"].values()))
        data[iteration]["density"] = graph.number_of_edges() / graph.number_of_nodes()
        graphs[iteration] = graph.copy()

        for node, degree in data[iteration]["degree"].items():
            if degree < data[iteration]["avg_degree"]:
                graph.remove_node(node)

        # Uncomment the following lines to visualize each iteration
        # nx.draw_networkx(graph, pos=nx.spring_layout(graph), with_labels=True)
        # plt.show()

        if iteration > 0 and data[iteration]["avg_degree"] == data[iteration - 1]["avg_degree"] \
                and data[iteration]["density"] == data[iteration - 1]["density"]:
            iteration = -1
        else:
            iteration += 1

    # Create DataFrames for analysis results
    degree_df = pd.DataFrame.from_records(value["degree"] for value in data.values()).fillna("r")
    others_df = pd.DataFrame.from_records({"avg_degree": value["avg_degree"], "density": value["density"]} for value in data.values())
    iterations_df = degree_df.merge(others_df, how="inner", left_index=True, right_index=True)
    print(iterations_df.to_latex())

    # Draw the graphs for different iterations
    for i in range(1, min(4, len(graphs))):
        nx.draw_networkx(graphs[i], pos=nx.spring_layout(graphs[i]), with_labels=True)
        plt.show()

# Analyze the sample graph
analyze_graph_iteration(create_sample_graph())

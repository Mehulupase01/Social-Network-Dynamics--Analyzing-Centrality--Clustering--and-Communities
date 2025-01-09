# Social Network Dynamics: Analyzing Centrality, Clustering, and Communities
 This project explores social network dynamics by analyzing real-world network datasets using various graph theory techniques. It implements algorithms to measure centrality, clustering, and community detection, providing a deep understanding of the structure and behavior of social networks.

## Overview

Social network analysis plays a critical role in understanding the structure and evolution of complex networks. This project delves into key topics in network theory, including **centrality measures**, **clustering coefficients**, **bipartite graph projections**, and **community detection**. The dataset used in this project represents a social network where nodes are individuals and edges represent their relationships or interactions.

In this project, we use two main social network datasets: **medium.in** and **large.in**, each containing directed edges between users in a social network. The project utilizes various Python libraries, such as **NetworkX** and **Networkit**, to analyze the networks, calculate centrality measures, and visualize key metrics.

### Key Features:
- **Centrality Measures**: We calculate degree centrality, betweenness centrality, and PageRank to identify the most important nodes within the network.
- **Clustering Coefficients**: Measures the tendency of a node’s neighbors to be interconnected.
- **Bipartite Graph Projections**: Analysis of bipartite graphs, both unweighted and weighted projections, to study relationships between two distinct node sets.
- **Community Detection**: Implementation of community detection algorithms to uncover dense subgroups within the network using modularity-based methods.
- **Visualization**: Various visualizations are generated to illustrate the network structure, degree distributions, and clustering results.

The code provides a comprehensive set of tools for analyzing and visualizing social networks, offering insights into the connectivity, structure, and behavior of individuals within the network.

## Project Goals

- **Measure Centrality**: Calculate the relative importance of nodes within the network using degree centrality, betweenness centrality, and PageRank.
- **Examine Clustering**: Calculate the clustering coefficients and analyze the density of ego networks to understand local network connectivity.
- **Bipartite Graph Analysis**: Project bipartite graphs to one-mode projections, both weighted and unweighted, to study relationships between different sets of nodes.
- **Detect Communities**: Apply community detection algorithms, such as modularity-based methods, to identify communities or clusters within the network.
- **Visualize Networks**: Use **Matplotlib** and **NetworkX** to create visual representations of the network, including degree distributions, connected components, and community structures.

## Data and Datasets

The datasets used in this project are from social networks, where nodes represent users, and directed edges represent relationships or interactions between them. The two primary datasets are:

- **medium.in**: A smaller social network dataset.
- **large.in**: A larger network dataset with more nodes and edges.

These datasets can be found in the following format:

- Each line represents a directed edge: `source_node [tab] target_node`

## Code Implementation

The project code is split into three primary files:

- **`exercise_1_and_2.py`**: Implements the core network analysis, including bipartite graph projections, centrality calculations, and network analysis.
- **`medium.py`**: Code for analyzing the **medium.in** dataset, including centrality measures, clustering coefficients, and graph visualization.
- **`large.py`**: Similar code to **medium.py**, but working with the larger **large.in** dataset for more complex analysis.

### Key Algorithms & Techniques:

1. **Bipartite Graph Projection**:
   - Project a bipartite graph into one-mode projections.
   - Calculate the relationship between grey nodes' degrees and the maximal clique size in the projected network.

2. **Charikar’s Greedy Algorithm**:
   - Use a greedy algorithm to find the densest subgraph in an undirected graph, iteratively removing nodes with the lowest degree.

3. **Centrality Measures**:
   - **Degree Centrality**: Measures the number of direct connections a node has.
   - **Betweenness Centrality**: Measures the frequency with which a node acts as a bridge along the shortest path between other nodes.
   - **PageRank**: A probabilistic measure of a node's importance based on its connections.

4. **Community Detection**:
   - Using the **Louvain** method, a modularity-based community detection algorithm, to identify groups or clusters within the network.
   - Modularity is used to quantify the structure of networks into dense subgraphs.

## Mathematical Formulas

- **Clustering Coefficient** of a node \(v\) is defined as:
  \[
  C(v) = \frac{2e}{k(k - 1)}
  \]
  where \(e\) is the number of edges between neighbors of node \(v\), and \(k\) is the degree of node \(v\).

- **Ego Network Density** is given by:
  \[
  D = \frac{2m}{n(n - 1)}
  \]
  where \(m\) is the number of edges and \(n\) is the number of nodes in the ego network.

- **Densest Subgraph** using Charikar’s algorithm calculates density as:
  \[
  d(G) = \frac{2|E|}{|V|}
  \]
  where \(|E|\) is the set of edges and \(|V|\) is the set of nodes in the graph.

## Visualizations

The project includes various visualizations to help analyze the network structure:

- **Degree Distribution**: A log-log plot showing the distribution of indegree and outdegree across nodes.
- **Connected Components**: Visualizations of both weakly and strongly connected components in the network.
- **Distance Distribution**: A bar plot of the distance between nodes in the largest weakly connected component.

## Results & Discussion

### Network Properties

The following key results are presented:
- Number of directed edges in the network.
- Number of nodes (users) in the network.
- Strongly and weakly connected components.
- Average clustering coefficient for the network.

### Centrality & Rankings

Centrality rankings are computed for each dataset, comparing the top nodes based on degree, betweenness, and PageRank centralities.

### Community Detection

Community detection results reveal the modularity score and the identified communities within the network. The results indicate the presence of dense subgroups, which can provide insights into the structure and functionality of the network.

## Conclusion

This project provides a thorough exploration of social network analysis, applying core graph theory techniques to understand the connectivity, centrality, and community structure of social networks. The code can be easily adapted to other datasets, allowing for further experimentation and analysis in the field of social network dynamics.


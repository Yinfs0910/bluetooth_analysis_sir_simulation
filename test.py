print("Hello, World!")

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import powerlaw
import random


def calcuate_networks():
    # Create a random graph with 100 nodes and a probability of 0.01 for each edge
    g = nx.gnp_random_graph(100, 0.01)
    # Print the degree distribution of the graph
    print(nx.degree_histogram(g))
    # Draw the graph
    nx.draw(g, with_labels=True)
    plt.show()
    # Calculate the degree moment for the 3rd order
    print(weight_moment(list(dict(g.degree).values()), 3))
    # Calculate the clustering coefficient
    print(nx.average_clustering(g))
    # Calculate the shortest path length between all pairs of nodes
    print(nx.floyd_warshall(g))
    # Calculate the diameter of the graph
    print(nx.diameter(g))
    # Calculate the radius of the graph
    print(nx.radius(g))
    # Calculate the eccentricity of each node
    print(nx.eccentricity(g))
    # Calculate the average shortest path length
    print(nx.average_shortest_path_length(g))
    # Calculate the average clustering coefficient
    print(nx.average_clustering(g))
    # Calculate the transitivity of the graph
    print(nx.transitivity(g))
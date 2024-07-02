import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats as stats
import powerlaw

# Number of nodes
n = 10000


def power_law(alpha, size):
    """
    Generate samples from a power-law distribution.

    Args:
    alpha (float): Power-law exponent.
    size (int): Number of samples to generate.

    Returns:
    numpy.ndarray: An array of power-law distributed samples.
    """
    C = (alpha - 1) / (size ** (1 - alpha) - 1)  # Normalization constant
    u = np.random.uniform(0, 1, size)
    samples = (1 - u) ** (-1 / (alpha - 1))//C
    return C * samples

def momment_of_degree_distribution(g, n):
    """
    Calculate the n-th moment of the degree distribution of a graph.

    Args:
    g (networkx.Graph): Input graph.
    n (int): The order of the moment.

    Returns:
    float: The n-th moment of the degree distribution.
    """
    degree_np = np.array(list(dict(g.degree).values()))**n
    return (sum(degree_np ** n) / len(g))

def weight_moment(weight, n):
    """
    Calculate the n-th moment of a list of weights.

    Args:
    weight (list): List of weights.
    n (int): The order of the moment.

    Returns:
    float: The n-th moment of the weight distribution.
    """
    return (sum(np.array(weight) ** n) / len(weight))

def calculate_joint_moment(samples_X, samples_Y):
    """
    Calculate the joint moment of two sets of samples.

    Args:
    samples_X (numpy.ndarray): First set of samples.
    samples_Y (numpy.ndarray): Second set of samples.

    Returns:
    float: The joint moment of the two sets of samples.
    """
    return np.mean(samples_X * samples_Y)

def generate_network(sequence, weight_list, path):
    """
    Generate a network based on degree sequence and weight list, and save it to a file.

    Args:
    sequence (list): Degree sequence.
    weight_list (list): List of weights for edges.
    path (str): File path to save the network.

    Returns:
    float: First moment of degree distribution (k1).
    float: Joint moment of degree and weight distributions (kl).
    """
    G1 = nx.configuration_model(sequence)
    G1.remove_edges_from(nx.selfloop_edges(G1))
    G1 = nx.Graph(G1)

    # Extract the largest component subgraph
    Gcc = sorted(nx.connected_components(G1), key=len, reverse=True)
    SG = G1.subgraph(Gcc[0])

    for index, item in enumerate(SG.edges()):
        SG[item[0]][item[1]]["weight"] = weight_list[index]

    # Save the network file
    with open(path, 'w') as outfile:
        for index, item in enumerate(SG.edges()):
            outfile.write(str(item[0]) + ',' +
                          str(item[1]) + ',' +
                          str(weight_list[index]) + '\n')

    # Calculate the result
    k1 = momment_of_degree_distribution(SG, 1)
    x = np.array(list(dict(SG.degree()).values()))
    y = np.array(list(dict(SG.degree(weight='weight')).values())).astype(int)
    kl = calculate_joint_moment(x, y)
    return k1, kl


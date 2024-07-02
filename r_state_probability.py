import numpy as np
from load_adjacency import load_adj
import matplotlib.pyplot as plt
import networkx as nx
import EoN

def calculate_node_weight_sums(graph):
    """
    Calculate the sum of weights of neighboring edges for each node in the graph.

    Args:
    graph (networkx.Graph): The input graph.

    Returns:
    dict: A dictionary mapping nodes to their total weight sums.
    """
    node_weight_sums = {}
    for node in graph.nodes():
        neighbors = graph.neighbors(node)
        total_weight = sum(graph[node][neighbor]["weight"] for neighbor in neighbors)
        node_weight_sums[node] = total_weight
    return node_weight_sums

def calculate_node_degrees(graph):
    """
    Calculate the degree of each node in the graph.

    Args:
    graph (networkx.Graph): The input graph.

    Returns:
    dict: A dictionary mapping nodes to their degrees.
    """
    node_degrees = {}
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        degree = len(neighbors)
        node_degrees[node] = degree+1
    return node_degrees

def calculate_node_degree_weight_product(graph):
    """
    Calculate the product of degree and weight sum for each node in the graph.

    Args:
    graph (networkx.Graph): The input graph.

    Returns:
    dict: A dictionary mapping nodes to their degree-weight product.
    """
    node_degree_weight_product = {}
    for node in graph.nodes():
        total_degree = len(list(graph.neighbors(node)))
        total_weight = sum(graph[node][neighbor]["weight"] for neighbor in graph.neighbors(node))
        product = total_degree * total_weight
        node_degree_weight_product[node] = product
    return node_degree_weight_product

def generate_network(data):
    """
    Generate a network from input data.

    Args:
    data (pandas.DataFrame): The input data containing network information.

    Returns:
    networkx.Graph: The generated network.
    """
    SG = nx.Graph()
    for index, row in data.iterrows():
        source = row["Source"]
        target = row["Target"]
        weight = row["Weight"]
        SG.add_node(source)
        SG.add_node(target)
        SG.add_edge(source, target, weight=weight)
    return SG

def simulate_sir(G, beta, gamma, num_simulations):
    """
    Simulate the SIR model on the network.

    Args:
    G (networkx.Graph): The input network.
    beta (float): The transmission rate.
    gamma (float): The recovery rate.
    num_simulations (int): The number of simulations to run.

    Returns:
    dict: A dictionary mapping nodes to their recovery counts.
    """
    nodes = G.nodes
    recovery_counts = {node: 0 for node in nodes}
    for _ in range(num_simulations):
        sim = EoN.fast_SIR(G, beta, gamma, transmission_weight='weight', return_full_data=True, rho=0.02)
        t = sim.t()
        t_max = t[-1]
        for node in nodes:
            final_statuses = sim.node_status(node, t_max)
            if final_statuses == 'R':
                recovery_counts[node] += np.round(1/num_simulations, 2)
    return recovery_counts

if __name__ == "__main__":
    # Parameters
    beta = 0.084
    gamma = 0.6
    num_simulations = 100

    # Load data
    data = load_adj("exp_pow.csv")
    G = generate_network(data)

    # Simulate SIR and calculate correlations
    p = simulate_sir(G, beta, gamma, num_simulations)
    weight = calculate_node_weight_sums(G)
    degree = calculate_node_degrees(G)
    d_w = calculate_node_degree_weight_product(G)

    # Create subplots
    plt.figure(figsize=(8.9 / 2.54, 6.75 / 2.54))
    plt.subplot(1, 3, 1)
    plt.loglog(list(degree.values()), list(p.values()), marker='o', linestyle='', c='b')
    plt.xlabel("Degree", fontsize=15)
    plt.ylabel("P", fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.subplot(1, 3, 2)
    plt.loglog(list(weight.values()), list(p.values()), marker='o', linestyle='', c='b')
    plt.xlabel("Weight", fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.subplot(1, 3, 3)
    plt.loglog(list(d_w.values()), list(p.values()), marker='o', linestyle='', c='b')
    plt.xlabel("Degree*Weight", fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.suptitle("exp_pow", fontsize=20)
    plt.savefig('exp_pow_correlation.png', dpi=300)
    plt.show()







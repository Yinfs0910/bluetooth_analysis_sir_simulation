import networkx as nx
import EoN
import matplotlib.pyplot as plt
import numpy as np
from load_adjacency import load_adj
import pandas as pd

def degree_moment(g, n):
    """
    Calculate the nth degree moment of a graph.

    Args:
    g (networkx.Graph): The input graph.
    n (int): The degree moment to calculate.

    Returns:
    float: The nth degree moment of the graph.
    """
    degree_np = np.array(list(dict(g.degree).values()))
    return (sum(degree_np**n)/len(g))

def weight_moment(weight, n):
    """
    Calculate the nth moment of a weight distribution.

    Args:
    weight (list): The list of weights.
    n (int): The moment to calculate.

    Returns:
    float: The nth moment of the weight distribution.
    """
    return (sum(np.array(weight)**n)/len(weight))

def calculate_joint_moment(samples_X, samples_Y):
    """
    Calculate the joint moment of two sets of samples.

    Args:
    samples_X (array-like): The first set of samples.
    samples_Y (array-like): The second set of samples.

    Returns:
    float: The joint moment of the two sets of samples.
    """
    return np.mean(samples_X * samples_Y)

def get_color(title):
    """
    Get a color code based on the title.

    Args:
    title (str): The title to determine the color for.

    Returns:
    str: The color code.
    """
    color = ""
    if title == "exp_pow":
        color = "ro--"
    elif title == "pow_pow":
        color = "yo--"
    else:
        color = "bo--"
    return color

def sir_sim(data, color, label):
    """
    Simulate SIR model and plot the results.

    Args:
    data (pandas.DataFrame): The input data containing network information.
    color (str): The color for plotting.
    label (str): The label for the legend.

    Returns:
    tuple: A tuple containing two lists (result_1, result_2).
    """
    SG = nx.Graph()

    for index, row in data.iterrows():
        source = row["Source"]
        target = row["Target"]
        weight = row["Weight"]

        SG.add_node(source)
        SG.add_node(target)
        SG.add_edge(source, target, weight=weight)

    x = np.array(list(dict(SG.degree()).values()))
    y = np.array(list(dict(SG.degree(weight='weight')).values())).astype(int)
    kw = calculate_joint_moment(x, y)

    r_all = []
    lbd = []

    mu = 0.6
    for beta in np.arange(0, 0.5, 0.002):
        r = []
        for _ in range(0, 100):
            t, S, I, R = EoN.fast_SIR(SG, beta, mu, transmission_weight='weight', rho=0.02)
            r.append(R[-1])
        r_all.append(np.mean(r))
        lbd.append(beta / mu)

    plt.plot(sorted(lbd), np.array(sorted(r_all)) / 10000, color, label=label)
    plt.ylim(0, 0.6)  # Set the y-axis range, adjust according to your data
    plt.yticks(np.arange(0, 0.7, 0.1))  # Set y-axis ticks
    lambda_c = degree_moment(SG, 1) / kw
    result_1 = sorted(lbd)
    result_2 = np.array(sorted(r_all)) / 10000

    plt.axvline(lambda_c, color=color[0])
    plt.xlabel("Lambda", fontsize=15)
    plt.ylabel("Fraction of infected nodes", fontsize=15)
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.title(str(data).split(".")[0])
    return result_1, result_2

if __name__ == "__main__":
    # Load data
    data_1 = load_adj("exp_pow.csv")
    data_3 = load_adj("pow_pow.csv")
    data_5 = load_adj("poi_pow.csv")

    plt.figure(figsize=(8.9 / 2.54, 6.75 / 2.54))

    plt.subplot(1, 3, 1)
    list1, list2 = sir_sim(data_1, "bo--", "<k> = 11.0 \n<w> = 1.0")
    plt.title("exp_pow.csv")
    df = pd.DataFrame({'Column1': list1, 'Column2': list2})
    df.to_csv("exp_data")

    plt.subplot(1, 3, 2)
    list3, list4 = sir_sim(data_3, "go--", "<k> = 11.0 \n<w> = 1.0")
    plt.title("pow_pow")
    df = pd.DataFrame({'Column1': list3, 'Column2': list4})
    df.to_csv("pow_data.csv")

    plt.subplot(1, 3, 3)
    list5, list6 = sir_sim(data_5, "ro--", "<k> = 11.0 \n<w> = 1.0")
    plt.title("poi_pow")
    df = pd.DataFrame({'Column1': list5, 'Column2': list6})
    df.to_csv("poi_data.csv")

    plt.savefig('compare_result.png', dpi=300)
    plt.show()





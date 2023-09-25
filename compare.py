import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_data(f):
    """
    Load data from a pickle file and convert it into a DataFrame.

    Args:
    f (file): The file object containing the pickled data.

    Returns:
    pandas.DataFrame: A DataFrame containing the loaded data with columns 'device_count' and 'contact_count'.
    """
    data = pickle.load(f)
    device = []
    contact = []
    for item in data:
        if len(item) != 0:
            device += item[0].keys()
            contact += item[0].values()

    df = pd.DataFrame({'device_count': device, 'contact_count': contact})
    df = df.sort_values(by='device_count').reset_index(drop=True)
    df = df[df['device_count'] != 0]
    return df


def make_plot(df, title):
    """
    Generate and display plots for data analysis.

    Args:
    df (pandas.DataFrame): The DataFrame containing the data.
    title (str): The title for the plot.

    Returns:
    None
    """
    plt.figure(figsize=(17, 5))

    # Plot 1: Log-log histogram of 'device_count'
    plt.subplot(1, 3, 1)
    n, bins, patches = plt.hist(df['device_count'], bins=1000)
    plt.cla()
    bins_mod = []
    for i in range(len(bins) - 1):
        bins_mod.append((bins[i] + bins[i + 1]) / 2)
    n_mod = [float(n[-1]) / len(df['device_count'])]
    for i in range(1, len(n)):
        to_append = n_mod[0] + float(n[-(i + 1)]) / len(df['device_count'])
        n_mod = [to_append] + n_mod
    plt.semilogy(bins_mod, n_mod, marker='o', linestyle='', label='ccdf')
    plt.xlabel('number of encountered devices', fontsize=15)
    plt.ylabel('ccdf', fontsize=16)
    plt.xticks([0, 1000, 2000, 3000], fontsize=16)
    plt.yticks(fontsize=16)

    # Plot 2: Log-log histogram of 'contact_count' / 'device_count'
    plt.subplot(1, 3, 2)
    n, bins, patches = plt.hist(np.array(df['contact_count']) / np.array(df['device_count']), bins=1000)
    plt.cla()
    bins_mod = []
    for i in range(len(bins) - 1):
        bins_mod.append((bins[i] + bins[i + 1]) / 2)
    n_mod = [float(n[-1]) / len(df['device_count'])]
    for i in range(1, len(n)):
        to_append = n_mod[0] + float(n[-(i + 1)]) / len(df['device_count'])
        n_mod = [to_append] + n_mod
    plt.loglog(bins_mod, n_mod, marker='o', linestyle='', label='ccdf')
    plt.xlabel('number of received signals per device', fontsize=15)
    plt.ylabel('ccdf', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Plot 3: Log-log histogram of 'contact_count'
    plt.subplot(1, 3, 3)
    n, bins, patches = plt.hist(df['contact_count'], bins=1000)
    plt.cla()
    bins_mod = []
    for i in range(len(bins) - 1):
        bins_mod.append((bins[i] + bins[i + 1]) / 2)
    n_mod = [float(n[-1]) / len(df['device_count'])]
    for i in range(1, len(n)):
        to_append = n_mod[0] + float(n[-(i + 1)]) / len(df['device_count'])
        n_mod = [to_append] + n_mod
    plt.loglog(bins_mod, n_mod, marker='o', linestyle='', label='ccdf')
    plt.xlabel('total number of received signals per participant', fontsize=15)
    plt.ylabel('ccdf', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    plt.savefig(title + ".png")


def compare(df):
    """
    Compare and plot multiple data distributions.

    Args:
    df (list of pandas.DataFrame): A list of DataFrames containing different data distributions.

    Returns:
    None
    """
    colors = ["black", "green", "blue", "red", "grey", "yellow"]
    labels = ["18-29", "30-39", "40-49", "50-59", "60-69", "total"]
    marker = ["o", "o", "o", "o", "o", "o"]
    all_data = []

    for data in df:
        n, bins, patches = plt.hist(data["average"], bins=10000)
        plt.clf()
        bins_mod = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
        n_mod = [float(n[-1]) / len(data)]
        for i in range(1, len(n)):
            to_append = n_mod[0] + float(n[-(i + 1)]) / len(data)
            n_mod = [to_append] + n_mod
        all_data.append((bins_mod, n_mod))

    for idx, (bins_mod, n_mod) in enumerate(all_data):
        plt.loglog(bins_mod, n_mod, marker=marker[idx], linestyle='', c=colors[idx], label=labels[idx])
        plt.legend()
        plt.xlabel('number of received signals per device', fontsize=15)
        plt.ylabel('ccdf', fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

    plt.tight_layout()
    plt.show()


def split_list_into_parts(lst, num_parts):
    """
    Split a list into approximately equal-sized parts.

    Args:
    lst (list): The list to be split.
    num_parts (int): The number of parts to split the list into.

    Returns:
    list of lists: A list of lists containing the split parts of the original list.
    """
    avg_part_size = len(lst) // num_parts
    remainder = len(lst) % num_parts
    parts = []
    start = 0

    for _ in range(num_parts):
        part_size = avg_part_size + 1 if remainder > 0 else avg_part_size
        parts.append(lst[start:start + part_size])
        start += part_size
        remainder -= 1

    return parts


if __name__ == "__main__":
    result = pd.read_csv("result.csv", index_col=[0])
    a = result[result['age'] == 1].sort_values(by='average')
    b = result[result['age'] == 2].sort_values(by='average')
    c = result[result['age'] == 3].sort_values(by='average')
    d = result[result['age'] == 4].sort_values(by='average')
    e = result[result['age'] == 5].sort_values(by='average')

    degree_1 = result[result["device"] <= 158]
    degree_2 = result[(result['device'] > 158) & (result['device'] <= 396)]
    degree_3 = result[(result['device'] > 396) & (result['device'] <= 714)]
    degree_4 = result[(result['device'] > 714)]

    n, bins, patches = plt.hist(result['average'], bins=100000)
    plt.cla()
    bins_mod = []
    for i in range(len(bins) - 1):
        bins_mod.append((bins[i] + bins[i + 1]) / 2)
    n_mod = [float(n[-1]) / len(result['average'])]
    for i in range(1, len(n)):
        to_append = n_mod[0] + float(n[-(i + 1)]) / len(result['average'])
        n_mod = [to_append] + n_mod
    plt.figure(figsize=(8.9 / 2.54, 6.75 / 2.54))
    plt.loglog(bins_mod, n_mod, marker='o', linestyle='', label='ccdf')
    plt.xlabel('number of received signals per device', fontsize=15)
    plt.ylabel('ccdf', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.show()

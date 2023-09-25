import pandas as pd


def load_adj(filename):
    """
    Load an adjacency list from a CSV file and return it as a DataFrame.

    Args:
    filename (str): The name of the CSV file containing the adjacency list.

    Returns:
    pandas.DataFrame: A DataFrame containing the adjacency list with columns "Source," "Target," and "Weight."
    """
    # Define custom column names for the DataFrame
    custom_cols = ["Source", "Target", "Weight"]

    # Read the CSV file into a DataFrame with custom column names
    df = pd.read_csv(filename, header=None, names=custom_cols)

    return df


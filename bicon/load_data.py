import pandas as pd
import numpy as np
from scipy import stats
import networkx as nx
import csv


def data_preprocessing(path_expr, path_net, log2=True, zscores=True, size=2000, no_zero=None, formats=None):
    """
    Raw data processing for further analysis

    :param path_expr: path for gene expression ATTENTION: expression data format:
    genes as rows (first column - gene ids), patients as columns path_ppi
    :param path_net: path for ppi
    :param log2: log2 transform (if needed)
    :param zscores: zscoring (if needed)
    :param size: specify size of the gene set  for standard deviation preselection. Usually optimal values are
    between 2000 and 5000
    :param no_zero: proportion of non-zero elements for each gene.
    If there are less values then a gene will not be maintained
    :param formats: format = list of data types for gene expression matrix and the ppi network.
    Example ["csv", "tsv"]. Used if the
    automatic delimiter identification needs to be omitted
    :return: expr: expression matrix
    G: networkX graph
    labels: dictionary used for conversion from real IDs to internal IDs
    rev_labels: dictionary to convert bach to real IDs
    """

    d_expr = None
    d_ppi = None

    if formats is not None:
        if formats[0] in ("csv", "tsv"):
            d_expr = formats[0]

        if formats[1] in ("csv", "tsv"):
            d_ppi = formats[1]

    expr = open_file(path_expr, d_expr)
    expr = expr.set_index(expr.columns[0])
    patients_new = list(set(expr.columns))
    tot_pats = len(patients_new)

    net = open_file(path_net, d_ppi, header=None)
    nodes_ppi1 = net[0].to_list()
    nodes_ppi1 = [str(x) for x in nodes_ppi1]
    nodes_ppi2 = net[1].to_list()
    nodes_ppi2 = [str(x) for x in nodes_ppi2]

    expr.drop_duplicates(inplace=True)
    genes_ge = list(expr.index)
    expr.index = [str(x) for x in genes_ge]
    genes_ge = list(set(genes_ge))
    if no_zero is not None:
        th = round(no_zero * tot_pats)
        valid_genes = []
        for x in genes_ge:
            try:
                if sum(list(expr.loc[x] > 1e-5)) > th:
                    valid_genes.append(x)
            except TypeError:
                print(
                    "WARNING: entity {0} appears more than once with different values. We recommend to double check "
                    "your IDs if you want to avoid information loss. You can also just continue with the analysis, "
                    "but all duplicates will be removed".format(
                        x))
        genes_ge = valid_genes
    new_genes_ge = set([str(x) for x in genes_ge])
    intersec_genes = set.intersection(new_genes_ge, set(nodes_ppi1 + nodes_ppi2))
    assert len(intersec_genes) > 0, "The identifiers in the expression file and network file do not match"

    intersec_genes_list = list(intersec_genes)
    expr = expr.loc[intersec_genes_list]
    if log2:
        minimal = expr.min().min()
        if minimal <= 0:
            expr += np.abs(minimal - 1)
        expr = np.log2(expr)

    if size is not None:  # std selection
        if len(intersec_genes) > size:
            std_genes = expr.std(axis=1)
            std_genes, intersec_genes_list = zip(*sorted(zip(std_genes, intersec_genes_list)))
            genes_for_expr = list(intersec_genes_list)[len(std_genes) - size:]
            intersec_genes = set(genes_for_expr)

        else:
            genes_for_expr = intersec_genes_list
    else:
        genes_for_expr = intersec_genes_list

    expr = expr.loc[genes_for_expr]
    if zscores:
        expr = pd.DataFrame((stats.zscore(expr.T)).T, columns=expr.columns, index=expr.index)
    nodes = list(expr.index) + list(expr.columns)
    labels = {i:nodes[i] for i in range(len(nodes))}
    rev_labels = {v:k for k,v in labels.items()}

    edges = []
    for i in range(len(nodes_ppi1)):
        if nodes_ppi1[i] in intersec_genes and nodes_ppi2[i] in intersec_genes:
            edges.append((rev_labels[nodes_ppi1[i]], rev_labels[nodes_ppi2[i]]))
    n, m = expr.shape
    G = nx.Graph()
    G.add_edges_from(edges)
    expr.index = np.arange(n)
    expr.columns = np.arange(n, n + m)
    return expr, G, labels, rev_labels


def open_file(file_name, d, **kwargs):
    """
    Alows to determine the delimiter automatically given the path or directly the object
    :param file_name: Name of the file to open
    :param d: delimeter
    :param kwargs: other pandas.read_csv() parameters
    :return: pandas dataframe
    """
    if d is None:
        if isinstance(file_name, str):
            with open(file_name, 'r') as csvfile:
                dialect = csv.Sniffer().sniff(csvfile.readline())
        else:  # the file is StringIO
            file_name.seek(0)
            dialect = csv.Sniffer().sniff(file_name.readline())

            file_name.seek(0)
        sp = dialect.delimiter
    else:
        if d == "csv":
            sp = ","
        elif d == "tsv":
            sp = "\t"
        else:
            raise TypeError(f'Wrong delimiter format {d} please do not pass any value for d')

    file = pd.read_csv(file_name, sep=sp, low_memory=False, **kwargs)
    return file

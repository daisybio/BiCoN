#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy import stats
import networkx as nx
import csv


def data_preprocessing(path_expr, path_net, log2=True, zscores=True, size=2000, no_zero=None, formats=None):
    """
    Raw data processing for further analysis

    Attributes:
    -----------
    non-default:
    path_expr - path for gene expression
    ATTENTION: expression data format: genes as rows (first column - gene ids), patients as columns
    path_ppi - path for ppi
    log2 - log2 transform (if needed)
    size -   specify size of the gene set  for standard deviation preselection. Usually optimal values are between 2000 and 5000\
    z-scores - indicates if z-scores normalization should be applied (default - True)
    no_zero - proportion of non-zero elements for each gene. If there are less values then a gene will not be maintained
    format = list of data types for gene expression matrix and the ppi network. Example ["csv", "tsv"]. Used if the automatic delimiter needs to be omitted
    """

    d_expr = None
    d_ppi = None

    if formats != None:
        if formats[0] in ("csv", "tsv"):
            d_expr = formats[0]

        if formats[1] in ("csv", "tsv"):
            d_ppi = formats[1]

    expr = open_file(path_expr, d_expr)
    expr = expr.set_index(expr.columns[0])
    patients_new = list(set(expr.columns))
    tot_pats = len(patients_new)

    net = open_file(path_net, d_ppi, header=None)
    nodes_ppi = list(set(net[0]).union(set(net[1])))
    expr.drop_duplicates(inplace=True)
    genes_ge = list(expr.index)
    expr.index = [str(x) for x in genes_ge]
    genes_ge = list(set(genes_ge))
    if no_zero != None:
        th = round(no_zero * tot_pats)
        valid_genes = []
        for x in genes_ge:
            try:
                if sum(list(expr.loc[x] > 1e-5)) > th:
                    valid_genes.append(x)
            except TypeError:
                print(
                    "WARNING: entity {0} appears more than once with different values. We recommend to double check your IDs if you want to avoid information loss. You can also just continue with the analysis, but all duplicates will be removed".format(
                        x))
        genes_ge = valid_genes
    new_genes_ge = set([str(x) for x in genes_ge])
    new_genes_ppi = set([str(x) for x in nodes_ppi])
    intersec_genes = list(set.intersection(new_genes_ge, new_genes_ppi))
    assert len(intersec_genes) > 0, "The identifiers in the expression file and network file do not match"

    expr = expr.loc[intersec_genes]
    if log2:
        minimal = expr.min().min()
        if minimal <= 0:
            expr += np.abs(minimal - 1)
        expr = np.log2(expr)

    if size != None:  # std selection
        if len(intersec_genes) > size:
            std_genes = expr.std(axis=1)
            std_genes, intersec_genes = zip(*sorted(zip(std_genes, intersec_genes)))
            genes_for_expr = list(intersec_genes)[len(std_genes) - size:]
        else:
            genes_for_expr = intersec_genes
    else:
        genes_for_expr = intersec_genes

    expr = expr.loc[genes_for_expr]
    if zscores:
        expr = pd.DataFrame((stats.zscore(expr.T)).T, columns=expr.columns, index=expr.index)

    labels = dict()
    rev_labels = dict()
    node = 0
    # nodes = set(deg_nodes + genes_aco)
    genes = list(expr.index)
    pts = list(expr.columns)
    for g in genes:
        labels[node] = g
        rev_labels[g] = node
        node = node + 1
    for p in pts:
        labels[node] = p
        rev_labels[p] = node
        node = node + 1
    n, m = expr.shape
    G = nx.Graph()
#    G.add_nodes_from(np.arange(n))
    for row in net.itertuples():
        node1 = str(row[1])
        node2 = str(row[2])
        if node1 in set(genes_for_expr) and node2 in set(genes_for_expr):
            G.add_edge(rev_labels[node1], rev_labels[node2])
    expr.index = np.arange(n)
    expr.columns = np.arange(n, n + m)
    return expr, G, labels, rev_labels


# allows to determine the delimiter automatically given the path or directly the object
def open_file(file_name, d, **kwargs):
    if d == None:
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
        if d == "tsv":
            sp = "\t"

    file = pd.read_csv(file_name, sep=sp, low_memory=False, **kwargs)
    return file

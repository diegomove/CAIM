import networkx as nx
from collections import deque, defaultdict
import os
import urllib.request
import gzip
import shutil

def download_data():
    files = ["email-Eu-core.txt.gz"]
    base_url = "https://snap.stanford.edu/data/"
    
    for filename in files:
        txt_name = filename[:-3]
        if not os.path.exists(txt_name):
            urllib.request.urlretrieve(base_url + filename, filename)
            with gzip.open(filename, 'rb') as f_in:
                with open(txt_name, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

download_data()

def normalize_edge(pair):
    return (min(pair), max(pair))

def edge_betweenness(G):
    """ Implementación del algoritmo de Brandes O(VE) """
    bet = defaultdict(float)
    V = list(G.nodes()) 

    for s in V:
        pred = {v: [] for v in V}
        sigma = {v: 0 for v in V}
        dist = {v: -1 for v in V}
        delta = {v: 0 for v in V}

        sigma[s] = 1
        dist[s]  = 0
        queue = deque([s])
        stack = []

        while queue:
            v = queue.popleft()
            stack.append(v)
            for w in G.neighbors(v):
                if dist[w] < 0:
                    queue.append(w)
                    dist[w] = dist[v] + 1
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)

        while stack:
            w = stack.pop()
            for v in pred[w]:
                if sigma[w] == 0: continue
                c = (sigma[v] / sigma[w]) * (1.0 + delta[w])
                bet[normalize_edge((v, w))] += c
                delta[v] += c

    return bet
G = nx.read_edgelist("email-Eu-core.txt", nodetype=int)
G.remove_edges_from(nx.selfloop_edges(G))

import numpy as np
np.random.seed(46562)
sample_nodes = np.random.choice(list(G.nodes()), 300, replace=False)
G_test = G.subgraph(sample_nodes).copy()
G_test = G_test.subgraph(max(nx.connected_components(G_test), key=len)).copy()

my_bet = edge_betweenness(G_test)

nx_bet = nx.edge_betweenness_centrality(G_test, normalized=False)

errores = 0
comparaciones = 0

top_edges = sorted(my_bet.items(), key=lambda x: x[1], reverse=True)[:10]

print(f"{'aresta':<15} | {'bet':<10} | {'NX ':<10} | {'diff'}")

for edge, val in top_edges:
    u, v = edge
    nx_val = nx_bet.get((u,v), nx_bet.get((v,u), 0)) * 2
    
    diff = abs(val - nx_val)
    if diff > 0.001: errores += 1
    comparaciones += 1
    
    print(f"{str(edge):<15} | {val:<10.1f} | {nx_val:<10.1f} | {diff:.4f}")

if errores == 0:
    print("\nNo diff")
else:
    print("\nError")
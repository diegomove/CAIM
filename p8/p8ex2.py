import networkx as nx
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from collections import deque, defaultdict
import numpy as np
import os
import urllib.request
import gzip
import shutil
import networkx.algorithms.community as nx_comm
from sklearn.metrics import adjusted_rand_score

def download_data():
    files = [
        "email-Eu-core.txt.gz",
        "email-Eu-core-department-labels.txt.gz"
    ]
    base_url = "https://snap.stanford.edu/data/"
    
    for filename in files:
        txt_name = filename[:-3]
        if not os.path.exists(txt_name):
            urllib.request.urlretrieve(base_url + filename, filename)
            
            with gzip.open(filename, 'rb') as f_in:
                with open(txt_name, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            print(f"Archivo ya existe: {txt_name}")

download_data()

def normalize_edge(pair):
    return (min(pair), max(pair))

def edge_betweenness(G):
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

def compute_modularity(G_original, components):
    m = G_original.number_of_edges()
    if m == 0: return 0

    Q = 0.0
    two_m = 2.0 * m
    
    for component in components:
        subgraph = G_original.subgraph(component)
        m_c = subgraph.number_of_edges()
        K_c = sum(G_original.degree(n) for n in component)
        
        term1 = m_c / m
        term2 = (K_c / two_m) ** 2
        Q += (term1 - term2)
        
    return Q

def girvan_newman(G):
    G0 = G.copy()
    G = G.copy()
    
    partitions = []
    modularities = []
    
    initial_comps = list(nx.connected_components(G))
    partitions.append(initial_comps)
    modularities.append(compute_modularity(G0, initial_comps))

    while G.number_of_edges() > 0:
        bet = edge_betweenness(G)
        
        if not bet: break
        edge_to_remove = max(bet.items(), key=lambda x: x[1])[0]
        G.remove_edge(*edge_to_remove)
        
        comps = list(nx.connected_components(G))
        
        if len(comps) > len(partitions[-1]):
            mod = compute_modularity(G0, comps)
            partitions.append(comps)
            modularities.append(mod)
            print(f"Comunidades: {len(comps)}, Modularidad: {mod:.4f}")

    return partitions, modularities

G_full = nx.read_edgelist("email-Eu-core.txt", nodetype=int)
G_full.remove_edges_from(nx.selfloop_edges(G_full))

def random_walk_sample(G, size, start=None):
    if start is None:
        start = np.random.choice(list(G.nodes()))
    sample = {start}
    current = start
    size = min(size, G.number_of_nodes())
    while len(sample) < size:
        neighbors = list(set(G.neighbors(current)) - sample)
        if not neighbors:
            current = np.random.choice(list(set(G.nodes()) - sample))
        else:
            current = np.random.choice(neighbors)
        sample.add(current)
    return sample

np.random.seed(4123814)
node_subset = random_walk_sample(G_full, size=250)
smallG = G_full.subgraph(node_subset).copy()
smallG = G_full.subgraph(max(nx.connected_components(smallG), key=len)).copy()

partitions, mods = girvan_newman(smallG)

best_idx = np.argmax(mods)
best_mod = mods[best_idx]
best_partition = partitions[best_idx]
print(f"\nBest mod GN: {best_mod:.4f} ( {len(best_partition)} comunidades)")

plt.figure(figsize=(10, 6))
plt.plot(range(len(mods)), mods, marker='o', linestyle='-', color='b')
plt.axvline(x=best_idx, color='r', linestyle='--', label='Máx Modularidad')
plt.title('Evolución de la Modularidad (Girvan-Newman)')
plt.xlabel('Pasos')
plt.ylabel('Modularidad Q')
plt.legend()
plt.grid(True)
plt.savefig("modularity_evolution.png")

louvain_comms = list(nx_comm.louvain_communities(smallG))
louvain_mod = compute_modularity(smallG, louvain_comms)

ground_truth_file = "email-Eu-core-department-labels.txt"
gt_dict = {}
with open(ground_truth_file) as f:
    for line in f:
        node, label = map(int, line.split())
        if node in smallG:
            gt_dict[node] = label

gt_comms_dict = defaultdict(list)
for node, label in gt_dict.items():
    gt_comms_dict[label].append(node)
gt_comms = list(gt_comms_dict.values())
gt_mod = compute_modularity(smallG, gt_comms)

def get_labels_array(nodes_order, partitions):
    label_map = {}
    for label_id, comm in enumerate(partitions):
        for node in comm:
            label_map[node] = label_id
    return [label_map.get(n, -1) for n in nodes_order]

nodes_ordered = sorted(list(smallG.nodes()))

labels_gt = get_labels_array(nodes_ordered, gt_comms)
labels_gn = get_labels_array(nodes_ordered, best_partition)
labels_louvain = get_labels_array(nodes_ordered, louvain_comms)

ari_gn = adjusted_rand_score(labels_gt, labels_gn)
ari_louvain = adjusted_rand_score(labels_gt, labels_louvain)

print(f"{'metode':<15} | {'mod (Q)':<15} | {'num comunitats':<12} | {'ARI vs REAL':<12}")
print(f"{'GN':<15} | {best_mod:<15.4f} | {len(best_partition):<12} | {ari_gn:<12.4f}")
print(f"{'Louvain':<15} | {louvain_mod:<15.4f} | {len(louvain_comms):<12} | {ari_louvain:<12.4f}")
print(f"{'realitat (GT)':<15} | {gt_mod:<15.4f} | {len(gt_comms):<12} | {'1.0000':<12}")
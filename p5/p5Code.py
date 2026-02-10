import hashlib
import numpy as np
import pickle
import os
import time
from itertools import combinations

print(f"test")

def _termhash(x : str, b : int) -> str:
    """returns bitstring of size b based on md5 algorithm"""
    assert b <= 128, 'this encoding scheme supports hashes of length at most 128; try smaller b'
    h = hashlib.md5(x.encode('utf8')).digest()
    return ''.join(format(byte, '08b') for byte in h)[:b]

fname = '05corpus.pkl'
with open(fname, 'rb') as f:
    corpus = pickle.load(f)

def _simhash(id, b):
    """
        id is the document id, b is the desired length of the simhash
        it should return a bitstring of length b as explained in the
        first section of the notebook
    """
    ## write your code
    sum_vector = np.zeros(b)
    doc_weights = corpus.get(id, {})
    
    for term, weight in doc_weights.items():
        term_hash_str = _termhash(term, b)
        h_prime = np.array([1 if bit == '1' else -1 for bit in term_hash_str])
        sum_vector += (h_prime * weight)
        
    final_hash_bits = ['1' if val > 0 else '0' for val in sum_vector]
    return "".join(final_hash_bits)

def calculate_cosine_similarity(doc_id1, doc_id2):
    doc1_vec = corpus.get(doc_id1, {})
    doc2_vec = corpus.get(doc_id2, {})
    
    similarity = 0.0
    if len(doc1_vec) < len(doc2_vec):
        for term, weight1 in doc1_vec.items():
            similarity += weight1 * doc2_vec.get(term, 0.0)
    else:
        for term, weight2 in doc2_vec.items():
            similarity += doc1_vec.get(term, 0.0) * weight2
    return similarity

parameter_sets = [
    (10,2),
    (10,3),
    (10,4),
    (12,2),
    (12,3),
    (12,4),
    (18, 2),  
    (18, 3),  
    (18, 4),    
    (27, 2)
]

for K, M in parameter_sets:
    
    ## constants
    B = M*K

    simhash = {id: _simhash(id, B) for id in corpus}

    start_time_lsh = time.time()
    hash_tables = [{} for _ in range(M)]

    for doc_id, hash_str in simhash.items():
        if hash_str is None or len(hash_str) != B: continue
            
        for m_index in range(M):
            start_index = m_index * K
            end_index = (m_index + 1) * K
            chunk = hash_str[start_index:end_index]
            
            table = hash_tables[m_index]
            if chunk not in table:
                table[chunk] = []
            table[chunk].append(doc_id)

    candidate_pairs = set()
    for table in hash_tables:
        for bucket in table.values():
            if len(bucket) > 1:
                for pair in combinations(bucket, 2):
                    sorted_pair = tuple(sorted(pair))
                    candidate_pairs.add(sorted_pair)

    lsh_duration = time.time() - start_time_lsh

    SIMILARITY_THRESHOLD = 0.9 
    true_positives_list = []
    false_positives = 0

    start_time_verification = time.time()

    for doc_id1, doc_id2 in candidate_pairs:
        similarity = calculate_cosine_similarity(doc_id1, doc_id2)
        
        if similarity >= SIMILARITY_THRESHOLD:
            true_positives_list.append((doc_id1, doc_id2, similarity))
        else:
            false_positives += 1

    verification_duration = time.time() - start_time_verification

    print(f"\nresultados del analisis (M={M}, K={K})")
    print(f"tiempo de ejecución (fases LSH): {lsh_duration:.4f} s")
    print(f"tiempo de ejecución (verificación): {verification_duration:.4f} s")
    print(f"tiempo total: {(lsh_duration + verification_duration):.4f} s")

    print(f"total pares candidatos:   {len(candidate_pairs)}")
    print(f"verdaderos positivos (sim >= {SIMILARITY_THRESHOLD}): {len(true_positives_list)}")
    print(f"falsos positivos (sim < {SIMILARITY_THRESHOLD}):    {false_positives}")

    true_positives_list.sort(key=lambda x: x[2], reverse=True)

    print("\ntop 10 pares de duplicados verdaderos")
    for i, (id1, id2, sim) in enumerate(true_positives_list[:10]):
        print(f"  {i+1}. Par ({id1}, {id2}) -> Similitud: {sim:.6f}")
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
from pprint import pprint
import tqdm
import numpy as np
import heapq
import time
import math
from collections import Counter

client = Elasticsearch("http://localhost:9200", request_timeout=1000)
INDEX_NAME = "toy"
FIELD_NAME = "text"

def preprocess_query_string(query_string):
    response = client.indices.analyze(
        index=INDEX_NAME, field=FIELD_NAME, text=query_string
    )
    return [token_info["token"] for token_info in response["tokens"]]


def get_total_docs():
    return int(client.cat.count(index=INDEX_NAME, format="json")[0]["count"])


def build_doc_term_freqs():
    start_pre = time.time()
    doc_term_freqs = {}
    for s in tqdm.tqdm(scan(client, index=INDEX_NAME, query={"query": {"match_all": {}}})):
        docid = s["_source"]["path"]
        text = s["_source"]["text"]
        analyzed = client.indices.analyze(index=INDEX_NAME, field=FIELD_NAME, text=text)
        tokens = [t["token"] for t in analyzed["tokens"]]
        doc_term_freqs[docid] = Counter(tokens)
    end_pre = time.time()
    print(f"\nPreprocessing time: {end_pre - start_pre:.2f}s")
    return doc_term_freqs


def compute_idf(doc_term_freqs, total_docs):
    df = Counter()
    for term_freqs in doc_term_freqs.values():
        for term in term_freqs.keys():
            df[term] += 1
    idf = {term: math.log(total_docs / (1 + freq)) for term, freq in df.items()}
    return idf


def compute_tf_idf_vector(doc_term_freqs, idf):
    doc_tfidf = {}
    for docid, term_freqs in tqdm.tqdm(doc_term_freqs.items()):
        tfidf = {t: (tf * idf.get(t, 0.0)) for t, tf in term_freqs.items()}
        norm = np.sqrt(sum(v ** 2 for v in tfidf.values())) or 1.0
        doc_tfidf[docid] = {t: v / norm for t, v in tfidf.items()}
    return doc_tfidf

def slow_search(query_tokens, doc_tfidf):
    sims = {}
    l2query = np.sqrt(len(query_tokens))
    for docid, weights in doc_tfidf.items():
        sim = 0.0
        for w in query_tokens:
            sim += weights.get(w, 0.0)
        sims[docid] = sim / l2query
    return sorted(sims.items(), key=lambda kv: kv[1], reverse=True)


def build_inverted_index(doc_tfidf):
    inverted = {}
    for docid, weights in doc_tfidf.items():
        for term, weight in weights.items():
            inverted.setdefault(term, []).append((docid, weight))
    return inverted


def quick_search(query_tokens, inverted_index):
    sims = {}
    l2query = np.sqrt(len(query_tokens))
    for w in query_tokens:
        if w in inverted_index:
            for docid, weight in inverted_index[w]:
                sims[docid] = sims.get(docid, 0.0) + weight
    for d in sims:
        sims[d] /= l2query
    return heapq.nlargest(10, sims.items(), key=lambda kv: kv[1])


if __name__ == "__main__":
    query_str = "this is a random large text just to test limits on the query to see if there is a big difference between algorythms I like animals houses with roofs and windows in the city and the countryside i do not know what else to write"
    query_tokens = preprocess_query_string(query_str)
    print(f"Query: '{query_str}'\n")

    total_docs = get_total_docs()
    doc_term_freqs = build_doc_term_freqs()
    idf = compute_idf(doc_term_freqs, total_docs)
    doc_tfidf = compute_tf_idf_vector(doc_term_freqs, idf)

    start_slow = time.time()
    result_slow = slow_search(query_tokens, doc_tfidf)
    end_slow = time.time()
    #pprint(result_slow[:10])
    #print(f"\nSlow time: {end_slow - start_slow:.2f}s")

    inverted_index = build_inverted_index(doc_tfidf)
    start_quick = time.time()
    result_quick = quick_search(query_tokens, inverted_index)
    end_quick = time.time()
    #pprint(result_quick[:10])
    #print(f"\nQuick time: {end_quick - start_quick:.2f}s")

    print(f"Tiempo slow:  {end_slow - start_slow:.2f}s")
    print(f"Tiempo quick: {end_quick - start_quick:.2f}s")
    overlap = len(set(d for d, _ in result_slow[:10]) & set(d for d, _ in result_quick[:10]))
    #print(f"Coincidencia en top 10 resultados: {overlap}/10")

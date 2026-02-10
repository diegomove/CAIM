from elasticsearch.helpers import scan
from pprint import pprint
from elasticsearch import Elasticsearch
import tqdm
import numpy as np
from collections import Counter, defaultdict

# ============================
# Funciones auxiliares
# ============================

def preprocess_query_string(query_string, client, index_name, field_name):
    response = client.indices.analyze(index=index_name, field=field_name, text=query_string)
    return [token_info["token"] for token_info in response["tokens"]]

def normalize(tfidf_list):
    norm = np.sqrt(sum(w**2 for _, w in tfidf_list)) or 1.0
    return [(t, w / norm) for t, w in tfidf_list]

def tf_idf(doc_tokens, total_docs, doc_freqs):
    tf = Counter(doc_tokens)
    tfidf = [(t, tf_val * np.log(total_docs / (1 + doc_freqs.get(t, 0)))) for t, tf_val in tf.items()]
    return normalize(tfidf)

# ============================
# Configuración
# ============================

client = Elasticsearch("http://localhost:9200", request_timeout=1000)
INDEX_NAME = "toy"
FIELD_NAME = "text"
TOP_R = 10

# ============================
# Construir índice invertido
# ============================

# Conteo de documentos y frecuencias de términos
total_docs = int(client.cat.count(index=INDEX_NAME, format="json")[0]["count"])
doc_freqs = Counter()
doc_tfidf = {}

for s in tqdm.tqdm(scan(client, index=INDEX_NAME, query={"query": {"match_all": {}}}), total=total_docs):
    docid = s["_source"]["path"]
    text = s["_source"][FIELD_NAME]
    analyzed = client.indices.analyze(index=INDEX_NAME, field=FIELD_NAME, text=text)
    tokens = [t["token"] for t in analyzed["tokens"]]
    for t in set(tokens):
        doc_freqs[t] += 1
    doc_tfidf[docid] = tf_idf(tokens, total_docs, doc_freqs)

# Construir índice invertido: término -> lista de (docid, peso)
inverted_index = defaultdict(list)
for docid, weights in doc_tfidf.items():
    for term, weight in weights:
        inverted_index[term].append((docid, weight))

# ============================
# Quick search
# ============================

query_str = "searching magic"
query_tokens = preprocess_query_string(query_str, client, INDEX_NAME, FIELD_NAME)
print(f"Executing search of query string '{query_str}' with tokens {query_tokens} over documents on index '{INDEX_NAME}'")

sims = defaultdict(float)
l2query = np.sqrt(len(query_tokens))

for w in query_tokens:
    for docid, weight in inverted_index.get(w, []):
        sims[docid] += weight

for d in sims:
    sims[d] /= l2query

# Ordenar y mostrar top resultados
sorted_answer = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
pprint(sorted_answer[:TOP_R])

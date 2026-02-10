from elasticsearch.helpers import scan
from pprint import pprint
from elasticsearch import Elasticsearch
import tqdm
import numpy as np
from collections import Counter

def preprocess_query_string(query_string, client, index_name, field_name):
    response = client.indices.analyze(
        index=index_name, field=field_name, text=query_string
    )
    return [token_info["token"] for token_info in response["tokens"]]

def normalize(tfidf_list):
    norm = np.sqrt(sum(w**2 for _, w in tfidf_list)) or 1.0
    return [(t, w / norm) for t, w in tfidf_list]

def tf_idf(doc_tokens, term_df, total_docs):
    tf = Counter(doc_tokens)
    tfidf = [(t, tf_val * np.log(total_docs / (1 + term_df[t]))) for t, tf_val in tf.items()]
    return normalize(tfidf)

client = Elasticsearch("http://localhost:9200", request_timeout=1000)

r = 10
query_str = "searching magic"
query_tokens = preprocess_query_string(query_str, client, "toy", "text")

print(f"Executing search of query string '{query_str}' with tokens {query_tokens} over documents on index 'toy'")

ndocs = int(client.cat.count(index="toy", format="json")[0]["count"])
doc_tokens_dict = {}
term_df = {}

for s in tqdm.tqdm(scan(client, index="toy", query={"query": {"match_all": {}}}), total=ndocs):
    docid = s["_source"]["path"]
    tokens = preprocess_query_string(s["_source"]["text"], client, "toy", "text")
    doc_tokens_dict[docid] = tokens
    for t in set(tokens):
        term_df[t] = term_df.get(t, 0) + 1

sims = dict()
l2query = np.sqrt(len(query_tokens))

for docid, tokens in doc_tokens_dict.items():
    weights = dict(tf_idf(tokens, term_df, ndocs))
    sims[docid] = 0.0
    for w in query_tokens:
        if w in weights:
            sims[docid] += weights[w]
    sims[docid] /= l2query

sorted_answer = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
pprint(sorted_answer[:r])

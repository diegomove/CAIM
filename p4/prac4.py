from elasticsearch import Elasticsearch
from pprint import pprint

# Conectar al cluster
client = Elasticsearch("http://localhost:9200", request_timeout=1000)

# Comprobar conexión
try:
    info = client.info()
    print(f"Conectado a Elasticsearch {info['version']['number']}")
except Exception as e:
    print("Error al conectar:", e)
    exit(1)

# Definir la query
atomic_query = {"match": {"text": "magic"}}

# Ejecutar búsqueda
try:
    response = client.search(
        index="toy",
        query=atomic_query,
        track_total_hits=True,
        size=5  # limitar resultados a 5
    )
except Exception as e:
    print("Error en la búsqueda:", e)
    exit(1)

# Mostrar resultados
total_hits = response['hits']['total']['value']
print(f"Found {total_hits} documents.\n")

for hit in response["hits"]["hits"]:
    doc_id = hit.get("_id", "N/A")
    score = hit.get("_score", 0)
    source = hit.get("_source", {})
    path = source.get("path", "N/A")
    text = source.get("text", "N/A")
    
    print(f"id: {doc_id}, score: {score:.2f}, path: {path}, text: {text}\n")

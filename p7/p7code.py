import re
import numpy as np
from pyspark.sql import SparkSession
from pprint import pprint

# Start Spark session
# local[*] runs locally using all available cores;
spark = SparkSession.builder.master("local[*]").config("spark.driver.memory", "8g").getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")
print("Using %i cores" % sc.defaultParallelism)

# Tokenization
regex = re.compile(r"[0-9]+|[a-zà-ÿ]+'?[a-zà-ÿ']*", re.IGNORECASE)

def tokens(s):
    return [m.group(0).lower() for m in regex.finditer(s)]

# Load text into an RDD
# We read the CSV with Wikipedia titles from last lab and turn it into an RDD of (doc_id, title string) pairs.
titles_rdd = (spark.read.csv("enwiki-2013-names.csv", header=True, escape="\\")
              .na.fill({"name": ""})
              .rdd
              .map(lambda row: (int(row.node_id), row.name))
              .repartition(6)
              .cache()
            )

print("%i titles" % titles_rdd.count())

# Exercise 1: Inverted index
# Find stopwords
counts = (titles_rdd
    .flatMap(lambda tup:( (word, 1) for word in tokens(tup[1]) ))        # mapper
    .reduceByKey(lambda a, b: a + b)                                     # reducer and combiner
)

stopwords_freq = counts.takeOrdered(40, key=lambda x:-x[1])
# We extract just the words for filtering
stopwords = set([x[0] for x in stopwords_freq])
stopwords_broadcast = sc.broadcast(stopwords)

def inverted_index(rdd):
    return (rdd
        .flatMap(lambda tup: ( (word, tup[0]) for word in tokens(tup[1]) if word not in stopwords_broadcast.value ))
        .groupByKey()
        .mapValues(lambda ids: sorted(list(ids))) 
        .sortByKey()                              
    )

# Execution
result = inverted_index(titles_rdd)
pprint(result.take(10))

edges_rdd = (sc.textFile("enwiki-2013.txt")
    .filter(lambda s: s[0] != '#')
    .map(lambda s: tuple(map(int, s.split())))
    .cache()
)

m = edges_rdd.count()
n = max(titles_rdd.count(), edges_rdd.map(max).max() + 1)
print("Loaded graph: %i vertices, %i directed edges" % (n, m))

# Given the outdegrees rdd, returns the list of dead-end nodes, 
# and a modified out-degree vector where their degree is 1.
def get_sinks(outdeg_rdd):
    # Collect the out degrees into the driver program's memory
    outdeg = np.zeros(n)
    for (key, val) in outdeg_rdd.collect():
        outdeg[key] = val

    sinks = np.where(outdeg == 0)[0]
    print("%i dead-end nodes" % len(sinks))
    outdeg[sinks] = 1                            # avoid division by zero
    return sinks, outdeg

def compute_outdegrees_rdd(edges_rdd):
    return edges_rdd.map(lambda x: (x[0], 1)).reduceByKey(lambda a, b: a + b)

def pagerank(edges_rdd, n, damping, teleport=None, tol=1e-5, max_iters=10):
    # Pagerank vector, initially uniform
    pr = np.full(n, 1.0 / n)
    
    outdeg_rdd_calc = compute_outdegrees_rdd(edges_rdd)
    sinks, outdeg = get_sinks(outdeg_rdd_calc)

    # Teleport vector, uniform if not provided
    if teleport is None:
        teleport = np.ones(n)

    # while not (termination condition):
    for i in range(max_iters):
        pr_nowhere = np.sum(pr[sinks])
        pr_teleport = (1.0 - damping) + damping * pr_nowhere

        # Compute probabilities without teleportation at the next step
        pr_divided = sc.broadcast(pr / outdeg)
        step = (edges_rdd
            .map(lambda x: (x[1], pr_divided.value[x[0]]))
            .reduceByKey(lambda a, b: a + b)
        )

        # Now account for teleportation
        pr = np.full(n, pr_teleport / n)
        for key, val in step.collect():
            pr[key] += damping * val
        
        print(f"Iteracion {i+1} completada")
                
    return pr

pr = pagerank(edges_rdd, n, damping=0.8)

def show_topk(pr, titles_rdd, topk=20):
    print("Top %i nodes in order of decreasing pagerank:" % topk)
    top = np.argsort(-pr)[:topk]
    f = titles_rdd.filter(lambda tup:tup[0] in top).cache()
    top_titles_map = dict(f.collect())
    pprint([ (pr[x], top_titles_map.get(x, "Unknown")) for x in top ])
        
show_topk(pr, titles_rdd)
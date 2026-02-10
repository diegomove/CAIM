from pprint import pprint
import nltk
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords') 
from nltk.tokenize import word_tokenize

import requests
url = "https://fegalaz.usc.es/~gamallo/aulas/lingcomputacional/corpus/quijote-es.txt"
response = requests.get(url)
response.encoding = "utf-8"
text = response.text

# Tokenización
tokenized_text = word_tokenize(text)

from nltk.corpus import stopwords
import string

spanish_sw = set(stopwords.words('spanish') + list(string.punctuation))
filtered_tokenized_text = [w.lower() for w in tokenized_text if w.lower() not in spanish_sw]

from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('spanish')
stemmed_text = [stemmer.stem(w) for w in filtered_tokenized_text]

from collections import Counter
import matplotlib.pyplot as plt

word_counts = Counter(filtered_tokenized_text)

# --- ORDENAR POR FRECUENCIA ---
sorted_counts = word_counts.most_common()
frequencies = [f for _, f in sorted_counts]

# --- GRAFICAR EN ESCALA LOG-LOG ---
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(frequencies)+1), frequencies, marker='', linestyle='-', markersize=2)

plt.xscale("log")
plt.yscale("log")

plt.title("Ley de Zipf en el Quijote (escala log-log)")
plt.xlabel("Rango de la palabra (log)")
plt.ylabel("Frecuencia de la palabra (log)")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()

plt.savefig("grafica_zipf.png", dpi=300)
plt.close()

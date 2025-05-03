#importamos las librerias
print("-" * 70)
import nltk
import string
import pandas as pd
from nltk import word_tokenize
from nltk import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

# Declaramos las funciones que necesitamos
def get_wordnet_pos(word): 
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV
    }
    return tag_dict.get(tag, wordnet.NOUN)

def remove_stopwords(text):
    englishSW = stopwords.words("english")
    clean_text = [w.lower() for w in text if w.lower() not in englishSW and w not in string.punctuation and w not in ["'s", "''", "--", ".-", "``", "|"]]
    return clean_text

def lemmatized(text):
    lemmatizer = WordNetLemmatizer()
    lemmatized_text = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in text]
    return lemmatized_text

# Tenemos el corpus. Dentro, lo tokenizamos, le quitamos las stopwords y lo lematizamos. 
corpus = [
lemmatized(remove_stopwords(word_tokenize("Python is an interpreted and high-level language, while CPlus is a compiled and low-level language .-"))),
lemmatized(remove_stopwords(word_tokenize("JavaScript runs in web browsers, while Python is used in various applications, including data science and artificial intelligence."))),
lemmatized(remove_stopwords(word_tokenize("JavaScript is dynamically and weakly typed, while Rust is statically typed and ensures greater data security .-"))),
lemmatized(remove_stopwords(word_tokenize("Python and JavaScript are interpreted languages, while Java, CPlus, and Rust require compilation before execution."))),
lemmatized(remove_stopwords(word_tokenize("JavaScript is widely used in web development, while Go is ideal for servers and cloud applications."))),
lemmatized(remove_stopwords(word_tokenize("Python is slower than CPlus and Rust due to its interpreted nature."))),
lemmatized(remove_stopwords(word_tokenize("JavaScript has a strong ecosystem with Node.js for backend development, while Python is widely used in data science .-"))),
lemmatized(remove_stopwords(word_tokenize("JavaScript does not require compilation, while CPlus and Rust require code compilation before execution .-"))),
lemmatized(remove_stopwords(word_tokenize("Python and JavaScript have large communities and an extensive number of available libraries."))),
lemmatized(remove_stopwords(word_tokenize("Python is ideal for beginners, while Rust and CPlus are more suitable for experienced programmers.")))
]

# Declaramos las variables donde vamos a guardar el texto limpio para vectorizarlo (final_corpus), y donde vamos a plotearlo (text).
final_corpus = []
text = ""

# Convertimos a cadena de texto las oraciones.
for sentence in corpus: 
    result = ' '.join(sentence)
    text += ' ' + ' '.join(sentence) 
    final_corpus.append(result)

print("Corpus preparado: ")
print(text)

print("-" * 70)

# Comenzamos con la vectorizacion.
vertorizer = TfidfVectorizer()

matrix = vertorizer.fit_transform(final_corpus)
words = vertorizer.get_feature_names_out()
df_tfidf = pd.DataFrame(matrix.toarray(), columns = words)
print("Matriz TF-IDF:")
print(df_tfidf.round(4))

print("-" * 70)

# Frequency distribution
tokenized_text = word_tokenize(text)
frequency = FreqDist(tokenized_text)

print("Las 6 palabras mas comunes: ")
for word, freq in frequency.most_common(20):
    rel_frequency = frequency.freq(word)
    print(f"{freq}\t{rel_frequency:.6f}\t{word}")

print("-" * 70)

# Buscamos la palabra menos usada del corpus
print("la palabra menos usada del corpus: ")

print(frequency.most_common()[-1])

print("-" * 70)

# Palabras mas repetidas en cada oracion
print("Las palabras mas repetidas en cada oracion:")
for sentence in final_corpus: 
    words = sentence.split()
    word_counts = Counter(words)
    most_common = word_counts.most_common(3)
    print(f"Oracion: {sentence}")
    print(f"Palabras mas repetidas: {most_common}")

print("-" * 70)

print("Realizado por: Núñez, Mauro Nicolás.")

print("-" * 70)

# Plot mostrando los datos
frequency.plot(20, show = True)
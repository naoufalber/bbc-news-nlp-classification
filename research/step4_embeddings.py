import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────
# 1. Chargement
# ──────────────────────────────────────────
df = pd.read_csv("outputs/results/data_clean.csv")
X_clean = df['text_clean'].fillna("").tolist()
y = df['label'].tolist()

# Tokenisation pour Word2Vec (liste de listes)
tokenized = [text.split() for text in X_clean]

# ──────────────────────────────────────────
# 2. Word2Vec
# ──────────────────────────────────────────
print("Entraînement Word2Vec...")
w2v_model = Word2Vec(
    sentences=tokenized,
    vector_size=100,   # dimension des vecteurs
    window=5,          # contexte = 5 mots autour
    min_count=2,       # ignore les mots rares
    workers=4,
    epochs=10
)

def doc_to_vec_w2v(tokens, model):
    vecs = [model.wv[w] for w in tokens if w in model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(model.vector_size)

X_w2v = np.array([doc_to_vec_w2v(tokens, w2v_model) for tokens in tokenized])
w2v_model.save("models/word2vec.model")
print(f"  Word2Vec terminé — shape : {X_w2v.shape}")

# ──────────────────────────────────────────
# 3. BERT (sentence-transformers)
# ──────────────────────────────────────────
print("\nEntraînement BERT embeddings...")
print("  (téléchargement du modèle si première fois, ~90MB)")

bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# BERT sur texte original (pas nettoyé) pour garder le contexte
X_raw = df['text'].fillna("").tolist()
X_bert = bert_model.encode(X_raw, batch_size=64, show_progress_bar=True)
print(f"  BERT terminé — shape : {X_bert.shape}")

# ──────────────────────────────────────────
# 4. Sauvegarde
# ──────────────────────────────────────────
np.save("outputs/results/X_w2v.npy", X_w2v)
np.save("outputs/results/X_bert.npy", X_bert)
np.save("outputs/results/y.npy", np.array(y))
print("\n  [Embeddings sauvegardés → outputs/results/]")

# ──────────────────────────────────────────
# 5. Résumé des dimensions
# ──────────────────────────────────────────
print("\n" + "=" * 50)
print("RÉSUMÉ DES EMBEDDINGS")
print("=" * 50)
print(f"  TF-IDF   : 2127 × 10000  (sparse, bag of words pondéré)")
print(f"  Word2Vec : {X_w2v.shape[0]} × {X_w2v.shape[1]}    (dense, sémantique locale)")
print(f"  BERT     : {X_bert.shape[0]} × {X_bert.shape[1]}   (dense, sémantique contextuelle)")

# ──────────────────────────────────────────
# 6. Test sémantique Word2Vec
# ──────────────────────────────────────────
print("\n" + "=" * 50)
print("TEST SÉMANTIQUE WORD2VEC")
print("=" * 50)
test_words = ['football', 'government', 'technology', 'film', 'bank']
for word in test_words:
    if word in w2v_model.wv:
        similar = w2v_model.wv.most_similar(word, topn=4)
        similar_str = " | ".join([f"{w} ({s:.2f})" for w, s in similar])
        print(f"  {word:<15} → {similar_str}")
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# ──────────────────────────────────────────
# 1. Chargement des données nettoyées
# ──────────────────────────────────────────
df = pd.read_csv("outputs/results/data_clean.csv")
X_clean = df['text_clean'].fillna("").tolist()
y = df['label'].tolist()

# ──────────────────────────────────────────
# 2. Bag of Words (CountVectorizer)
# ──────────────────────────────────────────
bow_vectorizer = CountVectorizer(max_features=10000)
X_bow = bow_vectorizer.fit_transform(X_clean)

# ──────────────────────────────────────────
# 3. TF-IDF
# ──────────────────────────────────────────
tfidf_vectorizer = TfidfVectorizer(max_features=10000)
X_tfidf = tfidf_vectorizer.fit_transform(X_clean)

# ──────────────────────────────────────────
# 4. Affichage des shapes
# ──────────────────────────────────────────
print("=" * 50)
print("DIMENSIONS DES MATRICES")
print("=" * 50)
print(f"  Bag of Words  : {X_bow.shape[0]} docs × {X_bow.shape[1]} features")
print(f"  TF-IDF        : {X_tfidf.shape[0]} docs × {X_tfidf.shape[1]} features")

# ──────────────────────────────────────────
# 5. Exemples de features
# ──────────────────────────────────────────
bow_features   = bow_vectorizer.get_feature_names_out()
tfidf_features = tfidf_vectorizer.get_feature_names_out()

print("\n" + "=" * 50)
print("EXEMPLES DE FEATURES (premiers et derniers)")
print("=" * 50)
print(f"  BoW   — premiers : {list(bow_features[:8])}")
print(f"  BoW   — derniers : {list(bow_features[-8:])}")
print(f"  TF-IDF — premiers : {list(tfidf_features[:8])}")
print(f"  TF-IDF — derniers : {list(tfidf_features[-8:])}")

# ──────────────────────────────────────────
# 6. Top 10 mots par classe (TF-IDF)
# ──────────────────────────────────────────
import numpy as np

df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_features)
df_tfidf['label'] = y

print("\n" + "=" * 50)
print("TOP 10 MOTS PAR CLASSE (TF-IDF moyen)")
print("=" * 50)
for label in sorted(df['label'].unique()):
    top = df_tfidf[df_tfidf['label'] == label].drop('label', axis=1).mean().nlargest(10)
    print(f"\n  [{label.upper()}]")
    print("  " + " | ".join(top.index.tolist()))

# ──────────────────────────────────────────
# 7. Densité des matrices
# ──────────────────────────────────────────
bow_density   = X_bow.nnz / (X_bow.shape[0] * X_bow.shape[1]) * 100
tfidf_density = X_tfidf.nnz / (X_tfidf.shape[0] * X_tfidf.shape[1]) * 100

print("\n" + "=" * 50)
print("DENSITÉ DES MATRICES (% valeurs non nulles)")
print("=" * 50)
print(f"  Bag of Words : {bow_density:.2f}%")
print(f"  TF-IDF       : {tfidf_density:.2f}%")
print("\n  [Matrices creuses → normal pour du texte]")
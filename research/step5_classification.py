import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)

# ──────────────────────────────────────────
# 1. Chargement
# ──────────────────────────────────────────
df      = pd.read_csv("outputs/results/data_clean.csv")
X_clean = df['text_clean'].fillna("").tolist()
y       = np.load("outputs/results/y.npy", allow_pickle=True)
X_w2v   = np.load("outputs/results/X_w2v.npy")
X_bert  = np.load("outputs/results/X_bert.npy")

# TF-IDF unigrammes
tfidf_uni = TfidfVectorizer(max_features=10000, ngram_range=(1,1))
X_tfidf   = tfidf_uni.fit_transform(X_clean)

# TF-IDF bigrammes
tfidf_bi  = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X_tfidf_bi = tfidf_bi.fit_transform(X_clean)

labels = sorted(set(y))

# ──────────────────────────────────────────
# 2. Fonction d'évaluation
# ──────────────────────────────────────────
def evaluate(X, y, model, embedding_name, model_name):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "Embedding" : embedding_name,
        "Modèle"    : model_name,
        "Accuracy"  : round(accuracy_score(y_test, y_pred) * 100, 2),
        "F1-score"  : round(f1_score(y_test, y_pred, average='weighted') * 100, 2),
        "y_test"    : y_test,
        "y_pred"    : y_pred
    }

# ──────────────────────────────────────────
# 3. Entraînement — toutes combinaisons
# ──────────────────────────────────────────
print("Entraînement en cours...")

configs = [
    (X_tfidf,    "TF-IDF (1,1)",  LinearSVC(max_iter=2000, dual=True),        "SVM"),
    (X_tfidf,    "TF-IDF (1,1)",  KNeighborsClassifier(n_neighbors=5),        "KNN"),
    (X_tfidf_bi, "TF-IDF (1,2)",  LinearSVC(max_iter=2000, dual=True),        "SVM"),
    (X_tfidf_bi, "TF-IDF (1,2)",  KNeighborsClassifier(n_neighbors=5),        "KNN"),
    (X_w2v,      "Word2Vec",      LinearSVC(max_iter=2000, dual=True),        "SVM"),
    (X_w2v,      "Word2Vec",      KNeighborsClassifier(n_neighbors=5),        "KNN"),
    (X_bert,     "BERT",          LinearSVC(max_iter=2000, dual=True),        "SVM"),
    (X_bert,     "BERT",          KNeighborsClassifier(n_neighbors=5),        "KNN"),
]

results = []
for X, emb, model, mod_name in configs:
    res = evaluate(X, y, model, emb, mod_name)
    results.append(res)
    print(f"  {emb:<15} + {mod_name:<5} → "
          f"Accuracy: {res['Accuracy']}%  F1: {res['F1-score']}%")

# ──────────────────────────────────────────
# 4. Tableau comparatif
# ──────────────────────────────────────────
df_res = pd.DataFrame([{k: v for k, v in r.items()
                         if k not in ['y_test','y_pred']} for r in results])

print("\n" + "=" * 58)
print("TABLEAU COMPARATIF COMPLET")
print("=" * 58)
print(df_res.to_string(index=False))
df_res.to_csv("outputs/results/metrics_comparison.csv", index=False)

# ──────────────────────────────────────────
# 5. Meilleur modèle → rapport détaillé
# ──────────────────────────────────────────
best = max(results, key=lambda r: r['Accuracy'])
print(f"\n  Meilleur modèle : {best['Embedding']} + {best['Modèle']} "
      f"→ {best['Accuracy']}%")

print("\n" + "=" * 58)
print(f"RAPPORT DÉTAILLÉ — {best['Embedding']} + {best['Modèle']}")
print("=" * 58)
print(classification_report(best['y_test'], best['y_pred'],
                             target_names=labels))

# ──────────────────────────────────────────
# 6. Matrices de confusion — meilleur vs BERT SVM
# ──────────────────────────────────────────
bert_svm = next(r for r in results
                if r['Embedding'] == 'BERT' and r['Modèle'] == 'SVM')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, res, cmap in zip(axes,
                          [best, bert_svm],
                          ['Blues', 'Greens']):
    cm = confusion_matrix(res['y_test'], res['y_pred'], labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f"{res['Embedding']} + {res['Modèle']}\n"
                 f"Accuracy: {res['Accuracy']}%", fontweight='bold')
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")

plt.suptitle("Matrices de confusion", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/figures/confusion_matrices.png", dpi=150)
plt.show()
print("\n  [Matrices sauvegardées → outputs/figures/confusion_matrices.png]")
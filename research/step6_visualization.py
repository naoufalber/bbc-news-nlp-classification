import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# ──────────────────────────────────────────
# 1. Chargement
# ──────────────────────────────────────────
df      = pd.read_csv("outputs/results/data_clean.csv")
X_clean = df['text_clean'].fillna("").tolist()
y       = np.load("outputs/results/y.npy", allow_pickle=True)
X_w2v   = np.load("outputs/results/X_w2v.npy")
X_bert  = np.load("outputs/results/X_bert.npy")

tfidf = TfidfVectorizer(max_features=10000)
X_tfidf = tfidf.fit_transform(X_clean).toarray()

classes = sorted(set(y))
palette = sns.color_palette("Set2", len(classes))
color_map = {cls: palette[i] for i, cls in enumerate(classes)}
colors = [color_map[label] for label in y]

# ──────────────────────────────────────────
# 2. Réduction t-SNE sur les 3 embeddings
# ──────────────────────────────────────────
print("Calcul t-SNE (peut prendre 1-2 min)...")

def compute_tsne(X, perplexity=30):
    # PCA d'abord pour accélérer t-SNE si haute dimension
    if X.shape[1] > 50:
        pca = PCA(n_components=50)
        X = pca.fit_transform(X)
    return TSNE(n_components=2, perplexity=perplexity,
                random_state=42, n_iter=1000).fit_transform(X)

tsne_tfidf = compute_tsne(X_tfidf)
print("  TF-IDF t-SNE : OK")
tsne_w2v = compute_tsne(X_w2v)
print("  Word2Vec t-SNE : OK")
tsne_bert = compute_tsne(X_bert)
print("  BERT t-SNE : OK")

# ──────────────────────────────────────────
# 3. Plot t-SNE — 3 embeddings
# ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

datasets = [
    (tsne_tfidf, "TF-IDF"),
    (tsne_w2v,   "Word2Vec"),
    (tsne_bert,  "BERT"),
]

for ax, (tsne_data, title) in zip(axes, datasets):
    for cls in classes:
        idx = [i for i, l in enumerate(y) if l == cls]
        ax.scatter(tsne_data[idx, 0], tsne_data[idx, 1],
                   c=[color_map[cls]], label=cls,
                   alpha=0.6, s=15, edgecolors='none')
    ax.set_title(f"t-SNE — {title}", fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, markerscale=2)
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle("Visualisation t-SNE des embeddings", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/figures/tsne_comparison.png", dpi=150)
plt.show()
print("\n  [t-SNE sauvegardé → outputs/figures/tsne_comparison.png]")

# ──────────────────────────────────────────
# 4. Graphique comparatif des performances
# ──────────────────────────────────────────
df_metrics = pd.read_csv("outputs/results/metrics_comparison.csv")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Accuracy
pivot_acc = df_metrics.pivot(index='Embedding', columns='Modèle', values='Accuracy')
pivot_acc.plot(kind='bar', ax=axes[0], color=['steelblue', 'coral'],
               edgecolor='white', width=0.6)
axes[0].set_title("Accuracy par embedding et modèle", fontweight='bold')
axes[0].set_ylabel("Accuracy (%)")
axes[0].set_ylim(88, 100)
axes[0].set_xticklabels(pivot_acc.index, rotation=0)
axes[0].legend(title="Modèle")
for container in axes[0].containers:
    axes[0].bar_label(container, fmt='%.1f', fontsize=8, padding=2)

# F1-score
pivot_f1 = df_metrics.pivot(index='Embedding', columns='Modèle', values='F1-score')
pivot_f1.plot(kind='bar', ax=axes[1], color=['steelblue', 'coral'],
              edgecolor='white', width=0.6)
axes[1].set_title("F1-score par embedding et modèle", fontweight='bold')
axes[1].set_ylabel("F1-score (%)")
axes[1].set_ylim(88, 100)
axes[1].set_xticklabels(pivot_f1.index, rotation=0)
axes[1].legend(title="Modèle")
for container in axes[1].containers:
    axes[1].bar_label(container, fmt='%.1f', fontsize=8, padding=2)

plt.suptitle("Comparaison des méthodes", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/figures/performance_comparison.png", dpi=150)
plt.show()
print("  [Comparaison sauvegardée → outputs/figures/performance_comparison.png]")

# ──────────────────────────────────────────
# 5. Synthèse texte
# ──────────────────────────────────────────
print("\n" + "=" * 55)
print("SYNTHÈSE — QUELLE MÉTHODE EST LA MEILLEURE ?")
print("=" * 55)
best = df_metrics.loc[df_metrics['Accuracy'].idxmax()]
print(f"\n  Meilleure combinaison : {best['Embedding']} + {best['Modèle']}")
print(f"  Accuracy : {best['Accuracy']}%  |  F1 : {best['F1-score']}%")
print("""
  Pourquoi TF-IDF + SVM domine :
  - BBC est un corpus thématiquement très distinct
  - Chaque classe possède un vocabulaire propre fort
  - TF-IDF capture ces mots discriminants efficacement
  - SVM trouve des hyperplans optimaux dans cet espace sparse
  - BERT est plus puissant en général mais moins nécessaire
    quand les classes sont aussi bien séparées lexicalement
""")
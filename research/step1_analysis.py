import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# ──────────────────────────────────────────
# 1. Chargement des données
# ──────────────────────────────────────────
DATA_PATH = "data/bbc"

texts = []
labels = []

for category in os.listdir(DATA_PATH):
    category_path = os.path.join(DATA_PATH, category)
    if not os.path.isdir(category_path):
        continue
    for filename in os.listdir(category_path):
        filepath = os.path.join(category_path, filename)
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
            labels.append(category)

X = texts
y = labels

# ──────────────────────────────────────────
# 2. Infos générales
# ──────────────────────────────────────────
df = pd.DataFrame({"text": X, "label": y})

print("=" * 50)
print("INFOS GÉNÉRALES")
print("=" * 50)
print(f"Nombre de documents   : {len(df)}")
print(f"Nombre de classes     : {df['label'].nunique()}")
print(f"Classes               : {sorted(df['label'].unique())}")
print(f"Doublons              : {df.duplicated().sum()}")
print(f"Valeurs manquantes    : {df['text'].isna().sum()}")

# ──────────────────────────────────────────
# 3. Distribution des classes
# ──────────────────────────────────────────
print("\n" + "=" * 50)
print("DISTRIBUTION DES CLASSES")
print("=" * 50)
dist = df['label'].value_counts()
for label, count in dist.items():
    pct = count / len(df) * 100
    bar = "█" * int(pct / 2)
    print(f"  {label:<15} {count:>4} docs  ({pct:.1f}%)  {bar}")

# ──────────────────────────────────────────
# 4. Stats sur la longueur des textes
# ──────────────────────────────────────────
df['num_words'] = df['text'].apply(lambda x: len(x.split()))

print("\n" + "=" * 50)
print("LONGUEUR DES TEXTES (en mots)")
print("=" * 50)
stats = df.groupby('label')['num_words'].agg(['mean', 'min', 'max'])
stats.columns = ['Moyenne', 'Min', 'Max']
stats['Moyenne'] = stats['Moyenne'].round(0).astype(int)
print(stats.to_string())
print(f"\n  Global — Moyenne : {df['num_words'].mean():.0f} mots | "
      f"Min : {df['num_words'].min()} | Max : {df['num_words'].max()}")

# ──────────────────────────────────────────
# 5. Exemples de textes
# ──────────────────────────────────────────
print("\n" + "=" * 50)
print("EXEMPLES DE TEXTES (100 premiers caractères)")
print("=" * 50)
for label in sorted(df['label'].unique()):
    sample = df[df['label'] == label]['text'].iloc[0]
    print(f"\n  [{label.upper()}]")
    print(f"  {sample[:150].strip()}...")

# ──────────────────────────────────────────
# 6. Graphique distribution
# ──────────────────────────────────────────
plt.figure(figsize=(8, 5))
colors = sns.color_palette("Set2", len(dist))
bars = plt.bar(dist.index, dist.values, color=colors, edgecolor='white', linewidth=1.2)

for bar, val in zip(bars, dist.values):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 5,
             str(val), ha='center', fontsize=11, fontweight='bold')

plt.title("Distribution des classes — BBC Dataset", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Catégorie", fontsize=12)
plt.ylabel("Nombre de documents", fontsize=12)
plt.ylim(0, max(dist.values) + 60)
plt.tight_layout()
plt.savefig("outputs/figures/class_distribution.png", dpi=150)
plt.show()
print("\n  [Graphique sauvegardé → outputs/figures/class_distribution.png]")
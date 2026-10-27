import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)

# ──────────────────────────────────────────
# 1. Rechargement des données
# ──────────────────────────────────────────
DATA_PATH = "data/bbc"
texts, labels = [], []

for category in os.listdir(DATA_PATH):
    category_path = os.path.join(DATA_PATH, category)
    if not os.path.isdir(category_path):
        continue
    for filename in os.listdir(category_path):
        with open(os.path.join(category_path, filename),
                  "r", encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
            labels.append(category)

df = pd.DataFrame({"text": texts, "label": labels})

before = len(df)
df = df.drop_duplicates(subset="text").reset_index(drop=True)
print(f"Doublons supprimés : {before - len(df)} | Documents restants : {len(df)}")

# ──────────────────────────────────────────
# 2. Outils NLP
# ──────────────────────────────────────────
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer    = PorterStemmer()

# ──────────────────────────────────────────
# 3. Fonctions de preprocessing
# ──────────────────────────────────────────
def clean_base(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def preprocess_lemma(text):
    text   = clean_base(text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

def preprocess_stem(text):
    text   = clean_base(text)
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(t) for t in tokens
              if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

# ──────────────────────────────────────────
# 4. Application sur tout le dataset
# ──────────────────────────────────────────
print("Preprocessing en cours...")
df['text_clean'] = df['text'].apply(preprocess_lemma)
df['text_stem']  = df['text'].apply(preprocess_stem)
print("  Done.")

# ──────────────────────────────────────────
# 5. Avant / Après — Lemmatisation
# ──────────────────────────────────────────
print("\n" + "=" * 58)
print("EXEMPLE AVANT / APRÈS — LEMMATISATION")
print("=" * 58)
for label in sorted(df['label'].unique()):
    sample = df[df['label'] == label].iloc[0]
    print(f"\n  [{label.upper()}]")
    print(f"  AVANT : {sample['text'][:150].strip()}...")
    print(f"  APRÈS : {sample['text_clean'][:150].strip()}...")

# ──────────────────────────────────────────
# 6. Lemmatisation vs Stemming
# ──────────────────────────────────────────
print("\n" + "=" * 58)
print("LEMMATISATION vs STEMMING — comparaison sur 3 exemples")
print("=" * 58)

test_words = ["running", "studies", "better", "companies", "playing"]
print(f"\n  {'Mot original':<15} {'Lemmatisé':<15} {'Stemmé':<15}")
print(f"  {'-'*45}")
for w in test_words:
    print(f"  {w:<15} {lemmatizer.lemmatize(w):<15} {stemmer.stem(w):<15}")

print("""
  Conclusion :
  - Lemmatisation → forme de base lisible ("studies" → "study")
  - Stemming      → troncature agressive ("studies" → "studi")
  - On garde la Lemmatisation pour le pipeline principal
    car elle préserve le sens et la lisibilité des tokens.
""")

# ──────────────────────────────────────────
# 7. Stats après preprocessing
# ──────────────────────────────────────────
df['num_words_raw']   = df['text'].apply(lambda x: len(x.split()))
df['num_words_clean'] = df['text_clean'].apply(lambda x: len(x.split()))
df['num_words_stem']  = df['text_stem'].apply(lambda x: len(x.split()))

print("=" * 58)
print("STATS — LONGUEUR EN MOTS PAR CLASSE (après lemmatisation)")
print("=" * 58)
stats = df.groupby('label')['num_words_clean'].agg(['mean','min','max'])
stats.columns = ['Moyenne', 'Min', 'Max']
stats['Moyenne'] = stats['Moyenne'].round(0).astype(int)
print(stats.to_string())
print(f"\n  Global — Moyenne : {df['num_words_clean'].mean():.0f} mots | "
      f"Min : {df['num_words_clean'].min()} | "
      f"Max : {df['num_words_clean'].max()}")

print(f"\n  Réduction moyenne brut → lemmatisé : "
      f"{df['num_words_raw'].mean():.0f} → {df['num_words_clean'].mean():.0f} mots "
      f"({(1 - df['num_words_clean'].mean()/df['num_words_raw'].mean())*100:.1f}% de réduction)")

# ──────────────────────────────────────────
# 8. Visualisation — distribution longueurs
# ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

for label in sorted(df['label'].unique()):
    subset = df[df['label'] == label]
    axes[0].hist(subset['num_words_raw'],   bins=30, alpha=0.5, label=label)
    axes[1].hist(subset['num_words_clean'], bins=30, alpha=0.5, label=label)

axes[0].set_title("Distribution longueurs — BRUT",        fontweight='bold')
axes[1].set_title("Distribution longueurs — LEMMATISÉ",   fontweight='bold')
for ax in axes:
    ax.set_xlabel("Nombre de mots")
    ax.set_ylabel("Fréquence")
    ax.legend(fontsize=8)

plt.suptitle("Impact du preprocessing sur la longueur des textes",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/figures/preprocessing_distribution.png", dpi=150)
plt.show()
print("  [Graphique sauvegardé → outputs/figures/preprocessing_distribution.png]")

# ──────────────────────────────────────────
# 9. Sauvegarde
# ──────────────────────────────────────────
df.to_csv("outputs/results/data_clean.csv", index=False)
print("\n  [Données sauvegardées → outputs/results/data_clean.csv]")

X_clean = df['text_clean'].tolist()
y       = df['label'].tolist()
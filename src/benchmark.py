"""BBC topic classification with a shared holdout and training-only feature fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = {"business", "entertainment", "politics", "sport", "tech"}


def load_corpus(folder):
    rows = []
    for category in sorted(CATEGORIES):
        for path in sorted((folder / category).glob("*.txt")):
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                rows.append({"text": text, "label": category,
                             "id": hashlib.sha256(text.encode()).hexdigest()})
    if not rows:
        raise ValueError("No articles found: expected data/bbc/<category>/*.txt")
    frame = pd.DataFrame(rows)
    if (frame.groupby("id").label.nunique() > 1).any():
        raise ValueError("Identical text has conflicting labels; review the dataset")
    return frame.drop_duplicates("id").reset_index(drop=True), len(rows)


def tokenize(text):
    return re.findall(r"[a-z]{3,}", text.lower())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/bbc")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/current")
    parser.add_argument("--embeddings", action="store_true", help="Also run Word2Vec and pretrained MiniLM")
    args = parser.parse_args()
    frame, raw_count = load_corpus(args.data)
    train, test = train_test_split(frame, test_size=.2, stratify=frame.label, random_state=42)
    args.output.mkdir(parents=True, exist_ok=True)
    representations = []
    for ngram in [(1, 1), (1, 2)]:
        vectorizer = TfidfVectorizer(max_features=10000, stop_words="english", ngram_range=ngram)
        representations.append((f"TF-IDF {ngram}", vectorizer.fit_transform(train.text), vectorizer.transform(test.text)))
    if args.embeddings:
        from gensim.models import Word2Vec
        from sentence_transformers import SentenceTransformer
        train_tokens = [tokenize(text) for text in train.text]
        model = Word2Vec(train_tokens, vector_size=100, window=5, min_count=2, workers=1, seed=42, epochs=10)
        def embed(texts):
            vectors = []
            for text in texts:
                words = [model.wv[w] for w in tokenize(text) if w in model.wv]
                vectors.append(np.mean(words, axis=0) if words else np.zeros(model.vector_size))
            return np.array(vectors)
        representations.append(("Word2Vec", embed(train.text), embed(test.text)))
        encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        representations.append(("MiniLM", encoder.encode(train.text.tolist(), show_progress_bar=True),
                                 encoder.encode(test.text.tolist(), show_progress_bar=True)))
    rows = []
    for representation, X_train, X_test in representations:
        for name, estimator in [("SVM", LinearSVC(max_iter=5000, random_state=42)),
                                ("KNN", KNeighborsClassifier(n_neighbors=5))]:
            estimator.fit(X_train, train.label)
            prediction = estimator.predict(X_test)
            rows.append({"representation": representation, "classifier": name,
                         "accuracy": accuracy_score(test.label, prediction),
                         "f1_macro": f1_score(test.label, prediction, average="macro"),
                         "f1_weighted": f1_score(test.label, prediction, average="weighted")})
            # Pre-specified primary baseline; do not choose a winner on the test set.
            if representation == "TF-IDF (1, 1)" and name == "SVM":
                fig, ax = plt.subplots(figsize=(8, 6))
                ConfusionMatrixDisplay.from_predictions(test.label, prediction, ax=ax, cmap="Blues", colorbar=False, xticks_rotation=30)
                ax.set_title("TF-IDF unigram + SVM | held-out articles")
                fig.tight_layout()
                fig.savefig(args.output / "confusion_matrix.png", dpi=160)
                plt.close(fig)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output / "metrics.csv", index=False)
    manifest = {"seed": 42, "raw_rows": raw_count, "deduplicated_rows": len(frame),
                "train_rows": len(train), "test_rows": len(test), "embeddings": args.embeddings,
                "test_document_hashes": test.id.tolist(), "class_counts": frame.label.value_counts().to_dict()}
    (args.output / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()

# Historical academic pipeline

These six scripts preserve the original exploratory workflow: analysis, preprocessing, vectorization, embeddings, classification and visualization. They expect execution from a directory with `data/bbc/`, `outputs/results/`, `outputs/figures/` and `models/`. They are reference material; the supported command is `python src/benchmark.py` from the repository root.

The historical TF-IDF and Word2Vec representations were fitted before the train/test split. Their saved scores in `../reports/historical_metrics.csv` are therefore exploratory, not the corrected benchmark. The old “BERT” label actually refers to pretrained `all-MiniLM-L6-v2` sentence embeddings; there was no BERT fine-tuning.

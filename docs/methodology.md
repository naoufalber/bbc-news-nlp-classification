# Protocol and reproducibility

Files and categories are sorted for deterministic loading. Exact duplicate texts are removed before a stratified 80/20 split (seed 42). Conflicting labels for identical texts raise an error. Every representation uses the same split.

TF-IDF vocabulary and IDF weights are fitted only on training texts. The portable baseline uses scikit-learn tokenization and English stopwords, unlike the historical NLTK lemmatization workflow. Results consequently differ from the old CSV and should not be treated as an isolated leakage-ablation experiment.

With `--embeddings`, Word2Vec is trained only on training articles, using one worker and seed 42. MiniLM uses a frozen pretrained sentence encoder, without task fine-tuning; long texts are truncated according to the model tokenizer limits. These optional paths were code-reviewed but not run in the preparation environment, which lacks their dependencies and downloaded model.

The unigram SVM is the pre-specified baseline for the confusion matrix. The test set is used for reporting, not hyperparameter tuning. Model changes should use validation folds inside the training partition, then an untouched final evaluation set. The corpus is small, dated and contains related news stories: exact deduplication does not guarantee story-level independence or performance on current news.

## Verified run

2,225 files loaded → 2,127 unique texts → 1,701 train / 426 test. All four TF-IDF combinations ran successfully on the supplied corpus. See `reports/validated/metrics.csv`, `run.json` and `confusion_matrix.png`. Metrics are proportions in CSV and percentages in the README. The 97.89% historical unigram SVM score remains separately labeled in `reports/historical_metrics.csv`.

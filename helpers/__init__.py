"""helpers — Shared evaluation and analysis utilities.

Modules
-------
decile_utils
    Popularity decile assignment, boundary computation, and corpus
    distribution helpers.
metrics
    Retrieval evaluation metrics (Recall@K, MRR, rank) and per-decile
    aggregation helpers (binned_stats, decile_stats).
bias_utils
    Popularity-bias analysis: wrong-document heatmaps, preference curves,
    score-to-distance transforms.
tfidf_service
    TF-IDF corpus statistics, vectorizer caching, and associated plots.
"""

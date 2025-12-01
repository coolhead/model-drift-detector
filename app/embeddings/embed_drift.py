import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model

def compute_embedding_drift(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    text_col: str,
    sample_size: int = 500,
) -> float:
    model = get_model()

    base_texts = baseline_df[text_col].dropna().astype(str).sample(
        min(sample_size, len(baseline_df)), random_state=42
    )
    curr_texts = current_df[text_col].dropna().astype(str).sample(
        min(sample_size, len(current_df)), random_state=42
    )

    base_emb = model.encode(base_texts.tolist(), show_progress_bar=False)
    curr_emb = model.encode(curr_texts.tolist(), show_progress_bar=False)

    # pairwise distance between means
    base_mean = base_emb.mean(axis=0, keepdims=True)
    curr_mean = curr_emb.mean(axis=0, keepdims=True)

    dist = cosine_distances(base_mean, curr_mean)[0, 0]
    return float(dist)

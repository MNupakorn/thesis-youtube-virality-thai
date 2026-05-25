"""TF-IDF features over Thai titles, using PyThaiNLP tokenization."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils import setup_logger

log = setup_logger("features.tfidf")


def _pythainlp_tokenize(text: str) -> list[str]:
    """Word-level tokenization with PyThaiNLP; falls back to whitespace if unavailable."""
    try:
        from pythainlp.tokenize import word_tokenize

        return [t for t in word_tokenize(text or "", engine="newmm") if t.strip()]
    except Exception:
        return (text or "").split()


def fit_tfidf(
    train_texts: list[str],
    max_features: int = 20000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 5,
    sublinear_tf: bool = True,
    tokenizer_name: str = "pythainlp",
) -> TfidfVectorizer:
    tok = _pythainlp_tokenize if tokenizer_name == "pythainlp" else None
    vec = TfidfVectorizer(
        tokenizer=tok,
        token_pattern=None if tok else r"(?u)\b\w+\b",
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
        lowercase=False,
    )
    log.info(
        f"fitting TF-IDF: max_features={max_features}, ngram={ngram_range}, "
        f"min_df={min_df}, tokenizer={tokenizer_name}, n_docs={len(train_texts)}"
    )
    vec.fit(train_texts)
    return vec


def transform_tfidf(vec: TfidfVectorizer, texts: list[str]) -> csr_matrix:
    return vec.transform(texts)


def save_vectorizer(vec: TfidfVectorizer, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vec, path)


def load_vectorizer(path: str | Path) -> TfidfVectorizer:
    return joblib.load(path)


def build_tfidf_for_splits(
    df: pd.DataFrame,
    text_col: str = "title",
    split_col: str = "split",
    fit_on: str = "train",
    max_features: int = 20000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 5,
    sublinear_tf: bool = True,
    tokenizer_name: str = "pythainlp",
) -> tuple[TfidfVectorizer, dict[str, csr_matrix]]:
    """Fit on the train split, transform all splits. Returns (vectorizer, matrices_by_split)."""
    fit_texts = df.loc[df[split_col] == fit_on, text_col].fillna("").astype(str).tolist()
    vec = fit_tfidf(
        fit_texts,
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
        tokenizer_name=tokenizer_name,
    )
    matrices: dict[str, csr_matrix] = {}
    for sp in df[split_col].unique():
        texts = df.loc[df[split_col] == sp, text_col].fillna("").astype(str).tolist()
        matrices[sp] = vec.transform(texts)
        log.info(f"TF-IDF {sp}: shape={matrices[sp].shape}, nnz={matrices[sp].nnz}")
    return vec, matrices

from pathlib import Path
import re
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


amount_pattern = re.compile(r"\b\d+[.,]\d{2}\b")
date_pattern = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
digit_pattern = re.compile(r"\d")


def build_features(df):

    out = pd.DataFrame(index=df.index)

    text = df["text"]

    out["n_chars"] = text.str.len()
    out["n_words"] = text.str.split().apply(len)

    out["digit_count"] = text.apply(lambda x: len(digit_pattern.findall(x)))
    out["amount_like_count"] = text.apply(lambda x: len(amount_pattern.findall(x)))
    out["date_like_count"] = text.apply(lambda x: len(date_pattern.findall(x)))

    out["has_total"] = text.str.contains("total", case=False).astype(int)
    out["has_tax"] = text.str.contains("tax", case=False).astype(int)

    return out


def train_detector(train_df, model_dir):

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    tfidf = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1,2),
        min_df=2,
        max_features=3000
    )

    X_text = tfidf.fit_transform(train_df["text"])

    X_manual = build_features(train_df)

    from scipy.sparse import hstack, csr_matrix

    X = hstack([csr_matrix(X_manual.values), X_text])

    y = train_df["is_forged"].astype(int)

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    )

    model.fit(X, y)

    joblib.dump(model, model_dir / "anomaly_model.joblib")
    joblib.dump(tfidf, model_dir / "tfidf.joblib")
    joblib.dump(list(X_manual.columns), model_dir / "manual_cols.joblib")

    return model_dir
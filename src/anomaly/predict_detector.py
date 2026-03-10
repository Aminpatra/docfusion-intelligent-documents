import joblib
import pandas as pd
from pathlib import Path
from scipy.sparse import hstack, csr_matrix

from .train_detector import build_features


def load_detector(model_dir):

    model_dir = Path(model_dir)

    model = joblib.load(model_dir / "anomaly_model.joblib")
    tfidf = joblib.load(model_dir / "tfidf.joblib")
    manual_cols = joblib.load(model_dir / "manual_cols.joblib")

    return model, tfidf, manual_cols


def predict_forgery(df, model, tfidf, manual_cols):

    X_text = tfidf.transform(df["text"])

    X_manual = build_features(df)[manual_cols]

    X = hstack([csr_matrix(X_manual.values), X_text])

    prob = model.predict_proba(X)[:,1]

    return prob
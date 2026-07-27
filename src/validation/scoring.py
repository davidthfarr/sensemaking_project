"""Post-annotation IAA and model performance scoring."""
import numpy as np
import pandas as pd

try:
    from sklearn.metrics import (
        cohen_kappa_score, confusion_matrix,
        classification_report, accuracy_score, f1_score,
    )
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False

LABELS = ["support", "oppose", "neutral"]


def _check_sklearn() -> None:
    if not _SKLEARN_OK:
        raise ImportError("scikit-learn is required: pip install scikit-learn")


def compute_iaa(a_df: pd.DataFrame, b_df: pd.DataFrame) -> dict:
    """
    Inter-annotator agreement between annotators A and B.

    Parameters
    ----------
    a_df, b_df : DataFrame
        Completed annotation files with post_id and stance_label columns.

    Returns
    -------
    dict: overall_kappa, per_class_kappa, agreement_rate, n_items, disagreements DataFrame
    """
    _check_sklearn()
    merged = (
        a_df[["post_id", "stance_label"]]
        .merge(b_df[["post_id", "stance_label"]], on="post_id", suffixes=("_a", "_b"))
        .dropna(subset=["stance_label_a", "stance_label_b"])
    )
    merged = merged[
        merged["stance_label_a"].isin(LABELS) & merged["stance_label_b"].isin(LABELS)
    ]

    overall_kappa = float(cohen_kappa_score(merged["stance_label_a"], merged["stance_label_b"]))
    per_class = {}
    for label in LABELS:
        a_bin = (merged["stance_label_a"] == label).astype(int)
        b_bin = (merged["stance_label_b"] == label).astype(int)
        try:
            per_class[label] = float(cohen_kappa_score(a_bin, b_bin))
        except Exception:
            per_class[label] = np.nan

    disagree = merged[merged["stance_label_a"] != merged["stance_label_b"]]
    return {
        "n_items":        len(merged),
        "overall_kappa":  overall_kappa,
        "per_class_kappa": per_class,
        "agreement_rate": float((merged["stance_label_a"] == merged["stance_label_b"]).mean()),
        "n_disagreements": len(disagree),
        "disagreements":  disagree,
    }


def compute_model_performance(gold_df: pd.DataFrame, key_df: pd.DataFrame) -> dict:
    """
    Evaluate model stance against adjudicated gold labels.

    Parameters
    ----------
    gold_df : DataFrame
        Adjudicated labels with post_id and stance_label (gold standard).
    key_df : DataFrame
        _key.csv with post_id and model_stance.

    Returns
    -------
    dict: accuracy, macro_f1, per_class report, confusion_matrix DataFrames
    """
    _check_sklearn()
    merged = (
        gold_df[["post_id", "stance_label"]]
        .merge(key_df[["post_id", "model_stance"]], on="post_id")
        .dropna()
    )
    merged = merged[
        merged["stance_label"].isin(LABELS) & merged["model_stance"].isin(LABELS)
    ]
    y_true, y_pred = merged["stance_label"], merged["model_stance"]

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    return {
        "n_items":    len(merged),
        "accuracy":   float(accuracy_score(y_true, y_pred)),
        "macro_f1":   float(f1_score(y_true, y_pred, average="macro", labels=LABELS)),
        "per_class":  classification_report(y_true, y_pred, labels=LABELS, output_dict=True),
        "confusion_matrix": pd.DataFrame(cm, index=LABELS, columns=LABELS),
        "confusion_matrix_normalized": pd.DataFrame(
            cm.astype(float) / cm.sum(axis=1, keepdims=True),
            index=LABELS, columns=LABELS,
        ),
    }


def simulate_c_bias(cm_normalized: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate bias in controversy score C induced by model classification errors.

    For each target C value, assumes the maximum-controversy configuration
    (s = o = C/2, neu = 1 - C), applies the row-normalized confusion matrix
    as a label-transition matrix to obtain the observed class distribution,
    then computes C_observed. Returns a DataFrame of (C_true, C_observed, induced_bias).

    A recommended_threshold attribute marks the minimum C where |bias| < 0.05.
    """
    T = cm_normalized.loc[LABELS, LABELS].values  # rows=true, cols=predicted

    rows = []
    for c_true in np.round(np.arange(0.0, 1.02, 0.05), 2):
        neu = max(0.0, 1.0 - float(c_true))
        s   = float(c_true) / 2.0
        o   = float(c_true) / 2.0
        p_true = np.array([s, o, neu])
        p_obs  = p_true @ T
        s_obs, o_obs, neu_obs = p_obs
        c_obs  = max(0.0, float(1.0 - neu_obs - abs(s_obs - o_obs)))
        rows.append({
            "C_true": round(float(c_true), 2),
            "C_observed": round(c_obs, 4),
            "induced_bias": round(c_obs - float(c_true), 4),
        })

    result = pd.DataFrame(rows)
    reliable = result[result["induced_bias"].abs() < 0.05]
    result.attrs["recommended_threshold"] = (
        float(reliable["C_true"].min()) if len(reliable) > 0 else np.nan
    )
    return result

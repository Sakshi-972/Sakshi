"""Train the customer churn model.

Usage:  python train_model.py

Outputs
  model/model.pkl          full sklearn Pipeline (preprocessing + classifier)
  model/metrics.json       metrics, top drivers and the input schema (used by API/UI)
  model/diagnostics.png    ROC curve, confusion matrix, feature importance
  report_images/eda_overview.png   exploratory charts for the report
"""
import json
from datetime import datetime

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import (CATEGORICAL_FEATURES, FEATURES, IMAGE_DIR, MODEL_DIR,
                    NUMERIC_FEATURES, TARGET, load_clean)

RANDOM_STATE = 42
THRESHOLD = 0.5


def build_pipeline(estimator):
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    prep = ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])
    return Pipeline([("prep", prep), ("model", estimator)])


def get_candidates():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1),
    }


def evaluate(pipe, X_test, y_test):
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)
    return proba, pred, {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
    }


def top_features(pipe, n=12):
    names = [c.split("__", 1)[1] for c in pipe.named_steps["prep"].get_feature_names_out()]
    est = pipe.named_steps["model"]
    if hasattr(est, "coef_"):
        values, kind = est.coef_[0], "coefficient (+ raises churn risk, - lowers it)"
    else:
        values, kind = est.feature_importances_, "feature importance"
    order = np.argsort(np.abs(values))[::-1][:n]
    return kind, [{"feature": names[i], "importance": round(float(values[i]), 4)} for i in order]


def save_diagnostics(y_test, proba, cm, feats, kind, model_name):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))

    fpr, tpr, _ = roc_curve(y_test, proba)
    ax[0].plot(fpr, tpr, color="#1f77b4", lw=2, label=f"AUC = {roc_auc_score(y_test, proba):.3f}")
    ax[0].plot([0, 1], [0, 1], "--", color="grey")
    ax[0].set(title=f"ROC Curve - {model_name}", xlabel="False positive rate", ylabel="True positive rate")
    ax[0].legend(loc="lower right")

    ax[1].imshow(cm, cmap="Blues")
    ax[1].set(title="Confusion Matrix", xlabel="Predicted", ylabel="Actual",
              xticks=[0, 1], yticks=[0, 1],
              xticklabels=["Stay", "Churn"], yticklabels=["Stay", "Churn"])
    for (i, j), v in np.ndenumerate(cm):
        ax[1].text(j, i, int(v), ha="center", va="center",
                   color="white" if v > cm.max() / 2 else "black", fontsize=13)

    names = [f["feature"] for f in feats][::-1]
    vals = [f["importance"] for f in feats][::-1]
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in vals] if "coefficient" in kind else "#1f77b4"
    ax[2].barh(names, vals, color=colors)
    ax[2].set(title="Top drivers of churn", xlabel=kind.split(" (")[0])
    ax[2].tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    fig.savefig(MODEL_DIR / "diagnostics.png", dpi=130)
    plt.close(fig)


def save_eda(df):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for col, a in [("Contract", ax[0, 0]), ("InternetService", ax[0, 1])]:
        rate = df.groupby(col)[TARGET].mean().mul(100).sort_values()
        a.bar(rate.index, rate.values, color="#1f77b4")
        a.set(title=f"Churn rate (%) by {col}", ylabel="% churned")
        for i, v in enumerate(rate.values):
            a.text(i, v + 0.5, f"{v:.1f}%", ha="center")

    ax[1, 0].hist([df[df[TARGET] == 0]["tenure"], df[df[TARGET] == 1]["tenure"]],
                  bins=24, label=["Stayed", "Churned"], color=["#2ca02c", "#d62728"])
    ax[1, 0].set(title="Tenure (months) distribution", xlabel="Tenure", ylabel="Customers")
    ax[1, 0].legend()

    ax[1, 1].boxplot([df[df[TARGET] == 0]["MonthlyCharges"], df[df[TARGET] == 1]["MonthlyCharges"]])
    ax[1, 1].set_xticklabels(["Stayed", "Churned"])
    ax[1, 1].set(title="Monthly charges by churn status", ylabel="USD / month")

    plt.tight_layout()
    fig.savefig(IMAGE_DIR / "eda_overview.png", dpi=130)
    plt.close(fig)


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    IMAGE_DIR.mkdir(exist_ok=True)

    df = load_clean()
    print(f"Loaded {len(df)} customers | churn rate = {df[TARGET].mean():.1%}")

    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    results, fitted = {}, {}
    for name, estimator in get_candidates().items():
        pipe = build_pipeline(estimator)
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc").mean()
        pipe.fit(X_train, y_train)
        proba, pred, m = evaluate(pipe, X_test, y_test)
        m["cv_roc_auc"] = round(float(cv_auc), 4)
        results[name], fitted[name] = m, (pipe, proba, pred)
        print(f"{name:20s} {m}")

    best = max(results, key=lambda k: results[k]["roc_auc"])
    pipe, proba, pred = fitted[best]
    print(f"\nBest model: {best}")

    cm = confusion_matrix(y_test, pred)
    kind, feats = top_features(pipe)

    schema = {
        "numeric": {c: {"min": float(df[c].min()), "max": float(df[c].max()),
                        "mean": round(float(df[c].mean()), 2)} for c in NUMERIC_FEATURES},
        "categorical": {c: sorted(df[c].dropna().unique().tolist()) for c in CATEGORICAL_FEATURES},
    }
    metrics = {
        "best_model": best,
        "threshold": THRESHOLD,
        "models": results,
        "confusion_matrix": cm.tolist(),
        "importance_type": kind,
        "top_features": feats,
        "n_samples": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "churn_rate": round(float(y.mean()), 4),
        "schema": schema,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    joblib.dump(pipe, MODEL_DIR / "model.pkl")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_diagnostics(y_test, proba, cm, feats, kind, best)
    save_eda(df)
    print("Saved: model/model.pkl, model/metrics.json, model/diagnostics.png, report_images/eda_overview.png")


if __name__ == "__main__":
    main()

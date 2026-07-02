# python -m app.train --data data.csv --out model.pkl
import argparse
import json
from collections import Counter
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .features import FEATURE_NAMES, extract_features


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}")

    if "url" in df.columns and "status" in df.columns:
        df = df.rename(columns={"url": "URL"})
        df["y"] = df["status"].map({"phishing": 1, "legitimate": 0})
    elif "URL" in df.columns and "label" in df.columns:
        df["y"] = 1 - df["label"].astype(int)
    else:
        raise ValueError(
            "Unrecognized dataset schema. Expected either "
            "('url', 'status') or ('URL', 'label') columns, got: "
            f"{list(df.columns)}"
        )

    df = df.dropna(subset=["URL", "y"]).reset_index(drop=True)
    print(f"Class balance -> phishing: {(df['y'] == 1).sum()}, "
          f"legitimate: {(df['y'] == 0).sum()}")
    return df


def build_lookup_tables(df: pd.DataFrame) -> dict:
    from urllib.parse import urlparse

    tlds = []
    for url, y in zip(df["URL"], df["y"]):
        try:
            domain = urlparse(url if "://" in url else "http://" + url).hostname or ""
        except ValueError:
            domain = ""
        parts = domain.split(".")
        tld = parts[-1] if len(parts) > 1 else ""
        tlds.append((tld, y))

    tld_counts = Counter(t for t, _ in tlds)
    tld_legit_counts = Counter(t for t, y in tlds if y == 0)  # y=0 -> legitimate
    tld_legit_prob = {
        tld: tld_legit_counts.get(tld, 0) / count
        for tld, count in tld_counts.items()
        if count >= 5
    }

    char_counter = Counter()
    total_chars = 0
    for url in df["URL"]:
        url_lower = str(url).lower()
        char_counter.update(url_lower)
        total_chars += len(url_lower)
    char_prob = {c: cnt / total_chars for c, cnt in char_counter.items()}

    print(f"Built TLD table ({len(tld_legit_prob)} TLDs) and "
          f"char-probability table ({len(char_prob)} chars)")
    return {"tld_legit_prob": tld_legit_prob, "char_prob": char_prob}


def build_feature_matrix(df: pd.DataFrame, lookup_tables: dict):
    feature_dicts = [extract_features(url, lookup_tables) for url in df["URL"]]
    X = pd.DataFrame(feature_dicts)[FEATURE_NAMES]
    y = df["y"]
    return X, y


def train_logistic_regression(X_train, y_train):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    lr.fit(X_scaled, y_train)
    return lr, scaler


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    rf_param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [15, 25, None],
        "min_samples_split": [2, 5],
    }
    rf_base = RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1)
    rf_grid = GridSearchCV(rf_base, rf_param_grid, cv=3, scoring="f1", n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    print(f"[RandomForest] best params: {rf_grid.best_params_}")
    return rf_grid.best_estimator_


def train_xgboost(X_train, y_train) -> XGBClassifier:
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    balance_ratio = neg_count / pos_count

    xgb_model = XGBClassifier(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=balance_ratio,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train)
    return xgb_model


def evaluate(model, X_test, y_test, name: str, scaler=None) -> dict:
    X_eval = scaler.transform(X_test) if scaler is not None else X_test
    y_pred = model.predict(X_eval)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"\n----- {name} -----")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=["legitimate", "phishing"]))
    return {"accuracy": acc, "f1": f1}


def main():
    parser = argparse.ArgumentParser(description="Train the phishing URL detector")
    parser.add_argument("--data", default="data.csv", help="Path to training CSV")
    parser.add_argument("--out", default="model.pkl", help="Where to save the winning model")
    parser.add_argument("--metrics-out", default="metrics.json", help="Where to save evaluation metrics")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional: randomly subsample N rows before training (useful for quick local runs)",
    )
    args = parser.parse_args()

    df = load_dataset(args.data)
    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=42).reset_index(drop=True)
        print(f"Subsampled down to {len(df)} rows")

    train_df, test_df = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df["y"]
    )

    lookup_tables = build_lookup_tables(train_df)

    X_train, y_train = build_feature_matrix(train_df, lookup_tables)
    X_test, y_test = build_feature_matrix(test_df, lookup_tables)
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}, "
          f"Features: {X_train.shape[1]}")

    results = {}

    lr_model, lr_scaler = train_logistic_regression(X_train, y_train)
    results["logistic_regression"] = (
        evaluate(lr_model, X_test, y_test, "LogisticRegression", scaler=lr_scaler),
        lr_model,
        lr_scaler,
    )

    rf_model = train_random_forest(X_train, y_train)
    results["random_forest"] = (
        evaluate(rf_model, X_test, y_test, "RandomForest"),
        rf_model,
        None,
    )

    xgb_model = train_xgboost(X_train, y_train)
    results["xgboost"] = (
        evaluate(xgb_model, X_test, y_test, "XGBoost"),
        xgb_model,
        None,
    )

    # Pick the best model by F1
    winner_name = max(results, key=lambda k: results[k][0]["f1"])
    winner_metrics, winner_model, winner_scaler = results[winner_name]
    print(f"\nSelected model for deployment: {winner_name} (f1={winner_metrics['f1']:.4f})")

    bundle = {
        "model": winner_model,
        "model_name": winner_name,
        "feature_names": FEATURE_NAMES,
        "lookup_tables": lookup_tables,
        "scaler": winner_scaler,  
    }
    joblib.dump(bundle, args.out)
    print(f"Saved model bundle -> {args.out}")

    Path(args.metrics_out).write_text(
        json.dumps(
            {name: metrics for name, (metrics, _, _) in results.items()}
            | {"selected": winner_name},
            indent=2,
        )
    )
    print(f"Saved metrics -> {args.metrics_out}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("synth4bench")


DATA_PATH = "~/synth4bench/dataset.tsv"

POSITIVE_CLASS = "TP"
NEGATIVE_CLASS = "FP"
EMITTED_CLASSES = (POSITIVE_CLASS, NEGATIVE_CLASS)


@dataclass
class FilteredDataset:
    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series
    numerical_features: List[str]
    categorical_features: List[str]


def preprocess_dataset(records: pd.DataFrame) -> FilteredDataset:
    categorical_features = ["Caller"]
    numeric_features = [
        "Coverage",
        "Read_length",
        "DP",
        "AF",
        "ref_len",
        "alt_len",
        "indel_len",
        "is_ref_eq_alt",
        "variant_type_code",
        "snp_substitution_code",
    ]
    numeric_features = [c for c in numeric_features if c in records.columns]

    X = records[categorical_features + numeric_features]
    y = records["label"].astype(int)
    groups = records["mutation_group"].astype(str)
    return FilteredDataset(X, y, groups, numeric_features, categorical_features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a per-record XGBoost credibility classifier (TP vs FP) on emitted calls."
    )
    parser.add_argument("--data-path", default=DATA_PATH)
    parser.add_argument("--folds", type=int, default=1)  # should be >=1
    parser.add_argument("--sample-rows", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--n-estimators", type=int, default=300)
    return parser.parse_args()


def is_transition(ref: str, alt: str) -> bool:
    ref = str(ref).upper()
    alt = str(alt).upper()
    return {ref, alt} in ({"A", "G"}, {"C", "T"})


def add_variant_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    ref = out["REF"].astype(str)
    alt = out["ALT"].astype(str)

    out["ref_len"] = ref.str.len()
    out["alt_len"] = alt.str.len()
    out["indel_len"] = out["alt_len"] - out["ref_len"]
    out["is_ref_eq_alt"] = (ref == alt).astype(int)

    is_snp = (out["ref_len"] == 1) & (out["alt_len"] == 1) & (out["is_ref_eq_alt"] == 0)
    is_insertion = out["indel_len"] > 0
    is_deletion = out["indel_len"] < 0

    # 0=other (including REF==ALT), 1=SNP, 2=INS, 3=DEL
    out["variant_type_code"] = np.select(
        [is_snp, is_insertion, is_deletion],
        [1, 2, 3],
        default=0,
    ).astype(int)

    # 0=NA (non-SNP), 1=transition, 2=transversion
    is_ti = np.array(
        [
            bool(is_transition(r, a)) if snp else False
            for r, a, snp in zip(ref, alt, is_snp.to_numpy(), strict=True)
        ],
        dtype=bool,
    )
    out["snp_substitution_code"] = np.where(
        is_snp.to_numpy(), np.where(is_ti, 1, 2), 0
    ).astype(int)
    return out


def build_emitted_record_table(
    data_path: str, sample_rows: int | None = None
) -> pd.DataFrame:
    """
    Builds a per-record training table aligned with inference-time inputs:
    one row corresponds to one emitted VCF record from a given caller.

    We train only on emitted calls: TP vs FP. FN/TN do not exist as records in the VCF,
    and therefore aren't suitable for this per-record credibility model.
    """
    usecols = [
        "Dataset",
        "Coverage",
        "Read_length",
        "POS",
        "REF",
        "ALT",
        "DP",
        "AF",
        "Caller",
        "Class",
    ]
    raw = pd.read_csv(
        data_path,
        sep="\t",
        usecols=usecols,
        nrows=sample_rows,
        low_memory=False,
    )

    raw = raw.loc[raw["Class"].isin(EMITTED_CLASSES)].copy()
    raw["label"] = (raw["Class"] == POSITIVE_CLASS).astype(int)

    raw = add_variant_features(raw)

    # Used only for grouped CV leakage control, not as a feature.
    raw["mutation_group"] = (
        raw["POS"].astype(str)
        + ":"
        + raw["REF"].astype(str)
        + ">"
        + raw["ALT"].astype(str)
    )

    dp = raw["DP"]
    af = raw["AF"]

    raw["AF"] = af
    return raw


def make_pipeline(
    y_train: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int,
    n_estimators: int,
) -> Pipeline:
    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    scale_pos_weight = negatives / positives if positives else 1.0

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ],
        remainder="drop",
    )

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline([("pre", pre), ("model", model)])


def plot_logloss(curves: list[list[float]], labels: list[str], path: str) -> None:
    fig, ax = plt.subplots()
    for label, curve in zip(labels, curves):
        ax.plot(curve, label=label)
    ax.set_xlabel("boosting round")
    ax.set_ylabel("logloss (validation)")
    ax.legend()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Loss curve saved to {path}")


def evaluate_fold(y_true: pd.Series, scores: np.ndarray) -> dict[str, float]:
    predictions = (scores >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def run_cross_validation(
    records: pd.DataFrame, folds: int, random_state: int, n_estimators: int
) -> pd.DataFrame:
    dataset = preprocess_dataset(records)
    groups = dataset.groups
    X = dataset.X
    y = dataset.y
    numeric_features = dataset.numerical_features
    categorical_features = dataset.categorical_features

    splitter = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=random_state,
    )
    metrics = []
    logloss_curves = []

    print(f"Records: {len(X):,}")
    print(f"Features: {len(categorical_features) + len(numeric_features):,}")
    print(f"Positive label rate: {y.mean():.4f}")
    print(f"Groups for CV: {groups.nunique():,}")
    print()

    for fold, (train_index, test_index) in enumerate(
        splitter.split(X, y, groups=groups),
        start=1,
    ):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        pipeline = make_pipeline(
            y_train=y_train,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            random_state=random_state + fold,
            n_estimators=n_estimators,
        )
        pre = pipeline[:-1]
        model = pipeline[-1]
        X_train_t = pre.fit_transform(X_train)
        X_test_t = pre.transform(X_test)
        model.fit(X_train_t, y_train, eval_set=[(X_test_t, y_test)], verbose=False)
        logloss_curves.append(model.evals_result()["validation_0"]["logloss"])
        scores = model.predict_proba(X_test_t)[:, 1]

        fold_metrics = evaluate_fold(y_test, scores)
        fold_metrics["fold"] = float(fold)
        metrics.append(fold_metrics)

        print(
            f"fold={fold} "
            f"roc_auc={fold_metrics['roc_auc']:.4f} "
            f"pr_auc={fold_metrics['pr_auc']:.4f} "
            f"precision={fold_metrics['precision']:.4f} "
            f"recall={fold_metrics['recall']:.4f} "
            f"f1={fold_metrics['f1']:.4f}"
        )

    plot_logloss(
        logloss_curves,
        [f"fold {i}" for i in range(1, folds + 1)],
        "figures/logloss_cv.png",
    )
    return pd.DataFrame(metrics)


def run_single_split(
    records: pd.DataFrame, random_state: int, n_estimators: int, test_size: float = 0.2
) -> dict[str, float]:
    """
    Single train/test split that keeps mutation groups intact and roughly preserves
    class balance by stratifying at the group level.
    """
    dataset = preprocess_dataset(records)
    groups = dataset.groups
    X = dataset.X
    y = dataset.y
    numeric_features = dataset.numerical_features
    categorical_features = dataset.categorical_features

    # Stratify groups by whether they contain at least one positive.
    group_frame = pd.DataFrame({"group": groups, "y": y})
    group_labels = group_frame.groupby("group", sort=False)["y"].max()
    group_names = group_labels.index.to_numpy()
    group_y = group_labels.to_numpy()

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_gi, test_gi = next(splitter.split(group_names, group_y))
    train_groups = set(group_names[train_gi])
    test_mask = groups.isin(group_names[test_gi])
    train_mask = groups.isin(train_groups)

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]
    X_test = X.loc[test_mask]
    y_test = y.loc[test_mask]

    pipeline = make_pipeline(
        y_train=y_train,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        random_state=random_state + 1,
        n_estimators=n_estimators,
    )
    pre = pipeline[:-1]
    model = pipeline[-1]
    X_train_t = pre.fit_transform(X_train)
    X_test_t = pre.transform(X_test)
    model.fit(X_train_t, y_train, eval_set=[(X_train_t, y_train), (X_test_t, y_test)], verbose=False)
    result = model.evals_result()
    plot_logloss(
        [result["validation_0"]["logloss"], result["validation_1"]["logloss"]],
        ["train", "validation"],
        "logloss_single.png",
    )
    scores = model.predict_proba(X_test_t)[:, 1]
    return evaluate_fold(y_test, scores)


def main() -> None:
    args = parse_args()
    log.info("Building emitted-record table (TP vs FP)...")
    records = build_emitted_record_table(args.data_path, sample_rows=args.sample_rows)
    log.info(f"Records: {len(records):,}")
    log.info(
        "Columns kept for modeling are derived; POS/REF/ALT/Dataset are not used as features."
    )

    if args.folds == 1:
        log.info("Running single split...")
        metrics = run_single_split(
            records,
            random_state=args.random_state,
            n_estimators=args.n_estimators,
        )
        print(
            "single_split "
            f"roc_auc={metrics['roc_auc']:.4f} "
            f"pr_auc={metrics['pr_auc']:.4f} "
            f"precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} "
            f"f1={metrics['f1']:.4f}"
        )
        return

    log.info(f"Running cross-validation with fold {args.folds}...")
    metrics = run_cross_validation(
        records,
        folds=args.folds,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
    )
    summary = metrics.drop(columns=["fold"]).agg(["mean", "std"])
    print("\nCross-validation summary:")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()

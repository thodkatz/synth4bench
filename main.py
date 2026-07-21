from __future__ import annotations

import argparse
import datetime
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import joblib
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
from sklearn.preprocessing import OneHotEncoder, label_binarize
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("synth4bench")


DATA_PATH = "./datasets/dataset.tsv"

CLASS_TO_CODE = {"FP": 0, "FN": 1, "TP": 2}
CODE_TO_CLASS = {v: k for k, v in CLASS_TO_CODE.items()}
CLASS_LABELS = [CODE_TO_CLASS[c] for c in sorted(CODE_TO_CLASS)]  # ["FP", "FN", "TP"]


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
        description="Train a per-record XGBoost credibility classifier (TP vs FP vs FN) on candidate mutations."
    )
    parser.add_argument("--data-path", default=DATA_PATH)
    parser.add_argument("--folds", type=int, default=1)  # should be >=1
    parser.add_argument("--sample-rows", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--early-stopping-rounds", type=int, default=20)
    parser.add_argument("--save-model", action=argparse.BooleanOptionalAction, default=True)
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


def build_record_table(
    data_path: str, sample_rows: int | None = None
) -> pd.DataFrame:
    """
    Builds a per-record training table for a 3-class TP/FP/FN classifier.

    TP/FP rows correspond to a caller's emitted VCF record (DP/AF observed by
    that caller). FN rows correspond to a ground-truth mutation `Caller` did
    not call at all — DP/AF for those rows are the ground-truth values, not
    anything the caller emitted.

    Deployment caveat: because FN rows are keyed to a known ground-truth
    mutation, scoring a candidate as TP/FP/FN requires a candidate mutation
    list up front (e.g. a truth set, or this synthetic benchmark itself).
    Unlike the old TP/FP-only filter, this model cannot be pointed at a
    caller's raw VCF output alone to discover what it missed — a caller
    produces no record for a mutation it didn't call.
    """
    usecols = [
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

    raw = raw.loc[raw["Class"].isin(CLASS_TO_CODE)].copy()
    raw["label"] = raw["Class"].map(CLASS_TO_CODE).astype(int)

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
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int,
    n_estimators: int,
    early_stopping_rounds: int = 20,
) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ],
        remainder="drop",
    )

    model = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=len(CLASS_TO_CODE),
        tree_method="hist",
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        early_stopping_rounds=early_stopping_rounds,
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline([("pre", pre), ("model", model)])


def plot_logloss(curves: list[list[float]], labels: list[str], path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    for label, curve in zip(labels, curves):
        log.info(f"Label for curve {label}")
        ax.plot(curve, label=label)
    ax.set_xlabel("boosting round")
    ax.set_ylabel("logloss (validation)")
    ax.legend()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Loss curve saved to {output_path}")


def save_model(pre, model) -> None:
    path = Path("saved_models") / f"model_{datetime.datetime.now(datetime.UTC).strftime('%Y%m%d')}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pre": pre, "model": model}, path)
    log.info(f"Model saved to {path}")


def load_model(path: str):
    obj = joblib.load(path)
    return obj["pre"], obj["model"]


LABELS = sorted(CODE_TO_CLASS)  # [0, 1, 2]


def evaluate_predictions(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    cm = confusion_matrix(y_true, predictions, labels=LABELS)
    precision = precision_score(y_true, predictions, labels=LABELS, average=None, zero_division=0)
    recall = recall_score(y_true, predictions, labels=LABELS, average=None, zero_division=0)
    f1 = f1_score(y_true, predictions, labels=LABELS, average=None, zero_division=0)
    support = cm.sum(axis=1)

    metrics: dict[str, float] = {
        "precision_macro": precision_score(y_true, predictions, labels=LABELS, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, predictions, labels=LABELS, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, predictions, labels=LABELS, average="macro", zero_division=0),
    }
    for code in LABELS:
        name = CODE_TO_CLASS[code]
        metrics[f"precision_{name}"] = float(precision[code])
        metrics[f"recall_{name}"] = float(recall[code])
        metrics[f"f1_{name}"] = float(f1[code])
        metrics[f"support_{name}"] = float(support[code])
        for pred_code in LABELS:
            metrics[f"cm_{name}_{CODE_TO_CLASS[pred_code]}"] = float(cm[code, pred_code])
    return metrics


def evaluate_fold(y_true: pd.Series, scores: np.ndarray) -> dict[str, float]:
    predictions = scores.argmax(axis=1)
    metrics = evaluate_predictions(y_true, predictions)

    unique_classes = pd.Series(y_true).nunique()
    roc_auc = float("nan")
    pr_auc = float("nan")

    if unique_classes < 2:
        log.warning("Fold contains a single class; roc_auc and pr_auc are undefined for this split.")
    else:
        roc_auc = roc_auc_score(y_true, scores, multi_class="ovr", average="macro", labels=LABELS)
        y_true_bin = label_binarize(y_true, classes=LABELS)
        pr_auc = average_precision_score(y_true_bin, scores, average="macro")

    metrics["roc_auc"] = roc_auc
    metrics["pr_auc"] = pr_auc
    return metrics


def evaluate_by_caller(
    callers: pd.Series, y_true: pd.Series, scores: np.ndarray
) -> pd.DataFrame:
    rows = []
    for caller in sorted(callers.astype(str).unique()):
        mask = (callers == caller).to_numpy()
        caller_y = y_true.loc[mask]
        caller_predictions = scores[mask].argmax(axis=1)

        metrics = evaluate_predictions(caller_y, caller_predictions)
        metrics["caller"] = caller
        metrics["records"] = float(mask.sum())
        rows.append(metrics)

    return pd.DataFrame(rows)


def print_caller_report(report: pd.DataFrame) -> None:
    print("\nPer-caller held-out metrics (3-class TP/FP/FN):")
    for row in report.sort_values("caller").itertuples(index=False):
        print(f"\n{row.caller} records={int(row.records):,}")
        for name in CLASS_LABELS:
            print(
                f"  {name}: "
                f"precision={getattr(row, f'precision_{name}'):.4f} "
                f"recall={getattr(row, f'recall_{name}'):.4f} "
                f"f1={getattr(row, f'f1_{name}'):.4f} "
                f"num_samples={int(getattr(row, f'support_{name}')):,}"
            )
        print(
            f"  macro: "
            f"precision={row.precision_macro:.4f} "
            f"recall={row.recall_macro:.4f} "
            f"f1={row.f1_macro:.4f}"
        )


def plot_confusion_matrix_by_caller(
    callers: pd.Series, y_true: pd.Series, scores: np.ndarray, path: str
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predictions = scores.argmax(axis=1)
    class_order = LABELS
    class_display = CLASS_LABELS

    caller_list = sorted(callers.astype(str).unique())
    ncols = min(len(caller_list), 3)
    nrows = -(-len(caller_list) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)

    for idx, caller in enumerate(caller_list):
        ax = axes[idx // ncols][idx % ncols]
        mask = (callers == caller).to_numpy()
        cm = confusion_matrix(y_true.loc[mask], predictions[mask], labels=class_order)

        ax.imshow(cm, cmap="Blues")
        ax.set_title(caller)
        ax.set_xticks(range(len(class_display)))
        ax.set_xticklabels(class_display)
        ax.set_yticks(range(len(class_display)))
        ax.set_yticklabels(class_display)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, f"{cm[i, j]:,}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )

    for idx in range(len(caller_list), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle("Per-caller confusion matrix (TP/FP/FN)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Per-caller confusion matrix saved to {output_path}")


def split_train_val_groups(
    groups: pd.Series,
    y: pd.Series,
    val_size: float,
    random_state: int,
) -> tuple[pd.Series, pd.Series]:
    group_frame = pd.DataFrame({"group": groups, "y": y})
    group_labels = group_frame.groupby("group", sort=False)["y"].max()
    group_names = group_labels.index.to_numpy()
    group_y = group_labels.to_numpy()

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    train_gi, val_gi = next(splitter.split(group_names, group_y))

    train_mask = groups.isin(set(group_names[train_gi]))
    val_mask = groups.isin(set(group_names[val_gi]))
    return train_mask, val_mask


def run_cross_validation(
    records: pd.DataFrame,
    folds: int,
    random_state: int,
    n_estimators: int,
    early_stopping_rounds: int = 20,
    val_size: float = 0.2,
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
    print(f"Unique mutation groups: {groups.nunique():,} ({groups.nunique() / len(X):.1%} of records)")
    print(f"Class rates: {y.value_counts(normalize=True).rename(index=CODE_TO_CLASS).round(4).to_dict()}")
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
        groups_train = groups.iloc[train_index]

        train_mask, val_mask = split_train_val_groups(
            groups=groups_train,
            y=y_train,
            val_size=val_size,
            random_state=random_state + fold,
        )
        X_fold_train = X_train.loc[train_mask]
        y_fold_train = y_train.loc[train_mask]
        X_fold_val = X_train.loc[val_mask]
        y_fold_val = y_train.loc[val_mask]

        pipeline = make_pipeline(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            random_state=random_state + fold,
            n_estimators=n_estimators,
            early_stopping_rounds=early_stopping_rounds,
        )
        pre = pipeline[:-1]
        model = pipeline[-1]
        X_train_t = pre.fit_transform(X_fold_train)
        X_val_t = pre.transform(X_fold_val)
        X_test_t = pre.transform(X_test)
        sample_weight = compute_sample_weight("balanced", y_fold_train)
        model.fit(
            X_train_t,
            y_fold_train,
            sample_weight=sample_weight,
            eval_set=[(X_val_t, y_fold_val)],
            verbose=False,
        )
        logloss_curves.append(model.evals_result()["validation_0"]["mlogloss"])
        scores = model.predict_proba(X_test_t)

        fold_metrics = evaluate_fold(y_test, scores)
        fold_metrics["fold"] = float(fold)
        metrics.append(fold_metrics)

        print(
            f"fold={fold} "
            f"roc_auc={fold_metrics['roc_auc']:.4f} "
            f"pr_auc={fold_metrics['pr_auc']:.4f} "
            f"precision_macro={fold_metrics['precision_macro']:.4f} "
            f"recall_macro={fold_metrics['recall_macro']:.4f} "
            f"f1_macro={fold_metrics['f1_macro']:.4f}"
        )

    plot_logloss(
        logloss_curves,
        [f"fold {i}" for i in range(1, folds + 1)],
        "figures/logloss_cv.png",
    )
    return pd.DataFrame(metrics)


def run_single_split(
    records: pd.DataFrame,
    random_state: int,
    n_estimators: int,
    early_stopping_rounds: int = 20,
    test_size: float = 0.2,
    val_size: float = 0.2,
) -> dict[str, float]:
    """
    Three-way split (train / val / test) that keeps mutation groups intact and
    roughly preserves class balance by stratifying at the group level.
    val is passed to eval_set for monitoring; test is touched only for final metrics.
    """
    dataset = preprocess_dataset(records)
    groups = dataset.groups
    X = dataset.X
    y = dataset.y
    numeric_features = dataset.numerical_features
    categorical_features = dataset.categorical_features

    group_frame = pd.DataFrame({"group": groups, "y": y})
    group_labels = group_frame.groupby("group", sort=False)["y"].max()
    group_names = group_labels.index.to_numpy()
    group_y = group_labels.to_numpy()

    # First split: hold out test.
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    trainval_gi, test_gi = next(splitter.split(group_names, group_y))

    # Second split: carve val out of trainval.
    trainval_names = group_names[trainval_gi]
    trainval_mask = groups.isin(set(trainval_names))
    train_mask_rel, val_mask_rel = split_train_val_groups(
        groups=groups.loc[trainval_mask],
        y=y.loc[trainval_mask],
        val_size=val_size,
        random_state=random_state,
    )

    train_mask = pd.Series(False, index=groups.index)
    val_mask = pd.Series(False, index=groups.index)
    train_mask.loc[trainval_mask] = train_mask_rel.to_numpy()
    val_mask.loc[trainval_mask] = val_mask_rel.to_numpy()
    test_mask = groups.isin(set(group_names[test_gi]))

    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_val, y_val = X.loc[val_mask], y.loc[val_mask]
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]

    total = len(X)
    log.info(
        f"Split — train: {len(X_train):,} ({len(X_train)/total:.0%})  "
        f"val: {len(X_val):,} ({len(X_val)/total:.0%})  "
        f"test: {len(X_test):,} ({len(X_test)/total:.0%})"
    )

    pipeline = make_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        random_state=random_state + 1,
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
    )
    pre = pipeline[:-1]
    model = pipeline[-1]
    X_train_t = pre.fit_transform(X_train)
    X_val_t = pre.transform(X_val)
    X_test_t = pre.transform(X_test)
    sample_weight = compute_sample_weight("balanced", y_train)
    model.fit(
        X_train_t,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(X_train_t, y_train), (X_val_t, y_val)],
        verbose=False,
    )
    result = model.evals_result()
    plot_logloss(
        [result["validation_0"]["mlogloss"], result["validation_1"]["mlogloss"]],
        ["train", "val"],
        "figures/logloss_single.png",
    )
    scores = model.predict_proba(X_test_t)
    caller_report = evaluate_by_caller(X_test["Caller"], y_test, scores)
    plot_confusion_matrix_by_caller(
        X_test["Caller"], y_test, scores, "figures/confusion_matrix_by_caller.png"
    )
    return evaluate_fold(y_test, scores), caller_report, pre, model


def save_single_split_results(
    metrics: dict[str, float], caller_report: pd.DataFrame, path: str = "results/single_split_metrics.csv"
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(output_path, index=False)
    caller_path = output_path.with_name(f"{output_path.stem}_by_caller.csv")
    caller_report.to_csv(caller_path, index=False)
    log.info(f"Results saved to {output_path} and {caller_path}")


def save_cv_results(
    metrics: pd.DataFrame, summary: pd.DataFrame, path: str = "results/cv_metrics.csv"
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_path, index=False)
    summary_path = output_path.with_name(f"{output_path.stem}_summary.csv")
    summary.to_csv(summary_path)
    log.info(f"Results saved to {output_path} and {summary_path}")


def main() -> None:
    args = parse_args()
    log.info("Building record table (TP vs FP vs FN)...")
    records = build_record_table(args.data_path, sample_rows=args.sample_rows)
    if args.sample_rows is None:
        log.info(f"Records: {len(records):,} (full dataset)")
    else:
        log.info(f"Records: {len(records):,} (subsample, requested {args.sample_rows:,})")
    log.info(
        "Columns kept for modeling are derived; POS/REF/ALT/Dataset are not used as features."
    )

    if args.folds == 1:
        log.info("Running single split...")
        metrics, caller_report, pre, model = run_single_split(
            records,
            random_state=args.random_state,
            n_estimators=args.n_estimators,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        print(
            "single_split "
            f"roc_auc={metrics['roc_auc']:.4f} "
            f"pr_auc={metrics['pr_auc']:.4f} "
            f"precision_macro={metrics['precision_macro']:.4f} "
            f"recall_macro={metrics['recall_macro']:.4f} "
            f"f1_macro={metrics['f1_macro']:.4f}"
        )
        print_caller_report(caller_report)
        save_single_split_results(metrics, caller_report)
        if args.save_model:
            save_model(pre, model)
        return

    log.info(f"Running cross-validation with fold {args.folds}...")
    metrics = run_cross_validation(
        records,
        folds=args.folds,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        early_stopping_rounds=args.early_stopping_rounds,
    )
    summary = metrics.drop(columns=["fold"]).agg(["mean", "std"])
    print("\nCross-validation summary:")
    print(summary.round(4).to_string())
    save_cv_results(metrics, summary)


if __name__ == "__main__":
    main()

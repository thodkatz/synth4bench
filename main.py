from __future__ import annotations

import argparse
from dataclasses import dataclass

import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("synth4bench")


DATA_PATH = "/mnt/data/documents/certh/synth4bench/dataset.tsv"

CALLERS = ("Freebayes", "LoFreq", "Mutect2", "VarDict", "VarScan")
CALLER_VALUE_COLUMNS = ("DP", "AD", "AF", "AF Deviation")
ID_COLUMNS = ("Dataset", "POS", "REF", "ALT")
TRUTH_CLASSES = ("TP", "FN")
EMITTED_CLASSES = ("TP", "FP")


@dataclass(frozen=True)
class DatasetBundle:
    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series
    candidate_ids: pd.Series


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a candidate-level XGBoost proof-of-concept ensemble."
    )
    parser.add_argument("--data-path", default=DATA_PATH)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--sample-rows", type=int, default=None)
    parser.add_argument("--include-unemitted-truth", action="store_true")
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--n-estimators", type=int, default=300)
    return parser.parse_args()


def is_transition(ref: str, alt: str) -> bool:
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

    # Group feature 1: SNP vs insertion vs deletion vs other.
    # 0=other (including REF==ALT), 1=SNP, 2=INS, 3=DEL
    out["variant_type_code"] = np.select(
        [is_snp, is_insertion, is_deletion],
        [1, 2, 3],
        default=0,
    ).astype(int)

    # Group feature 2 (SNP-only): transition vs transversion vs NA.
    # 0=NA (non-SNP), 1=transition, 2=transversion
    is_ti = np.array(
        [
            bool(is_transition(r, a)) if snp else False
            for r, a, snp in zip(ref, alt, is_snp.to_numpy(), strict=True)
        ],
        dtype=bool,
    )
    out["snp_substitution_code"] = np.where(is_snp.to_numpy(), np.where(is_ti, 1, 2), 0).astype(
        int
    )
    return out


def flatten_columns(columns: pd.MultiIndex) -> list[str]:
    return [f"{caller}_{feature}" for feature, caller in columns]


def build_candidate_table(
    data_path: str,
    sample_rows: int | None = None,
    include_unemitted_truth: bool = False,
) -> DatasetBundle:
    usecols = [
        "Dataset",
        "Coverage",
        "Read_length",
        "POS",
        "REF",
        "ALT",
        "DP",
        "AD",
        "AF",
        "AF Deviation",
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

    raw["is_truth"] = raw["Class"].isin(TRUTH_CLASSES).astype(int)
    raw["is_emitted"] = raw["Class"].isin(EMITTED_CLASSES).astype(int)

    base = (
        raw.groupby(list(ID_COLUMNS), dropna=False)
        .agg(
            Coverage=("Coverage", "first"),
            Read_length=("Read_length", "first"),
            label=("is_truth", "max"),
            n_callers_supporting=("is_emitted", "sum"),
        )
        .reset_index()
    )

    emitted = raw.loc[raw["is_emitted"] == 1, [*ID_COLUMNS, "Caller", *CALLER_VALUE_COLUMNS]]
    called_by = (
        emitted.assign(called=1)
        .pivot_table(
            index=list(ID_COLUMNS),
            columns="Caller",
            values="called",
            aggfunc="max",
            fill_value=0,
        )
        .reindex(columns=CALLERS, fill_value=0)
    )
    called_by.columns = [f"called_by_{caller}" for caller in called_by.columns]

    caller_values = emitted.pivot_table(
        index=list(ID_COLUMNS),
        columns="Caller",
        values=list(CALLER_VALUE_COLUMNS),
        aggfunc="first",
    )
    caller_values = caller_values.reindex(
        columns=pd.MultiIndex.from_product([CALLER_VALUE_COLUMNS, CALLERS])
    )
    caller_values.columns = flatten_columns(caller_values.columns)

    candidates = (
        base.set_index(list(ID_COLUMNS))
        .join(called_by, how="left")
        .join(caller_values, how="left")
        .reset_index()
    )

    called_cols = [f"called_by_{caller}" for caller in CALLERS]
    candidates[called_cols] = candidates[called_cols].fillna(0).astype(int)
    candidates["n_callers_supporting"] = candidates[called_cols].sum(axis=1)

    if not include_unemitted_truth:
        candidates = candidates.loc[candidates["n_callers_supporting"] > 0].copy()

    candidates = add_variant_features(candidates)
    candidates["candidate_id"] = (
        candidates["Dataset"].astype(str)
        + ":"
        + candidates["POS"].astype(str)
        + ":"
        + candidates["REF"].astype(str)
        + ">"
        + candidates["ALT"].astype(str)
    )
    candidates["mutation_group"] = (
        candidates["POS"].astype(str)
        + ":"
        + candidates["REF"].astype(str)
        + ">"
        + candidates["ALT"].astype(str)
    )

    for caller in CALLERS:
        called = candidates[f"called_by_{caller}"].astype(bool)
        dp = candidates[f"{caller}_DP"]
        ad_raw = candidates[f"{caller}_AD"]
        af_raw = candidates[f"{caller}_AF"]

        # Only impute within "called" rows. If a caller didn't emit the variant,
        # we should not synthesize evidence for that caller.
        ad_inferred = np.where(
            called & ad_raw.isna() & dp.notna() & af_raw.notna() & (dp > 0),
            af_raw * dp,
            np.nan,
        )
        af_inferred = np.where(
            called & af_raw.isna() & dp.notna() & ad_raw.notna() & (dp > 0),
            ad_raw / dp,
            np.nan,
        )

        ad_filled = ad_raw.fillna(pd.Series(ad_inferred, index=candidates.index))
        af_filled = af_raw.fillna(pd.Series(af_inferred, index=candidates.index))

        candidates[f"{caller}_AD_inferred"] = ad_inferred
        candidates[f"{caller}_AF_inferred"] = af_inferred
        candidates[f"{caller}_AD_filled"] = ad_filled
        candidates[f"{caller}_AF_filled"] = af_filled
        candidates[f"{caller}_AD_was_imputed"] = (
            called & ad_raw.isna() & pd.notna(ad_inferred)
        ).astype(int)
        candidates[f"{caller}_AF_was_imputed"] = (
            called & af_raw.isna() & pd.notna(af_inferred)
        ).astype(int)

        candidates[f"{caller}_AD_over_DP"] = ad_filled / dp.replace(0, np.nan)
        candidates[f"{caller}_missing_AD"] = (called & ad_filled.isna()).astype(int)

        dev_raw = candidates[f"{caller}_AF Deviation"]
        candidates[f"{caller}_missing_AF_deviation"] = (called & dev_raw.isna()).astype(
            int
        )
        candidates[f"{caller}_AF_deviation_filled"] = np.where(
            called.to_numpy(),
            dev_raw.fillna(0.0).to_numpy(),
            np.nan,
        )

    excluded = {
        *ID_COLUMNS,
        "label",
        "candidate_id",
        "mutation_group",
    }
    feature_columns = [column for column in candidates.columns if column not in excluded]

    return DatasetBundle(
        X=candidates[feature_columns],
        y=candidates["label"].astype(int),
        groups=candidates["mutation_group"],
        candidate_ids=candidates["candidate_id"],
    )


def make_model(y_train: pd.Series, random_state: int, n_estimators: int) -> XGBClassifier:
    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    scale_pos_weight = negatives / positives if positives else 1.0

    return XGBClassifier(
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


def run_cross_validation(bundle: DatasetBundle, folds: int, random_state: int, n_estimators: int) -> pd.DataFrame:
    splitter = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=random_state,
    )
    metrics = []

    print(f"Candidates: {len(bundle.X):,}")
    print(f"Features: {len(bundle.X.columns):,}")
    print(f"Positive label rate: {bundle.y.mean():.4f}")
    print(f"Groups for CV: {bundle.groups.nunique():,}")
    print()

    for fold, (train_index, test_index) in enumerate(
        splitter.split(bundle.X, bundle.y, groups=bundle.groups),
        start=1,
    ):
        X_train = bundle.X.iloc[train_index]
        X_test = bundle.X.iloc[test_index]
        y_train = bundle.y.iloc[train_index]
        y_test = bundle.y.iloc[test_index]

        model = make_model(y_train, random_state + fold, n_estimators)
        model.fit(X_train, y_train)
        scores = model.predict_proba(X_test)[:, 1]

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

    return pd.DataFrame(metrics)


def main() -> None:
    args = parse_args()
    log.info("Building candidate table...")
    bundle = build_candidate_table(
        args.data_path,
        sample_rows=args.sample_rows,
        include_unemitted_truth=args.include_unemitted_truth,
    )

    log.info("Candidate table built.")
    log.info(f"Columns: {list(bundle.X.columns)}")

    # metrics = run_cross_validation(
    #     bundle,
    #     folds=args.folds,
    #     random_state=args.random_state,
    #     n_estimators=args.n_estimators,
    # )

    summary = metrics.drop(columns=["fold"]).agg(["mean", "std"])
    print("\nCross-validation summary:")
    print(summary.round(4).to_string())



if __name__ == "__main__":
    main()

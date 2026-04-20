"""
Label extraction for K-MIMIC MEDS — two candidate binary mortality tasks.

Tasks
-----
icu_mortality_24h      : died in ICU, predicted from first 24h of ICU stay
inhospital_mortality_24h : died in hospital, predicted from first 24h of admission

Output format (MEDS-DEV compatible)
------------------------------------
  subject_id (int64) | prediction_time (datetime64[us]) | boolean_value (bool)
  saved to data/labels/{task}/{split}/0.parquet
"""

import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("data/output")
LABELS_DIR = Path("data/labels")
GAP        = pd.Timedelta(hours=24)


def load_full_dataset() -> pd.DataFrame:
    dfs = []
    for split in ["train", "tuning", "held_out"]:
        df = pd.read_parquet(OUTPUT_DIR / f"data/{split}/0.parquet")
        df["split"] = split
        dfs.append(df)
    full = pd.concat(dfs, ignore_index=True)
    full["time"] = pd.to_datetime(full["time"])
    return full


def extract_events(full: pd.DataFrame):
    def first(df, prefix, col):
        return (
            df[df["code"].str.startswith(prefix)][["subject_id", "time"]]
            .rename(columns={"time": col})
            .groupby("subject_id")[col].min()
            .reset_index()
        )

    icu_adm  = first(full, "ICU_ADMISSION",      "icu_intime")
    icu_dis  = first(full, "ICU_DISCHARGE",       "icu_outtime")
    hosp_adm = first(full, "HOSPITAL_ADMISSION",  "adm_time")
    hosp_dis = first(full, "HOSPITAL_DISCHARGE",  "dis_time")
    deaths   = first(full, "MEDS_DEATH",          "death_time")

    split_map = (
        full[["subject_id", "split"]]
        .drop_duplicates("subject_id")
        .set_index("subject_id")["split"]
    )
    return icu_adm, icu_dis, hosp_adm, hosp_dis, deaths, split_map


def build_icu_mortality(icu_adm, icu_dis, deaths, split_map) -> pd.DataFrame:
    """
    Index:      ICU_ADMISSION
    Prediction: ICU_ADMISSION + 24h
    Label:      MEDS_DEATH before ICU_DISCHARGE
    Exclusions: patients discharged or dead before prediction_time
    """
    df = icu_adm.merge(icu_dis, on="subject_id", how="inner")
    df = df.merge(deaths, on="subject_id", how="left")
    df["prediction_time"] = df["icu_intime"] + GAP

    mask = (df["prediction_time"] <= df["icu_outtime"]) & (
        df["death_time"].isna() | (df["death_time"] >= df["prediction_time"])
    )
    df = df[mask].copy()

    df["boolean_value"] = df["death_time"].notna() & (
        df["death_time"] <= df["icu_outtime"]
    )
    df["split"] = df["subject_id"].map(split_map)
    return df[["subject_id", "prediction_time", "boolean_value", "split"]]


def build_inhospital_mortality(hosp_adm, hosp_dis, deaths, split_map) -> pd.DataFrame:
    """
    Index:      HOSPITAL_ADMISSION
    Prediction: HOSPITAL_ADMISSION + 24h
    Label:      MEDS_DEATH before HOSPITAL_DISCHARGE
    Exclusions: patients discharged or dead before prediction_time
    """
    df = hosp_adm.merge(hosp_dis, on="subject_id", how="inner")
    df = df.merge(deaths, on="subject_id", how="left")
    df["prediction_time"] = df["adm_time"] + GAP

    mask = (df["prediction_time"] <= df["dis_time"]) & (
        df["death_time"].isna() | (df["death_time"] >= df["prediction_time"])
    )
    df = df[mask].copy()

    df["boolean_value"] = df["death_time"].notna() & (
        df["death_time"] <= df["dis_time"]
    )
    df["split"] = df["subject_id"].map(split_map)
    return df[["subject_id", "prediction_time", "boolean_value", "split"]]


def save_labels(labels: pd.DataFrame, task_name: str) -> None:
    for split in ["train", "tuning", "held_out"]:
        out_dir = LABELS_DIR / task_name / split
        out_dir.mkdir(parents=True, exist_ok=True)
        subset = labels[labels["split"] == split][
            ["subject_id", "prediction_time", "boolean_value"]
        ].reset_index(drop=True)
        subset.to_parquet(out_dir / "0.parquet", index=False)


def print_prevalence(labels: pd.DataFrame, task_name: str) -> None:
    print(f"\n{'=' * 55}")
    print(f"  {task_name}")
    print(f"{'=' * 55}")
    total_pos = labels["boolean_value"].sum()
    total_n   = len(labels)
    print(f"  Total cohort : {total_n:>5}  positives: {int(total_pos):>4}  ({total_pos/total_n*100:.1f}%)")
    print(f"  {'Split':<12} {'N':>6}  {'Pos':>5}  {'Prevalence':>11}")
    print(f"  {'-'*40}")
    for split in ["train", "tuning", "held_out"]:
        sub = labels[labels["split"] == split]
        n   = len(sub)
        pos = int(sub["boolean_value"].sum())
        pct = pos / n * 100 if n > 0 else 0.0
        flag = "  [SPARSE]" if pos < 10 else ""
        print(f"  {split:<12} {n:>6}  {pos:>5}  {pct:>10.1f}%{flag}")


def main():
    print("Loading MEDS dataset...")
    full = load_full_dataset()
    icu_adm, icu_dis, hosp_adm, hosp_dis, deaths, split_map = extract_events(full)

    icu_labels  = build_icu_mortality(icu_adm, icu_dis, deaths, split_map)
    hosp_labels = build_inhospital_mortality(hosp_adm, hosp_dis, deaths, split_map)

    print_prevalence(icu_labels,  "icu_mortality_24h")
    print_prevalence(hosp_labels, "inhospital_mortality_24h")

    save_labels(icu_labels,  "icu_mortality_24h")
    save_labels(hosp_labels, "inhospital_mortality_24h")

    print(f"\nLabels saved to {LABELS_DIR.resolve()}")

    # Task selection guidance
    print("\n" + "=" * 55)
    print("  TASK SELECTION")
    print("=" * 55)
    for name, ldf in [("icu_mortality_24h", icu_labels), ("inhospital_mortality_24h", hosp_labels)]:
        min_pos = min(ldf[ldf["split"] == s]["boolean_value"].sum() for s in ["train", "tuning", "held_out"])
        viable = min_pos >= 10
        status = "OK " if viable else "LOW"
        print(f"  [{status}]  {name:<30}  min positives per split: {int(min_pos)}")


if __name__ == "__main__":
    main()

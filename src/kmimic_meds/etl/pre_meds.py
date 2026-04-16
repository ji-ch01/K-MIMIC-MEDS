"""
Pre-MEDS transformation step for K-MIMIC (Synthetic SYN-ICU).

Responsibilities:
- Load raw K-MIMIC .xlsx files
- Build a stable UUID → int64 mapping for all subject/hadm/stay IDs
- Resolve timestamp inconsistencies (mixed formats, date-only columns)
- Compute derived columns (year_of_birth from anchor_age + anchor_year)
- Rename non-standard columns (icustay_id → stay_id)
- Write cleaned Parquet files to intermediate_dir for meds_convert.py
"""

import argparse
import hashlib
from datetime import timedelta
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Table configuration
# ---------------------------------------------------------------------------

# Tables whose absence makes the MEDS output invalid — abort on missing.
CRITICAL_TABLES = {
    "syn_patients",
    "syn_admissions",
    "syn_icustays",
    "syn_chartevents",
    "syn_labevents",
}

# Minimum required columns per table (subset check — extra columns are fine).
EXPECTED_COLUMNS: dict[str, set[str]] = {
    "syn_patients":        {"subject_id", "anchor_age", "anchor_year", "sex"},
    "syn_admissions":      {"subject_id", "hadm_id", "admittime", "dischtime"},
    "syn_icustays":        {"subject_id", "hadm_id", "stay_id", "intime", "outtime"},
    "syn_chartevents":     {"subject_id", "hadm_id", "stay_id", "itemid", "charttime", "valuenum"},
    "syn_labevents":       {"subject_id", "hadm_id", "itemid", "charttime", "valuenum"},
    "syn_diagnoses_icd":   {"subject_id", "hadm_id", "icd_code"},
    "syn_procedures_icd":  {"subject_id", "hadm_id", "icd_code", "chartdate"},
    "syn_inputevents":     {"subject_id", "hadm_id", "itemid", "starttime", "amount"},
    "syn_outputevents":    {"subject_id", "hadm_id", "itemid", "charttime"},
    "syn_procedureevents": {"subject_id", "hadm_id", "stay_id", "itemid", "starttime"},
    "syn_emar":            {"subject_id", "hadm_id", "itemid", "charttime"},
}


# ---------------------------------------------------------------------------
# UUID → int64 helpers
# ---------------------------------------------------------------------------

def uuid_to_int(uuid_str: str) -> int:
    """Convert a UUID string to a stable positive int64 via SHA-256."""
    h = hashlib.sha256(str(uuid_str).encode()).digest()
    return int.from_bytes(h[:8], "big") & 0x7FFFFFFFFFFFFFFF


def build_all_id_maps(raw: dict) -> tuple:
    """
    Scan all loaded DataFrames and build stable UUID→int64 maps
    for subject_id, hadm_id, and stay_id (including icustay_id).
    Returns (subject_map, hadm_map, stay_map).
    """
    subject_uuids, hadm_uuids, stay_uuids = set(), set(), set()

    for df in raw.values():
        if "subject_id" in df.columns:
            subject_uuids.update(df["subject_id"].dropna().unique())
        if "hadm_id" in df.columns:
            hadm_uuids.update(df["hadm_id"].dropna().unique())
        for col in ["stay_id", "icustay_id"]:
            if col in df.columns:
                stay_uuids.update(df[col].dropna().unique())

    subject_map = {v: uuid_to_int(v) for v in subject_uuids}
    hadm_map    = {v: uuid_to_int(v) for v in hadm_uuids}
    stay_map    = {v: uuid_to_int(v) for v in stay_uuids}

    # Verify no SHA-256 collision after UUID → int64 conversion.
    # This is a production safety invariant — never disable with -O.
    for name, id_map in [("subject", subject_map), ("hadm", hadm_map), ("stay", stay_map)]:
        int_ids = list(id_map.values())
        if len(int_ids) != len(set(int_ids)):
            raise ValueError(
                f"SHA-256 int64 collision detected in {name} ID mapping "
                f"({len(int_ids)} UUIDs → {len(set(int_ids))} unique ints). "
                "This should never happen for datasets of this size. "
                "Verify the source UUID format has not changed."
            )

    print(f"  ID maps: {len(subject_map)} subjects, {len(hadm_map)} hadm, {len(stay_map)} stays — no collisions")
    return subject_map, hadm_map, stay_map


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def read_xlsx(path: Path) -> pd.DataFrame:
    print(f"  reading {path.name}...")
    return pd.read_excel(path, engine="openpyxl")


# ---------------------------------------------------------------------------
# Timestamp helper — wraps pd.to_datetime with a coercion loss warning
# ---------------------------------------------------------------------------

def coerce_datetime(series: pd.Series, label: str = "") -> pd.Series:
    """
    Parse a Series as datetime with errors='coerce'.
    Prints a warning if any non-null values could not be parsed.
    """
    before = series.notna().sum()
    result = pd.to_datetime(series, errors="coerce")
    lost = int(before - result.notna().sum())
    if lost > 0:
        tag = f" ({label})" if label else ""
        print(f"  WARNING{tag}: {lost}/{before} non-null values could not be parsed as timestamps → set to NaT")
    return result


# ---------------------------------------------------------------------------
# Table-specific transformations
# ---------------------------------------------------------------------------

def transform_patients(df: pd.DataFrame, subject_map: dict) -> pd.DataFrame:
    df = df.copy()
    df["subject_id"] = df["subject_id"].map(subject_map)
    df["anchor_age"] = pd.to_numeric(df["anchor_age"], errors="coerce")
    df["anchor_year"] = pd.to_numeric(df["anchor_year"], errors="coerce")
    # Store as Int64 (nullable integer) — downstream extract_patients converts to str for date parsing
    df["year_of_birth"] = (df["anchor_year"] - df["anchor_age"]).astype("Int64")
    df["dod"] = coerce_datetime(df["dod"], "patients.dod")
    return df


def transform_admissions(df: pd.DataFrame, subject_map: dict, hadm_map: dict) -> pd.DataFrame:
    df = df.copy()
    df["subject_id"] = df["subject_id"].map(subject_map)
    df["hadm_id"] = df["hadm_id"].map(hadm_map)
    for col in ["admittime", "dischtime", "deathtime", "edregtime", "edouttime"]:
        if col in df.columns:
            df[col] = coerce_datetime(df[col], f"admissions.{col}")
    return df


def transform_transfers(df: pd.DataFrame, subject_map: dict, hadm_map: dict) -> pd.DataFrame:
    df = df.copy()
    df["subject_id"] = df["subject_id"].map(subject_map)
    df["hadm_id"] = df["hadm_id"].map(hadm_map)
    for col in ["intime", "outtime"]:
        if col in df.columns:
            df[col] = coerce_datetime(df[col], f"transfers.{col}")
    return df


def transform_icustays(df: pd.DataFrame, subject_map: dict, hadm_map: dict, stay_map: dict) -> pd.DataFrame:
    df = df.copy()
    df["subject_id"] = df["subject_id"].map(subject_map)
    df["hadm_id"] = df["hadm_id"].map(hadm_map)
    df["stay_id"] = df["stay_id"].map(stay_map)
    # intime/outtime have mixed formats in SYN-ICU (some date-only, some with time)
    for col in ["intime", "outtime"]:
        if col in df.columns:
            df[col] = coerce_datetime(df[col], f"icustays.{col}")
    return df


def transform_chartevents(df, subject_map, hadm_map, stay_map):
    df = df.copy()
    df["subject_id"] = df["subject_id"].map(subject_map)
    df["hadm_id"] = df["hadm_id"].map(hadm_map)
    df["stay_id"] = df["stay_id"].map(stay_map)
    for col in ["charttime", "storetime"]:
        if col in df.columns:
            df[col] = coerce_datetime(df[col], f"chartevents.{col}")
    if "value" in df.columns:
        df["value"] = df["value"].astype(str)
    df["valuenum"] = pd.to_numeric(df["valuenum"], errors="coerce")
    return df


def transform_labevents(df: pd.DataFrame, subject_map: dict, hadm_map: dict, stay_map: dict) -> pd.DataFrame:
    df = df.copy()
    df["subject_id"] = df["subject_id"].map(subject_map)
    df["hadm_id"] = df["hadm_id"].map(hadm_map)
    if "stay_id" in df.columns:
        df["stay_id"] = df["stay_id"].map(stay_map)
    for col in ["charttime", "storetime"]:
        if col in df.columns:
            df[col] = coerce_datetime(df[col], f"labevents.{col}")
    df["valuenum"] = pd.to_numeric(df["valuenum"], errors="coerce")
    return df


def transform_diagnoses_icd(df: pd.DataFrame, subject_map: dict, hadm_map: dict, stay_map: dict) -> pd.DataFrame:
    df = df.copy()
    df["subject_id"] = df["subject_id"].map(subject_map)
    df["hadm_id"] = df["hadm_id"].map(hadm_map)
    if "stay_id" in df.columns:
        df["stay_id"] = df["stay_id"].map(stay_map)
    # is_icu is 0/1 integer in SYN-ICU (doc says Y/N)
    if "is_icu" in df.columns:
        df["is_icu"] = df["is_icu"].astype(str)
    return df


def transform_procedures_icd(df: pd.DataFrame, subject_map: dict, hadm_map: dict, stay_map: dict) -> pd.DataFrame:
    df = df.copy()
    df["subject_id"] = df["subject_id"].map(subject_map)
    df["hadm_id"] = df["hadm_id"].map(hadm_map)
    if "stay_id" in df.columns:
        df["stay_id"] = df["stay_id"].map(stay_map)
    # chartdate is date-only → place at end of day to avoid temporal leakage
    if "chartdate" in df.columns:
        df["chartdate"] = (
            coerce_datetime(df["chartdate"], "procedures_icd.chartdate")
            + timedelta(hours=23, minutes=59, seconds=59)
        )
    return df


def transform_inputevents(df: pd.DataFrame, subject_map: dict, hadm_map: dict, stay_map: dict) -> pd.DataFrame:
    df = df.copy()
    df["subject_id"] = df["subject_id"].map(subject_map)
    df["hadm_id"] = df["hadm_id"].map(hadm_map)
    # SYN-ICU uses icustay_id instead of stay_id
    if "icustay_id" in df.columns:
        df["stay_id"] = df["icustay_id"].map(stay_map)
        df = df.drop(columns=["icustay_id"])
    elif "stay_id" in df.columns:
        df["stay_id"] = df["stay_id"].map(stay_map)
    for col in ["starttime", "endtime", "storetime"]:
        if col in df.columns:
            df[col] = coerce_datetime(df[col], f"inputevents.{col}")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    return df


def transform_outputevents(df: pd.DataFrame, subject_map: dict, hadm_map: dict, stay_map: dict) -> pd.DataFrame:
    df = df.copy()
    df["subject_id"] = df["subject_id"].map(subject_map)
    df["hadm_id"] = df["hadm_id"].map(hadm_map)
    if "icustay_id" in df.columns:
        df["stay_id"] = df["icustay_id"].map(stay_map)
        df = df.drop(columns=["icustay_id"])
    elif "stay_id" in df.columns:
        df["stay_id"] = df["stay_id"].map(stay_map)
    for col in ["charttime", "storetime"]:
        if col in df.columns:
            df[col] = coerce_datetime(df[col], f"outputevents.{col}")
    if "value" in df.columns:
        df["value"] = df["value"].astype(str)
    return df


def transform_procedureevents(df: pd.DataFrame, subject_map: dict, hadm_map: dict, stay_map: dict) -> pd.DataFrame:
    df = df.copy()
    df["subject_id"] = df["subject_id"].map(subject_map)
    df["hadm_id"] = df["hadm_id"].map(hadm_map)
    if "stay_id" in df.columns:
        df["stay_id"] = df["stay_id"].map(stay_map)
    for col in ["starttime", "endtime", "storetime"]:
        if col in df.columns:
            df[col] = coerce_datetime(df[col], f"procedureevents.{col}")
    return df


def transform_emar(df: pd.DataFrame, subject_map: dict, hadm_map: dict, stay_map: dict) -> pd.DataFrame:
    df = df.copy()
    df["subject_id"] = df["subject_id"].map(subject_map)
    df["hadm_id"] = df["hadm_id"].map(hadm_map)
    if "stay_id" in df.columns:
        df["stay_id"] = df["stay_id"].map(stay_map)
    for col in ["charttime", "storetime"]:
        if col in df.columns:
            df[col] = coerce_datetime(df[col], f"emar.{col}")
    return df


def transform_emar_detail(df: pd.DataFrame, subject_map: dict, hadm_map: dict, stay_map: dict) -> pd.DataFrame:
    df = df.copy()
    df["subject_id"] = df["subject_id"].map(subject_map)
    df["hadm_id"] = df["hadm_id"].map(hadm_map)
    if "stay_id" in df.columns:
        df["stay_id"] = df["stay_id"].map(stay_map)
    return df


def transform_d_items(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy()


def transform_d_labitems(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

XLSX_FILES = [
    "syn_patients",
    "syn_admissions",
    "syn_transfers",
    "syn_icustays",
    "syn_chartevents",
    "syn_labevents",
    "syn_diagnoses_icd",
    "syn_procedures_icd",
    "syn_inputevents",
    "syn_outputevents",
    "syn_procedureevents",
    "syn_emar",
    "syn_emar_detail",
    "syn_d_items",
    "syn_d_labitems",
]


def run(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load all raw xlsx files
    print("Loading raw xlsx files...")
    raw: dict = {}
    for name in XLSX_FILES:
        path = input_dir / f"{name}.xlsx"
        if path.exists():
            raw[name] = read_xlsx(path)
        elif name in CRITICAL_TABLES:
            raise FileNotFoundError(
                f"Critical table '{name}.xlsx' not found in {input_dir}. "
                "This table is required — the MEDS output would be invalid without it."
            )
        else:
            print(f"  WARNING: {name}.xlsx not found (optional table), skipping.")

    # 2. Validate required columns before any transformation
    print("Validating input schemas...")
    for name, df in raw.items():
        required = EXPECTED_COLUMNS.get(name, set())
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Table '{name}' is missing required columns: {sorted(missing)}. "
                f"Found columns: {sorted(df.columns)}"
            )
    print("  Schema validation passed.")

    # 3. Build UUID → int64 maps from all tables
    print("Building ID maps...")
    subject_map, hadm_map, stay_map = build_all_id_maps(raw)

    # 4. Transform and write each table
    print("Transforming tables...")
    transforms = {
        "syn_patients":        lambda df: transform_patients(df, subject_map),
        "syn_admissions":      lambda df: transform_admissions(df, subject_map, hadm_map),
        "syn_transfers":       lambda df: transform_transfers(df, subject_map, hadm_map),
        "syn_icustays":        lambda df: transform_icustays(df, subject_map, hadm_map, stay_map),
        "syn_chartevents":     lambda df: transform_chartevents(df, subject_map, hadm_map, stay_map),
        "syn_labevents":       lambda df: transform_labevents(df, subject_map, hadm_map, stay_map),
        "syn_diagnoses_icd":   lambda df: transform_diagnoses_icd(df, subject_map, hadm_map, stay_map),
        "syn_procedures_icd":  lambda df: transform_procedures_icd(df, subject_map, hadm_map, stay_map),
        "syn_inputevents":     lambda df: transform_inputevents(df, subject_map, hadm_map, stay_map),
        "syn_outputevents":    lambda df: transform_outputevents(df, subject_map, hadm_map, stay_map),
        "syn_procedureevents": lambda df: transform_procedureevents(df, subject_map, hadm_map, stay_map),
        "syn_emar":            lambda df: transform_emar(df, subject_map, hadm_map, stay_map),
        "syn_emar_detail":     lambda df: transform_emar_detail(df, subject_map, hadm_map, stay_map),
        "syn_d_items":         lambda df: transform_d_items(df),
        "syn_d_labitems":      lambda df: transform_d_labitems(df),
    }

    for name, df in raw.items():
        if name in transforms:
            print(f"  transforming {name}...")
            out = transforms[name](df)
            out_path = output_dir / f"{name}.parquet"
            out.to_parquet(out_path, index=False)
            print(f"    -> {out_path.name} ({len(out)} rows)")

    print("Pre-MEDS done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-MEDS transformation for K-MIMIC SYN-ICU")
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory with raw .xlsx files")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for intermediate .parquet files")
    args = parser.parse_args()
    run(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()

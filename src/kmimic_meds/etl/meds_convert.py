"""
Standalone MEDS conversion pipeline for K-MIMIC SYN-ICU.

Bypasses MEDS-Extract CLI entirely (which has Windows compatibility issues).
Reads intermediate Parquet files produced by pre_meds.py and produces a
fully MEDS-compliant dataset directly using pandas + pyarrow.

Output structure:
    data/output/
    ├── data/
    │   ├── train/0.parquet
    │   ├── tuning/0.parquet
    │   └── held_out/0.parquet
    └── metadata/
        ├── codes.parquet
        ├── dataset.json
        └── subject_splits.parquet
"""

import json
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# MEDS schema
# ---------------------------------------------------------------------------

MEDS_SCHEMA = pa.schema([
    pa.field("subject_id", pa.int64()),
    pa.field("time", pa.timestamp("us")),
    pa.field("code", pa.string()),
    pa.field("numeric_value", pa.float32()),
])

CODES_SCHEMA = pa.schema([
    pa.field("code", pa.string()),
    pa.field("description", pa.string()),
    pa.field("parent_codes", pa.list_(pa.string())),
])

SPLITS_SCHEMA = pa.schema([
    pa.field("subject_id", pa.int64()),
    pa.field("split", pa.string()),
])

# Values considered as empty / missing
_EMPTY = {"", "nan", "None", "UNK", "NaN", "none", "null", "NULL", "<NA>"}

# Mapping of Korean/non-standard units to UCUM-aligned equivalents.
# Note: 회/min and 회/분 both mean "times per minute" — UCUM form is /min.
UNIT_MAP = {
    "회/min":    "/min",
    "회/분":     "/min",
    "℃":        "Cel",
    "㎍/dL":    "ug/dL",
    "㎍/mL":    "ug/mL",
    "㎍/L":     "ug/L",
    "㎎/dL":    "mg/dL",
    "㎎/L":     "mg/L",
    "㎝":       "cm",
    "㎜":       "mm",
    "㎏":       "kg",
    "㎖":       "mL",
    "㎕":       "uL",
    "μg/dL":   "ug/dL",
    "μg/mL":   "ug/mL",
    "μg/L":    "ug/L",
    "μmol/L":  "umol/L",
    "μU/mL":   "uU/mL",
    "/㎕":      "per_uL",
    "μℓ":      "uL",
    "/μℓ":     "per_uL",
    "×10^6/㎕": "x10e6/uL",
    "×10³/㎕":  "x10e3/uL",
    "×10^3/㎕": "x10e3/uL",
    "x10^6/㎕": "x10e6/uL",
    "x10^3/㎕": "x10e3/uL",
    "L%/R%":   "L%/R%",
}


def normalize_unit(unit):
    """Normalize a unit string to a standard equivalent if known, otherwise return as-is."""
    if unit is None or not isinstance(unit, str) or unit.strip() in _EMPTY:
        return None
    return UNIT_MAP.get(unit.strip(), unit.strip())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_code(*parts):
    """
    Builds a MEDS code by joining non-empty parts with //.

    Example:
        make_code("CHARTEVENT", "001C_102", "mmHg") -> "CHARTEVENT//001C_102//mmHg"
        make_code("HOSPITAL_ADMISSION", "nan", "Home") -> "HOSPITAL_ADMISSION//Home"
    """
    clean_parts = [
        str(p).strip()
        for p in parts
        if p is not None and str(p).strip() not in _EMPTY
    ]
    return "//".join(clean_parts) if clean_parts else "UNKNOWN"


def clean_col(series):
    """Replaces empty/nan values with None in a Series."""
    s = series.astype(str).str.strip()
    return s.where(~s.isin(_EMPTY), other=None)


def _clean_part(series: pd.Series) -> pd.Series:
    """
    Vectorized: return series with empty/nan-like strings replaced by pd.NA,
    suitable for use in vectorized code building.
    """
    s = series.astype(str).str.strip()
    return s.where(~s.isin(_EMPTY), other=None)


def _build_code(df: pd.DataFrame, prefix: str, *col_names: str) -> pd.Series:
    """
    Vectorized code builder: concatenates prefix with optional column values using //.
    Skips null/empty values. Returns a Series of MEDS code strings.

    Example:
        _build_code(df, "HOSPITAL_ADMISSION", "admission_type", "admission_location")
        → "HOSPITAL_ADMISSION//Emergency department//..."
    """
    code = pd.Series(prefix, index=df.index, dtype=str)
    for col in col_names:
        if col not in df.columns:
            continue
        part = _clean_part(df[col])
        has_part = part.notna()
        code = np.where(has_part, code + "//" + part.fillna(""), code)
        code = pd.Series(code, index=df.index)
    return code


# ---------------------------------------------------------------------------
# Event extractors — fully vectorized
# Each function returns a DataFrame [subject_id, time, code, numeric_value]
# ---------------------------------------------------------------------------

def extract_patients(df, deathtime_map=None):
    """
    Extracts from syn_patients:
    - GENDER//sex (static, time=NaT)
    - MEDS_BIRTH at year_of_birth-01-01 00:00:01
    - MEDS_DEATH at dod (or precise deathtime from admissions if available via deathtime_map)

    Parameters
    ----------
    deathtime_map : dict, optional
        {subject_id: deathtime} built from syn_admissions.deathtime.
        When available, the precise in-hospital deathtime overrides the
        date-only dod field, eliminating the need for a temporal tolerance.
    """
    df = df[df["subject_id"].notna()].copy()
    results = []

    # --- birth ---
    birth = df[["subject_id", "year_of_birth"]].copy()
    birth = birth[birth["year_of_birth"].notna()]
    birth = birth[~birth["year_of_birth"].astype(str).isin(_EMPTY)]
    birth["time"] = pd.to_datetime(
        birth["year_of_birth"].astype(str) + "-01-01 00:00:01",
        errors="coerce",
        utc=False,
    ).astype("datetime64[us]")
    birth["code"] = "MEDS_BIRTH"
    birth["numeric_value"] = None
    results.append(birth[["subject_id", "time", "code", "numeric_value"]])

    # --- gender (static) ---
    gender = df[["subject_id", "sex"]].copy()
    gender = gender[gender["sex"].notna()]
    gender["sex"] = clean_col(gender["sex"])
    gender = gender[gender["sex"].notna()]
    gender["time"] = pd.NaT
    gender["time"] = gender["time"].astype("datetime64[us]")
    gender["code"] = "GENDER//" + gender["sex"]
    gender["numeric_value"] = None
    results.append(gender[["subject_id", "time", "code", "numeric_value"]])

    # --- death ---
    death = df[["subject_id", "dod"]].copy()
    death = death[death["dod"].notna()]
    death["time"] = pd.to_datetime(death["dod"], errors="coerce").astype("datetime64[us]")
    death = death[death["time"].notna()]

    # Use precise in-hospital deathtime when available (sub-day precision,
    # avoids the 48-hour tolerance workaround for date-only dod).
    if deathtime_map:
        precise = death["subject_id"].map(deathtime_map)
        precise = pd.to_datetime(precise, errors="coerce").astype("datetime64[us]")
        death["time"] = precise.where(precise.notna(), death["time"])

    death["code"] = "MEDS_DEATH"
    death["numeric_value"] = None
    results.append(death[["subject_id", "time", "code", "numeric_value"]])

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def extract_admissions(df):
    """
    Extracts from syn_admissions:
    - HOSPITAL_ADMISSION//type//location (at admittime)
    - INSURANCE, MARITAL_STATUS, ETHNICITY demographics (at admittime)
    - HOSPITAL_DISCHARGE//location (at dischtime)
    - ED_REGISTRATION / ED_OUT

    Note: MEDS_DEATH is handled in extract_patients using syn_patients.dod
    (with optional override from the precise admissions.deathtime passed via deathtime_map).
    """
    results = []

    # --- admission ---
    adm = df[df["admittime"].notna()].copy()
    adm["code"] = _build_code(adm, "HOSPITAL_ADMISSION", "admission_type", "admission_location")
    adm["numeric_value"] = None
    results.append(adm[["subject_id", "admittime", "code", "numeric_value"]].rename(
        columns={"admittime": "time"}))

    # --- demographics at admission ---
    for col, prefix in [("insurance", "INSURANCE"),
                        ("marital_status", "MARITAL_STATUS"),
                        ("ethnicity", "ETHNICITY")]:
        if col in df.columns:
            demo = df[df["admittime"].notna() & df[col].notna()].copy()
            demo["val"] = clean_col(demo[col])
            demo = demo[demo["val"].notna()]
            demo["code"] = prefix + "//" + demo["val"]
            demo["numeric_value"] = None
            results.append(demo[["subject_id", "admittime", "code", "numeric_value"]].rename(
                columns={"admittime": "time"}))

    # --- discharge ---
    dis = df[df["dischtime"].notna()].copy()
    dis["code"] = _build_code(dis, "HOSPITAL_DISCHARGE", "discharge_location")
    dis["numeric_value"] = None
    results.append(dis[["subject_id", "dischtime", "code", "numeric_value"]].rename(
        columns={"dischtime": "time"}))

    # --- emergency department ---
    for col, code in [("edregtime", "ED_REGISTRATION"), ("edouttime", "ED_OUT")]:
        if col in df.columns:
            ed = df[df[col].notna()].copy()
            ed["code"] = code
            ed["numeric_value"] = None
            results.append(ed[["subject_id", col, "code", "numeric_value"]].rename(
                columns={col: "time"}))

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def extract_icustays(df):
    """
    Extracts from syn_icustays:
    - ICU_ADMISSION//careunit (at intime)
    - ICU_DISCHARGE//careunit (at outtime)
    """
    results = []

    adm = df[df["intime"].notna()].copy()
    adm["code"] = _build_code(adm, "ICU_ADMISSION", "first_careunit")
    adm["numeric_value"] = None
    results.append(adm[["subject_id", "intime", "code", "numeric_value"]].rename(
        columns={"intime": "time"}))

    dis = df[df["outtime"].notna()].copy()
    dis["code"] = _build_code(dis, "ICU_DISCHARGE", "last_careunit")
    dis["numeric_value"] = None
    results.append(dis[["subject_id", "outtime", "code", "numeric_value"]].rename(
        columns={"outtime": "time"}))

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def extract_chartevents(df):
    """
    Extracts from syn_chartevents.
    Vectorized: builds CHARTEVENT//itemid//uom codes in a single pass.
    """
    df = df[df["charttime"].notna()].copy()

    itemid = df["itemid"].astype(str).str.strip()
    # astype(object) after map ensures None values have object dtype, not an
    # arrow null type — the latter causes PyArrow errors in empty-slice concat.
    uom = df["valueuom"].map(normalize_unit).astype(object)

    has_uom = uom.notna() & ~uom.isin(_EMPTY)
    df["code"] = "CHARTEVENT//" + itemid
    df.loc[has_uom, "code"] = "CHARTEVENT//" + itemid[has_uom] + "//" + uom[has_uom].astype(str)

    df["numeric_value"] = pd.to_numeric(df["valuenum"], errors="coerce")

    return df[["subject_id", "charttime", "code", "numeric_value"]].rename(
        columns={"charttime": "time"})


def extract_labevents(df):
    """
    Extracts from syn_labevents.
    Vectorized: builds LAB//itemid//uom codes in a single pass.
    """
    df = df[df["charttime"].notna()].copy()

    itemid = df["itemid"].astype(str).str.strip()
    # astype(object) ensures None from normalize_unit has object dtype, not arrow null.
    uom = df["valueuom"].map(normalize_unit).astype(object)

    # Both guards required: notna() catches None returned by normalize_unit;
    # isin(_EMPTY) catches passthrough empty strings from the source.
    has_uom = uom.notna() & ~uom.isin(_EMPTY)
    df["code"] = "LAB//" + itemid
    df.loc[has_uom, "code"] = "LAB//" + itemid[has_uom] + "//" + uom[has_uom].astype(str)

    df["numeric_value"] = pd.to_numeric(df["valuenum"], errors="coerce")

    return df[["subject_id", "charttime", "code", "numeric_value"]].rename(
        columns={"charttime": "time"})


def extract_diagnoses_icd(df, admissions_df=None):
    """
    Extracts from syn_diagnoses_icd.
    Joins with admissions to retrieve admittime, providing a real timestamp
    to each diagnosis instead of NaT.
    """
    hadm_to_time = {}
    if admissions_df is not None:
        hadm_to_time = dict(zip(
            admissions_df["hadm_id"],
            admissions_df["admittime"],
        ))

    df = df.copy()
    df["icd_code"] = clean_col(df["icd_code"].astype(str))
    df = df[df["icd_code"].notna()]

    # Vectorized code building.
    # dtype=object on fallback ensures empty-slice string concat doesn't
    # fail when numpy tries to add a string literal to a float64 array.
    icd_ver = (
        _clean_part(df["icd_version"].astype(str))
        if "icd_version" in df.columns
        else pd.Series(dtype=object, index=df.index)
    )
    has_ver = icd_ver.notna()
    df["code"] = "DIAGNOSIS//" + df["icd_code"]
    df.loc[has_ver, "code"] = "DIAGNOSIS//" + icd_ver[has_ver].astype(str) + "//" + df.loc[has_ver, "icd_code"]

    df["time"] = pd.to_datetime(df["hadm_id"].map(hadm_to_time), errors="coerce")
    df["numeric_value"] = None

    return df[["subject_id", "time", "code", "numeric_value"]]


def extract_procedures_icd(df):
    """
    Extracts from syn_procedures_icd.
    chartdate is already resolved to 23:59:59 by pre_meds.py.
    """
    df = df.copy()
    df["icd_code"] = clean_col(df["icd_code"].astype(str))
    df = df[df["icd_code"].notna()]

    icd_ver = (
        _clean_part(df["icd_version"].astype(str))
        if "icd_version" in df.columns
        else pd.Series(dtype=object, index=df.index)
    )
    has_ver = icd_ver.notna()
    df["code"] = "PROCEDURE_ICD//" + df["icd_code"]
    df.loc[has_ver, "code"] = "PROCEDURE_ICD//" + icd_ver[has_ver].astype(str) + "//" + df.loc[has_ver, "icd_code"]

    df["time"] = pd.to_datetime(df["chartdate"], errors="coerce")
    df["numeric_value"] = None

    return df[["subject_id", "time", "code", "numeric_value"]]


def extract_inputevents(df):
    """
    Extracts from syn_inputevents.
    Vectorized: builds INPUT_START//itemid//uom codes.
    """
    df = df[df["starttime"].notna()].copy()

    itemid = df["itemid"].astype(str).str.strip()
    uom = df["amountuom"].map(normalize_unit).astype(object)

    has_uom = uom.notna() & ~uom.isin(_EMPTY)
    df["code"] = "INPUT_START//" + itemid
    df.loc[has_uom, "code"] = "INPUT_START//" + itemid[has_uom] + "//" + uom[has_uom].astype(str)

    df["numeric_value"] = pd.to_numeric(df["amount"], errors="coerce")

    return df[["subject_id", "starttime", "code", "numeric_value"]].rename(
        columns={"starttime": "time"})


def extract_outputevents(df):
    """
    Extracts from syn_outputevents.
    Vectorized: builds OUTPUT//itemid//uom codes.
    """
    df = df[df["charttime"].notna()].copy()

    itemid = df["itemid"].astype(str).str.strip()
    uom = df["valueuom"].map(normalize_unit).astype(object)

    has_uom = uom.notna() & ~uom.isin(_EMPTY)
    df["code"] = "OUTPUT//" + itemid
    df.loc[has_uom, "code"] = "OUTPUT//" + itemid[has_uom] + "//" + uom[has_uom].astype(str)

    df["numeric_value"] = pd.to_numeric(df["value"], errors="coerce")

    return df[["subject_id", "charttime", "code", "numeric_value"]].rename(
        columns={"charttime": "time"})


def extract_emar(df):
    """
    Extracts from syn_emar.
    Vectorized: builds MEDICATION//itemid codes.
    """
    df = df[df["charttime"].notna()].copy()
    df["code"] = "MEDICATION//" + df["itemid"].astype(str).str.strip()
    df["numeric_value"] = None

    return df[["subject_id", "charttime", "code", "numeric_value"]].rename(
        columns={"charttime": "time"})


def extract_procedureevents(df):
    """
    Extracts from syn_procedureevents.
    Each procedure generates two events:
    - PROCEDURE_START//itemid at starttime
    - PROCEDURE_END//itemid at endtime (if present)
    """
    results = []

    start = df[df["starttime"].notna()].copy()
    start["code"] = "PROCEDURE_START//" + start["itemid"].astype(str).str.strip()
    start["numeric_value"] = None
    results.append(start[["subject_id", "starttime", "code", "numeric_value"]].rename(
        columns={"starttime": "time"}))

    end = df[df["endtime"].notna()].copy()
    end["code"] = "PROCEDURE_END//" + end["itemid"].astype(str).str.strip()
    end["numeric_value"] = None
    results.append(end[["subject_id", "endtime", "code", "numeric_value"]].rename(
        columns={"endtime": "time"}))

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


# ---------------------------------------------------------------------------
# Metadata builders
# ---------------------------------------------------------------------------

def build_codes_parquet(events_df, intermediate_dir=None):
    """
    Build codes.parquet: one row per unique code with optional description
    and parent_codes (EDI) enriched from d_labitems and d_items.
    Uses vectorized merges instead of iterrows.
    """
    codes = sorted(events_df["code"].dropna().unique())
    df = pd.DataFrame({
        "code": codes,
        "description": [None] * len(codes),
        "parent_codes": [None] * len(codes),
    })

    if intermediate_dir is None:
        return df

    label_map: dict[str, str] = {}
    edi_map: dict[str, str] = {}

    # syn_d_labitems — labels + EDI parent codes
    d_lab_path = intermediate_dir / "syn_d_labitems.parquet"
    if d_lab_path.exists():
        d_lab = pd.read_parquet(d_lab_path)
        d_lab["itemid"] = d_lab["itemid"].astype(str).str.strip()

        if "label" in d_lab.columns:
            d_lab["label"] = d_lab["label"].astype(str).str.strip()
            valid_labels = d_lab[~d_lab["label"].isin(_EMPTY) & d_lab["label"].notna()]
            label_map.update(zip(valid_labels["itemid"], valid_labels["label"]))

        if "edi_code" in d_lab.columns:
            d_lab["edi_code"] = d_lab["edi_code"].astype(str).str.strip()
            valid_edi = d_lab[
                ~d_lab["edi_code"].isin(_EMPTY | {"KMM90000"}) & d_lab["edi_code"].notna()
            ]
            edi_map.update(zip(valid_edi["itemid"], "EDI/" + valid_edi["edi_code"]))

    # syn_d_items — labels only (d_items takes precedence over d_labitems for labels)
    d_items_path = intermediate_dir / "syn_d_items.parquet"
    if d_items_path.exists():
        d_items = pd.read_parquet(d_items_path)
        d_items["itemid"] = d_items["itemid"].astype(str).str.strip()
        if "label" in d_items.columns:
            d_items["label"] = d_items["label"].astype(str).str.strip()
            valid_items = d_items[~d_items["label"].isin(_EMPTY) & d_items["label"].notna()]
            label_map.update(zip(valid_items["itemid"], valid_items["label"]))

    # Vectorized lookup: extract itemid (parts[1]) from code and map
    code_parts = df["code"].str.split("//", expand=True)
    itemids = code_parts[1] if code_parts.shape[1] > 1 else pd.Series(None, index=df.index)

    df["description"] = itemids.map(label_map)
    df["parent_codes"] = itemids.map(edi_map).map(
        lambda v: [v] if pd.notna(v) else None
    )

    return df


def build_subject_splits(subject_ids, train_frac=0.8, tuning_frac=0.1):
    """
    Assigns each patient to a train/tuning/held_out split.
    Uses numpy.random.default_rng(42) for reproducibility across NumPy versions.
    """
    rng = np.random.default_rng(42)
    ids = np.array(sorted(subject_ids))
    ids = rng.permutation(ids)
    n = len(ids)
    n_train = int(n * train_frac)
    n_tuning = int(n * tuning_frac)

    split_labels = (
        ["train"] * n_train
        + ["tuning"] * n_tuning
        + ["held_out"] * (n - n_train - n_tuning)
    )
    return pd.DataFrame({"subject_id": ids, "split": split_labels})


def build_dataset_json(output_dir, dataset_name, dataset_version):
    meta = {
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "etl_name": "kmimic-meds",
        "etl_version": "0.2.0",
        "meds_version": "0.3.3",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "metadata" / "dataset.json").write_text(json.dumps(meta, indent=2))


# ---------------------------------------------------------------------------
# Conversion to PyArrow with strict MEDS schema
# ---------------------------------------------------------------------------

def to_meds_table(df):
    """
    Converts a pandas DataFrame to a PyArrow table compliant with MEDS schema.
    Applies explicit type casts before passing to PyArrow (safe=True) so that
    schema violations raise immediately rather than being silently coerced.
    Sorts by subject_id then time (NaT first for static events).
    """
    if df.empty:
        df = pd.DataFrame(columns=["subject_id", "time", "code", "numeric_value"])

    df = df.copy()

    # Explicit casts — errors surface here, not silently in PyArrow
    df["subject_id"] = pd.to_numeric(df["subject_id"], errors="coerce").astype("Int64")
    df["time"] = pd.to_datetime(df["time"], errors="coerce").astype("datetime64[us]")
    df["code"] = df["code"].astype(str)
    df["numeric_value"] = pd.to_numeric(df["numeric_value"], errors="coerce").astype("float32")

    # Drop rows where subject_id is null (should never happen after filtering in extractors)
    n_before = len(df)
    df = df[df["subject_id"].notna()].reset_index(drop=True)
    if len(df) < n_before:
        print(f"  WARNING: dropped {n_before - len(df)} rows with null subject_id")

    df = df.sort_values(["subject_id", "time"], na_position="first").reset_index(drop=True)

    return pa.Table.from_pandas(df, schema=MEDS_SCHEMA, safe=True)


def write_parquet(table, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(intermediate_dir: Path, output_dir: Path, dataset_name: str, dataset_version: str):
    import time
    t0 = time.time()

    print("Loading intermediate Parquet files...")
    all_events = []

    # Load admissions early — needed for diagnosis timestamp join and deathtime map
    admissions_df = None
    admissions_path = intermediate_dir / "syn_admissions.parquet"
    if admissions_path.exists():
        admissions_df = pd.read_parquet(admissions_path)
        print(f"  loaded syn_admissions.parquet ({len(admissions_df)} rows)")

    # Build subject_id → precise deathtime map from admissions.
    # Overrides date-only dod in extract_patients, removing the need for a
    # 48-hour temporal tolerance for patients who died in-hospital.
    deathtime_map: dict = {}
    if admissions_df is not None and "deathtime" in admissions_df.columns:
        dt = admissions_df[admissions_df["deathtime"].notna()][["subject_id", "deathtime"]]
        deathtime_map = dict(zip(dt["subject_id"], dt["deathtime"]))
        print(f"  built deathtime_map: {len(deathtime_map)} patients with precise death timestamps")

    extractors = {
        "syn_patients":        (extract_patients,        {"deathtime_map": deathtime_map}),
        "syn_admissions":      (extract_admissions,      {}),
        "syn_icustays":        (extract_icustays,        {}),
        "syn_chartevents":     (extract_chartevents,     {}),
        "syn_labevents":       (extract_labevents,       {}),
        "syn_diagnoses_icd":   (extract_diagnoses_icd,   {"admissions_df": admissions_df}),
        "syn_procedures_icd":  (extract_procedures_icd,  {}),
        "syn_procedureevents": (extract_procedureevents, {}),
        "syn_inputevents":     (extract_inputevents,     {}),
        "syn_outputevents":    (extract_outputevents,    {}),
        "syn_emar":            (extract_emar,            {}),
    }

    for name, (extractor, kwargs) in extractors.items():
        path = intermediate_dir / f"{name}.parquet"
        if not path.exists():
            print(f"  WARNING: {name}.parquet not found, skipping.")
            continue
        print(f"  extracting events from {name}.parquet...")
        t1 = time.time()
        df = pd.read_parquet(path)
        events = extractor(df, **kwargs)
        if events is not None and not events.empty:
            all_events.append(events)
            print(f"    -> {len(events)} events ({time.time() - t1:.1f}s)")

    print("Merging all events...")
    events_df = pd.concat(all_events, ignore_index=True)
    print(f"  total: {len(events_df)} events, {events_df['subject_id'].nunique()} subjects")

    print("Building subject splits...")
    subject_ids = events_df["subject_id"].dropna().unique().astype(int).tolist()
    splits_df = build_subject_splits(subject_ids)

    print("Writing MEDS data files...")
    split_id_sets = {
        row["split"]: set() for _, row in splits_df.iterrows()
    }
    for _, row in splits_df.iterrows():
        split_id_sets[row["split"]].add(row["subject_id"])

    for split_name in ["train", "tuning", "held_out"]:
        split_ids = split_id_sets.get(split_name, set())
        split_events = events_df[events_df["subject_id"].isin(split_ids)]
        table = to_meds_table(split_events)
        out_path = output_dir / "data" / split_name / "0.parquet"
        write_parquet(table, out_path)
        print(f"  {split_name}/0.parquet — {len(split_events)} events, {len(split_ids)} subjects")

    print("Writing metadata...")
    meta_dir = output_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    codes_df = build_codes_parquet(events_df, intermediate_dir)
    codes_table = pa.Table.from_pandas(codes_df, schema=CODES_SCHEMA, safe=True)
    write_parquet(codes_table, meta_dir / "codes.parquet")
    print(f"  codes.parquet — {len(codes_df)} unique codes")

    splits_table = pa.Table.from_pandas(splits_df, schema=SPLITS_SCHEMA, safe=True)
    write_parquet(splits_table, meta_dir / "subject_splits.parquet")
    print(f"  subject_splits.parquet — {len(splits_df)} subjects")

    build_dataset_json(output_dir, dataset_name, dataset_version)
    print("  dataset.json")

    print(f"MEDS conversion done in {time.time() - t0:.1f}s")
    print(f"Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Standalone MEDS conversion for K-MIMIC SYN-ICU")
    parser.add_argument("--intermediate_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--dataset_name", type=str, default="K-MIMIC-MEDS")
    parser.add_argument("--dataset_version", type=str, default="0.2.0")
    args = parser.parse_args()
    run(args.intermediate_dir, args.output_dir, args.dataset_name, args.dataset_version)


if __name__ == "__main__":
    main()

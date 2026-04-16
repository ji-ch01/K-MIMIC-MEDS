"""
Standalone MEDS conversion pipeline for K-MIMIC SYN-ICU.

Bypasses MEDS-Extract CLI entirely (which has Windows compatibility issues).
Reads intermediate Parquet files produced by pre_meds.py and produces a
fully MEDS-compliant dataset directly using pandas + pyarrow.

Improvements over v0.1.0:
- Vectorized extractors (no more iterrows) — 10-50x faster
- Timestamps for diagnoses_icd via join with admissions

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

# Values considered as empty
_EMPTY = {"", "nan", "None", "UNK", "NaN", "none", "null", "NULL"}

# Mapping of Korean/non-standard units to standard equivalents
UNIT_MAP = {
    "회/min": "per_min",
    "회/분": "per_min",
    "℃": "Cel",
    "㎍/dL": "ug/dL",
    "㎍/mL": "ug/mL",
    "㎍/L": "ug/L",
    "㎎/dL": "mg/dL",
    "㎎/L": "mg/L",
    "㎝": "cm",
    "㎜": "mm",
    "㎏": "kg",
    "㎖": "mL",
    "㎕": "uL",
    "μg/dL": "ug/dL",
    "μg/mL": "ug/mL",
    "μg/L": "ug/L",
    "μmol/L": "umol/L",
    "μU/mL": "uU/mL",
    "㎕": "uL",
    "/㎕": "per_uL",
    "μℓ": "uL",
    "/μℓ": "per_uL",
    "×10^6/㎕": "x10e6/uL",
    "×10³/㎕": "x10e3/uL",
    "×10^3/㎕": "x10e3/uL",
    "x10^6/㎕": "x10e6/uL",
    "x10^3/㎕": "x10e3/uL",
    "L%/R%": "L%/R%",
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
    Builds a MEDS code by joining parts with //.
    Empty or nan parts are ignored.

    Example:
        make_code("CHARTEVENT", "001C_102", "times/min") -> "CHARTEVENT//001C_102//times/min"
        make_code("HOSPITAL_ADMISSION", "nan", "Home") -> "HOSPITAL_ADMISSION//Home"
    """
    clean_parts = [
        str(p).strip()
        for p in parts
        if p is not None and str(p).strip() not in _EMPTY
    ]
    return "//".join(clean_parts) if clean_parts else "UNKNOWN"


def vec_make_code(df, *col_names, prefix=None):
    """
    Vectorized version of make_code to build a column of codes.
    Takes a DataFrame and a list of column names.
    Returns a Series of MEDS codes.

    Example:
        vec_make_code(df, "itemid", "valueuom", prefix="CHARTEVENT")
        -> "CHARTEVENT//001C_102//mmHg"
    """
    parts = []
    if prefix:
        parts.append(pd.Series(prefix, index=df.index))

    for col in col_names:
        if col in df.columns:
            s = df[col].astype(str).str.strip()
            s = s.where(~s.isin(_EMPTY), other=None)
            parts.append(s)

    if not parts:
        return pd.Series("UNKNOWN", index=df.index)

    result = parts[0].fillna("")
    for p in parts[1:]:
        if p is not None:
            mask = p.notna() & (p != "None") & (p != "nan")
            result = result.where(
                ~mask,
                result + "//" + p.fillna("")
            )

    # Clean leading or trailing //
    result = result.str.strip("//")
    result = result.replace("", "UNKNOWN")
    return result


def clean_col(series):
    """Replaces empty/nan values with None in a Series."""
    s = series.astype(str).str.strip()
    return s.where(~s.isin(_EMPTY), other=None)


def to_meds_df(df, subject_col="subject_id", time_col="time",
                code_col="code", value_col="numeric_value"):
    """
    Selects and renames columns to produce a MEDS DataFrame.
    Returns a DataFrame with exactly the MEDS columns.
    """
    cols = {
        subject_col: "subject_id",
        time_col: "time",
        code_col: "code",
    }
    result = df[list(cols.keys())].rename(columns=cols)

    if value_col and value_col in df.columns:
        result["numeric_value"] = pd.to_numeric(df[value_col], errors="coerce")
    else:
        result["numeric_value"] = None

    return result[["subject_id", "time", "code", "numeric_value"]]


# ---------------------------------------------------------------------------
# Event extractors — vectorized version
# Each function returns a DataFrame [subject_id, time, code, numeric_value]
# ---------------------------------------------------------------------------

def extract_patients(df):
    df = df[df["subject_id"].notna()].copy()
    results = []

    # --- birth ---
    birth = df[["subject_id", "year_of_birth"]].copy()
    birth = birth[birth["year_of_birth"].notna()]
    birth = birth[~birth["year_of_birth"].astype(str).isin(_EMPTY | {"<NA>"})]
    birth["time"] = pd.to_datetime(
        birth["year_of_birth"].astype(str) + "-01-01 00:00:01",
        errors="coerce",
        utc=False
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
    death["code"] = "MEDS_DEATH"
    death["numeric_value"] = None
    results.append(death[["subject_id", "time", "code", "numeric_value"]])

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def extract_admissions(df):
    """
    Extracts from syn_admissions:
    - HOSPITAL_ADMISSION (with type and location)
    - demographic data at admission (INSURANCE, MARITAL_STATUS, ETHNICITY)
    - HOSPITAL_DISCHARGE
    - MEDS_DEATH (if deathtime is present)
    - ED_REGISTRATION / ED_OUT
    """
    results = []

    # --- admission ---
    adm = df[df["admittime"].notna()].copy()
    adm["code"] = adm.apply(
        lambda r: make_code("HOSPITAL_ADMISSION", r.get("admission_type"), r.get("admission_location")),
        axis=1
    )
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
    dis["code"] = dis.apply(
        lambda r: make_code("HOSPITAL_DISCHARGE", r.get("discharge_location")),
        axis=1
    )
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
    adm["code"] = adm.apply(
        lambda r: make_code("ICU_ADMISSION", r.get("first_careunit")), axis=1)
    adm["numeric_value"] = None
    results.append(adm[["subject_id", "intime", "code", "numeric_value"]].rename(
        columns={"intime": "time"}))

    dis = df[df["outtime"].notna()].copy()
    dis["code"] = dis.apply(
        lambda r: make_code("ICU_DISCHARGE", r.get("last_careunit")), axis=1)
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
    uom = df["valueuom"].map(normalize_unit)

    # Build code: CHARTEVENT//itemid//uom (if uom not empty)
    has_uom = ~uom.isin(_EMPTY) & uom.notna()
    df["code"] = "CHARTEVENT//" + itemid
    df.loc[has_uom, "code"] = "CHARTEVENT//" + itemid[has_uom] + "//" + uom[has_uom]
    
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
    uom = df["valueuom"].map(normalize_unit)

    has_uom = ~uom.isin(_EMPTY)
    df["code"] = "LAB//" + itemid
    df.loc[has_uom, "code"] = "LAB//" + itemid[has_uom] + "//" + uom[has_uom]

    df["numeric_value"] = pd.to_numeric(df["valuenum"], errors="coerce")

    return df[["subject_id", "charttime", "code", "numeric_value"]].rename(
        columns={"charttime": "time"})


def extract_diagnoses_icd(df, admissions_df=None):
    """
    Extracts from syn_diagnoses_icd.
    v0.2 Improvement: join with admissions to retrieve admittime
    and provide a real timestamp to each diagnosis instead of NaT.
    """
    # Build hadm_id -> admittime dictionary
    hadm_to_time = {}
    if admissions_df is not None:
        hadm_to_time = dict(zip(
            admissions_df["hadm_id"],
            admissions_df["admittime"]
        ))

    df = df.copy()

    # Filter rows without icd_code
    df["icd_code"] = clean_col(df["icd_code"].astype(str))
    df = df[df["icd_code"].notna()]

    # Build MEDS codes
    df["code"] = df.apply(
        lambda r: make_code("DIAGNOSIS", r.get("icd_version"), r.get("icd_code")),
        axis=1
    )

    # Retrieve timestamp from admissions
    df["time"] = df["hadm_id"].map(hadm_to_time)
    # If no hadm_id or not found -> NaT (static)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

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

    df["code"] = df.apply(
        lambda r: make_code("PROCEDURE_ICD", r.get("icd_version"), r.get("icd_code")),
        axis=1
    )
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
    uom = df["amountuom"].map(normalize_unit)

    has_uom = ~uom.isin(_EMPTY) & uom.notna()
    df["code"] = "INPUT_START//" + itemid
    df.loc[has_uom, "code"] = "INPUT_START//" + itemid[has_uom] + "//" + uom[has_uom]

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
    uom = df["valueuom"].map(normalize_unit)

    has_uom = uom.notna() & ~uom.isin(_EMPTY)
    df["code"] = "OUTPUT//" + itemid
    df.loc[has_uom, "code"] = "OUTPUT//" + itemid[has_uom] + "//" + uom[has_uom]

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

    # procedure start
    start = df[df["starttime"].notna()].copy()
    start["code"] = "PROCEDURE_START//" + start["itemid"].astype(str).str.strip()
    start["numeric_value"] = None
    results.append(start[["subject_id", "starttime", "code", "numeric_value"]].rename(
        columns={"starttime": "time"}))

    # procedure end
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
    codes = sorted(events_df["code"].dropna().unique())
    df = pd.DataFrame({
        "code": codes,
        "description": [None] * len(codes),
        "parent_codes": [None] * len(codes),
    })

    if intermediate_dir is not None:
        label_map = {}
        edi_map = {}

        # syn_d_labitems — labels + codes EDI
        d_lab_path = intermediate_dir / "syn_d_labitems.parquet"
        if d_lab_path.exists():
            d_lab = pd.read_parquet(d_lab_path)
            for _, r in d_lab.iterrows():
                itemid = str(r.get("itemid", "")).strip()
                label = str(r.get("label", "")).strip()
                edi = str(r.get("edi_code", "")).strip()
                if itemid and itemid not in _EMPTY:
                    if label and label not in _EMPTY:
                        label_map[itemid] = label
                    if edi and edi not in _EMPTY and edi != "KMM90000":
                        # KMM90000 = generic "non-billed" code, not useful
                        edi_map[itemid] = f"EDI/{edi}"

        # syn_d_items — labels only (no external ontology)
        d_items_path = intermediate_dir / "syn_d_items.parquet"
        if d_items_path.exists():
            d_items = pd.read_parquet(d_items_path)
            for _, r in d_items.iterrows():
                itemid = str(r.get("itemid", "")).strip()
                label = str(r.get("label", "")).strip()
                if itemid and label and itemid not in _EMPTY and label not in _EMPTY:
                    label_map[itemid] = label

        def find_description(code):
            parts = code.split("//")
            if len(parts) >= 2:
                return label_map.get(parts[1])
            return None

        def find_parent_codes(code):
            parts = code.split("//")
            if len(parts) >= 2:
                edi = edi_map.get(parts[1])
                if edi:
                    return [edi]
            return None

        df["description"] = df["code"].apply(find_description)
        df["parent_codes"] = df["code"].apply(find_parent_codes)

    return df


def build_subject_splits(subject_ids, train_frac=0.8, tuning_frac=0.1):
    """
    Assigns each patient to a train/tuning/held_out split.
    seed=42 ensures reproducibility.
    """
    import random
    random.seed(42)
    ids = sorted(subject_ids)
    random.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_frac)
    n_tuning = int(n * tuning_frac)

    splits = {}
    for i, sid in enumerate(ids):
        if i < n_train:
            splits[sid] = "train"
        elif i < n_train + n_tuning:
            splits[sid] = "tuning"
        else:
            splits[sid] = "held_out"

    return pd.DataFrame([{"subject_id": sid, "split": s} for sid, s in splits.items()])


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
    Sorts by subject_id then time (NaT first for static events).
    """
    if df.empty:
        df = pd.DataFrame(columns=["subject_id", "time", "code", "numeric_value"])

    df = df.copy()
    df["subject_id"] = pd.to_numeric(df["subject_id"], errors="coerce").astype("Int64")
    df["time"] = pd.to_datetime(df["time"], errors="coerce").astype("datetime64[us]")
    df["code"] = df["code"].astype(str)
    df["numeric_value"] = pd.to_numeric(df["numeric_value"], errors="coerce").astype("float32")

    df = df.sort_values(["subject_id", "time"], na_position="first").reset_index(drop=True)

    return pa.Table.from_pandas(df, schema=MEDS_SCHEMA, safe=False)


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

    # Load admissions early for diagnostics join
    admissions_df = None
    admissions_path = intermediate_dir / "syn_admissions.parquet"
    if admissions_path.exists():
        admissions_df = pd.read_parquet(admissions_path)
        print(f"  loaded syn_admissions.parquet ({len(admissions_df)} rows)")

    extractors = {
        "syn_patients":       (extract_patients, {}),
        "syn_admissions":     (extract_admissions, {}),
        "syn_icustays":       (extract_icustays, {}),
        "syn_chartevents":    (extract_chartevents, {}),
        "syn_labevents":      (extract_labevents, {}),
        "syn_diagnoses_icd":  (extract_diagnoses_icd, {"admissions_df": admissions_df}),
        "syn_procedures_icd": (extract_procedures_icd, {}),
        "syn_procedureevents": (extract_procedureevents, {}),
        "syn_inputevents":    (extract_inputevents, {}),
        "syn_outputevents":   (extract_outputevents, {}),
        "syn_emar":           (extract_emar, {}),
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
    for split_name in ["train", "tuning", "held_out"]:
        split_ids = set(splits_df[splits_df["split"] == split_name]["subject_id"])
        split_events = events_df[events_df["subject_id"].isin(split_ids)]
        table = to_meds_table(split_events)
        out_path = output_dir / "data" / split_name / "0.parquet"
        write_parquet(table, out_path)
        print(f"  {split_name}/0.parquet — {len(split_events)} events, {len(split_ids)} subjects")

    print("Writing metadata...")
    meta_dir = output_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    codes_df = build_codes_parquet(events_df, intermediate_dir)
    codes_table = pa.Table.from_pandas(codes_df, schema=CODES_SCHEMA, safe=False)
    write_parquet(codes_table, meta_dir / "codes.parquet")
    print(f"  codes.parquet — {len(codes_df)} unique codes")

    splits_table = pa.Table.from_pandas(splits_df, schema=SPLITS_SCHEMA, safe=False)
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
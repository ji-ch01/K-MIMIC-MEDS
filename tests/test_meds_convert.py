"""Unit tests for meds_convert.py"""

import pandas as pd
import numpy as np
import pytest
import pyarrow as pa
from kmimic_meds.etl.meds_convert import (
    make_code,
    normalize_unit,
    clean_col,
    _build_code,
    extract_patients,
    extract_admissions,
    extract_icustays,
    extract_chartevents,
    extract_labevents,
    extract_diagnoses_icd,
    extract_inputevents,
    extract_outputevents,
    extract_emar,
    extract_procedureevents,
    build_subject_splits,
    build_codes_parquet,
    to_meds_table,
    UNIT_MAP,
    MEDS_SCHEMA,
)
from kmimic_meds.etl.pre_meds import uuid_to_int


# ---------------------------------------------------------------------------
# make_code
# ---------------------------------------------------------------------------

def test_make_code_basic():
    assert make_code("CHARTEVENT", "001C_102", "mmHg") == "CHARTEVENT//001C_102//mmHg"

def test_make_code_filters_nan():
    assert make_code("HOSPITAL_ADMISSION", "nan", "Home") == "HOSPITAL_ADMISSION//Home"

def test_make_code_filters_none():
    assert make_code("LAB", None, "mg/dL") == "LAB//mg/dL"

def test_make_code_filters_empty():
    assert make_code("LAB", "", "mg/dL") == "LAB//mg/dL"

def test_make_code_all_empty():
    assert make_code("", None, "nan") == "UNKNOWN"

def test_make_code_single_part():
    assert make_code("MEDS_BIRTH") == "MEDS_BIRTH"


# ---------------------------------------------------------------------------
# normalize_unit
# ---------------------------------------------------------------------------

def test_normalize_unit_korean_frequency():
    # 회/min means "times per minute" — UCUM form is /min
    assert normalize_unit("회/min") == "/min"

def test_normalize_unit_korean_frequency_hangul():
    assert normalize_unit("회/분") == "/min"

def test_normalize_unit_celsius():
    assert normalize_unit("℃") == "Cel"

def test_normalize_unit_standard_passthrough():
    assert normalize_unit("mmHg") == "mmHg"

def test_normalize_unit_none():
    assert normalize_unit(None) is None

def test_normalize_unit_float_nan():
    assert normalize_unit(float("nan")) is None

def test_normalize_unit_empty():
    assert normalize_unit("") is None

def test_normalize_unit_unknown_kept():
    assert normalize_unit("some_unknown_unit") == "some_unknown_unit"

def test_normalize_unit_microgram():
    assert normalize_unit("㎍/dL") == "ug/dL"

def test_normalize_unit_no_duplicate_keys():
    # Sanity check: all UNIT_MAP values are unique strings (no accidental duplicates)
    values = list(UNIT_MAP.values())
    keys = list(UNIT_MAP.keys())
    assert len(keys) == len(set(keys)), "UNIT_MAP has duplicate keys"


# ---------------------------------------------------------------------------
# _build_code (vectorized code builder)
# ---------------------------------------------------------------------------

def test_build_code_single_col():
    df = pd.DataFrame({"col": ["Emergency department"]})
    result = _build_code(df, "HOSPITAL_ADMISSION", "col")
    assert result.iloc[0] == "HOSPITAL_ADMISSION//Emergency department"

def test_build_code_skips_empty():
    df = pd.DataFrame({"col": ["nan"]})
    result = _build_code(df, "ICU_ADMISSION", "col")
    assert result.iloc[0] == "ICU_ADMISSION"

def test_build_code_missing_col():
    df = pd.DataFrame({"other": ["x"]})
    result = _build_code(df, "PROCEDURE_ICD", "col_not_present")
    assert result.iloc[0] == "PROCEDURE_ICD"

def test_build_code_two_cols():
    df = pd.DataFrame({"type": ["EMERGENCY"], "loc": ["ED"]})
    result = _build_code(df, "HOSPITAL_ADMISSION", "type", "loc")
    assert result.iloc[0] == "HOSPITAL_ADMISSION//EMERGENCY//ED"

def test_build_code_second_col_null():
    df = pd.DataFrame({"type": ["EMERGENCY"], "loc": [None]})
    result = _build_code(df, "HOSPITAL_ADMISSION", "type", "loc")
    assert result.iloc[0] == "HOSPITAL_ADMISSION//EMERGENCY"


# ---------------------------------------------------------------------------
# uuid_to_int (from pre_meds)
# ---------------------------------------------------------------------------

def test_uuid_to_int_stable():
    uid = "550e8400-e29b-41d4-a716-446655440000"
    assert uuid_to_int(uid) == uuid_to_int(uid)

def test_uuid_to_int_positive():
    uid = "550e8400-e29b-41d4-a716-446655440000"
    assert uuid_to_int(uid) > 0

def test_uuid_to_int_fits_int64():
    uid = "550e8400-e29b-41d4-a716-446655440000"
    val = uuid_to_int(uid)
    assert val <= 0x7FFFFFFFFFFFFFFF

def test_uuid_to_int_different_uuids():
    uid1 = "550e8400-e29b-41d4-a716-446655440000"
    uid2 = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
    assert uuid_to_int(uid1) != uuid_to_int(uid2)


# ---------------------------------------------------------------------------
# extract_patients
# ---------------------------------------------------------------------------

def test_extract_patients_birth():
    df = pd.DataFrame([{
        "subject_id": 1001,
        "year_of_birth": 2090,
        "sex": "M",
        "dod": None,
    }])
    result = extract_patients(df)
    birth = result[result["code"] == "MEDS_BIRTH"]
    assert len(birth) == 1
    assert birth.iloc[0]["subject_id"] == 1001

def test_extract_patients_gender_static():
    df = pd.DataFrame([{
        "subject_id": 1001,
        "year_of_birth": 2090,
        "sex": "F",
        "dod": None,
    }])
    result = extract_patients(df)
    gender = result[result["code"] == "GENDER//F"]
    assert len(gender) == 1
    assert pd.isna(gender.iloc[0]["time"])

def test_extract_patients_death():
    df = pd.DataFrame([{
        "subject_id": 1001,
        "year_of_birth": 2090,
        "sex": "M",
        "dod": pd.Timestamp("2150-06-15"),
    }])
    result = extract_patients(df)
    death = result[result["code"] == "MEDS_DEATH"]
    assert len(death) == 1

def test_extract_patients_skips_null_subject():
    df = pd.DataFrame([{
        "subject_id": None,
        "year_of_birth": 2090,
        "sex": "M",
        "dod": None,
    }])
    result = extract_patients(df)
    assert len(result) == 0

def test_extract_patients_death_precise_deathtime():
    """deathtime_map should override the date-only dod with a precise timestamp."""
    df = pd.DataFrame([{
        "subject_id": 1001,
        "year_of_birth": 2090,
        "sex": "M",
        "dod": pd.Timestamp("2150-06-15"),          # midnight, date-only
    }])
    precise = pd.Timestamp("2150-06-15 14:32:00")   # hour-level precision
    result = extract_patients(df, deathtime_map={1001: precise})
    death = result[result["code"] == "MEDS_DEATH"]
    assert len(death) == 1
    assert death.iloc[0]["time"] == pd.Timestamp("2150-06-15 14:32:00")

def test_extract_patients_no_deathtime_falls_back_to_dod():
    df = pd.DataFrame([{
        "subject_id": 1001,
        "year_of_birth": 2090,
        "sex": "M",
        "dod": pd.Timestamp("2150-06-15"),
    }])
    # deathtime_map for a different patient — should not affect patient 1001
    result = extract_patients(df, deathtime_map={9999: pd.Timestamp("2150-06-15 10:00:00")})
    death = result[result["code"] == "MEDS_DEATH"]
    assert len(death) == 1
    assert pd.Timestamp(death.iloc[0]["time"]).date() == pd.Timestamp("2150-06-15").date()


# ---------------------------------------------------------------------------
# extract_admissions
# ---------------------------------------------------------------------------

def _admission_row(**kwargs):
    base = {
        "subject_id": 1001,
        "hadm_id": 999,
        "admittime": pd.Timestamp("2130-01-01 08:00:00"),
        "dischtime": pd.Timestamp("2130-01-10 10:00:00"),
        "admission_type": "Emergency department",
        "admission_location": None,
        "discharge_location": "Home",
        "deathtime": None,
        "edregtime": None,
        "edouttime": None,
        "insurance": "Others",
        "marital_status": "single",
        "ethnicity": None,
    }
    base.update(kwargs)
    return pd.DataFrame([base])

def test_extract_admissions_hospital_admission_code():
    result = extract_admissions(_admission_row())
    adm = result[result["code"].str.startswith("HOSPITAL_ADMISSION")]
    assert len(adm) == 1
    assert adm.iloc[0]["code"] == "HOSPITAL_ADMISSION//Emergency department"

def test_extract_admissions_hospital_discharge_code():
    result = extract_admissions(_admission_row())
    dis = result[result["code"].str.startswith("HOSPITAL_DISCHARGE")]
    assert len(dis) == 1
    assert dis.iloc[0]["code"] == "HOSPITAL_DISCHARGE//Home"

def test_extract_admissions_insurance():
    result = extract_admissions(_admission_row())
    ins = result[result["code"].str.startswith("INSURANCE")]
    assert len(ins) == 1
    assert ins.iloc[0]["code"] == "INSURANCE//Others"

def test_extract_admissions_ed_events():
    df = _admission_row(
        edregtime=pd.Timestamp("2130-01-01 06:00:00"),
        edouttime=pd.Timestamp("2130-01-01 07:45:00"),
    )
    result = extract_admissions(df)
    assert "ED_REGISTRATION" in result["code"].values
    assert "ED_OUT" in result["code"].values

def test_extract_admissions_no_nan_codes():
    result = extract_admissions(_admission_row())
    assert not result["code"].str.contains("//nan", na=False).any()


# ---------------------------------------------------------------------------
# extract_icustays
# ---------------------------------------------------------------------------

def _icustay_row(**kwargs):
    base = {
        "subject_id": 1001,
        "hadm_id": 999,
        "stay_id": 888,
        "intime": pd.Timestamp("2130-01-02 10:00:00"),
        "outtime": pd.Timestamp("2130-01-05 09:00:00"),
        "first_careunit": "RICU",
        "last_careunit": "RICU",
    }
    base.update(kwargs)
    return pd.DataFrame([base])

def test_extract_icustays_admission_code():
    result = extract_icustays(_icustay_row())
    adm = result[result["code"].str.startswith("ICU_ADMISSION")]
    assert len(adm) == 1
    assert adm.iloc[0]["code"] == "ICU_ADMISSION//RICU"

def test_extract_icustays_discharge_code():
    result = extract_icustays(_icustay_row())
    dis = result[result["code"].str.startswith("ICU_DISCHARGE")]
    assert len(dis) == 1
    assert dis.iloc[0]["code"] == "ICU_DISCHARGE//RICU"

def test_extract_icustays_missing_careunit():
    result = extract_icustays(_icustay_row(first_careunit=None, last_careunit=None))
    adm = result[result["code"].str.startswith("ICU_ADMISSION")]
    assert adm.iloc[0]["code"] == "ICU_ADMISSION"

def test_extract_icustays_skips_null_intime():
    result = extract_icustays(_icustay_row(intime=None))
    adm = result[result["code"].str.startswith("ICU_ADMISSION")]
    assert len(adm) == 0


# ---------------------------------------------------------------------------
# extract_chartevents
# ---------------------------------------------------------------------------

def test_extract_chartevents_basic():
    df = pd.DataFrame([{
        "subject_id": 1001,
        "charttime": pd.Timestamp("2130-01-01 08:00:00"),
        "storetime": pd.Timestamp("2130-01-01 08:05:00"),
        "itemid": "001C_102",
        "value": "83.5",
        "valuenum": 83.5,
        "valueuom": "mmHg",
        "warning": 0,
        "hadm_id": 999,
        "stay_id": 888,
    }])
    result = extract_chartevents(df)
    assert len(result) == 1
    assert result.iloc[0]["code"] == "CHARTEVENT//001C_102//mmHg"
    assert result.iloc[0]["numeric_value"] == pytest.approx(83.5, abs=0.01)

def test_extract_chartevents_normalizes_korean_unit():
    df = pd.DataFrame([{
        "subject_id": 1001,
        "charttime": pd.Timestamp("2130-01-01 08:00:00"),
        "storetime": pd.Timestamp("2130-01-01 08:00:00"),
        "itemid": "001C_102",
        "value": "83",
        "valuenum": 83.0,
        "valueuom": "회/min",
        "warning": 0,
        "hadm_id": 999,
        "stay_id": 888,
    }])
    result = extract_chartevents(df)
    assert result.iloc[0]["code"] == "CHARTEVENT//001C_102///min"

def test_extract_chartevents_skips_no_timestamp():
    df = pd.DataFrame([{
        "subject_id": 1001,
        "charttime": None,
        "storetime": None,
        "itemid": "001C_102",
        "value": "83",
        "valuenum": 83.0,
        "valueuom": "mmHg",
        "warning": 0,
        "hadm_id": 999,
        "stay_id": 888,
    }])
    result = extract_chartevents(df)
    assert len(result) == 0

def test_extract_chartevents_empty_uom_no_nan_code():
    """Empty valueuom must not produce a //nan code."""
    df = pd.DataFrame([{
        "subject_id": 1001,
        "charttime": pd.Timestamp("2130-01-01 08:00:00"),
        "storetime": pd.Timestamp("2130-01-01 08:00:00"),
        "itemid": "001C_102",
        "value": "83",
        "valuenum": 83.0,
        "valueuom": "",
        "warning": 0,
        "hadm_id": 999,
        "stay_id": 888,
    }])
    result = extract_chartevents(df)
    assert not result["code"].str.contains("//nan", na=False).any()
    assert result.iloc[0]["code"] == "CHARTEVENT//001C_102"


# ---------------------------------------------------------------------------
# extract_labevents
# ---------------------------------------------------------------------------

def test_extract_labevents_basic():
    df = pd.DataFrame([{
        "subject_id": 1001,
        "hadm_id": 999,
        "stay_id": 888,
        "specimen_id": "S1",
        "itemid": "001L3005",
        "charttime": pd.Timestamp("2130-01-01 10:00:00"),
        "storetime": pd.Timestamp("2130-01-01 10:30:00"),
        "value": "132.9",
        "valuenum": 132.9,
        "valueuom": "mg/dL",
        "ref_range_lower": 70,
        "ref_range_upper": 110,
        "flag": "abnormal",
        "comments": None,
    }])
    result = extract_labevents(df)
    assert len(result) == 1
    assert result.iloc[0]["code"] == "LAB//001L3005//mg/dL"
    assert result.iloc[0]["numeric_value"] == pytest.approx(132.9, abs=0.01)

def test_extract_labevents_empty_uom_no_nan_code():
    """Regression test for missing uom.notna() guard — must not produce //nan codes."""
    df = pd.DataFrame([{
        "subject_id": 1001,
        "hadm_id": 999,
        "itemid": "001L3005",
        "charttime": pd.Timestamp("2130-01-01 10:00:00"),
        "storetime": None,
        "value": "132.9",
        "valuenum": 132.9,
        "valueuom": "",          # empty unit
        "ref_range_lower": None,
        "ref_range_upper": None,
        "flag": None,
        "comments": None,
    }])
    result = extract_labevents(df)
    assert len(result) == 1
    assert not result["code"].str.contains("//nan", na=False).any()
    assert result.iloc[0]["code"] == "LAB//001L3005"

def test_extract_labevents_nan_uom_no_nan_code():
    """NaN valueuom must not produce a //nan code."""
    df = pd.DataFrame([{
        "subject_id": 1001,
        "hadm_id": 999,
        "itemid": "001L3005",
        "charttime": pd.Timestamp("2130-01-01 10:00:00"),
        "storetime": None,
        "value": "132.9",
        "valuenum": 132.9,
        "valueuom": float("nan"),
        "ref_range_lower": None,
        "ref_range_upper": None,
        "flag": None,
        "comments": None,
    }])
    result = extract_labevents(df)
    assert len(result) == 1
    assert not result["code"].str.contains("//nan", na=False).any()


# ---------------------------------------------------------------------------
# extract_diagnoses_icd
# ---------------------------------------------------------------------------

def _diag_row(**kwargs):
    base = {
        "subject_id": 1001,
        "hadm_id": 999,
        "icd_version": "KCD8",
        "icd_code": "I251",
    }
    base.update(kwargs)
    return pd.DataFrame([base])

def _admissions_df():
    return pd.DataFrame([{"hadm_id": 999, "admittime": pd.Timestamp("2130-01-01 08:00:00")}])

def test_extract_diagnoses_icd_basic():
    result = extract_diagnoses_icd(_diag_row(), _admissions_df())
    assert len(result) == 1
    assert result.iloc[0]["code"] == "DIAGNOSIS//KCD8//I251"

def test_extract_diagnoses_icd_timestamp_from_admissions():
    result = extract_diagnoses_icd(_diag_row(), _admissions_df())
    assert pd.Timestamp(result.iloc[0]["time"]) == pd.Timestamp("2130-01-01 08:00:00")

def test_extract_diagnoses_icd_no_admissions_gives_nat():
    result = extract_diagnoses_icd(_diag_row(), admissions_df=None)
    assert pd.isna(result.iloc[0]["time"])

def test_extract_diagnoses_icd_skips_null_code():
    df = _diag_row(icd_code=None)
    result = extract_diagnoses_icd(df, _admissions_df())
    assert len(result) == 0

def test_extract_diagnoses_icd_no_version():
    df = _diag_row()
    df = df.drop(columns=["icd_version"])
    result = extract_diagnoses_icd(df, _admissions_df())
    assert result.iloc[0]["code"] == "DIAGNOSIS//I251"


# ---------------------------------------------------------------------------
# extract_inputevents
# ---------------------------------------------------------------------------

def test_extract_inputevents_basic():
    df = pd.DataFrame([{
        "subject_id": 1001,
        "hadm_id": 999,
        "stay_id": 888,
        "itemid": "001I_1315",
        "starttime": pd.Timestamp("2130-01-03 09:00:00"),
        "endtime": pd.Timestamp("2130-01-03 09:30:00"),
        "amount": 250.0,
        "amountuom": "cc",
        "storetime": None,
    }])
    result = extract_inputevents(df)
    assert len(result) == 1
    assert result.iloc[0]["code"] == "INPUT_START//001I_1315//cc"
    assert result.iloc[0]["numeric_value"] == pytest.approx(250.0, abs=0.01)

def test_extract_inputevents_skips_null_starttime():
    df = pd.DataFrame([{
        "subject_id": 1001,
        "hadm_id": 999,
        "stay_id": 888,
        "itemid": "001I_1315",
        "starttime": None,
        "amount": 250.0,
        "amountuom": "cc",
    }])
    result = extract_inputevents(df)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# extract_outputevents
# ---------------------------------------------------------------------------

def test_extract_outputevents_basic():
    df = pd.DataFrame([{
        "subject_id": 1001,
        "hadm_id": 999,
        "stay_id": 888,
        "itemid": "001O_148",
        "charttime": pd.Timestamp("2130-01-04 12:00:00"),
        "storetime": None,
        "value": 300.0,
        "valueuom": "cc",
    }])
    result = extract_outputevents(df)
    assert len(result) == 1
    assert result.iloc[0]["code"] == "OUTPUT//001O_148//cc"

def test_extract_outputevents_empty_uom_no_nan():
    df = pd.DataFrame([{
        "subject_id": 1001,
        "hadm_id": 999,
        "stay_id": 888,
        "itemid": "001O_148",
        "charttime": pd.Timestamp("2130-01-04 12:00:00"),
        "storetime": None,
        "value": 300.0,
        "valueuom": "",
    }])
    result = extract_outputevents(df)
    assert not result["code"].str.contains("//nan", na=False).any()


# ---------------------------------------------------------------------------
# extract_emar
# ---------------------------------------------------------------------------

def test_extract_emar_basic():
    df = pd.DataFrame([{
        "subject_id": 1001,
        "hadm_id": 999,
        "stay_id": 888,
        "itemid": "12005122",
        "charttime": pd.Timestamp("2130-01-02 08:00:00"),
        "storetime": None,
    }])
    result = extract_emar(df)
    assert len(result) == 1
    assert result.iloc[0]["code"] == "MEDICATION//12005122"

def test_extract_emar_skips_null_charttime():
    df = pd.DataFrame([{
        "subject_id": 1001,
        "hadm_id": 999,
        "itemid": "12005122",
        "charttime": None,
    }])
    result = extract_emar(df)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# extract_procedureevents
# ---------------------------------------------------------------------------

def test_extract_procedureevents_start_and_end():
    df = pd.DataFrame([{
        "subject_id": 1001,
        "hadm_id": 999,
        "stay_id": 888,
        "itemid": "001P_OPFG130303",
        "starttime": pd.Timestamp("2130-01-03 10:00:00"),
        "endtime": pd.Timestamp("2130-01-03 12:00:00"),
        "storetime": None,
    }])
    result = extract_procedureevents(df)
    starts = result[result["code"].str.startswith("PROCEDURE_START")]
    ends = result[result["code"].str.startswith("PROCEDURE_END")]
    assert len(starts) == 1
    assert len(ends) == 1
    assert starts.iloc[0]["code"] == "PROCEDURE_START//001P_OPFG130303"
    assert ends.iloc[0]["code"] == "PROCEDURE_END//001P_OPFG130303"

def test_extract_procedureevents_no_endtime():
    df = pd.DataFrame([{
        "subject_id": 1001,
        "hadm_id": 999,
        "stay_id": 888,
        "itemid": "001P_OPFG130303",
        "starttime": pd.Timestamp("2130-01-03 10:00:00"),
        "endtime": None,
        "storetime": None,
    }])
    result = extract_procedureevents(df)
    assert len(result[result["code"].str.startswith("PROCEDURE_START")]) == 1
    assert len(result[result["code"].str.startswith("PROCEDURE_END")]) == 0


# ---------------------------------------------------------------------------
# build_subject_splits
# ---------------------------------------------------------------------------

def test_build_subject_splits_proportions():
    ids = list(range(1000))
    splits = build_subject_splits(ids)
    counts = splits["split"].value_counts()
    assert 75 <= counts["train"] / 1000 * 100 <= 85
    assert 8  <= counts["tuning"] / 1000 * 100 <= 12
    assert 8  <= counts["held_out"] / 1000 * 100 <= 12

def test_build_subject_splits_no_overlap():
    ids = list(range(500))
    splits = build_subject_splits(ids)
    train_ids   = set(splits[splits["split"] == "train"]["subject_id"])
    tuning_ids  = set(splits[splits["split"] == "tuning"]["subject_id"])
    held_ids    = set(splits[splits["split"] == "held_out"]["subject_id"])
    assert len(train_ids & tuning_ids) == 0
    assert len(train_ids & held_ids) == 0
    assert len(tuning_ids & held_ids) == 0

def test_build_subject_splits_covers_all():
    ids = list(range(100))
    splits = build_subject_splits(ids)
    assert set(splits["subject_id"]) == set(ids)

def test_build_subject_splits_reproducible():
    ids = list(range(200))
    s1 = build_subject_splits(ids)
    s2 = build_subject_splits(ids)
    assert s1["subject_id"].tolist() == s2["subject_id"].tolist()
    assert s1["split"].tolist() == s2["split"].tolist()


# ---------------------------------------------------------------------------
# build_codes_parquet
# ---------------------------------------------------------------------------

def test_build_codes_parquet_basic():
    events = pd.DataFrame({"code": ["LAB//001L3005//mg/dL", "CHARTEVENT//001C_102//mmHg", "MEDS_BIRTH"]})
    result = build_codes_parquet(events)
    assert set(result["code"]) == {"LAB//001L3005//mg/dL", "CHARTEVENT//001C_102//mmHg", "MEDS_BIRTH"}
    assert "description" in result.columns
    assert "parent_codes" in result.columns

def test_build_codes_parquet_no_duplicate_codes():
    events = pd.DataFrame({"code": ["LAB//001L3005//mg/dL", "LAB//001L3005//mg/dL", "MEDS_BIRTH"]})
    result = build_codes_parquet(events)
    assert result["code"].nunique() == len(result)

def test_build_codes_parquet_sorted():
    events = pd.DataFrame({"code": ["MEDS_DEATH", "CHARTEVENT//X", "LAB//Y"]})
    result = build_codes_parquet(events)
    assert result["code"].tolist() == sorted(result["code"].tolist())


# ---------------------------------------------------------------------------
# to_meds_table
# ---------------------------------------------------------------------------

def test_to_meds_table_schema():
    df = pd.DataFrame({
        "subject_id": [1001],
        "time": [pd.Timestamp("2130-01-01")],
        "code": ["MEDS_BIRTH"],
        "numeric_value": [None],
    })
    table = to_meds_table(df)
    assert table.schema.equals(MEDS_SCHEMA)

def test_to_meds_table_sorted():
    df = pd.DataFrame({
        "subject_id": [1001, 1001, 1001],
        "time": [pd.Timestamp("2130-01-03"), None, pd.Timestamp("2130-01-01")],
        "code": ["C", "A", "B"],
        "numeric_value": [None, None, None],
    })
    table = to_meds_table(df)
    result = table.to_pandas()
    # Static event (NaT) must come first
    assert pd.isna(result.iloc[0]["time"])

def test_to_meds_table_float32():
    df = pd.DataFrame({
        "subject_id": [1001],
        "time": [pd.Timestamp("2130-01-01")],
        "code": ["LAB//X"],
        "numeric_value": [3.14],
    })
    table = to_meds_table(df)
    assert table.schema.field("numeric_value").type == pa.float32()

def test_to_meds_table_empty():
    df = pd.DataFrame(columns=["subject_id", "time", "code", "numeric_value"])
    table = to_meds_table(df)
    assert table.num_rows == 0
    assert table.schema.equals(MEDS_SCHEMA)

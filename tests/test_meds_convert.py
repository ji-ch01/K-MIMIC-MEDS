"""Unit tests for meds_convert.py"""

import pandas as pd
import pytest
from kmimic_meds.etl.meds_convert import (
    make_code,
    normalize_unit,
    extract_patients,
    extract_chartevents,
    extract_labevents,
    UNIT_MAP,
)


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
    assert normalize_unit("회/min") == "/min"

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


# ---------------------------------------------------------------------------
# extract_patients
# ---------------------------------------------------------------------------

def test_extract_patients_birth():
    df = pd.DataFrame([{
        "subject_id": 1001,
        "year_of_birth": "2090",
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
        "year_of_birth": "2090",
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
        "year_of_birth": "2090",
        "sex": "M",
        "dod": pd.Timestamp("2150-06-15"),
    }])
    result = extract_patients(df)
    death = result[result["code"] == "MEDS_DEATH"]
    assert len(death) == 1

def test_extract_patients_skips_null_subject():
    df = pd.DataFrame([{
        "subject_id": None,
        "year_of_birth": "2090",
        "sex": "M",
        "dod": None,
    }])
    result = extract_patients(df)
    assert len(result) == 0


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
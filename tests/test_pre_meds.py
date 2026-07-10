"""Unit tests for pre_meds.py"""

import pandas as pd
import pytest

from kmimic_meds.etl.pre_meds import coerce_datetime, run, uuid_to_int, validate_input_schema


def test_coerce_datetime_mixed_formats_future_years():
    series = pd.Series([
        "3021-04-22 23:35:19",
        "2023-01-01",
        None,
        "not-a-date",
    ])

    result = coerce_datetime(series, "mixed")

    assert str(result.dtype) == "datetime64[us]"
    assert result.iloc[0] == pd.Timestamp("3021-04-22 23:35:19")
    assert result.iloc[1] == pd.Timestamp("2023-01-01 00:00:00")
    assert pd.isna(result.iloc[2])
    assert pd.isna(result.iloc[3])


def test_validate_input_schema_accepts_icustay_id_alternative():
    df = pd.DataFrame(columns=[
        "subject_id",
        "hadm_id",
        "icustay_id",
        "itemid",
        "starttime",
        "amount",
        "amountuom",
    ])

    validate_input_schema("syn_inputevents", df)


def test_validate_input_schema_reports_missing_columns():
    df = pd.DataFrame(columns=["subject_id", "hadm_id"])

    with pytest.raises(ValueError, match="valueuom"):
        validate_input_schema("syn_labevents", df)


def test_run_writes_core_tables_from_xlsx(tmp_path):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "intermediate"
    raw_dir.mkdir()

    subject_id = "550e8400-e29b-41d4-a716-446655440000"
    hadm_id = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
    stay_id = "6ba7b811-9dad-11d1-80b4-00c04fd430c8"

    pd.DataFrame([{
        "subject_id": subject_id,
        "anchor_age": 40,
        "anchor_year": 3021,
        "sex": "F",
        "dod": None,
    }]).to_excel(raw_dir / "syn_patients.xlsx", index=False)

    pd.DataFrame([{
        "subject_id": subject_id,
        "hadm_id": hadm_id,
        "admittime": "3021-04-22 23:35:19",
        "dischtime": "3021-04-23",
    }]).to_excel(raw_dir / "syn_admissions.xlsx", index=False)

    pd.DataFrame([{
        "subject_id": subject_id,
        "hadm_id": hadm_id,
        "stay_id": stay_id,
        "intime": "3021-04-23T01:00:00",
        "outtime": "3021-04-24T01:00:00",
    }]).to_excel(raw_dir / "syn_icustays.xlsx", index=False)

    pd.DataFrame([{
        "subject_id": subject_id,
        "hadm_id": hadm_id,
        "stay_id": stay_id,
        "itemid": "001C_102",
        "charttime": "3021-04-23 01:30:00",
        "valuenum": 80.5,
        "valueuom": "mmHg",
    }]).to_excel(raw_dir / "syn_chartevents.xlsx", index=False)

    pd.DataFrame([{
        "subject_id": subject_id,
        "hadm_id": hadm_id,
        "itemid": "001L3005",
        "charttime": "3021-04-23",
        "valuenum": 132.9,
        "valueuom": "mg/dL",
    }]).to_excel(raw_dir / "syn_labevents.xlsx", index=False)

    run(raw_dir, out_dir)

    for name in [
        "syn_patients",
        "syn_admissions",
        "syn_icustays",
        "syn_chartevents",
        "syn_labevents",
    ]:
        assert (out_dir / f"{name}.parquet").exists()

    patients = pd.read_parquet(out_dir / "syn_patients.parquet")
    assert patients.iloc[0]["subject_id"] == uuid_to_int(subject_id)
    assert patients.iloc[0]["year_of_birth"] == 2981

    chartevents = pd.read_parquet(out_dir / "syn_chartevents.parquet")
    assert str(chartevents["charttime"].dtype) == "datetime64[us]"
    assert chartevents.iloc[0]["charttime"] == pd.Timestamp("3021-04-23 01:30:00")

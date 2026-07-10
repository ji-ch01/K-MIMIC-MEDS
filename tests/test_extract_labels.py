"""Unit tests for mortality label extraction."""

import pandas as pd

import extract_labels


def _event(subject_id: int, time: str | None, code: str, split: str = "train") -> dict:
    return {
        "subject_id": subject_id,
        "time": pd.Timestamp(time) if time is not None else pd.NaT,
        "code": code,
        "numeric_value": None,
        "split": split,
    }


def test_build_inhospital_mortality_24h_labels_and_exclusions():
    full = pd.DataFrame([
        # Included positive: alive at 24h, dies before hospital discharge.
        _event(1, "2130-01-01 00:00:00", "HOSPITAL_ADMISSION//Emergency", "train"),
        _event(1, "2130-01-04 00:00:00", "HOSPITAL_DISCHARGE//Expired", "train"),
        _event(1, "2130-01-03 00:00:00", "MEDS_DEATH", "train"),
        # Included negative: alive at 24h, discharged without death.
        _event(2, "2130-01-01 00:00:00", "HOSPITAL_ADMISSION//Emergency", "tuning"),
        _event(2, "2130-01-03 00:00:00", "HOSPITAL_DISCHARGE//Home", "tuning"),
        # Excluded: discharged before prediction time.
        _event(3, "2130-01-01 00:00:00", "HOSPITAL_ADMISSION//Emergency", "held_out"),
        _event(3, "2130-01-01 12:00:00", "HOSPITAL_DISCHARGE//Home", "held_out"),
        # Excluded: died before prediction time.
        _event(4, "2130-01-01 00:00:00", "HOSPITAL_ADMISSION//Emergency", "held_out"),
        _event(4, "2130-01-04 00:00:00", "HOSPITAL_DISCHARGE//Expired", "held_out"),
        _event(4, "2130-01-01 12:00:00", "MEDS_DEATH", "held_out"),
    ])

    _, _, hosp_adm, hosp_dis, deaths, split_map = extract_labels.extract_events(full)
    labels = extract_labels.build_inhospital_mortality(
        hosp_adm,
        hosp_dis,
        deaths,
        split_map,
    ).sort_values("subject_id").reset_index(drop=True)

    assert labels["subject_id"].tolist() == [1, 2]
    assert labels["prediction_time"].tolist() == [
        pd.Timestamp("2130-01-02 00:00:00"),
        pd.Timestamp("2130-01-02 00:00:00"),
    ]
    assert labels["boolean_value"].tolist() == [True, False]
    assert labels["split"].tolist() == ["train", "tuning"]


def test_build_icu_mortality_24h_uses_icu_window():
    full = pd.DataFrame([
        # Hospital death after ICU discharge should not be an ICU mortality positive.
        _event(10, "2130-01-01 00:00:00", "ICU_ADMISSION//MICU", "train"),
        _event(10, "2130-01-03 00:00:00", "ICU_DISCHARGE//MICU", "train"),
        _event(10, "2130-01-04 00:00:00", "MEDS_DEATH", "train"),
        # ICU death after 24h and before ICU discharge is positive.
        _event(11, "2130-01-01 00:00:00", "ICU_ADMISSION//MICU", "held_out"),
        _event(11, "2130-01-05 00:00:00", "ICU_DISCHARGE//MICU", "held_out"),
        _event(11, "2130-01-03 00:00:00", "MEDS_DEATH", "held_out"),
    ])

    icu_adm, icu_dis, _, _, deaths, split_map = extract_labels.extract_events(full)
    labels = extract_labels.build_icu_mortality(
        icu_adm,
        icu_dis,
        deaths,
        split_map,
    ).sort_values("subject_id").reset_index(drop=True)

    assert labels["subject_id"].tolist() == [10, 11]
    assert labels["boolean_value"].tolist() == [False, True]
    assert labels["split"].tolist() == ["train", "held_out"]


def test_save_labels_writes_meds_dev_split_files(tmp_path, monkeypatch):
    monkeypatch.setattr(extract_labels, "LABELS_DIR", tmp_path)
    labels = pd.DataFrame([
        {
            "subject_id": 1,
            "prediction_time": pd.Timestamp("2130-01-02"),
            "boolean_value": True,
            "split": "train",
        },
        {
            "subject_id": 2,
            "prediction_time": pd.Timestamp("2130-01-02"),
            "boolean_value": False,
            "split": "held_out",
        },
    ])

    extract_labels.save_labels(labels, "inhospital_mortality_24h")

    train = pd.read_parquet(tmp_path / "inhospital_mortality_24h" / "train" / "0.parquet")
    tuning = pd.read_parquet(tmp_path / "inhospital_mortality_24h" / "tuning" / "0.parquet")
    held_out = pd.read_parquet(tmp_path / "inhospital_mortality_24h" / "held_out" / "0.parquet")

    assert train.columns.tolist() == ["subject_id", "prediction_time", "boolean_value"]
    assert train["boolean_value"].tolist() == [True]
    assert tuning.empty
    assert held_out["subject_id"].tolist() == [2]

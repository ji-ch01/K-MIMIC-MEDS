"""
CLI validation script for K-MIMIC-MEDS output.
Runs the same checks as the validation notebook but from the command line.

Usage:
    python validate.py --output_dir data/output
"""

import json
import argparse
import sys
from pathlib import Path

import pandas as pd


def validate(output_dir: Path) -> bool:
    passed = 0
    failed = 0

    def check(name, result, detail=""):
        nonlocal passed, failed
        status = "PASS" if result else "FAIL"
        detail_str = f" — {detail}" if detail else ""
        print(f"  [{status}] {name}{detail_str}")
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\nValidating MEDS dataset at: {output_dir}\n")

    # 1. File structure
    print("1. File structure")
    expected_files = [
        "data/train/0.parquet",
        "data/tuning/0.parquet",
        "data/held_out/0.parquet",
        "metadata/codes.parquet",
        "metadata/dataset.json",
        "metadata/subject_splits.parquet",
    ]
    for f in expected_files:
        check(f, (output_dir / f).exists())

    # 2. MEDS schema
    print("\n2. MEDS schema")
    train = pd.read_parquet(output_dir / "data/train/0.parquet")
    check("subject_id is int64", str(train["subject_id"].dtype) in ("int64", "Int64"))
    check("time is datetime", "datetime" in str(train["time"].dtype))
    check("code is string", str(train["code"].dtype) in ("object", "string", "str"))
    check("numeric_value is float32", str(train["numeric_value"].dtype) == "float32")
    check("exactly 4 columns", len(train.columns) == 4)

    # 3. Code quality
    print("\n3. Code quality")
    nan_codes = train[train["code"].str.contains("//nan", na=False)]
    check("no //nan in codes", len(nan_codes) == 0, f"{len(nan_codes)} found")
    unknown_codes = train[train["code"] == "UNKNOWN"]
    check("no UNKNOWN codes", len(unknown_codes) == 0, f"{len(unknown_codes)} found")
    korean = train[train["code"].str.contains("회|℃|㎍|㎎|μ|×", na=False, regex=True)]
    check("no Korean units in codes", len(korean) == 0, f"{len(korean)} found")
    # Debug — show what was found
    if len(korean) > 0:
        print(f"    Sample: {korean['code'].unique()[:5]}")

    # 4. Static events
    print("\n4. Static events")
    static = train[train["time"].isna()]
    check("static events exist", len(static) > 0, f"{len(static)} found")
    if len(static) > 0:
        only_gender = static["code"].str.startswith(("GENDER",)).all()
        check("static events are GENDER only", only_gender)

    # 5. Splits
    print("\n5. Splits")
    splits = pd.read_parquet(output_dir / "metadata/subject_splits.parquet")
    total = len(splits)
    counts = splits["split"].value_counts()
    train_pct = counts.get("train", 0) / total * 100
    tuning_pct = counts.get("tuning", 0) / total * 100
    held_pct = counts.get("held_out", 0) / total * 100
    check("train ~80%", 75 <= train_pct <= 85, f"{train_pct:.1f}%")
    check("tuning ~10%", 8 <= tuning_pct <= 12, f"{tuning_pct:.1f}%")
    check("held_out ~10%", 8 <= held_pct <= 12, f"{held_pct:.1f}%")

    # 6. Metadata
    print("\n6. Metadata")
    codes = pd.read_parquet(output_dir / "metadata/codes.parquet")
    check("codes.parquet has descriptions", codes["description"].notna().sum() > 0,
          f"{codes['description'].notna().sum()}/{len(codes)} codes")
    check("codes.parquet has parent_codes", codes["parent_codes"].notna().sum() > 0,
          f"{codes['parent_codes'].notna().sum()}/{len(codes)} codes")
    meta = json.loads((output_dir / "metadata/dataset.json").read_text())
    check("dataset.json has dataset_name", "dataset_name" in meta)
    check("dataset.json has meds_version", "meds_version" in meta)

    # 7. Global statistics
    print("\n7. Global statistics")
    all_dfs = []
    for split_name in ["train", "tuning", "held_out"]:
        df = pd.read_parquet(output_dir / f"data/{split_name}/0.parquet")
        all_dfs.append(df)
    full = pd.concat(all_dfs, ignore_index=True)
    total_events = len(full)
    total_patients = full["subject_id"].nunique()
    check("total events > 0", total_events > 0, f"{total_events:,}")
    check("total patients > 0", total_patients > 0, f"{total_patients:,}")
    check("events with numeric value exist", full["numeric_value"].notna().sum() > 0,
          f"{full['numeric_value'].notna().sum():,}")

    # Summary
    total_checks = passed + failed
    print(f"\n{'='*50}")
    print(f"Result: {passed}/{total_checks} checks passed")
    if failed > 0:
        print(f"        {failed} checks FAILED")
    print(f"{'='*50}\n")

    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="Validate a K-MIMIC-MEDS output dataset")
    parser.add_argument("--output_dir", type=Path, default=Path("data/output"))
    args = parser.parse_args()

    success = validate(args.output_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
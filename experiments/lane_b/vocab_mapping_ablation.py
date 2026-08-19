"""
Lane B — Vocabulary-mapping ablation.

Quantifies how many of the 204 K-MIMIC codes can be linked to a MIMIC-IV
code under three matching strategies. This makes explicit what the
"Vocabulary mismatch" limitation in the paper means in practice: how much
of the vocabulary the concept-level crosswalk in concepts.yaml actually
covers, versus what naive automatic matching would achieve. CPU-only;
runs entirely against metadata already produced by the ETL pipeline and
the MIMIC-IV-MEDS extraction (no model training, no GPU).

Strategies
----------
1. naive_itemid   — exact string match between K-MIMIC's local item ID and
                     MIMIC-IV's lab item ID (bare code, ignoring prefixes
                     and units). Represents matching with zero domain
                     knowledge.
2. naive_label    — case-insensitive whole-word containment between
                     K-MIMIC's English abbreviation (the part of its
                     bilingual "ENG > KOR" description before the
                     separator) and MIMIC-IV lab descriptions (LOINC long
                     common names). Represents an automatic text-similarity
                     pass with no clinical review.
3. edi_crosswalk  — the expert-curated concept crosswalk actually used in
                     concepts.yaml / the paper's benchmark. Reports how many
                     of the 204 codes participate, and how many of those are
                     independently anchored by an EDI parent code on the
                     K-MIMIC side (a Korean insurance billing code, present
                     for lab items only) versus relying on clinical-concept
                     judgement alone (vitals, which carry no coded standard
                     on either side in this dataset).

Usage:
    python experiments/lane_b/vocab_mapping_ablation.py \\
        --kmimic_codes data/output/metadata/codes.parquet \\
        --mimic_codes  data/MEDS_cohort/extract_code_metadata/codes.parquet \\
        --mimic_labs   data/MEDS_cohort/extract_code_metadata/d_labitems_to_loinc.parquet \\
        --concepts     experiments/concepts.yaml \\
        --output_dir   experiments/lane_b/results
"""

import argparse
import json
import logging
import re
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MIN_TOKEN_LEN = 3  # naive_label: ignore abbreviations shorter than this (too noisy)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_kmimic_codes(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["local_itemid"] = df["code"].str.split("//").str[1]
    df["eng_abbrev"] = df["description"].apply(_eng_abbrev)
    df["has_edi"] = df["parent_codes"].apply(lambda v: v is not None and len(v) > 0)
    return df


def _eng_abbrev(desc) -> str | None:
    """K-MIMIC descriptions are bilingual, 'ENG > KOR'; take the English part
    and strip it down to ASCII letters/digits/spaces so it can be used as a
    naive text-matching key."""
    if pd.isna(desc):
        return None
    eng = str(desc).split(">")[0]
    eng = re.sub(r"[^A-Za-z0-9 ]+", " ", eng).strip().lower()
    return eng or None


def resolve_prefixes(all_codes: list[str], prefixes: list[str]) -> list[str]:
    """Codes equal to a prefix or starting with '{prefix}//' (same rule as
    feature_extract.code_matches, applied to the code catalog instead of
    the event stream)."""
    matched = []
    for c in all_codes:
        for p in prefixes:
            if c == p or c.startswith(p + "//"):
                matched.append(c)
                break
    return matched


# ---------------------------------------------------------------------------
# Strategy 1 — naive itemid match
# ---------------------------------------------------------------------------

def naive_itemid_strategy(kmimic: pd.DataFrame, mimic_lab_itemids: set[str]) -> dict:
    kmimic_itemids = set(kmimic["local_itemid"].dropna())
    matched = sorted(kmimic_itemids & mimic_lab_itemids)
    n = len(kmimic)
    return {
        "description": "Exact string match on bare item ID (K-MIMIC local ID vs MIMIC-IV itemid).",
        "n_kmimic_codes_matched": len(matched),
        "coverage_pct_of_204": round(100 * len(matched) / n, 1),
        "matched_itemids": matched,
    }


# ---------------------------------------------------------------------------
# Strategy 2 — naive label-text match
# ---------------------------------------------------------------------------

def naive_label_strategy(kmimic: pd.DataFrame, mimic_labs: pd.DataFrame) -> dict:
    mimic_labs = mimic_labs.copy()
    mimic_labs["desc_lower"] = mimic_labs["description"].str.lower()

    candidates = kmimic[kmimic["eng_abbrev"].notna() & (kmimic["eng_abbrev"].str.len() >= MIN_TOKEN_LEN)]

    matches = []
    for row in candidates.itertuples(index=False):
        token = row.eng_abbrev
        pattern = re.compile(r"\b" + re.escape(token) + r"\b")
        hits = mimic_labs[mimic_labs["desc_lower"].apply(lambda d: bool(pattern.search(d)))]
        if len(hits) > 0:
            matches.append({
                "kmimic_code": row.code,
                "eng_abbrev": token,
                "n_mimic_candidates": int(len(hits)),
                "mimic_candidates": hits["code"].tolist()[:5],
            })

    n = len(kmimic)
    n_eligible = len(candidates)
    n_matched = len(matches)
    mean_candidates = (
        sum(m["n_mimic_candidates"] for m in matches) / n_matched if n_matched else 0.0
    )
    return {
        "description": (
            "Case-insensitive whole-word containment of the K-MIMIC English "
            "abbreviation in a MIMIC-IV lab (LOINC) description. No clinical "
            "review; candidates are not disambiguated."
        ),
        "n_kmimic_codes_with_text_label": n_eligible,
        "n_kmimic_codes_matched": n_matched,
        "coverage_pct_of_204": round(100 * n_matched / n, 1),
        "mean_mimic_candidates_per_match": round(mean_candidates, 2),
        "matches": matches,
    }


# ---------------------------------------------------------------------------
# Strategy 3 — expert-curated EDI / concept crosswalk (concepts.yaml)
# ---------------------------------------------------------------------------

def edi_crosswalk_strategy(kmimic: pd.DataFrame, mimic_codes: pd.DataFrame, concepts_path: str) -> dict:
    with open(concepts_path) as f:
        cfg = yaml.safe_load(f)

    kmimic_all_codes = kmimic["code"].tolist()
    mimic_all_codes = mimic_codes["code"].tolist()
    mimic_parent = dict(zip(mimic_codes["code"], mimic_codes["parent_codes"]))

    concept_rows = []
    kmimic_matched_codes: set[str] = set()
    mimic_loinc_anchored = 0
    mimic_no_catalog_entry = 0

    for concept, spec in cfg["kmimic"]["numeric_concepts"].items():
        km_prefixes = spec["codes"]
        km_matches = resolve_prefixes(kmimic_all_codes, km_prefixes)
        kmimic_matched_codes.update(km_matches)
        km_has_edi = any(
            bool(kmimic.loc[kmimic["code"] == c, "has_edi"].iloc[0]) for c in km_matches
        )

        mm_prefixes = cfg["mimic"]["numeric_concepts"][concept]["codes"]
        mm_matches = resolve_prefixes(mimic_all_codes, mm_prefixes)
        if mm_matches:
            has_loinc = any(
                mimic_parent.get(c) is not None and len(mimic_parent.get(c)) > 0
                for c in mm_matches
            )
        else:
            has_loinc = False

        if mm_matches and has_loinc:
            mimic_loinc_anchored += 1
        elif not mm_matches:
            mimic_no_catalog_entry += 1

        concept_rows.append({
            "concept": concept,
            "kmimic_codes": km_matches,
            "kmimic_anchored_by_edi": km_has_edi,
            "mimic_codes": mm_prefixes,
            "mimic_found_in_catalog": bool(mm_matches),
            "mimic_anchored_by_loinc": has_loinc,
        })

    n = len(kmimic)
    n_used = len(kmimic_matched_codes)
    n_with_edi = sum(1 for c in kmimic_matched_codes if kmimic.loc[kmimic["code"] == c, "has_edi"].iloc[0])

    return {
        "description": (
            "Expert-curated clinical-concept crosswalk (concepts.yaml), the "
            "strategy actually used for the Lane B benchmark. Codes are "
            "aligned by clinical meaning, not by an automated code-to-code "
            "mapping."
        ),
        "n_concepts": len(concept_rows),
        "n_kmimic_codes_used": n_used,
        "coverage_pct_of_204": round(100 * n_used / n, 1),
        "n_kmimic_codes_anchored_by_edi": n_with_edi,
        "n_kmimic_codes_vitals_only_no_coded_standard": n_used - n_with_edi,
        "mimic_side": {
            "n_concepts": len(concept_rows),
            "n_anchored_by_loinc": mimic_loinc_anchored,
            "n_absent_from_code_catalog": mimic_no_catalog_entry,
        },
        "concepts": concept_rows,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Vocabulary-mapping ablation (Lane B).")
    parser.add_argument("--kmimic_codes", default="data/output/metadata/codes.parquet")
    parser.add_argument("--mimic_codes", default="data/MEDS_cohort/extract_code_metadata/codes.parquet")
    parser.add_argument("--mimic_labs", default="data/MEDS_cohort/extract_code_metadata/d_labitems_to_loinc.parquet")
    parser.add_argument("--concepts", default="experiments/concepts.yaml")
    parser.add_argument("--output_dir", default="experiments/lane_b/results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading code catalogs...")
    kmimic = load_kmimic_codes(args.kmimic_codes)
    mimic_codes = pd.read_parquet(args.mimic_codes)
    mimic_labs = pd.read_parquet(args.mimic_labs)
    mimic_lab_itemids = set(mimic_labs["itemid"].astype(str))

    log.info(f"K-MIMIC: {len(kmimic)} codes ({kmimic['description'].notna().sum()} with description, "
              f"{kmimic['has_edi'].sum()} with EDI parent code)")
    log.info(f"MIMIC-IV: {len(mimic_codes)} codes in catalog, {len(mimic_labs)} lab codes with LOINC")

    results = {
        "n_kmimic_codes": len(kmimic),
        "strategies": {
            "naive_itemid": naive_itemid_strategy(kmimic, mimic_lab_itemids),
            "naive_label_text": naive_label_strategy(kmimic, mimic_labs),
            "edi_concept_crosswalk": edi_crosswalk_strategy(kmimic, mimic_codes, args.concepts),
        },
    }

    out_path = output_dir / "vocab_mapping_ablation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info(f"Saved: {out_path}")

    print("\n" + "=" * 78)
    print(f"  Vocabulary-mapping ablation — {len(kmimic)} K-MIMIC codes")
    print("=" * 78)
    print(f"  {'Strategy':<28} {'Matched':>10} {'Coverage':>12}")
    print("  " + "-" * 74)
    for name, r in results["strategies"].items():
        if name == "edi_concept_crosswalk":
            matched, cov = r["n_kmimic_codes_used"], r["coverage_pct_of_204"]
        else:
            matched, cov = r["n_kmimic_codes_matched"], r["coverage_pct_of_204"]
        print(f"  {name:<28} {matched:>10} {cov:>10.1f}%")
    print("=" * 78)
    edi = results["strategies"]["edi_concept_crosswalk"]
    print(f"  Of the {edi['n_kmimic_codes_used']} codes used by the paper's crosswalk, "
          f"{edi['n_kmimic_codes_anchored_by_edi']} are anchored by an EDI billing code "
          f"(labs); {edi['n_kmimic_codes_vitals_only_no_coded_standard']} are vitals aligned "
          f"by clinical judgement alone (no coded standard on either side).")
    print("=" * 78)


if __name__ == "__main__":
    main()

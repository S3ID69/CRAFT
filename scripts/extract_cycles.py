"""
extract_cycles.py — Phase 1 data preparation script.

Reads raw ICBHI 2017 files and produces:
  1. data/cycles/  — one .wav file per annotated respiratory cycle
  2. data/metadata.csv — ground-truth CSV with all labels and patient info

ICBHI file format:
  Each recording has:
    - <id>.wav           : the audio recording
    - <id>.txt           : annotation file, one cycle per line:
                           <start_time> <end_time> <crackle_flag> <wheeze_flag>
    - ICBHI_patient_diagnosis.txt : patient info (id, age, sex, location, mode, device)

Usage:
    python scripts/extract_cycles.py \
        --raw_dir data/raw \
        --cycles_dir data/cycles \
        --output_csv data/metadata.csv \
        --sample_rate 22050

The official ICBHI train/test split file (patient_list_foldwise.txt or
official_split.txt, included in the dataset) is used to assign splits.
If it is absent, the script falls back to a 60/40 random patient-level split.
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from tqdm import tqdm


# ── Label encoding ───────────────────────────────────────────────────────────
def _encode_label(crackle: int, wheeze: int) -> str:
    if crackle == 0 and wheeze == 0:
        return "normal"
    elif crackle == 1 and wheeze == 0:
        return "crackle"
    elif crackle == 0 and wheeze == 1:
        return "wheeze"
    else:
        return "both"


# ── Parse annotation file ─────────────────────────────────────────────────────
def parse_annotation_file(txt_path: Path) -> list[dict]:
    """
    Parse a single ICBHI annotation .txt file.

    Each line: <start> <end> <crackle> <wheeze>
    Returns a list of dicts with keys: start, end, crackle, wheeze, label
    """
    cycles = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            start   = float(parts[0])
            end     = float(parts[1])
            crackle = int(parts[2])
            wheeze  = int(parts[3])
            cycles.append({
                "start":   start,
                "end":     end,
                "crackle": crackle,
                "wheeze":  wheeze,
                "label":   _encode_label(crackle, wheeze),
            })
    return cycles


# ── Parse patient demographics ────────────────────────────────────────────────
def parse_patient_info(raw_dir: Path) -> dict[str, dict]:
    """
    Parse ICBHI_patient_diagnosis.txt (if present).

    Returns a dict: patient_id → {age, sex, diagnosis}
    """
    info_path = raw_dir / "ICBHI_patient_diagnosis.txt"
    if not info_path.exists():
        return {}

    patient_info = {}
    with open(info_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                pid = str(parts[0]).zfill(3)
                patient_info[pid] = {
                    "age": parts[1] if parts[1] != "NaN" else "40",
                    "sex": parts[2] if len(parts) > 2 else "unknown",
                    "diagnosis": parts[3] if len(parts) > 3 else "unknown",
                }
    return patient_info


# ── Parse recording metadata from filename ────────────────────────────────────
def parse_filename_metadata(stem: str) -> dict:
    """
    ICBHI filenames encode: <patient_id>_<recording_idx>_<mode>_<location>_<device>
    e.g. 101_1b1_Al_sc_Meditron
    """
    parts = stem.split("_")
    result = {
        "patient_id": parts[0].zfill(3) if len(parts) > 0 else "000",
        "recording_index": parts[1] if len(parts) > 1 else "unknown",
        "mode": parts[2] if len(parts) > 2 else "unknown",     # sc / mc / tc
        "location": parts[3] if len(parts) > 3 else "unknown",  # Tc, Al, Ar ...
        "device": parts[4] if len(parts) > 4 else "unknown",    # AKGC417L, Litt...
    }
    return result


# ── Load official train/test split ───────────────────────────────────────────
def load_official_split(raw_dir: Path) -> dict[str, str] | None:
    """
    Load the official ICBHI train/test split.

    The split file lists recording stems, one per line, with their split.
    Returns dict: recording_stem → "train" | "test", or None if not found.
    """
    for candidate in ["official_split.txt", "patient_list_foldwise.txt", "train_test_split.txt"]:
        split_path = raw_dir / candidate
        if split_path.exists():
            splits = {}
            with open(split_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        splits[parts[0]] = parts[1].lower()
            print(f"  Loaded official split from {candidate} ({len(splits)} entries)")
            return splits
    return None


# ── Main extraction logic ─────────────────────────────────────────────────────
def extract_cycles(
    raw_dir: Path,
    cycles_dir: Path,
    output_csv: Path,
    sample_rate: int = 22050,
) -> pd.DataFrame:
    """
    Extract all respiratory cycles from ICBHI raw files.

    For each recording:
      - Parse its .txt annotation file to get cycle timestamps
      - Slice the corresponding segment from the .wav
      - Save as <cycle_id>.wav in cycles_dir
      - Record all metadata in a DataFrame row

    Returns the complete metadata DataFrame.
    """
    cycles_dir.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    patient_info = parse_patient_info(raw_dir)
    official_split = load_official_split(raw_dir)

    # Collect all .wav files that have a corresponding .txt
    wav_files = sorted(raw_dir.glob("*.wav"))
    print(f"Found {len(wav_files)} .wav files in {raw_dir}")

    records = []
    skipped = 0

    for wav_path in tqdm(wav_files, desc="Extracting cycles"):
        stem = wav_path.stem
        txt_path = raw_dir / f"{stem}.txt"

        if not txt_path.exists():
            skipped += 1
            continue

        # Parse recording metadata from filename
        rec_meta = parse_filename_metadata(stem)
        pid = rec_meta["patient_id"]

        # Patient demographics
        demo = patient_info.get(pid, {})
        age  = float(demo.get("age", 40))
        sex  = demo.get("sex", "unknown")
        diagnosis = demo.get("diagnosis", "unknown")

        # Determine split
        if official_split is not None:
            split = official_split.get(stem, "train")
        else:
            # Deterministic fallback: hash patient_id to train/test
            # 60% train, 40% test — patient-level, never cycle-level
            split = "train" if (int(pid) % 10) < 6 else "test"

        # Load full waveform once per recording
        try:
            waveform, orig_sr = librosa.load(str(wav_path), sr=sample_rate, mono=True)
        except Exception as e:
            print(f"  Warning: could not load {wav_path}: {e}")
            skipped += 1
            continue

        # Parse annotation and extract cycles
        cycles = parse_annotation_file(txt_path)

        for cycle_idx, cycle in enumerate(cycles):
            start_sample = int(cycle["start"] * sample_rate)
            end_sample   = int(cycle["end"]   * sample_rate)

            # Guard against out-of-bounds
            start_sample = max(0, start_sample)
            end_sample   = min(len(waveform), end_sample)

            if end_sample <= start_sample:
                continue

            cycle_audio = waveform[start_sample:end_sample]

            # Skip extremely short cycles (< 0.1 s) — likely annotation errors
            if len(cycle_audio) < 0.1 * sample_rate:
                continue

            cycle_id = f"{stem}_{cycle_idx:03d}"
            out_path = cycles_dir / f"{cycle_id}.wav"
            sf.write(str(out_path), cycle_audio, sample_rate)

            records.append({
                "cycle_id":        cycle_id,
                "filename":        f"{cycle_id}.wav",
                "patient_id":      pid,
                "recording_stem":  stem,
                "cycle_index":     cycle_idx,
                "start_sec":       cycle["start"],
                "end_sec":         cycle["end"],
                "duration_sec":    cycle["end"] - cycle["start"],
                "label":           cycle["label"],
                "crackle":         cycle["crackle"],
                "wheeze":          cycle["wheeze"],
                "age":             age,
                "sex":             sex,
                "diagnosis":       diagnosis,
                "device":          rec_meta["device"],
                "location":        rec_meta["location"],
                "mode":            rec_meta["mode"],
                "split":           split,
            })

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)

    print(f"\n{'='*55}")
    print(f"  Extraction complete")
    print(f"  Total cycles extracted : {len(df)}")
    print(f"  Skipped recordings     : {skipped}")
    print(f"  Saved to               : {output_csv}")
    if len(df) > 0:
        print(f"\n  Label distribution:")
        for label, count in df["label"].value_counts().items():
            pct = 100 * count / len(df)
            print(f"    {label:10s}: {count:5d}  ({pct:.1f}%)")
        print(f"\n  Split distribution:")
        for split, count in df["split"].value_counts().items():
            print(f"    {split:6s}: {count:5d} cycles")
    else:
        print("\n  WARNING: No cycles extracted. Check --raw_dir path.")
    print(f"{'='*55}\n")

    return df


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ICBHI cycle extraction (Phase 1)")
    parser.add_argument("--raw_dir",    type=str, default="data/raw",
                        help="Directory containing ICBHI .wav and .txt files")
    parser.add_argument("--cycles_dir", type=str, default="data/cycles",
                        help="Output directory for extracted cycle .wav files")
    parser.add_argument("--output_csv", type=str, default="data/metadata.csv",
                        help="Path for the output metadata CSV")
    parser.add_argument("--sample_rate", type=int, default=22050,
                        help="Target sample rate (default: 22050)")
    args = parser.parse_args()

    extract_cycles(
        raw_dir=Path(args.raw_dir),
        cycles_dir=Path(args.cycles_dir),
        output_csv=Path(args.output_csv),
        sample_rate=args.sample_rate,
    )


if __name__ == "__main__":
    main()
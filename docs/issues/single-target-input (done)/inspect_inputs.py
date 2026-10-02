"""Read-only reference calculation, independent of the application loader.

Usage: python inspect_inputs.py PATH_TO_CONTAINER_OR_EXPERIMENT
Outputs JSON to stdout. Missing predictions are not zero-valued predictions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        sample = stream.read(8192)
        stream.seek(0)
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
        except csv.Error:
            delimiter = ","
        return list(csv.DictReader(stream, delimiter=delimiter))


def number(value: str) -> float:
    result = float(value.strip().replace(",", "."))
    if not math.isfinite(result):
        raise ValueError("Non-finite input")
    return result


def inspect(root: Path) -> dict:
    rows = read_csv(root / "results.csv")
    result = {"directory": root.name, "record_count": len(rows)}
    for target, column in (("fuel", "fuel"), ("steam", "diluent")):
        absolute, percentage = [], []
        for row in rows:
            predicted = row[f"predicted_{column}_flow"].strip()
            if not predicted:
                continue
            factor = {"g/h": 1.0, "kg/h": 1000.0, "g/s": 3600.0}[row["unit"]]
            expected = number(row[f"expected_{column}_flow"]) * factor
            error = abs(number(predicted) * factor - expected)
            absolute.append(error)
            if expected != 0:
                percentage.append(100 * error / abs(expected))
        result[target] = {
            "prediction_count": len(absolute),
            "mae_g_h": statistics.fmean(absolute) if absolute else None,
            "mape_percent": statistics.fmean(percentage) if percentage else None,
            "mape_count": len(percentage),
        }
    counts = Counter((number(r["expected_fuel_flow"]),
                      number(r["expected_diluent_flow"])) for r in rows)
    result["regimes"] = [
        {"fuel": fuel, "steam": steam, "record_count": count}
        for (fuel, steam), count in sorted(counts.items())
    ]
    result["sha256"] = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("results.csv", "video_index.csv", "source.json")
        if (root / name).is_file()
    }
    if (root / "video_index.csv").is_file():
        index = read_csv(root / "video_index.csv")
        ids = [r["frame_id"] for r in index]
        result["video_index"] = {
            "record_count": len(index),
            "unique_ids": len(set(ids)),
            "same_ids_as_results": set(ids) == {r["frame_id"] for r in rows},
            "records_by_video": dict(Counter(
                r["source_video"].replace("\\", "/").rsplit("/", 1)[-1]
                for r in index)),
        }
    if (root / "source.json").is_file():
        videos = json.loads((root / "source.json").read_text(encoding="utf-8-sig"))["videos"]
        result["source"] = {
            "video_count": len(videos),
            "statuses": dict(Counter(v["status"] for v in videos)),
            "completed_records": sum(v["completed_records"] for v in videos),
        }
    frames = root / "frames"
    if frames.is_dir():
        children = list(frames.iterdir())
        windows = [p for p in children if p.is_dir()]
        result["frame_inventory"] = {
            "primary_jpeg_count": sum(p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}
                                      for p in children),
            "window_count": len(windows),
            "windows_by_jpeg_count": dict(Counter(
                sum(p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}
                    for p in window.iterdir()) for window in windows)),
            "readability_verified": False,
        }
    return result


def discover(root: Path) -> list[Path]:
    if (root / "results.csv").is_file():
        return [root]
    # Deliberately limited depth, to avoid picking up generated report CSVs.
    return sorted({p.parent for pattern in ("*/results.csv", "*/*/results.csv")
                   for p in root.glob(pattern)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    roots = discover(args.root)
    if not roots:
        parser.error("No results.csv found at the supplied path or within two levels")
    print(json.dumps([inspect(root) for root in roots], ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

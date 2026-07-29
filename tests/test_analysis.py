from __future__ import annotations

import csv
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from PIL import Image

from burner_eye_report.analysis import analyze_experiment, calculate_metrics, split_three
from burner_eye_report.loader import load_experiment
from burner_eye_report.models import PredictionRow


FIELDS = [
    "timestamp",
    "frame_id",
    "unit",
    "expected_fuel_flow",
    "expected_diluent_flow",
    "trained_regime",
    "predicted_fuel_flow",
    "predicted_diluent_flow",
    "regime",
    "regime_confidence",
]


def write_experiment(root: Path, rows: list[dict[str, object]]) -> None:
    frames = root / "frames"
    frames.mkdir()
    with (root / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        frame_id = str(row["frame_id"])
        image = Image.new("RGB", (320, 180), color=(40 + int(float(frame_id)) * 10, 90, 130))
        image.save(frames / f"{frame_id}.jpg", quality=85)


def csv_row(
    frame_id: int,
    *,
    timestamp: datetime,
    unit: str = "g/h",
    fuel: float = 1000,
    steam: float = 800,
    trained: int = 1,
    predicted_fuel: float = 1010,
    predicted_steam: float = 790,
) -> dict[str, object]:
    return {
        "timestamp": timestamp.isoformat(timespec="milliseconds"),
        "frame_id": frame_id,
        "unit": unit,
        "expected_fuel_flow": fuel,
        "expected_diluent_flow": steam,
        "trained_regime": trained,
        "predicted_fuel_flow": predicted_fuel,
        "predicted_diluent_flow": predicted_steam,
        "regime": "model-label",
        "regime_confidence": 0.95,
    }


class LoaderTests(unittest.TestCase):
    def test_equivalent_units_form_one_conflicting_regime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            now = datetime(2026, 7, 1, 12, 0, 0)
            write_experiment(
                root,
                [
                    csv_row(
                        1,
                        timestamp=now,
                        unit="kg/h",
                        fuel=1.0,
                        steam=0.8,
                        trained=1,
                        predicted_fuel=1.01,
                        predicted_steam=0.79,
                    ),
                    csv_row(
                        2,
                        timestamp=now + timedelta(seconds=1),
                        unit="g/h",
                        fuel=1000,
                        steam=800,
                        trained=0,
                    ),
                ],
            )
            experiment = load_experiment(root)

            self.assertEqual(len(experiment.rows), 2)
            self.assertEqual(len(experiment.regimes), 1)
            regime = experiment.regimes[0]
            self.assertTrue(regime.is_conflict)
            self.assertIsNone(regime.final_status)
            self.assertFalse(experiment.can_generate)
            regime.final_status = True
            self.assertTrue(experiment.can_generate)
            self.assertEqual(regime.fuel_g_h, 1000)
            self.assertEqual(regime.steam_g_h, 800)

    def test_duplicate_frame_id_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            now = datetime(2026, 7, 1, 12, 0, 0)
            write_experiment(
                root,
                [
                    csv_row(1, timestamp=now),
                    csv_row(1, timestamp=now + timedelta(seconds=1)),
                ],
            )
            experiment = load_experiment(root)
            self.assertEqual(len(experiment.rows), 1)
            self.assertEqual(experiment.excluded_rows, 1)
            self.assertTrue(
                any(issue.code == "duplicate_frame_id" for issue in experiment.issues)
            )


class MetricsTests(unittest.TestCase):
    def test_zero_expected_is_kept_in_mae_and_excluded_from_mape(self) -> None:
        row = PredictionRow(
            timestamp=datetime(2026, 7, 1),
            frame_id="1",
            source_unit="g/h",
            expected_fuel=0,
            expected_steam=100,
            predicted_fuel=20,
            predicted_steam=110,
            expected_fuel_g_h=0,
            expected_steam_g_h=100,
            predicted_fuel_g_h=20,
            predicted_steam_g_h=110,
            trained_regime=True,
            predicted_regime="x",
            regime_confidence=0.9,
            source_row_number=2,
        )
        metrics = calculate_metrics([row])
        self.assertEqual(metrics.mae_fuel, 20)
        self.assertIsNone(metrics.mape_fuel)
        self.assertEqual(metrics.mape_fuel_count, 0)
        self.assertEqual(metrics.mape_steam, 10)

    def test_split_three_differs_by_at_most_one(self) -> None:
        base = datetime(2026, 7, 1)
        rows = [
            PredictionRow(
                timestamp=base + timedelta(seconds=index),
                frame_id=str(index),
                source_unit="g/h",
                expected_fuel=1,
                expected_steam=1,
                predicted_fuel=1,
                predicted_steam=1,
                expected_fuel_g_h=1,
                expected_steam_g_h=1,
                predicted_fuel_g_h=1,
                predicted_steam_g_h=1,
                trained_regime=True,
                predicted_regime="x",
                regime_confidence=None,
                source_row_number=index + 2,
            )
            for index in range(10)
        ]
        parts = split_three(reversed(rows))
        self.assertEqual([len(part) for part in parts], [4, 3, 3])
        self.assertEqual([part[0].frame_id for part in parts], ["0", "4", "7"])

    def test_frame_selection_never_repeats_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            now = datetime(2026, 7, 1, 12, 0)
            rows = [
                csv_row(
                    index,
                    timestamp=now + timedelta(seconds=index),
                    predicted_fuel=1000 + index * 10,
                    predicted_steam=800 - index * 5,
                )
                for index in range(1, 6)
            ]
            write_experiment(root, rows)
            analysis = analyze_experiment(load_experiment(root))
            ids = [item.row.frame_id for item in analysis.selected_frames]
            self.assertEqual(len(ids), 4)
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(
                sum(item.label.startswith("Лучший") for item in analysis.selected_frames),
                2,
            )
            self.assertEqual(
                sum(item.label.startswith("Худший") for item in analysis.selected_frames),
                2,
            )


if __name__ == "__main__":
    unittest.main()


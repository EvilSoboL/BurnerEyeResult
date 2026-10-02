from __future__ import annotations

import csv
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from pypdf import PdfReader

from burner_eye_report.comparison import analyze_comparison, metric_winner
from burner_eye_report.comparison_reporting import generate_comparison_report
from burner_eye_report.loader import load_experiment


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


def write_results(root: Path, rows: list[dict[str, object]]) -> None:
    root.mkdir()
    with (root / "results.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def result_row(
    frame_id: str,
    timestamp: datetime,
    *,
    fuel: float,
    steam: float,
    predicted_fuel: float,
    predicted_steam: float,
    unit: str = "g/h",
) -> dict[str, object]:
    return {
        "timestamp": timestamp.isoformat(timespec="milliseconds"),
        "frame_id": frame_id,
        "unit": unit,
        "expected_fuel_flow": fuel,
        "expected_diluent_flow": steam,
        "trained_regime": 1,
        "predicted_fuel_flow": predicted_fuel,
        "predicted_diluent_flow": predicted_steam,
        "regime": "model-label",
        "regime_confidence": 0.95,
    }


class ComparisonTests(unittest.TestCase):
    def test_compares_only_common_targets(self) -> None:
        cases = (("fuel", "fuel", {"fuel"}), ("steam", "steam", {"steam"}), ("both", "fuel", {"fuel"}))
        for first_target, second_target, expected in cases:
            with self.subTest(first=first_target, second=second_target), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                start = datetime(2026, 7, 30, 12, 0)
                first_rows = [result_row(f"a{index}", start + timedelta(seconds=index), fuel=100, steam=50, predicted_fuel=105 if first_target in {"fuel", "both"} else "", predicted_steam=48 if first_target in {"steam", "both"} else "") for index in range(3)]
                second_rows = [result_row(f"b{index}", start + timedelta(seconds=index), fuel=100, steam=50, predicted_fuel=105 if second_target in {"fuel", "both"} else "", predicted_steam=48 if second_target in {"steam", "both"} else "") for index in range(3)]
                first_root, second_root = root / "first", root / "second"
                write_results(first_root, first_rows)
                write_results(second_root, second_rows)

                comparison = analyze_comparison(load_experiment(first_root), load_experiment(second_root))

                self.assertEqual(comparison.comparable_targets, expected)
                if first_target == second_target:
                    pdf = generate_comparison_report(comparison)
                    self.assertGreaterEqual(len(PdfReader(str(pdf)).pages), 1)
                    self.assertTrue((pdf.parent / "csv" / "overall_model_comparison.csv").is_file())
        self.assertEqual(metric_winner(None, 1.0, "first", "second"), "Нет сопоставимых данных")
        self.assertEqual(metric_winner(1.0, 1.0, "first", "second"), "Одинаково")

    def test_different_targets_fail_even_with_shared_physical_regime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            start = datetime(2026, 7, 30, 12, 0)
            fuel_rows = [result_row(f"f{index}", start + timedelta(seconds=index), fuel=100, steam=50, predicted_fuel=105, predicted_steam="") for index in range(3)]
            steam_rows = [result_row(f"s{index}", start + timedelta(seconds=index), fuel=100, steam=50, predicted_fuel="", predicted_steam=48) for index in range(3)]
            first, second = root / "fuel", root / "steam"
            write_results(first, fuel_rows)
            write_results(second, steam_rows)

            with self.assertRaisesRegex(ValueError, "общей предсказываемой величины"):
                analyze_comparison(load_experiment(first), load_experiment(second))

    def test_common_target_must_exist_inside_regime_intersection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            start = datetime(2026, 7, 30, 12, 0)
            first_rows = [result_row(f"a{index}", start + timedelta(seconds=index), fuel=100, steam=50, predicted_fuel=105, predicted_steam="") for index in range(3)]
            first_rows += [result_row(f"b{index}", start + timedelta(seconds=10 + index), fuel=200, steam=80, predicted_fuel="", predicted_steam=78) for index in range(3)]
            second_rows = [result_row(f"c{index}", start + timedelta(seconds=index), fuel=100, steam=50, predicted_fuel="", predicted_steam=48) for index in range(3)]
            second_rows += [result_row(f"d{index}", start + timedelta(seconds=10 + index), fuel=300, steam=90, predicted_fuel=305, predicted_steam="") for index in range(3)]
            first, second = root / "first", root / "second"
            write_results(first, first_rows)
            write_results(second, second_rows)

            with self.assertRaisesRegex(ValueError, "общей предсказываемой величины"):
                analyze_comparison(load_experiment(first), load_experiment(second))

    def test_targets_in_different_common_regimes_are_not_comparable_overall(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            start = datetime(2026, 7, 30, 12, 0)
            first_rows = [result_row(f"a{index}", start + timedelta(seconds=index), fuel=100, steam=50, predicted_fuel=105, predicted_steam="") for index in range(3)]
            first_rows += [result_row(f"b{index}", start + timedelta(seconds=10 + index), fuel=200, steam=80, predicted_fuel="", predicted_steam=78) for index in range(3)]
            second_rows = [result_row(f"c{index}", start + timedelta(seconds=index), fuel=100, steam=50, predicted_fuel="", predicted_steam=48) for index in range(3)]
            second_rows += [result_row(f"d{index}", start + timedelta(seconds=10 + index), fuel=200, steam=80, predicted_fuel=205, predicted_steam="") for index in range(3)]
            first, second = root / "first", root / "second"
            write_results(first, first_rows)
            write_results(second, second_rows)

            with self.assertRaisesRegex(ValueError, "общей предсказываемой величины"):
                analyze_comparison(load_experiment(first), load_experiment(second))

    def test_zero_fact_mape_does_not_block_mae_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            start = datetime(2026, 7, 30, 12, 0)
            first, second = root / "first", root / "second"
            write_results(first, [result_row(f"a{index}", start + timedelta(seconds=index), fuel=0, steam=50, predicted_fuel=10, predicted_steam="") for index in range(3)])
            write_results(second, [result_row(f"b{index}", start + timedelta(seconds=index), fuel=0, steam=50, predicted_fuel=20, predicted_steam="") for index in range(3)])

            comparison = analyze_comparison(load_experiment(first), load_experiment(second))
            pdf = generate_comparison_report(comparison)
            with (pdf.parent / "csv" / "metrics_by_common_regime.csv").open(encoding="utf-8-sig", newline="") as stream:
                regime = next(csv.DictReader(stream))
            text = "\n".join(page.extract_text() or "" for page in PdfReader(str(pdf)).pages)

            self.assertEqual(comparison.common_regimes[0].first_metrics.mae_fuel, 10)
            self.assertIsNone(comparison.common_regimes[0].first_metrics.mape_fuel)
            self.assertEqual(regime["mae_fuel_winner"], "first")
            self.assertEqual(regime["mape_fuel_winner"], "Нет сопоставимых данных")
            self.assertIn("топливо: MAE — first", text)

    def test_mixed_regime_coverage_has_no_false_winner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            start = datetime(2026, 7, 30, 12, 0)
            first_rows = [result_row(f"a{index}", start + timedelta(seconds=index), fuel=100, steam=50, predicted_fuel=105, predicted_steam="") for index in range(3)]
            first_rows += [result_row(f"b{index}", start + timedelta(seconds=10 + index), fuel=200, steam=80, predicted_fuel="", predicted_steam=78) for index in range(3)]
            second_rows = [result_row(f"c{index}", start + timedelta(seconds=index), fuel=100, steam=50, predicted_fuel=103, predicted_steam="") for index in range(3)]
            second_rows += [result_row(f"d{index}", start + timedelta(seconds=10 + index), fuel=200, steam=80, predicted_fuel=203, predicted_steam="") for index in range(3)]
            first, second = root / "first", root / "second"
            write_results(first, first_rows)
            write_results(second, second_rows)

            comparison = analyze_comparison(load_experiment(first), load_experiment(second))
            pdf = generate_comparison_report(comparison)
            with (pdf.parent / "csv" / "metrics_by_common_regime.csv").open(encoding="utf-8-sig", newline="") as stream:
                regimes = list(csv.DictReader(stream))

            self.assertEqual(regimes[0]["mae_fuel_winner"], "second")
            self.assertEqual(regimes[1]["mae_steam_winner"], "Нет сопоставимых данных")
            self.assertEqual(regimes[1]["mae_fuel_winner"], "Нет сопоставимых данных")

    def test_uses_only_physical_regime_intersection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_root = root / "model_alpha"
            second_root = root / "model_beta"
            start = datetime(2026, 7, 30, 12, 0)
            first_rows = [
                result_row(
                    f"a{index}",
                    start + timedelta(seconds=index),
                    fuel=100,
                    steam=50,
                    predicted_fuel=101,
                    predicted_steam=49,
                )
                for index in range(2)
            ]
            first_rows.extend(
                result_row(
                    f"b{index}",
                    start + timedelta(seconds=10 + index),
                    fuel=200,
                    steam=100,
                    predicted_fuel=210,
                    predicted_steam=95,
                )
                for index in range(3)
            )
            second_rows = [
                result_row(
                    f"b{index}",
                    start + timedelta(seconds=10 + index),
                    unit="kg/h",
                    fuel=0.2,
                    steam=0.1,
                    predicted_fuel=0.204,
                    predicted_steam=0.102,
                )
                for index in range(4)
            ]
            second_rows.extend(
                result_row(
                    f"c{index}",
                    start + timedelta(seconds=20 + index),
                    fuel=300,
                    steam=150,
                    predicted_fuel=290,
                    predicted_steam=145,
                )
                for index in range(2)
            )
            write_results(first_root, first_rows)
            write_results(second_root, second_rows)

            comparison = analyze_comparison(
                load_experiment(first_root),
                load_experiment(second_root),
            )

            self.assertEqual(len(comparison.common_regimes), 1)
            self.assertEqual(len(comparison.first_only_regimes), 1)
            self.assertEqual(len(comparison.second_only_regimes), 1)
            self.assertEqual(comparison.first_overall_metrics.record_count, 3)
            self.assertEqual(comparison.second_overall_metrics.record_count, 4)
            self.assertEqual(comparison.first_overall_metrics.mae_fuel, 10)
            self.assertAlmostEqual(
                comparison.second_overall_metrics.mae_fuel or 0,
                4,
            )
            self.assertEqual(comparison.common_regimes[0].key, (200.0, 100.0))

    def test_generates_comparison_pdf_and_csv_without_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_root = root / "first_model"
            second_root = root / "second_model"
            start = datetime(2026, 7, 30, 12, 0)
            write_results(
                first_root,
                [
                    result_row(
                        f"first-{index}",
                        start + timedelta(seconds=index),
                        fuel=1000,
                        steam=800,
                        predicted_fuel=1000 + index * 10,
                        predicted_steam=800 - index * 5,
                    )
                    for index in range(1, 6)
                ],
            )
            write_results(
                second_root,
                [
                    result_row(
                        f"second-{index}",
                        start + timedelta(seconds=index),
                        fuel=1000,
                        steam=800,
                        predicted_fuel=1000 + index * 5,
                        predicted_steam=800 - index * 2,
                    )
                    for index in range(1, 6)
                ],
            )

            comparison = analyze_comparison(
                load_experiment(first_root),
                load_experiment(second_root),
            )
            pdf_path = generate_comparison_report(comparison)

            self.assertTrue(pdf_path.is_file())
            self.assertGreater(pdf_path.stat().st_size, 10_000)
            self.assertGreaterEqual(len(PdfReader(str(pdf_path)).pages), 4)
            self.assertEqual(
                {path.name for path in (pdf_path.parent / "csv").iterdir()},
                {
                    "overall_model_comparison.csv",
                    "metrics_by_common_regime.csv",
                    "prediction_timeline_comparison.csv",
                    "excluded_regimes.csv",
                },
            )
            self.assertTrue((pdf_path.parent / "processing.log").is_file())


if __name__ == "__main__":
    unittest.main()

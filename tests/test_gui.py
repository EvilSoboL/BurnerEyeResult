from __future__ import annotations

import csv
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from PIL import Image

from burner_eye_report.gui import load_selected_experiment, resolve_experiment_directory
from burner_eye_report.gui import BurnerEyeReportApp
from burner_eye_report.models import MetricSet


class ExperimentDirectoryResolutionTests(unittest.TestCase):
    @staticmethod
    def make_experiment(path: Path) -> Path:
        path.mkdir(parents=True)
        (path / "results.csv").write_text("frame_id\n", encoding="utf-8")
        return path.resolve()

    def test_explicit_experiment_path_is_kept(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_experiment(Path(directory) / "experiment")
            self.assertEqual(resolve_experiment_directory(root), root)

    def test_unique_candidate_resolves_at_one_or_two_levels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = self.make_experiment(root / "Расход пара" / "video_prediction")
            report = root / "Расход пара" / "video_prediction" / "prediction_report_2026-01-01"
            report.mkdir()
            (report / "results.csv").write_text("report should not be selected", encoding="utf-8")

            self.assertEqual(resolve_experiment_directory(root), experiment)
            self.assertEqual(resolve_experiment_directory(root / "Расход пара"), experiment)

    def test_multiple_candidates_are_listed_without_arbitrary_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.make_experiment(root / "Расход пара" / "experiment-a")
            second = self.make_experiment(root / "Расход топлива" / "experiment-b")

            with self.assertRaisesRegex(ValueError, "несколько экспериментов") as raised:
                resolve_experiment_directory(root)

            self.assertIn(str(first), str(raised.exception))
            self.assertIn(str(second), str(raised.exception))

    def test_no_candidate_leaves_original_path_for_loader_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(resolve_experiment_directory(root), root.resolve())

    def test_selected_parent_loads_unique_single_target_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / "Расход топлива"
            experiment = parent / "video_prediction"
            frames = experiment / "frames"
            frames.mkdir(parents=True)
            fields = ["timestamp", "frame_id", "unit", "expected_fuel_flow", "expected_diluent_flow", "trained_regime", "predicted_fuel_flow", "predicted_diluent_flow", "regime", "regime_confidence"]
            now = datetime(2026, 7, 1, 12, 0)
            with (experiment / "results.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for index in range(1, 5):
                    writer.writerow({
                        "timestamp": (now + timedelta(seconds=index)).isoformat(),
                        "frame_id": index,
                        "unit": "g/h",
                        "expected_fuel_flow": 100,
                        "expected_diluent_flow": 50,
                        "trained_regime": 1,
                        "predicted_fuel_flow": 101,
                        "predicted_diluent_flow": "",
                        "regime": "",
                        "regime_confidence": "",
                    })
                    Image.new("RGB", (16, 16), color=(20, 50, 80)).save(frames / f"{index}.jpg")

            loaded = load_selected_experiment(parent)

            self.assertEqual(loaded.root, experiment.resolve())
            self.assertEqual(len(loaded.rows), 4)
            self.assertEqual(loaded.predicted_fuel_count, 4)
            self.assertEqual(loaded.predicted_steam_count, 0)
            self.assertTrue(loaded.can_generate)

    def test_comparison_channel_display_handles_steam_only_and_zero_fact(self) -> None:
        first = MetricSet(4, None, None, 0, 5.0, 10.0, 4, 0, 4)
        second = MetricSet(4, None, None, 0, 3.0, 6.0, 4, 0, 4)
        first_view = BurnerEyeReportApp._format_comparison_channel(first, second, "steam", "first", "second")
        fuel_view = BurnerEyeReportApp._format_comparison_channel(first, second, "fuel", "first", "second")
        self.assertEqual(first_view, ("10.000 %", "6.000 %", "second"))
        self.assertEqual(fuel_view, ("Нет сопоставимых данных",) * 3)

        first.mape_steam = None
        second.mape_steam = None
        first.mape_steam_count = second.mape_steam_count = 0
        zero_fact_view = BurnerEyeReportApp._format_comparison_channel(first, second, "steam", "first", "second")
        self.assertEqual(zero_fact_view, ("5.000 г/ч", "3.000 г/ч", "second"))


if __name__ == "__main__":
    unittest.main()

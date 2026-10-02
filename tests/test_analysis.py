from __future__ import annotations

import csv
import json
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
    def test_video_metadata_bom_linking_and_nonblocking_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            now = datetime(2026, 7, 1, 12, 0, 0)
            write_experiment(root, [csv_row(1, timestamp=now), csv_row(2, timestamp=now + timedelta(seconds=1))])
            (root / "video_index.csv").write_text(
                "\ufeffframe_id,source_video,start_s,end_s,source_frame_indices\n"
                "1,clip.mkv,0,1,0;2;4\n1,dup.mkv,0,1,0;1\n2,clip.mkv,1,2,60;62;64\n9,extra.mkv,0,1,0;1\n",
                encoding="utf-8",
            )
            source = {
                "videos": [{
                    "source_video": "clip.mkv", "status": "error", "stop_reason": "read failed",
                    "completed_records": 1, "expected": {"fuel_flow": 999, "diluent_flow": 800, "unit": "g/h", "trained_regime": True},
                    "window": {"num_frames": 3, "clip_duration_s": 1.0}, "model_path": "missing/model.pt",
                }]
            }
            (root / "source.json").write_text("\ufeff" + json.dumps(source), encoding="utf-8")

            experiment = load_experiment(root)

            self.assertEqual(len(experiment.rows), 2)
            self.assertIsNone(experiment.rows[0].video_metadata)
            self.assertEqual(experiment.rows[1].video_metadata.source_frame_indices, [60, 62, 64])
            self.assertEqual(experiment.video_sources[0].model_path, "missing/model.pt")
            self.assertFalse(experiment.blocking_issues)
            codes = {issue.code for issue in experiment.issues}
            self.assertTrue({"video_index_duplicate_id", "video_index_missing_ids", "video_index_extra_ids", "source_completed_records_mismatch", "source_video_partial", "source_expected_mismatch"}.issubset(codes))

    def test_damaged_optional_metadata_never_blocks_results(self) -> None:
        for filename, contents, expected_code in (
            ("source.json", "{bad json", "source_json_unreadable"),
            ("video_index.csv", "wrong,columns\n1,x\n", "video_index_schema"),
        ):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                write_experiment(root, [csv_row(1, timestamp=datetime(2026, 7, 1, 12, 0))])
                (root / filename).write_text(contents, encoding="utf-8")

                experiment = load_experiment(root)

                self.assertEqual(len(experiment.rows), 1)
                self.assertFalse(experiment.blocking_issues)
                self.assertIn(expected_code, {issue.code for issue in experiment.issues})

    def test_invalid_video_index_records_are_skipped_individually(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            now = datetime(2026, 7, 1, 12, 0, 0)
            write_experiment(root, [csv_row(1, timestamp=now), csv_row(2, timestamp=now + timedelta(seconds=1))])
            (root / "video_index.csv").write_text(
                "frame_id,source_video,start_s,end_s,source_frame_indices\n"
                "1,clip.mkv,2,1,0;1\n2,clip.mkv,1,2,4;2\n",
                encoding="utf-8",
            )

            experiment = load_experiment(root)

            self.assertEqual(len(experiment.rows), 2)
            self.assertTrue(all(row.video_metadata is None for row in experiment.rows))
            invalid = [issue for issue in experiment.issues if issue.code == "video_index_record_invalid"]
            self.assertEqual(len(invalid), 2)
            self.assertFalse(experiment.blocking_issues)

    def test_optional_prediction_channels_and_unit_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            now = datetime(2026, 7, 1, 12, 0, 0)
            rows = [
                csv_row(1, timestamp=now, unit="kg/h", predicted_fuel="", predicted_steam=0),
                csv_row(2, timestamp=now + timedelta(seconds=1), unit="g/s", predicted_fuel=2, predicted_steam=" "),
            ]
            write_experiment(root, rows)
            experiment = load_experiment(root)

            self.assertEqual(len(experiment.rows), 2)
            self.assertEqual(experiment.predicted_fuel_count, 1)
            self.assertEqual(experiment.predicted_steam_count, 1)
            self.assertEqual(experiment.available_targets, {"fuel", "steam"})
            self.assertIsNone(experiment.rows[0].predicted_fuel_g_h)
            self.assertEqual(experiment.rows[0].predicted_steam_g_h, 0)
            self.assertEqual(experiment.rows[1].predicted_fuel_g_h, 7200)
            self.assertIsNone(experiment.rows[1].predicted_steam_g_h)

    def test_both_empty_and_invalid_optional_prediction_are_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            now = datetime(2026, 7, 1, 12, 0, 0)
            rows = [
                csv_row(1, timestamp=now, predicted_fuel="", predicted_steam="  "),
                csv_row(2, timestamp=now + timedelta(seconds=1), predicted_fuel="NaN", predicted_steam=2),
                csv_row(3, timestamp=now + timedelta(seconds=2), predicted_fuel=3, predicted_steam="Inf"),
                csv_row(4, timestamp=now + timedelta(seconds=3), predicted_fuel=0, predicted_steam=""),
            ]
            write_experiment(root, rows)
            experiment = load_experiment(root)

            self.assertEqual([row.frame_id for row in experiment.rows], ["4"])
            self.assertEqual(experiment.excluded_rows, 3)
            self.assertEqual(experiment.predicted_fuel_count, 1)
            self.assertEqual(experiment.predicted_steam_count, 0)

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
    def test_mixed_channel_metrics_and_frame_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            now = datetime(2026, 7, 1, 12, 0)
            rows = [
                csv_row(1, timestamp=now, fuel=100, steam=50, predicted_fuel=110, predicted_steam=""),
                csv_row(2, timestamp=now + timedelta(seconds=1), fuel=100, steam=50, predicted_fuel="", predicted_steam=45),
            ]
            write_experiment(root, rows)
            analysis = analyze_experiment(load_experiment(root))

            metrics = analysis.overall_metrics
            self.assertEqual(metrics.record_count, 2)
            self.assertEqual((metrics.mae_fuel, metrics.mae_fuel_count), (10, 1))
            self.assertEqual((metrics.mape_fuel, metrics.mape_fuel_count), (10, 1))
            self.assertEqual((metrics.mae_steam, metrics.mae_steam_count), (5, 1))
            self.assertEqual((metrics.mape_steam, metrics.mape_steam_count), (10, 1))
            self.assertIsNone(analysis.row_metrics["1"]["signed_error_steam"])
            self.assertIsNone(analysis.row_metrics["2"]["absolute_error_fuel"])
            self.assertTrue(any("Состав каналов" in note for note in analysis.processing_notes))
            self.assertTrue(all(frame.abs_error_fuel is None or frame.abs_error_steam is None for frame in analysis.selected_frames))

    def test_fully_missing_channel_and_direction_conclusion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            now = datetime(2026, 7, 1, 12, 0)
            rows = [
                csv_row(index, timestamp=now + timedelta(seconds=index), predicted_fuel=1000 + index, predicted_steam="")
                for index in range(1, 5)
            ]
            write_experiment(root, rows)
            analysis = analyze_experiment(load_experiment(root))

            self.assertIsNone(analysis.overall_metrics.mae_steam)
            self.assertIsNone(analysis.overall_metrics.mape_steam)
            self.assertEqual(analysis.overall_metrics.mae_steam_count, 0)
            self.assertEqual(analysis.overall_metrics.mape_steam_count, 0)
            self.assertIn("недоступно", analysis.error_conclusions["overall_steam"])
            self.assertTrue(all(frame.abs_error_steam is None for frame in analysis.selected_frames))

    def test_stage_conclusion_requires_a_common_channel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            now = datetime(2026, 7, 1, 12, 0)
            rows = []
            for index in range(6):
                fuel_prediction = 1010 if index < 2 else (1010 if index < 4 else "")
                steam_prediction = 790 if index >= 2 else ""
                rows.append(
                    csv_row(
                        index + 1,
                        timestamp=now + timedelta(seconds=index),
                        predicted_fuel=fuel_prediction,
                        predicted_steam=steam_prediction,
                    )
                )
            write_experiment(root, rows)
            analysis = analyze_experiment(load_experiment(root))

            self.assertIn("нет каналов с сопоставимыми прогнозами", analysis.overall_conclusion)

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


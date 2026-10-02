from __future__ import annotations

import csv
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from PIL import Image
from pypdf import PdfReader

from burner_eye_report.loader import load_experiment
from burner_eye_report.reporting import generate_report


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


class ReportIntegrationTests(unittest.TestCase):
    def test_single_channel_reports_export_empty_values_and_provenance(self) -> None:
        expected_files = {
            "overall_metrics.csv",
            "trained_untrained_metrics.csv",
            "metrics_by_regime.csv",
            "metrics_by_regime_stage.csv",
            "overall_stage_metrics.csv",
            "prediction_timeline.csv",
            "error_distribution.csv",
            "selected_frames.csv",
            "regime_classification.csv",
        }
        for channel in ("fuel", "steam"):
            with self.subTest(channel=channel), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                frames = root / "frames"
                frames.mkdir()
                rows: list[dict[str, object]] = []
                start = datetime(2026, 7, 20, 10, 0)
                for index in range(1, 5):
                    rows.append({
                        "timestamp": (start + timedelta(seconds=index)).isoformat(timespec="milliseconds"),
                        "frame_id": index,
                        "unit": "g/h",
                        "expected_fuel_flow": 100,
                        "expected_diluent_flow": 0 if channel == "fuel" else 50,
                        "trained_regime": 1,
                        "predicted_fuel_flow": 100 + (index - 1) * 10 if channel == "fuel" else "",
                        "predicted_diluent_flow": 50 if channel == "steam" else "",
                        "regime": "model",
                        "regime_confidence": "",
                    })
                    Image.new("RGB", (320, 180), color=(20 * index, 80, 120)).save(frames / f"{index}.jpg")
                with (root / "results.csv").open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=FIELDS)
                    writer.writeheader()
                    writer.writerows(rows)
                (root / "video_index.csv").write_text(
                    "frame_id,source_video,start_s,end_s,source_frame_indices\n" +
                    "".join(f"{index},clip.mkv,{index - 1},{index},0;2;4\n" for index in range(1, 5)),
                    encoding="utf-8",
                )
                (root / "source.json").write_text(
                    '{"videos":[{"source_video":"clip.mkv","status":"error",'
                    '"stop_reason":"fixture partial","completed_records":4}]}' ,
                    encoding="utf-8",
                )

                pdf_path = generate_report(load_experiment(root))

                csv_dir = pdf_path.parent / "csv"
                self.assertEqual({path.name for path in csv_dir.iterdir()}, expected_files)
                self.assertTrue((pdf_path.parent / "processing.log").is_file())
                text = "\n".join(page.extract_text() or "" for page in PdfReader(str(pdf_path)).pages)
                self.assertIn("Прогнозы по каналам", text)
                self.assertIn("fixture partial", text)
                self.assertIn("clip.mkv", text)
                self.assertIn("Время обработки", text)
                self.assertIn("Нет прогнозов", text)
                with (csv_dir / "overall_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
                    metric = next(csv.DictReader(stream))
                predicted_key = "mae_fuel_record_count" if channel == "fuel" else "mae_steam_record_count"
                absent_key = "mae_steam_record_count" if channel == "fuel" else "mae_fuel_record_count"
                self.assertEqual(metric[predicted_key], "4")
                self.assertEqual(metric[absent_key], "0")
                with (csv_dir / "prediction_timeline.csv").open(encoding="utf-8-sig", newline="") as stream:
                    timeline = list(csv.DictReader(stream))
                absent_prediction = "predicted_steam_g_h" if channel == "fuel" else "predicted_fuel_g_h"
                absent_reason = "steam_mape_exclusion_reason" if channel == "fuel" else "fuel_mape_exclusion_reason"
                self.assertTrue(all(row[absent_prediction] == "" for row in timeline))
                self.assertTrue(all(row[absent_reason] == "prediction_missing" for row in timeline))
                self.assertEqual(timeline[0]["timestamp_type"], "processing_time")
                self.assertEqual(timeline[0]["source_video"], "clip.mkv")
                with (csv_dir / "error_distribution.csv").open(encoding="utf-8-sig", newline="") as stream:
                    errors = list(csv.DictReader(stream))
                absent_error = "signed_error_steam_g_h" if channel == "fuel" else "signed_error_fuel_g_h"
                self.assertTrue(all(row[absent_error] == "" for row in errors))
                if channel == "fuel":
                    self.assertEqual(errors[0]["signed_error_fuel_g_h"], "0")

    def test_generates_openable_pdf_and_all_csv_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = root / "frames"
            frames.mkdir()
            rows: list[dict[str, object]] = []
            start = datetime(2026, 7, 20, 10, 0)
            for index in range(1, 9):
                second_regime = index > 4
                fuel = 1500 if not second_regime else 1200
                steam = 800 if not second_regime else 600
                rows.append(
                    {
                        "timestamp": (start + timedelta(seconds=index)).isoformat(
                            timespec="milliseconds"
                        ),
                        "frame_id": index,
                        "unit": "g/h",
                        "expected_fuel_flow": fuel,
                        "expected_diluent_flow": steam,
                        "trained_regime": 1 if not second_regime else 0,
                        "predicted_fuel_flow": fuel + (index - 4) * 8,
                        "predicted_diluent_flow": steam - (index - 3) * 5,
                        "regime": f"model-{int(second_regime)}",
                        "regime_confidence": 0.93,
                    }
                )
                Image.new(
                    "RGB",
                    (640, 360),
                    color=(35 + index * 12, 75 + index * 4, 125),
                ).save(frames / f"{index}.jpg", quality=88)
            with (root / "results.csv").open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)

            experiment = load_experiment(root)
            self.assertTrue(experiment.can_generate)
            pdf_path = generate_report(experiment)

            self.assertTrue(pdf_path.is_file())
            self.assertGreater(pdf_path.stat().st_size, 20_000)
            reader = PdfReader(str(pdf_path))
            self.assertGreaterEqual(len(reader.pages), 8)
            expected_files = {
                "overall_metrics.csv",
                "trained_untrained_metrics.csv",
                "metrics_by_regime.csv",
                "metrics_by_regime_stage.csv",
                "overall_stage_metrics.csv",
                "prediction_timeline.csv",
                "error_distribution.csv",
                "selected_frames.csv",
                "regime_classification.csv",
            }
            csv_dir = pdf_path.parent / "csv"
            self.assertEqual({path.name for path in csv_dir.iterdir()}, expected_files)
            self.assertTrue((pdf_path.parent / "processing.log").is_file())

            with (csv_dir / "selected_frames.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as stream:
                selected = list(csv.DictReader(stream))
            self.assertEqual(len(selected), 4)
            self.assertEqual(len({row["frame_id"] for row in selected}), 4)


if __name__ == "__main__":
    unittest.main()


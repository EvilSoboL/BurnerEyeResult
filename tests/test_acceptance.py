from __future__ import annotations

import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from pypdf import PdfReader

from burner_eye_report.analysis import analyze_experiment
from burner_eye_report.loader import load_experiment
from burner_eye_report.reporting import generate_report


PACKAGE = Path(__file__).resolve().parents[1] / "docs" / "issues" / "single-target-input"


class SampleAcceptanceTests(unittest.TestCase):
    def test_sample_to_analysis_and_report(self) -> None:
        evidence = json.loads((PACKAGE / "evidence.json").read_text(encoding="utf-8"))
        for expected in evidence["samples"]:
            with self.subTest(sample=expected["directory"]), tempfile.TemporaryDirectory() as tmp:
                sample_root = PACKAGE / "samples" / expected["directory"]
                root = Path(tmp) / expected["directory"]
                root.mkdir()
                for name in ("results.csv", "video_index.csv", "source.json"):
                    shutil.copyfile(sample_root / name, root / name)

                with (root / "results.csv").open(encoding="utf-8-sig", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                frames = root / "frames"
                frames.mkdir()
                for row in rows:
                    Image.new("RGB", (32, 24), color=(30, 80, 120)).save(
                        frames / f"{row['frame_id']}.jpg", format="JPEG"
                    )

                experiment = load_experiment(root)
                analysis = analyze_experiment(experiment)
                target = expected["target"]
                other = "steam" if target == "fuel" else "fuel"
                metrics = analysis.overall_metrics
                self.assertEqual(len(experiment.rows), 18)
                self.assertEqual(experiment.excluded_rows, 0)
                self.assertEqual(len(experiment.regimes), 6)
                self.assertTrue(all(regime.final_status for regime in experiment.regimes))
                self.assertEqual(getattr(metrics, f"mae_{target}_count"), 18)
                self.assertEqual(getattr(metrics, f"mae_{other}_count"), 0)
                self.assertAlmostEqual(
                    getattr(metrics, f"mae_{target}"),
                    expected[target]["mae_g_h"],
                    delta=1e-9,
                )
                self.assertAlmostEqual(
                    getattr(metrics, f"mape_{target}"),
                    expected[target]["mape_percent"],
                    delta=1e-9,
                )
                self.assertEqual(len(analysis.selected_frames), 4)
                self.assertEqual(
                    len({selected.row.frame_id for selected in analysis.selected_frames}), 4
                )

                pdf_path = generate_report(experiment)
                self.assertGreater(len(PdfReader(str(pdf_path)).pages), 1)
                self.assertTrue((pdf_path.parent / "processing.log").is_file())
                processing_log = (pdf_path.parent / "processing.log").read_text(encoding="utf-8")
                self.assertIn("source_video_partial", processing_log)
                self.assertIn("Ошибка чтения последнего кадра", processing_log)
                csv_dir = pdf_path.parent / "csv"
                self.assertEqual(len(list(csv_dir.glob("*.csv"))), 9)
                with (csv_dir / "prediction_timeline.csv").open(
                    encoding="utf-8-sig", newline=""
                ) as stream:
                    timeline = list(csv.DictReader(stream))
                self.assertEqual(len(timeline), 18)
                prediction_key = f"predicted_{target}_g_h"
                missing_key = f"predicted_{other}_g_h"
                self.assertTrue(all(row[prediction_key] for row in timeline))
                self.assertTrue(all(row[missing_key] == "" for row in timeline))
                self.assertTrue(all(row["source_video"] for row in timeline))
                self.assertTrue(all(row["source_frame_indices"] for row in timeline))


if __name__ == "__main__":
    unittest.main()

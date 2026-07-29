from __future__ import annotations

import csv
import math
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .analysis import analyze_experiment
from .models import (
    AnalysisResult,
    ExperimentData,
    MetricSet,
    Regime,
    SelectedFrame,
    StageResult,
    status_label,
)


ProgressCallback = Callable[[int, str], None]

INK = colors.HexColor("#172033")
MUTED = colors.HexColor("#64748B")
GRID = colors.HexColor("#D9E2EC")
PANEL = colors.HexColor("#F5F8FC")
ACCENT = colors.HexColor("#0F6CBD")
TEAL = colors.HexColor("#0F9D8A")
ORANGE = colors.HexColor("#E47C22")
RED = colors.HexColor("#C43D4B")
PURPLE = colors.HexColor("#7C5AC7")


def generate_report(
    experiment: ExperimentData,
    threshold_percent: float = 5.0,
    *,
    progress: ProgressCallback | None = None,
) -> Path:
    _progress(progress, 5, "Расчёт метрик")
    analysis = analyze_experiment(experiment, threshold_percent)
    output_dir = _create_output_directory(experiment.root)
    csv_dir = output_dir / "csv"
    try:
        csv_dir.mkdir()
    except OSError as exc:
        raise RuntimeError(f"Не удалось создать папку CSV: {exc}") from exc

    _progress(progress, 20, "Сохранение расчётных CSV")
    _write_all_csv(analysis, csv_dir)
    _write_processing_log(analysis, output_dir / "processing.log")

    pdf_path = output_dir / "prediction_report.pdf"
    _progress(progress, 45, "Формирование PDF")
    _build_pdf(analysis, pdf_path)
    _progress(progress, 90, "Проверка PDF")
    _verify_pdf(pdf_path)
    _progress(progress, 100, "Отчёт готов")
    return pdf_path


def _create_output_directory(root: Path) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base = root / f"prediction_report_{stamp}"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = root / f"{base.name}_{suffix:02d}"
        suffix += 1
    try:
        candidate.mkdir()
    except OSError as exc:
        raise RuntimeError(f"Не удалось создать папку отчёта: {exc}") from exc
    return candidate


def _write_all_csv(analysis: AnalysisResult, csv_dir: Path) -> None:
    overall_rows = [{"scope": "experiment", "status": "all", **analysis.overall_metrics.as_dict()}]
    _write_csv(
        csv_dir / "overall_metrics.csv",
        overall_rows,
        ["scope", "status", *_metric_fields()],
    )

    status_rows = [
        {
            "status": "trained" if status else "untrained",
            **metrics.as_dict(),
        }
        for status, metrics in analysis.trained_untrained_metrics.items()
    ]
    _write_csv(
        csv_dir / "trained_untrained_metrics.csv",
        status_rows,
        ["status", *_metric_fields()],
    )

    regime_rows = []
    for regime in analysis.experiment.regimes:
        regime_rows.append(
            {
                "regime_id": regime.regime_id,
                "regime_name": regime.display_name,
                "status": "trained" if regime.final_status else "untrained",
                "expected_fuel_g_h": regime.fuel_g_h,
                "expected_steam_g_h": regime.steam_g_h,
                "source_units": "|".join(regime.source_units),
                **analysis.regime_metrics[regime.regime_id].as_dict(),
            }
        )
    _write_csv(
        csv_dir / "metrics_by_regime.csv",
        regime_rows,
        [
            "regime_id",
            "regime_name",
            "status",
            "expected_fuel_g_h",
            "expected_steam_g_h",
            "source_units",
            *_metric_fields(),
        ],
    )

    stage_rows: list[dict[str, object]] = []
    for regime_id, stages in analysis.regime_stages.items():
        for stage in stages:
            stage_rows.append(_stage_csv_row(stage, regime_id=regime_id))
    _write_csv(
        csv_dir / "metrics_by_regime_stage.csv",
        stage_rows,
        ["regime_id", "stage", "start_time", "end_time", *_metric_fields()],
    )

    overall_stage_rows = [
        _stage_csv_row(stage, analysis_scope="experiment")
        for stage in analysis.overall_stages
    ]
    for status, stages in analysis.group_stages.items():
        scope = "trained" if status else "untrained"
        overall_stage_rows.extend(
            _stage_csv_row(stage, analysis_scope=scope) for stage in stages
        )
    _write_csv(
        csv_dir / "overall_stage_metrics.csv",
        overall_stage_rows,
        ["analysis_scope", "stage", "start_time", "end_time", *_metric_fields()],
    )

    regime_lookup = {
        regime.regime_id: regime for regime in analysis.experiment.regimes
    }
    timeline_rows = []
    distribution_rows = []
    for row in analysis.experiment.rows:
        regime = regime_lookup[row.regime_id]
        metrics = analysis.row_metrics[row.frame_id]
        base = {
            "timestamp": row.timestamp.isoformat(timespec="milliseconds"),
            "frame_id": row.frame_id,
            "regime_id": row.regime_id,
            "regime_name": regime.display_name,
            "status": "trained" if regime.final_status else "untrained",
            "source_unit": row.source_unit,
        }
        timeline_rows.append(
            {
                **base,
                "expected_fuel_g_h": row.expected_fuel_g_h,
                "predicted_fuel_g_h": row.predicted_fuel_g_h,
                "absolute_error_fuel_g_h": metrics["absolute_error_fuel"],
                "absolute_percentage_error_fuel_percent": metrics["ape_fuel"],
                "fuel_mape_excluded": int(metrics["ape_fuel"] is None),
                "expected_steam_g_h": row.expected_steam_g_h,
                "predicted_steam_g_h": row.predicted_steam_g_h,
                "absolute_error_steam_g_h": metrics["absolute_error_steam"],
                "absolute_percentage_error_steam_percent": metrics["ape_steam"],
                "steam_mape_excluded": int(metrics["ape_steam"] is None),
            }
        )
        distribution_rows.append(
            {
                **base,
                "signed_error_fuel_g_h": metrics["signed_error_fuel"],
                "signed_error_steam_g_h": metrics["signed_error_steam"],
            }
        )
    _write_csv(
        csv_dir / "prediction_timeline.csv",
        timeline_rows,
        [
            "timestamp",
            "frame_id",
            "regime_id",
            "regime_name",
            "status",
            "source_unit",
            "expected_fuel_g_h",
            "predicted_fuel_g_h",
            "absolute_error_fuel_g_h",
            "absolute_percentage_error_fuel_percent",
            "fuel_mape_excluded",
            "expected_steam_g_h",
            "predicted_steam_g_h",
            "absolute_error_steam_g_h",
            "absolute_percentage_error_steam_percent",
            "steam_mape_excluded",
        ],
    )
    _write_csv(
        csv_dir / "error_distribution.csv",
        distribution_rows,
        [
            "timestamp",
            "frame_id",
            "regime_id",
            "regime_name",
            "status",
            "source_unit",
            "signed_error_fuel_g_h",
            "signed_error_steam_g_h",
        ],
    )

    selected_rows = []
    for selected in analysis.selected_frames:
        row = selected.row
        regime = regime_lookup[row.regime_id]
        selected_rows.append(
            {
                "selection": "best" if selected.label.startswith("Лучший") else "worst",
                "rank": selected.rank,
                "frame_id": row.frame_id,
                "timestamp": row.timestamp.isoformat(timespec="milliseconds"),
                "regime_id": row.regime_id,
                "regime_name": regime.display_name,
                "status": "trained" if regime.final_status else "untrained",
                "image_path": str(row.image_path or ""),
                "expected_fuel_g_h": row.expected_fuel_g_h,
                "predicted_fuel_g_h": row.predicted_fuel_g_h,
                "absolute_error_fuel_g_h": selected.abs_error_fuel,
                "absolute_percentage_error_fuel_percent": selected.ape_fuel,
                "expected_steam_g_h": row.expected_steam_g_h,
                "predicted_steam_g_h": row.predicted_steam_g_h,
                "absolute_error_steam_g_h": selected.abs_error_steam,
                "absolute_percentage_error_steam_percent": selected.ape_steam,
                "frame_error_score": selected.score,
                "score_method": selected.score_method,
            }
        )
    _write_csv(
        csv_dir / "selected_frames.csv",
        selected_rows,
        [
            "selection",
            "rank",
            "frame_id",
            "timestamp",
            "regime_id",
            "regime_name",
            "status",
            "image_path",
            "expected_fuel_g_h",
            "predicted_fuel_g_h",
            "absolute_error_fuel_g_h",
            "absolute_percentage_error_fuel_percent",
            "expected_steam_g_h",
            "predicted_steam_g_h",
            "absolute_error_steam_g_h",
            "absolute_percentage_error_steam_percent",
            "frame_error_score",
            "score_method",
        ],
    )

    classification_rows = []
    for regime in analysis.experiment.regimes:
        classification_rows.append(
            {
                "regime_id": regime.regime_id,
                "expected_fuel_g_h": regime.fuel_g_h,
                "expected_steam_g_h": regime.steam_g_h,
                "original_status": (
                    "conflict"
                    if regime.is_conflict
                    else "trained"
                    if regime.source_status
                    else "untrained"
                ),
                "final_status": "trained" if regime.final_status else "untrained",
                "manually_changed": int(regime.manually_changed),
            }
        )
    _write_csv(
        csv_dir / "regime_classification.csv",
        classification_rows,
        [
            "regime_id",
            "expected_fuel_g_h",
            "expected_steam_g_h",
            "original_status",
            "final_status",
            "manually_changed",
        ],
    )


def _metric_fields() -> list[str]:
    return [
        "record_count",
        "mae_fuel_g_h",
        "mape_fuel_percent",
        "mape_fuel_record_count",
        "mae_steam_g_h",
        "mape_steam_percent",
        "mape_steam_record_count",
    ]


def _stage_csv_row(
    stage: StageResult,
    *,
    regime_id: str | None = None,
    analysis_scope: str | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {}
    if regime_id is not None:
        result["regime_id"] = regime_id
    if analysis_scope is not None:
        result["analysis_scope"] = analysis_scope
    result.update(
        {
            "stage": stage.stage,
            "start_time": (
                stage.start_time.isoformat(timespec="milliseconds")
                if stage.start_time
                else ""
            ),
            "end_time": (
                stage.end_time.isoformat(timespec="milliseconds")
                if stage.end_time
                else ""
            ),
            **stage.metrics.as_dict(),
        }
    )
    return result


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    try:
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: _csv_value(row.get(key))
                        for key in fields
                    }
                )
    except OSError as exc:
        raise RuntimeError(f"Не удалось сохранить {path.name}: {exc}") from exc


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.10g}"
    return value


def _write_processing_log(analysis: AnalysisResult, path: Path) -> None:
    lines = [
        f"BurnerEye Prediction Report - {datetime.now().isoformat(timespec='seconds')}",
        f"Experiment: {analysis.experiment.root}",
        f"CSV rows: {analysis.experiment.total_csv_rows}",
        f"Valid rows: {len(analysis.experiment.rows)}",
        f"Excluded rows: {analysis.experiment.excluded_rows}",
        f"Regimes: {len(analysis.experiment.regimes)}",
        f"Primary frames: {analysis.experiment.primary_frame_count}",
        f"Temporal windows: {analysis.experiment.temporal_window_count}",
        f"Significance threshold: {analysis.threshold_percent:g}%",
        "",
        "Validation:",
    ]
    for issue in analysis.experiment.issues:
        row = f" row={issue.row_number}" if issue.row_number is not None else ""
        lines.append(
            f"[{issue.severity.upper()}] {issue.code}{row}: {issue.message}"
        )
    lines.extend(["", "Processing notes:"])
    lines.extend(f"- {note}" for note in analysis.processing_notes)
    lines.extend(["", "Regime classification:"])
    for regime in analysis.experiment.regimes:
        lines.append(
            f"- {regime.regime_id}: original={regime.source_status_label}; "
            f"final={regime.final_status_label}; manual={int(regime.manually_changed)}"
        )
    try:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Не удалось сохранить processing.log: {exc}") from exc


def _build_pdf(analysis: AnalysisResult, path: Path) -> None:
    regular_font, bold_font = _register_fonts()
    styles = _styles(regular_font, bold_font)
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=18 * mm,
        bottomMargin=16 * mm,
        title="BurnerEye - отчёт по результатам предсказания",
        author="BurnerEye Prediction Report",
    )
    story: list[Flowable] = []
    experiment = analysis.experiment
    generated_at = datetime.now()
    regime_lookup = {regime.regime_id: regime for regime in experiment.regimes}

    story.extend(
        [
            Spacer(1, 22 * mm),
            Paragraph("BURNEREYE", styles["eyebrow"]),
            Spacer(1, 4 * mm),
            Paragraph("Отчёт по результатам предсказания", styles["title"]),
            Spacer(1, 8 * mm),
            Paragraph(
                f"<b>Эксперимент:</b> {escape(experiment.root.name)}",
                styles["lead"],
            ),
            Paragraph(
                f"<b>Путь:</b> {escape(str(experiment.root))}",
                styles["path"],
            ),
            Paragraph(
                f"<b>Дата формирования:</b> {generated_at.strftime('%d.%m.%Y %H:%M:%S')}",
                styles["body"],
            ),
            Spacer(1, 12 * mm),
            _summary_panel(analysis, styles),
            Spacer(1, 8 * mm),
            Paragraph(
                "Итоговые метрики: только MAE и MAPE. Все расходы и MAE "
                "приведены к базовой единице г/ч.",
                styles["note"],
            ),
            PageBreak(),
        ]
    )

    story.extend(_validation_section(analysis, styles))
    story.extend(_classification_section(analysis, styles))
    story.extend(_overall_metrics_section(analysis, styles))
    story.extend(_overall_dynamics_section(analysis, styles, regular_font))
    story.extend(_regime_summary_section(analysis, styles))

    for regime_index, regime in enumerate(experiment.regimes):
        story.extend(
            _regime_section(
                analysis,
                regime,
                styles,
                regular_font,
                start_new_page=regime_index > 0,
            )
        )

    story.extend(_overall_distribution_section(analysis, styles, regular_font))
    story.extend(
        _selected_frames_section(
            analysis,
            styles,
            regime_lookup,
        )
    )
    story.extend(_conclusions_section(analysis, styles))

    def draw_page(canvas, document) -> None:
        canvas.saveState()
        canvas.setTitle("BurnerEye - отчёт по результатам предсказания")
        canvas.setAuthor("BurnerEye Prediction Report")
        canvas.setFont(regular_font, 8)
        canvas.setFillColor(MUTED)
        if document.page > 1:
            canvas.drawString(15 * mm, A4[1] - 10 * mm, "BurnerEye - отчёт по предсказаниям")
        canvas.drawRightString(
            A4[0] - 15 * mm,
            9 * mm,
            f"Страница {document.page}",
        )
        canvas.setStrokeColor(GRID)
        canvas.line(15 * mm, 13 * mm, A4[0] - 15 * mm, 13 * mm)
        canvas.restoreState()

    try:
        doc.build(story, onFirstPage=draw_page, onLaterPages=draw_page)
    except Exception as exc:
        raise RuntimeError(f"Не удалось сформировать PDF: {exc}") from exc


def _summary_panel(analysis: AnalysisResult, styles: dict[str, ParagraphStyle]) -> Table:
    experiment = analysis.experiment
    data = [
        [
            _panel_value(str(len(experiment.rows)), "корректных записей", styles),
            _panel_value(str(len(experiment.regimes)), "режимов", styles),
            _panel_value(str(experiment.primary_frame_count), "основных кадров", styles),
        ],
        [
            _panel_value(
                str(experiment.trained_regime_count), "обученных режимов", styles
            ),
            _panel_value(
                str(experiment.untrained_regime_count), "необученных режимов", styles
            ),
            _panel_value(str(experiment.warning_count), "предупреждений", styles),
        ],
    ]
    table = Table(data, colWidths=[56 * mm] * 3, rowHeights=[24 * mm, 24 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PANEL),
                ("BOX", (0, 0), (-1, -1), 0.6, GRID),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, GRID),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    return table


def _panel_value(
    value: str,
    label: str,
    styles: dict[str, ParagraphStyle],
) -> Paragraph:
    return Paragraph(
        f'<font size="18" color="#0F6CBD"><b>{escape(value)}</b></font><br/>'
        f'<font size="8" color="#64748B">{escape(label)}</font>',
        styles["body"],
    )


def _validation_section(
    analysis: AnalysisResult,
    styles: dict[str, ParagraphStyle],
) -> list[Flowable]:
    experiment = analysis.experiment
    rows = [
        ["Проверка", "Результат"],
        ["Строк в results.csv", str(experiment.total_csv_rows)],
        ["Корректных строк", str(len(experiment.rows))],
        ["Исключённых строк", str(experiment.excluded_rows)],
        ["Обнаружено режимов", str(len(experiment.regimes))],
        ["Основных кадров", str(experiment.primary_frame_count)],
        ["Строк без кадра", str(experiment.missing_frame_count)],
        ["Несвязанных кадров", str(len(experiment.orphan_frames))],
        ["Temporal-окон", str(experiment.temporal_window_count)],
        ["Готовность", "Данные прошли обязательные проверки"],
    ]
    result: list[Flowable] = [
        Paragraph("1. Проверка входных данных", styles["h1"]),
        _table(rows, [75 * mm, 95 * mm], styles),
        Spacer(1, 4 * mm),
    ]
    if experiment.issues:
        issue_rows = [["Уровень", "Сообщение"]]
        for issue in experiment.issues:
            row_prefix = (
                f"Строка {issue.row_number}. " if issue.row_number is not None else ""
            )
            issue_rows.append([issue.severity_label, row_prefix + issue.message])
        result.extend(
            [
                Paragraph("Сообщения проверки", styles["h2"]),
                _table(issue_rows, [32 * mm, 138 * mm], styles, repeat_rows=1),
            ]
        )
    else:
        result.append(Paragraph("Ошибок и предупреждений не обнаружено.", styles["success"]))
    result.append(Spacer(1, 5 * mm))
    return result


def _classification_section(
    analysis: AnalysisResult,
    styles: dict[str, ParagraphStyle],
) -> list[Flowable]:
    rows = [["ID", "Режим", "Исходный статус", "Итоговый статус", "Ручное изменение"]]
    for regime in analysis.experiment.regimes:
        rows.append(
            [
                regime.regime_id,
                regime.display_name,
                regime.source_status_label,
                regime.final_status_label,
                "Да" if regime.manually_changed else "Нет",
            ]
        )
    result: list[Flowable] = [
        Paragraph("2. Классификация режимов", styles["h1"]),
        _table(
            rows,
            [14 * mm, 60 * mm, 36 * mm, 36 * mm, 24 * mm],
            styles,
            repeat_rows=1,
        ),
        Spacer(1, 3 * mm),
    ]
    changed = [regime for regime in analysis.experiment.regimes if regime.manually_changed]
    if changed:
        result.append(
            Paragraph(
                "Ручные исправления применены к режимам: "
                + ", ".join(regime.regime_id for regime in changed)
                + ". Исходный results.csv не изменялся.",
                styles["note"],
            )
        )
    else:
        result.append(
            Paragraph(
                "Ручных исправлений нет. Исходный results.csv не изменялся.",
                styles["note"],
            )
        )
    result.append(Spacer(1, 5 * mm))
    return result


def _overall_metrics_section(
    analysis: AnalysisResult,
    styles: dict[str, ParagraphStyle],
) -> list[Flowable]:
    rows = [["Группа", "Записей", "MAE топлива", "MAPE топлива", "MAE пара", "MAPE пара"]]
    groups: list[tuple[str, MetricSet]] = [("Весь эксперимент", analysis.overall_metrics)]
    for status in (True, False):
        metrics = analysis.trained_untrained_metrics.get(status)
        if metrics:
            groups.append((status_label(status) + " режимы", metrics))
    for label, metrics in groups:
        rows.append(_metrics_row(label, metrics))

    group_labels = [item[0] for item in groups]
    mae_series = {
        "Топливо": [item[1].mae_fuel for item in groups],
        "Пар": [item[1].mae_steam for item in groups],
    }
    mape_series = {
        "Топливо": [item[1].mape_fuel for item in groups],
        "Пар": [item[1].mape_steam for item in groups],
    }
    return [
        Paragraph("3. Общий анализ качества", styles["h1"]),
        Paragraph(
            "MAE приведён в г/ч. MAPE рассчитан только по строкам с ненулевым фактическим расходом.",
            styles["note"],
        ),
        Spacer(1, 2 * mm),
        _table(rows, [42 * mm, 18 * mm, 28 * mm, 28 * mm, 27 * mm, 27 * mm], styles, repeat_rows=1),
        Spacer(1, 5 * mm),
        KeepTogether(
            [
                Paragraph("Сравнение MAE", styles["h2"]),
                BarChartFlowable(group_labels, mae_series, "MAE, г/ч"),
            ]
        ),
        Spacer(1, 3 * mm),
        KeepTogether(
            [
                Paragraph("Сравнение MAPE", styles["h2"]),
                BarChartFlowable(group_labels, mape_series, "MAPE, %"),
            ]
        ),
        Spacer(1, 5 * mm),
    ]


def _overall_dynamics_section(
    analysis: AnalysisResult,
    styles: dict[str, ParagraphStyle],
    font_name: str,
) -> list[Flowable]:
    result: list[Flowable] = [Paragraph("4. Динамика качества эксперимента", styles["h1"])]
    if not analysis.overall_stages:
        result.append(
            Paragraph(
                "Недостаточно данных: для деления на начало, середину и конец нужно не менее трёх записей.",
                styles["warning"],
            )
        )
        return result
    result.extend(
        [
            _stage_table(analysis.overall_stages, styles),
            Spacer(1, 4 * mm),
            _stage_charts(analysis.overall_stages, font_name),
            Spacer(1, 3 * mm),
            Paragraph(analysis.overall_conclusion, styles["conclusion"]),
        ]
    )
    if (
        len(analysis.group_stages.get(True, [])) == 3
        and len(analysis.group_stages.get(False, [])) == 3
    ):
        result.extend(
            [
                Spacer(1, 4 * mm),
                Paragraph("Обученные и необученные режимы в динамике", styles["h2"]),
                _group_stage_chart(analysis, "mae", font_name),
                Spacer(1, 3 * mm),
                _group_stage_chart(analysis, "mape", font_name),
            ]
        )
    else:
        result.append(
            Paragraph(
                "Сравнение динамики обученных и необученных режимов не выполнялось: "
                "в эксперименте присутствует только одна группа.",
                styles["note"],
            )
        )
    result.append(Spacer(1, 5 * mm))
    return result


def _regime_summary_section(
    analysis: AnalysisResult,
    styles: dict[str, ParagraphStyle],
) -> list[Flowable]:
    rows = [["ID", "Режим", "Статус", "Записей", "MAE топл.", "MAPE топл.", "MAE пара", "MAPE пара"]]
    for regime in analysis.experiment.regimes:
        metrics = analysis.regime_metrics[regime.regime_id]
        rows.append(
            [
                regime.regime_id,
                regime.display_name,
                status_label(bool(regime.final_status)),
                str(metrics.record_count),
                _fmt(metrics.mae_fuel),
                _fmt_mape(metrics.mape_fuel),
                _fmt(metrics.mae_steam),
                _fmt_mape(metrics.mape_steam),
            ]
        )
    return [
        Paragraph("5. Метрики по режимам", styles["h1"]),
        _table(
            rows,
            [13 * mm, 47 * mm, 23 * mm, 15 * mm, 18 * mm, 19 * mm, 18 * mm, 17 * mm],
            styles,
            repeat_rows=1,
            font_size=7,
        ),
        Spacer(1, 5 * mm),
    ]


def _regime_section(
    analysis: AnalysisResult,
    regime: Regime,
    styles: dict[str, ParagraphStyle],
    font_name: str,
    *,
    start_new_page: bool,
) -> list[Flowable]:
    metrics = analysis.regime_metrics[regime.regime_id]
    rows = sorted(regime.rows, key=lambda row: row.timestamp)
    x_values = [row.timestamp.timestamp() for row in rows]
    x_labels = [row.timestamp.strftime("%d.%m %H:%M:%S") for row in rows]
    row_metrics = analysis.row_metrics
    stages = analysis.regime_stages[regime.regime_id]
    result: list[Flowable] = []
    if start_new_page:
        result.append(PageBreak())
    result.extend(
        [
        Paragraph(
            f"6.{int(regime.regime_id[1:])}. {escape(regime.regime_id)} - "
            f"{escape(regime.display_name)}",
            styles["h1"],
        ),
        Paragraph(
            f"<b>Статус:</b> {status_label(bool(regime.final_status))}. "
            f"<b>Записей:</b> {len(rows)}. "
            f"<b>Период:</b> {regime.first_timestamp.strftime('%d.%m.%Y %H:%M:%S')} - "
            f"{regime.last_timestamp.strftime('%d.%m.%Y %H:%M:%S')}. "
            f"<b>Исходные единицы:</b> {escape(', '.join(regime.source_units))}.",
            styles["body"],
        ),
        Spacer(1, 3 * mm),
        _table(
            [
                ["Показатель", "Топливо", "Пар"],
                ["MAE, г/ч", _fmt(metrics.mae_fuel), _fmt(metrics.mae_steam)],
                ["MAPE, %", _fmt_mape(metrics.mape_fuel), _fmt_mape(metrics.mape_steam)],
            ],
            [55 * mm, 57 * mm, 57 * mm],
            styles,
            repeat_rows=1,
        ),
        Spacer(1, 4 * mm),
        KeepTogether(
            [
                Paragraph("Фактический и предсказанный расход топлива", styles["h2"]),
                LineChartFlowable(
                    x_values,
                    x_labels,
                    {
                        "Фактический": [row.expected_fuel_g_h for row in rows],
                        "Предсказанный": [row.predicted_fuel_g_h for row in rows],
                    },
                    "Расход, г/ч",
                    font_name=font_name,
                ),
            ]
        ),
        Spacer(1, 3 * mm),
        KeepTogether(
            [
                Paragraph("Фактический и предсказанный расход пара", styles["h2"]),
                LineChartFlowable(
                    x_values,
                    x_labels,
                    {
                        "Фактический": [row.expected_steam_g_h for row in rows],
                        "Предсказанный": [row.predicted_steam_g_h for row in rows],
                    },
                    "Расход, г/ч",
                    font_name=font_name,
                ),
            ]
        ),
        Spacer(1, 4 * mm),
        Paragraph("Начало, середина и конец режима", styles["h2"]),
        ]
    )
    if stages:
        result.extend(
            [
                _stage_table(stages, styles),
                Spacer(1, 3 * mm),
                _stage_charts(stages, font_name),
                Spacer(1, 3 * mm),
                Paragraph(
                    analysis.regime_conclusions[regime.regime_id],
                    styles["conclusion"],
                ),
            ]
        )
    else:
        result.append(
            Paragraph(
                "Недостаточно данных: для сравнения участков нужно не менее трёх корректных записей.",
                styles["warning"],
            )
        )
    result.extend(
        [
            PageBreak(),
            Paragraph("Распределение знаковой ошибки", styles["h2"]),
            Paragraph(
                "Положительная ошибка означает завышение, отрицательная - занижение.",
                styles["note"],
            ),
            Spacer(1, 2 * mm),
            KeepTogether(
                [
                    Paragraph("Топливо", styles["h2"]),
                    HistogramFlowable(
                        [
                            float(row_metrics[row.frame_id]["signed_error_fuel"] or 0.0)
                            for row in rows
                        ],
                        "Ошибка топлива, г/ч",
                        font_name=font_name,
                        color=ACCENT,
                    ),
                    Paragraph(
                        analysis.error_conclusions[f"{regime.regime_id}_fuel"],
                        styles["conclusion"],
                    ),
                ]
            ),
            Spacer(1, 3 * mm),
            KeepTogether(
                [
                    Paragraph("Пар", styles["h2"]),
                    HistogramFlowable(
                        [
                            float(row_metrics[row.frame_id]["signed_error_steam"] or 0.0)
                            for row in rows
                        ],
                        "Ошибка пара, г/ч",
                        font_name=font_name,
                        color=TEAL,
                    ),
                    Paragraph(
                        analysis.error_conclusions[f"{regime.regime_id}_steam"],
                        styles["conclusion"],
                    ),
                ]
            ),
        ]
    )
    return result


def _overall_distribution_section(
    analysis: AnalysisResult,
    styles: dict[str, ParagraphStyle],
    font_name: str,
) -> list[Flowable]:
    metrics = analysis.row_metrics
    return [
        PageBreak(),
        Paragraph("7. Общее распределение ошибок", styles["h1"]),
        Paragraph(
            "Знаковая ошибка используется только для анализа направления и не является итоговой метрикой.",
            styles["note"],
        ),
        Spacer(1, 3 * mm),
        HistogramFlowable(
            [float(item["signed_error_fuel"] or 0.0) for item in metrics.values()],
            "Ошибка топлива, г/ч",
            font_name=font_name,
            color=ACCENT,
        ),
        Paragraph(analysis.error_conclusions["overall_fuel"], styles["conclusion"]),
        Spacer(1, 5 * mm),
        HistogramFlowable(
            [float(item["signed_error_steam"] or 0.0) for item in metrics.values()],
            "Ошибка пара, г/ч",
            font_name=font_name,
            color=TEAL,
        ),
        Paragraph(analysis.error_conclusions["overall_steam"], styles["conclusion"]),
    ]


def _selected_frames_section(
    analysis: AnalysisResult,
    styles: dict[str, ParagraphStyle],
    regime_lookup: dict[str, Regime],
) -> list[Flowable]:
    result: list[Flowable] = [
        PageBreak(),
        Paragraph("8. Лучшие и худшие кадры", styles["h1"]),
    ]
    if len(analysis.selected_frames) < 4:
        result.append(
            Paragraph(
                f"Доступно {len(analysis.selected_frames)} корректных кадров; "
                "все они включены в отчёт без повторов.",
                styles["warning"],
            )
        )
    for selected in analysis.selected_frames:
        result.extend(
            [
                Spacer(1, 4 * mm),
                _frame_card(selected, regime_lookup[selected.row.regime_id], styles),
            ]
        )
    return result


def _frame_card(
    selected: SelectedFrame,
    regime: Regime,
    styles: dict[str, ParagraphStyle],
) -> Flowable:
    row = selected.row
    image = Image(str(row.image_path))
    max_width = 73 * mm
    max_height = 50 * mm
    scale = min(max_width / image.imageWidth, max_height / image.imageHeight)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    details = [
        [f"{selected.label} #{selected.rank}", f"frame_id: {row.frame_id}"],
        ["Дата и время", row.timestamp.strftime("%d.%m.%Y %H:%M:%S.%f")[:-3]],
        ["Режим", regime.display_name],
        ["Статус", status_label(bool(regime.final_status))],
        ["Топливо факт / прогноз", f"{_fmt(row.expected_fuel_g_h)} / {_fmt(row.predicted_fuel_g_h)} г/ч"],
        ["Ошибка топлива", f"{_fmt(selected.abs_error_fuel)} г/ч; {_fmt_mape(selected.ape_fuel)}"],
        ["Пар факт / прогноз", f"{_fmt(row.expected_steam_g_h)} / {_fmt(row.predicted_steam_g_h)} г/ч"],
        ["Ошибка пара", f"{_fmt(selected.abs_error_steam)} г/ч; {_fmt_mape(selected.ape_steam)}"],
        ["Оценка кадра", f"{selected.score:.3f}; {selected.score_method}"],
    ]
    details_table = _table(details, [34 * mm, 65 * mm], styles, font_size=7)
    card = Table([[image, details_table]], colWidths=[76 * mm, 100 * mm])
    card.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PANEL),
                ("BOX", (0, 0), (-1, -1), 0.7, GRID),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return KeepTogether([card])


def _conclusions_section(
    analysis: AnalysisResult,
    styles: dict[str, ParagraphStyle],
) -> list[Flowable]:
    result: list[Flowable] = [
        PageBreak(),
        Paragraph("9. Итоговые автоматические выводы", styles["h1"]),
        Paragraph(f"<b>Эксперимент:</b> {escape(analysis.overall_conclusion)}", styles["body"]),
        Paragraph(
            f"<b>Направление ошибки топлива:</b> "
            f"{escape(analysis.error_conclusions['overall_fuel'])}",
            styles["body"],
        ),
        Paragraph(
            f"<b>Направление ошибки пара:</b> "
            f"{escape(analysis.error_conclusions['overall_steam'])}",
            styles["body"],
        ),
        Spacer(1, 4 * mm),
        Paragraph("Выводы по режимам", styles["h2"]),
    ]
    for regime in analysis.experiment.regimes:
        result.append(
            Paragraph(
                f"<b>{regime.regime_id}:</b> "
                f"{escape(analysis.regime_conclusions[regime.regime_id])}",
                styles["body"],
            )
        )
    result.extend(
        [
            Spacer(1, 5 * mm),
            Paragraph("Предупреждения и исключённые строки", styles["h2"]),
        ]
    )
    if analysis.experiment.issues:
        for issue in analysis.experiment.issues:
            prefix = f"Строка {issue.row_number}: " if issue.row_number else ""
            result.append(
                Paragraph(
                    f"- {escape(prefix + issue.message)}",
                    styles["body"],
                )
            )
    else:
        result.append(Paragraph("Предупреждений нет.", styles["success"]))
    if analysis.processing_notes:
        result.extend([Spacer(1, 3 * mm), Paragraph("Примечания обработки", styles["h2"])])
        for note in analysis.processing_notes:
            result.append(Paragraph(f"- {escape(note)}", styles["body"]))
    return result


def _stage_table(
    stages: list[StageResult],
    styles: dict[str, ParagraphStyle],
) -> Table:
    rows = [["Участок", "Записей", "MAE топлива", "MAPE топлива", "MAE пара", "MAPE пара"]]
    for stage in stages:
        rows.append(_metrics_row(stage.stage, stage.metrics))
    return _table(
        rows,
        [35 * mm, 19 * mm, 29 * mm, 29 * mm, 29 * mm, 29 * mm],
        styles,
        repeat_rows=1,
    )


def _stage_charts(stages: list[StageResult], font_name: str) -> Flowable:
    labels = [stage.stage for stage in stages]
    mae = {
        "Топливо": [stage.metrics.mae_fuel for stage in stages],
        "Пар": [stage.metrics.mae_steam for stage in stages],
    }
    mape = {
        "Топливо": [stage.metrics.mape_fuel for stage in stages],
        "Пар": [stage.metrics.mape_steam for stage in stages],
    }
    return KeepTogether(
        [
            LineChartFlowable(
                list(range(len(labels))),
                labels,
                mae,
                "MAE, г/ч",
                font_name=font_name,
                height=150,
            ),
            Spacer(1, 2 * mm),
            LineChartFlowable(
                list(range(len(labels))),
                labels,
                mape,
                "MAPE, %",
                font_name=font_name,
                height=150,
            ),
        ]
    )


def _group_stage_chart(
    analysis: AnalysisResult,
    metric_kind: str,
    font_name: str,
) -> Flowable:
    labels = ["Начало", "Середина", "Конец"]
    series: dict[str, list[float | None]] = {}
    for status in (True, False):
        stages = analysis.group_stages[status]
        label_prefix = "Обуч." if status else "Необуч."
        if metric_kind == "mae":
            series[f"{label_prefix} топливо"] = [
                stage.metrics.mae_fuel for stage in stages
            ]
            series[f"{label_prefix} пар"] = [
                stage.metrics.mae_steam for stage in stages
            ]
            y_label = "MAE, г/ч"
        else:
            series[f"{label_prefix} топливо"] = [
                stage.metrics.mape_fuel for stage in stages
            ]
            series[f"{label_prefix} пар"] = [
                stage.metrics.mape_steam for stage in stages
            ]
            y_label = "MAPE, %"
    return LineChartFlowable(
        [0, 1, 2],
        labels,
        series,
        y_label,
        font_name=font_name,
        height=165,
    )


def _metrics_row(label: str, metrics: MetricSet) -> list[str]:
    return [
        label,
        str(metrics.record_count),
        _fmt(metrics.mae_fuel),
        _fmt_mape(metrics.mape_fuel),
        _fmt(metrics.mae_steam),
        _fmt_mape(metrics.mape_steam),
    ]


def _table(
    data: list[list[object]],
    col_widths: list[float],
    styles: dict[str, ParagraphStyle],
    *,
    repeat_rows: int = 0,
    font_size: int = 8,
) -> Table:
    body_style = ParagraphStyle(
        "TableCell",
        parent=styles["body"],
        fontSize=font_size,
        leading=font_size + 2,
    )
    header_style = ParagraphStyle(
        "TableHeader",
        parent=body_style,
        fontName=styles["h1"].fontName,
        textColor=colors.white,
    )
    wrapped: list[list[Paragraph]] = []
    for row_index, row in enumerate(data):
        style = header_style if repeat_rows and row_index < repeat_rows else body_style
        wrapped.append([Paragraph(escape(str(value)), style) for value in row])
    table = Table(wrapped, colWidths=col_widths, repeatRows=repeat_rows)
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.45, GRID),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if repeat_rows:
        commands.append(("BACKGROUND", (0, 0), (-1, repeat_rows - 1), ACCENT))
    for row_index in range(repeat_rows, len(data)):
        if (row_index - repeat_rows) % 2:
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), PANEL))
    table.setStyle(TableStyle(commands))
    return table


def _register_fonts() -> tuple[str, str]:
    candidates = [
        (
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("C:/Windows/Fonts/arialbd.ttf"),
        ),
        (
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ),
    ]
    for regular, bold in candidates:
        if regular.is_file() and bold.is_file():
            try:
                pdfmetrics.registerFont(TTFont("BurnerEyeRegular", str(regular)))
                pdfmetrics.registerFont(TTFont("BurnerEyeBold", str(bold)))
                return "BurnerEyeRegular", "BurnerEyeBold"
            except Exception:
                continue
    raise RuntimeError(
        "Не найден Unicode-шрифт для PDF. Установите Arial или DejaVu Sans."
    )


def _styles(regular_font: str, bold_font: str) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName=bold_font,
            fontSize=26,
            leading=31,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=6,
        ),
        "eyebrow": ParagraphStyle(
            "Eyebrow",
            parent=base["Normal"],
            fontName=bold_font,
            fontSize=10,
            leading=12,
            textColor=ACCENT,
            tracking=1.5,
        ),
        "lead": ParagraphStyle(
            "Lead",
            parent=base["Normal"],
            fontName=regular_font,
            fontSize=12,
            leading=17,
            textColor=INK,
            spaceAfter=5,
        ),
        "h1": ParagraphStyle(
            "Heading1",
            parent=base["Heading1"],
            fontName=bold_font,
            fontSize=15,
            leading=19,
            textColor=INK,
            spaceBefore=8,
            spaceAfter=7,
        ),
        "h2": ParagraphStyle(
            "Heading2",
            parent=base["Heading2"],
            fontName=bold_font,
            fontSize=11,
            leading=14,
            textColor=INK,
            spaceBefore=5,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName=regular_font,
            fontSize=9,
            leading=13,
            textColor=INK,
            spaceAfter=4,
        ),
        "path": ParagraphStyle(
            "Path",
            parent=base["BodyText"],
            fontName=regular_font,
            fontSize=9,
            leading=13,
            textColor=INK,
            spaceAfter=4,
            wordWrap="CJK",
        ),
        "note": ParagraphStyle(
            "Note",
            parent=base["BodyText"],
            fontName=regular_font,
            fontSize=8,
            leading=11,
            textColor=MUTED,
            leftIndent=4,
            borderColor=GRID,
            borderWidth=0.5,
            borderPadding=6,
            backColor=PANEL,
        ),
        "warning": ParagraphStyle(
            "Warning",
            parent=base["BodyText"],
            fontName=regular_font,
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#8A4B08"),
            backColor=colors.HexColor("#FFF5E6"),
            borderPadding=6,
        ),
        "success": ParagraphStyle(
            "Success",
            parent=base["BodyText"],
            fontName=regular_font,
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#176B50"),
        ),
        "conclusion": ParagraphStyle(
            "Conclusion",
            parent=base["BodyText"],
            fontName=bold_font,
            fontSize=9,
            leading=13,
            textColor=INK,
            backColor=colors.HexColor("#EAF3FC"),
            borderColor=colors.HexColor("#B5D3F0"),
            borderWidth=0.5,
            borderPadding=6,
        ),
    }


class LineChartFlowable(Flowable):
    def __init__(
        self,
        x_values: list[float],
        x_labels: list[str],
        series: dict[str, list[float | None]],
        y_label: str,
        *,
        font_name: str = "Helvetica",
        width: float = 480,
        height: float = 210,
    ) -> None:
        super().__init__()
        self.width = width
        self.height = height
        self.x_values = x_values
        self.x_labels = x_labels
        self.series = series
        self.y_label = y_label
        self.font_name = font_name

    def draw(self) -> None:
        canvas = self.canv
        left, right, bottom, top = 50, 12, 34, 30
        chart_w = self.width - left - right
        chart_h = self.height - bottom - top
        values = [
            float(value)
            for sequence in self.series.values()
            for value in sequence
            if value is not None and math.isfinite(float(value))
        ]
        canvas.setFillColor(colors.white)
        canvas.rect(0, 0, self.width, self.height, fill=1, stroke=0)
        if not values:
            canvas.setFont(self.font_name, 9)
            canvas.setFillColor(MUTED)
            canvas.drawCentredString(self.width / 2, self.height / 2, "Недостаточно данных")
            return
        y_min, y_max = min(values), max(values)
        if y_min == y_max:
            padding = abs(y_min) * 0.1 or 1.0
        else:
            padding = (y_max - y_min) * 0.08
        y_min -= padding
        y_max += padding
        if y_min > 0:
            y_min = 0
        x_min = min(self.x_values) if self.x_values else 0.0
        x_max = max(self.x_values) if self.x_values else 1.0
        if x_min == x_max:
            x_max = x_min + 1.0

        canvas.setFont(self.font_name, 7)
        canvas.setStrokeColor(GRID)
        canvas.setFillColor(MUTED)
        for index in range(5):
            ratio = index / 4
            y = bottom + chart_h * ratio
            value = y_min + (y_max - y_min) * ratio
            canvas.line(left, y, left + chart_w, y)
            canvas.drawRightString(left - 5, y - 2, _axis_number(value))
        canvas.setStrokeColor(INK)
        canvas.line(left, bottom, left, bottom + chart_h)
        canvas.line(left, bottom, left + chart_w, bottom)
        canvas.saveState()
        canvas.translate(10, bottom + chart_h / 2)
        canvas.rotate(90)
        canvas.drawCentredString(0, 0, self.y_label)
        canvas.restoreState()

        palette = [ACCENT, TEAL, ORANGE, PURPLE, RED, colors.HexColor("#556270")]
        for series_index, (name, sequence) in enumerate(self.series.items()):
            color = palette[series_index % len(palette)]
            canvas.setStrokeColor(color)
            canvas.setFillColor(color)
            canvas.setLineWidth(1.5)
            previous: tuple[float, float] | None = None
            for x_value, y_value in zip(self.x_values, sequence):
                if y_value is None:
                    previous = None
                    continue
                x = left + (x_value - x_min) / (x_max - x_min) * chart_w
                y = bottom + (float(y_value) - y_min) / (y_max - y_min) * chart_h
                if previous is not None:
                    canvas.line(previous[0], previous[1], x, y)
                if len(sequence) <= 40:
                    canvas.circle(x, y, 1.8, fill=1, stroke=0)
                previous = (x, y)
            legend_x = left + series_index * (chart_w / max(1, len(self.series)))
            canvas.rect(legend_x, self.height - 14, 8, 3, fill=1, stroke=0)
            canvas.setFont(self.font_name, 7)
            canvas.drawString(legend_x + 11, self.height - 15, name[:28])

        if self.x_labels:
            indices = sorted({0, len(self.x_labels) // 2, len(self.x_labels) - 1})
            canvas.setFillColor(MUTED)
            canvas.setFont(self.font_name, 6.5)
            for index in indices:
                if index >= len(self.x_values):
                    continue
                x = left + (self.x_values[index] - x_min) / (x_max - x_min) * chart_w
                if index == 0:
                    canvas.drawString(x, 16, self.x_labels[index])
                elif index == len(self.x_labels) - 1:
                    canvas.drawRightString(x, 16, self.x_labels[index])
                else:
                    canvas.drawCentredString(x, 16, self.x_labels[index])


class BarChartFlowable(Flowable):
    def __init__(
        self,
        groups: list[str],
        series: dict[str, list[float | None]],
        y_label: str,
        *,
        width: float = 480,
        height: float = 190,
    ) -> None:
        super().__init__()
        self.width = width
        self.height = height
        self.groups = groups
        self.series = series
        self.y_label = y_label

    def draw(self) -> None:
        canvas = self.canv
        left, right, bottom, top = 50, 12, 42, 28
        chart_w = self.width - left - right
        chart_h = self.height - bottom - top
        values = [
            float(value)
            for sequence in self.series.values()
            for value in sequence
            if value is not None
        ]
        max_value = max(values, default=1.0) or 1.0
        canvas.setFont("BurnerEyeRegular", 7)
        canvas.setFillColor(MUTED)
        canvas.setStrokeColor(GRID)
        for index in range(5):
            ratio = index / 4
            y = bottom + chart_h * ratio
            canvas.line(left, y, left + chart_w, y)
            canvas.drawRightString(left - 5, y - 2, _axis_number(max_value * ratio))
        group_width = chart_w / max(1, len(self.groups))
        bar_count = max(1, len(self.series))
        bar_width = min(22.0, group_width * 0.7 / bar_count)
        palette = [ACCENT, TEAL, ORANGE, PURPLE]
        for series_index, (name, sequence) in enumerate(self.series.items()):
            color = palette[series_index % len(palette)]
            canvas.setFillColor(color)
            for group_index, value in enumerate(sequence):
                if value is None:
                    continue
                center = left + group_width * (group_index + 0.5)
                group_total = bar_width * bar_count
                x = center - group_total / 2 + series_index * bar_width
                height = max(0.5, float(value) / max_value * chart_h)
                canvas.rect(x, bottom, bar_width - 2, height, fill=1, stroke=0)
            legend_x = left + series_index * 115
            canvas.rect(legend_x, self.height - 14, 8, 4, fill=1, stroke=0)
            canvas.drawString(legend_x + 11, self.height - 15, name)
        canvas.setFillColor(MUTED)
        for index, group in enumerate(self.groups):
            x = left + group_width * (index + 0.5)
            canvas.drawCentredString(x, 23, group[:22])
        canvas.saveState()
        canvas.translate(10, bottom + chart_h / 2)
        canvas.rotate(90)
        canvas.drawCentredString(0, 0, self.y_label)
        canvas.restoreState()


class HistogramFlowable(Flowable):
    def __init__(
        self,
        values: list[float],
        x_label: str,
        *,
        font_name: str,
        color,
        width: float = 480,
        height: float = 170,
        bins: int = 12,
    ) -> None:
        super().__init__()
        self.width = width
        self.height = height
        self.values = values
        self.x_label = x_label
        self.font_name = font_name
        self.color = color
        self.bins = bins

    def draw(self) -> None:
        canvas = self.canv
        left, right, bottom, top = 42, 12, 38, 12
        chart_w = self.width - left - right
        chart_h = self.height - bottom - top
        values = self.values or [0.0]
        minimum, maximum = min(values), max(values)
        if minimum == maximum:
            minimum -= 1.0
            maximum += 1.0
        bin_count = min(self.bins, max(3, int(math.sqrt(len(values))) + 1))
        width = (maximum - minimum) / bin_count
        counts = [0] * bin_count
        for value in values:
            index = min(bin_count - 1, max(0, int((value - minimum) / width)))
            counts[index] += 1
        max_count = max(counts) or 1
        canvas.setFont(self.font_name, 7)
        canvas.setStrokeColor(GRID)
        canvas.setFillColor(MUTED)
        for index in range(5):
            ratio = index / 4
            y = bottom + chart_h * ratio
            canvas.line(left, y, left + chart_w, y)
            canvas.drawRightString(left - 5, y - 2, str(round(max_count * ratio)))
        bar_width = chart_w / bin_count
        canvas.setFillColor(self.color)
        for index, count in enumerate(counts):
            height = count / max_count * chart_h
            canvas.rect(
                left + index * bar_width + 1,
                bottom,
                max(1, bar_width - 2),
                height,
                fill=1,
                stroke=0,
            )
        if minimum <= 0 <= maximum:
            zero_x = left + (0 - minimum) / (maximum - minimum) * chart_w
            canvas.setStrokeColor(RED)
            canvas.setLineWidth(1)
            canvas.line(zero_x, bottom, zero_x, bottom + chart_h)
        canvas.setFillColor(MUTED)
        canvas.drawString(left, 23, _axis_number(minimum))
        canvas.drawCentredString(left + chart_w / 2, 23, _axis_number((minimum + maximum) / 2))
        canvas.drawRightString(left + chart_w, 23, _axis_number(maximum))
        canvas.drawCentredString(left + chart_w / 2, 10, self.x_label)


def _fmt(value: float | None) -> str:
    return "Нет данных" if value is None else f"{value:.3f}"


def _fmt_mape(value: float | None) -> str:
    return (
        "Недостаточно данных для расчёта MAPE"
        if value is None
        else f"{value:.3f}%"
    )


def _axis_number(value: float) -> str:
    absolute = abs(value)
    if absolute >= 1000:
        return f"{value:.0f}"
    if absolute >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _verify_pdf(path: Path) -> None:
    try:
        if not path.is_file() or path.stat().st_size < 1024:
            raise ValueError("файл пуст или слишком мал")
        reader = PdfReader(str(path))
        if not reader.pages:
            raise ValueError("PDF не содержит страниц")
        _ = reader.pages[-1].mediabox
    except Exception as exc:
        raise RuntimeError(f"Сформированный PDF не прошёл проверку: {exc}") from exc


def _progress(callback: ProgressCallback | None, value: int, message: str) -> None:
    if callback is not None:
        callback(value, message)

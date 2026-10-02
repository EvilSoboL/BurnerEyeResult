from __future__ import annotations

import csv
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import (
    Flowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

from .comparison import (
    STAGES,
    ComparisonAnalysis,
    CommonRegimeComparison,
    metric_winner,
)
from .models import MetricSet, PredictionRow
from .reporting import (
    GRID,
    MUTED,
    BarChartFlowable,
    LineChartFlowable,
    _fmt,
    _fmt_mae,
    _fmt_mape_metric,
    _register_fonts,
    _styles,
    _table,
    _verify_pdf,
)


ProgressCallback = Callable[[int, str], None]


def generate_comparison_report(
    comparison: ComparisonAnalysis,
    *,
    progress: ProgressCallback | None = None,
) -> Path:
    _progress(progress, 10, "Подготовка сравнительных таблиц")
    output_dir = _create_output_directory(comparison.first.root)
    csv_dir = output_dir / "csv"
    try:
        csv_dir.mkdir()
    except OSError as exc:
        raise RuntimeError(f"Не удалось создать папку CSV: {exc}") from exc

    _progress(progress, 25, "Сохранение сравнительных CSV")
    _write_csv_files(comparison, csv_dir)
    _write_processing_log(comparison, output_dir / "processing.log")

    pdf_path = output_dir / "model_comparison_report.pdf"
    _progress(progress, 45, "Формирование сравнительного PDF")
    _build_pdf(comparison, pdf_path)
    _progress(progress, 90, "Проверка PDF")
    _verify_pdf(pdf_path)
    _progress(progress, 100, "Сравнительный отчёт готов")
    return pdf_path


def _create_output_directory(root: Path) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base = root / f"model_comparison_{stamp}"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = root / f"{base.name}_{suffix:02d}"
        suffix += 1
    try:
        candidate.mkdir()
    except OSError as exc:
        raise RuntimeError(
            f"Не удалось создать папку сравнительного отчёта: {exc}"
        ) from exc
    return candidate


def _write_csv_files(comparison: ComparisonAnalysis, csv_dir: Path) -> None:
    metric_fields = [
        "record_count",
        "mae_fuel_g_h",
        "mae_fuel_record_count",
        "mape_fuel_percent",
        "mape_fuel_record_count",
        "mae_steam_g_h",
        "mae_steam_record_count",
        "mape_steam_percent",
        "mape_steam_record_count",
    ]
    overall_rows = [
        {
            "model": comparison.first_name,
            "folder": str(comparison.first.root),
            **comparison.first_overall_metrics.as_dict(),
        },
        {
            "model": comparison.second_name,
            "folder": str(comparison.second.root),
            **comparison.second_overall_metrics.as_dict(),
        },
    ]
    _write_csv(
        csv_dir / "overall_model_comparison.csv",
        overall_rows,
        ["model", "folder", *metric_fields],
    )

    regime_rows: list[dict[str, object]] = []
    for item in comparison.common_regimes:
        row: dict[str, object] = {
            "regime": item.display_name,
            "expected_fuel_g_h": item.key[0],
            "expected_steam_g_h": item.key[1],
        }
        for prefix, metrics in (
            ("model_1", item.first_metrics),
            ("model_2", item.second_metrics),
        ):
            row.update(
                {
                    f"{prefix}_{key}": value
                    for key, value in metrics.as_dict().items()
                }
            )
        row.update(
            {
                "mae_fuel_winner": metric_winner(
                    item.first_metrics.mae_fuel,
                    item.second_metrics.mae_fuel,
                    comparison.first_name,
                    comparison.second_name,
                ),
                "mape_fuel_winner": metric_winner(
                    item.first_metrics.mape_fuel,
                    item.second_metrics.mape_fuel,
                    comparison.first_name,
                    comparison.second_name,
                ),
                "mae_steam_winner": metric_winner(
                    item.first_metrics.mae_steam,
                    item.second_metrics.mae_steam,
                    comparison.first_name,
                    comparison.second_name,
                ),
                "mape_steam_winner": metric_winner(
                    item.first_metrics.mape_steam,
                    item.second_metrics.mape_steam,
                    comparison.first_name,
                    comparison.second_name,
                ),
            }
        )
        regime_rows.append(row)
    regime_fields = [
        "regime",
        "expected_fuel_g_h",
        "expected_steam_g_h",
        *[f"model_1_{field}" for field in metric_fields],
        *[f"model_2_{field}" for field in metric_fields],
        "mae_fuel_winner",
        "mape_fuel_winner",
        "mae_steam_winner",
        "mape_steam_winner",
    ]
    _write_csv(
        csv_dir / "metrics_by_common_regime.csv",
        regime_rows,
        regime_fields,
    )

    timeline_rows: list[dict[str, object]] = []
    for item in comparison.common_regimes:
        for model_name, rows in (
            (comparison.first_name, item.first_regime.rows),
            (comparison.second_name, item.second_regime.rows),
        ):
            for index, row in enumerate(
                sorted(rows, key=lambda candidate: candidate.timestamp),
                start=1,
            ):
                timeline_rows.append(
                    {
                        "model": model_name,
                        "regime": item.display_name,
                        "observation": index,
                        "timestamp": row.timestamp.isoformat(timespec="milliseconds"),
                        "timestamp_type": "processing_time",
                        "frame_id": row.frame_id,
                        "expected_fuel_g_h": row.expected_fuel_g_h,
                        "predicted_fuel_g_h": row.predicted_fuel_g_h,
                        "expected_steam_g_h": row.expected_steam_g_h,
                        "predicted_steam_g_h": row.predicted_steam_g_h,
                    }
                )
    _write_csv(
        csv_dir / "prediction_timeline_comparison.csv",
        timeline_rows,
        [
            "model",
            "regime",
            "observation",
            "timestamp",
            "timestamp_type",
            "frame_id",
            "expected_fuel_g_h",
            "predicted_fuel_g_h",
            "expected_steam_g_h",
            "predicted_steam_g_h",
        ],
    )

    excluded_rows = [
        {
            "model": comparison.first_name,
            "regime": regime.display_name,
            "reason": "режим отсутствует во второй модели",
        }
        for regime in comparison.first_only_regimes
    ]
    excluded_rows.extend(
        {
            "model": comparison.second_name,
            "regime": regime.display_name,
            "reason": "режим отсутствует в первой модели",
        }
        for regime in comparison.second_only_regimes
    )
    _write_csv(
        csv_dir / "excluded_regimes.csv",
        excluded_rows,
        ["model", "regime", "reason"],
    )


def _write_csv(
    path: Path,
    rows: list[dict[str, object]],
    fields: list[str],
) -> None:
    try:
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: "" if value is None else value
                        for key, value in row.items()
                    }
                )
    except OSError as exc:
        raise RuntimeError(f"Не удалось записать {path.name}: {exc}") from exc


def _write_processing_log(
    comparison: ComparisonAnalysis,
    path: Path,
) -> None:
    lines = [
        f"generated_at={datetime.now().isoformat(timespec='seconds')}",
        f"model_1={comparison.first.root}",
        f"model_2={comparison.second.root}",
        f"common_regimes={len(comparison.common_regimes)}",
        f"model_1_common_rows={comparison.first_overall_metrics.record_count}",
        f"model_2_common_rows={comparison.second_overall_metrics.record_count}",
        f"model_1_excluded_regimes={len(comparison.first_only_regimes)}",
        f"model_2_excluded_regimes={len(comparison.second_only_regimes)}",
        "comparison_scope=physical regime intersection by expected fuel and steam in g/h",
    ]
    try:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Не удалось записать журнал обработки: {exc}") from exc


def _build_pdf(comparison: ComparisonAnalysis, path: Path) -> None:
    regular_font, bold_font = _register_fonts()
    styles = _styles(regular_font, bold_font)
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=18 * mm,
        bottomMargin=16 * mm,
        title="BurnerEye - сравнение моделей",
        author="BurnerEye Prediction Report",
    )
    story: list[Flowable] = [
        Spacer(1, 18 * mm),
        Paragraph("BURNEREYE", styles["eyebrow"]),
        Spacer(1, 4 * mm),
        Paragraph("Сравнение моделей нейронной сети", styles["title"]),
        Spacer(1, 8 * mm),
        Paragraph(
            f"<b>Модель 1:</b> {escape(comparison.first_name)}",
            styles["lead"],
        ),
        Paragraph(
            f"<b>Путь:</b> {escape(str(comparison.first.root))}",
            styles["path"],
        ),
        Paragraph(
            f"<b>Модель 2:</b> {escape(comparison.second_name)}",
            styles["lead"],
        ),
        Paragraph(
            f"<b>Путь:</b> {escape(str(comparison.second.root))}",
            styles["path"],
        ),
        Spacer(1, 7 * mm),
        Paragraph(
            "В расчёты, таблицы и графики включены только физические режимы, "
            "которые присутствуют в обеих папках. Режим определяется парой "
            "фактических расходов топлива и пара после приведения к г/ч.",
            styles["note"],
        ),
        Paragraph(
            "Сопоставимые цели: "
            + ", ".join("топливо" if target == "fuel" else "пар" for target in sorted(comparison.comparable_targets))
            + ".",
            styles["note"],
        ),
        Spacer(1, 5 * mm),
        _table(
            [
                ["Показатель", comparison.first_name, comparison.second_name],
                [
                    "Общих режимов",
                    len(comparison.common_regimes),
                    len(comparison.common_regimes),
                ],
                [
                    "Строк в общих режимах",
                    comparison.first_overall_metrics.record_count,
                    comparison.second_overall_metrics.record_count,
                ],
                [
                    "Исключено уникальных режимов",
                    len(comparison.first_only_regimes),
                    len(comparison.second_only_regimes),
                ],
            ],
            [58 * mm, 58 * mm, 58 * mm],
            styles,
            repeat_rows=1,
        ),
        PageBreak(),
    ]
    story.extend(_scope_section(comparison, styles))
    story.extend(_overall_section(comparison, styles))
    story.extend(_regime_overview_section(comparison, styles))
    for item in comparison.common_regimes:
        story.extend(
            _regime_section(
                comparison,
                item,
                styles,
                regular_font,
            )
        )

    def draw_page(canvas, document) -> None:
        canvas.saveState()
        canvas.setTitle("BurnerEye - сравнение моделей")
        canvas.setAuthor("BurnerEye Prediction Report")
        canvas.setFont(regular_font, 8)
        canvas.setFillColor(MUTED)
        if document.page > 1:
            canvas.drawString(
                15 * mm,
                A4[1] - 10 * mm,
                "BurnerEye - сравнение моделей",
            )
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
        raise RuntimeError(f"Не удалось сформировать сравнительный PDF: {exc}") from exc


def _scope_section(comparison: ComparisonAnalysis, styles) -> list[Flowable]:
    rows: list[list[object]] = [
        [
            "Код",
            "Общий режим",
            f"Строк: {comparison.first_name}",
            f"Строк: {comparison.second_name}",
        ]
    ]
    rows.extend(
        [
            f"R{index:02d}",
            item.display_name,
            len(item.first_regime.rows),
            len(item.second_regime.rows),
        ]
        for index, item in enumerate(comparison.common_regimes, start=1)
    )
    result: list[Flowable] = [
        Paragraph("1. Область сравнения", styles["h1"]),
        _table(
            rows,
            [14 * mm, 80 * mm, 40 * mm, 40 * mm],
            styles,
            repeat_rows=1,
        ),
    ]
    coverage_rows = [["Общий режим", "Топливо: модель 1", "Топливо: модель 2", "Пар: модель 1", "Пар: модель 2", "Общие цели"]]
    for index, item in enumerate(comparison.common_regimes, start=1):
        first, second = item.first_metrics, item.second_metrics
        shared = []
        if first.mae_fuel_count and second.mae_fuel_count:
            shared.append("топливо")
        if first.mae_steam_count and second.mae_steam_count:
            shared.append("пар")
        coverage_rows.append([
            f"R{index:02d}",
            str(first.mae_fuel_count),
            str(second.mae_fuel_count),
            str(first.mae_steam_count),
            str(second.mae_steam_count),
            ", ".join(shared) if shared else "Нет сопоставимых данных",
        ])
    result.extend([
        Spacer(1, 4 * mm),
        Paragraph("Покрытие прогнозами в общих режимах", styles["h2"]),
        _table(coverage_rows, [31 * mm, 29 * mm, 29 * mm, 26 * mm, 26 * mm, 39 * mm], styles, repeat_rows=1, font_size=7),
    ])
    excluded = [
        (comparison.first_name, regime)
        for regime in comparison.first_only_regimes
    ] + [
        (comparison.second_name, regime)
        for regime in comparison.second_only_regimes
    ]
    if excluded:
        result.extend(
            [
                Spacer(1, 4 * mm),
                Paragraph("Режимы вне пересечения", styles["h2"]),
                Paragraph(
                    "Следующие режимы не участвуют ни в метриках, ни в графиках:",
                    styles["body"],
                ),
                _table(
                    [
                        ["Модель", "Исключённый режим"],
                        *[
                            [model_name, regime.display_name]
                            for model_name, regime in excluded
                        ],
                    ],
                    [58 * mm, 116 * mm],
                    styles,
                    repeat_rows=1,
                ),
            ]
        )
    return result


def _overall_section(comparison: ComparisonAnalysis, styles) -> list[Flowable]:
    first = comparison.first_overall_metrics
    second = comparison.second_overall_metrics
    rows = [
        [
            "Модель",
            "Строк",
            "MAE топлива, г/ч",
            "MAPE топлива",
            "MAE пара, г/ч",
            "MAPE пара",
        ],
        _metrics_row(comparison.first_name, first),
        _metrics_row(comparison.second_name, second),
    ]
    return [
        Spacer(1, 5 * mm),
        Paragraph("2. Итоговые метрики по общим режимам", styles["h1"]),
        _table(
            rows,
            [42 * mm, 17 * mm, 30 * mm, 27 * mm, 30 * mm, 27 * mm],
            styles,
            repeat_rows=1,
            font_size=7,
        ),
        Spacer(1, 4 * mm),
        Paragraph("Сравнение MAE", styles["h2"]),
        BarChartFlowable(
            [comparison.first_name, comparison.second_name],
            {
                "Топливо": _paired_values(first.mae_fuel, second.mae_fuel),
                "Пар": _paired_values(first.mae_steam, second.mae_steam),
            },
            "MAE, г/ч",
        ),
        Paragraph("Сравнение MAPE", styles["h2"]),
        BarChartFlowable(
            [comparison.first_name, comparison.second_name],
            {
                "Топливо": _paired_values(first.mape_fuel, second.mape_fuel),
                "Пар": _paired_values(first.mape_steam, second.mape_steam),
            },
            "MAPE, %",
        ),
    ]


def _regime_overview_section(
    comparison: ComparisonAnalysis,
    styles,
) -> list[Flowable]:
    groups = [f"R{index:02d}" for index in range(1, len(comparison.common_regimes) + 1)]
    result: list[Flowable] = [
        PageBreak(),
        Paragraph("3. Сравнение по общим режимам", styles["h1"]),
        Paragraph(
            "Обозначения R01, R02 и далее соответствуют порядку режимов "
            "в таблице области сравнения.",
            styles["note"],
        ),
        KeepTogether(
            [
                Paragraph("MAE топлива по режимам", styles["h2"]),
                BarChartFlowable(
                    groups,
                    {
                        comparison.first_name: _paired_regime_values(comparison, "mae_fuel", "mae_fuel_count"),
                        comparison.second_name: _paired_regime_values(comparison, "mae_fuel", "mae_fuel_count", second=True),
                    },
                    "MAE топлива, г/ч",
                ),
            ]
        ),
        KeepTogether(
            [
                Paragraph("MAE пара по режимам", styles["h2"]),
                BarChartFlowable(
                    groups,
                    {
                        comparison.first_name: _paired_regime_values(comparison, "mae_steam", "mae_steam_count"),
                        comparison.second_name: _paired_regime_values(comparison, "mae_steam", "mae_steam_count", second=True),
                    },
                    "MAE пара, г/ч",
                ),
            ]
        ),
        KeepTogether(
            [
                Paragraph("MAPE топлива по режимам", styles["h2"]),
                BarChartFlowable(
                    groups,
                    {
                        comparison.first_name: _paired_regime_values(comparison, "mape_fuel", "mape_fuel_count"),
                        comparison.second_name: _paired_regime_values(comparison, "mape_fuel", "mape_fuel_count", second=True),
                    },
                    "MAPE топлива, %",
                ),
            ]
        ),
        KeepTogether(
            [
                Paragraph("MAPE пара по режимам", styles["h2"]),
                BarChartFlowable(
                    groups,
                    {
                        comparison.first_name: _paired_regime_values(comparison, "mape_steam", "mape_steam_count"),
                        comparison.second_name: _paired_regime_values(comparison, "mape_steam", "mape_steam_count", second=True),
                    },
                    "MAPE пара, %",
                ),
            ]
        ),
    ]
    return result


def _regime_section(
    comparison: ComparisonAnalysis,
    item: CommonRegimeComparison,
    styles,
    font_name: str,
) -> list[Flowable]:
    first_rows = sorted(
        item.first_regime.rows,
        key=lambda row: row.timestamp,
    )
    second_rows = sorted(
        item.second_regime.rows,
        key=lambda row: row.timestamp,
    )
    observation_count = max(len(first_rows), len(second_rows))
    x_values = [float(index) for index in range(1, observation_count + 1)]
    x_labels = [str(index) for index in range(1, observation_count + 1)]
    expected_fuel = [item.key[0]] * observation_count
    expected_steam = [item.key[1]] * observation_count
    fuel_comparable = bool(item.first_metrics.mae_fuel_count and item.second_metrics.mae_fuel_count)
    steam_comparable = bool(item.first_metrics.mae_steam_count and item.second_metrics.mae_steam_count)
    result: list[Flowable] = [
        PageBreak(),
        Paragraph(item.display_name, styles["h1"]),
        Paragraph(
            "Ось X показывает порядковый номер наблюдения внутри режима. "
            "Если число записей различается, каждая линия заканчивается на "
            "последней доступной записи своей модели.",
            styles["note"],
        ),
        Spacer(1, 3 * mm),
        _table(
            [
                [
                    "Модель",
                    "Строк",
                    "MAE топлива",
                    "MAPE топлива",
                    "MAE пара",
                    "MAPE пара",
                ],
                _metrics_row(comparison.first_name, item.first_metrics),
                _metrics_row(comparison.second_name, item.second_metrics),
            ],
            [42 * mm, 17 * mm, 29 * mm, 28 * mm, 29 * mm, 28 * mm],
            styles,
            repeat_rows=1,
            font_size=7,
        ),
        Spacer(1, 4 * mm),
        KeepTogether(
            [
                Paragraph(
                    "Фактический и предсказанный расход топлива" if fuel_comparable else "Топливо: нет сопоставимых прогнозов",
                    styles["h2"],
                ),
                LineChartFlowable(
                    x_values,
                    x_labels,
                    {
                        "Факт": expected_fuel,
                        comparison.first_name: _padded_values(
                            first_rows,
                            "predicted_fuel_g_h",
                            observation_count,
                        ) if fuel_comparable else [None] * observation_count,
                        comparison.second_name: _padded_values(
                            second_rows,
                            "predicted_fuel_g_h",
                            observation_count,
                        ) if fuel_comparable else [None] * observation_count,
                    },
                    "Топливо, г/ч",
                    font_name=font_name,
                ),
            ]
        ),
        KeepTogether(
            [
                Paragraph(
                    "Фактический и предсказанный расход пара" if steam_comparable else "Пар: нет сопоставимых прогнозов",
                    styles["h2"],
                ),
                LineChartFlowable(
                    x_values,
                    x_labels,
                    {
                        "Факт": expected_steam,
                        comparison.first_name: _padded_values(
                            first_rows,
                            "predicted_steam_g_h",
                            observation_count,
                        ) if steam_comparable else [None] * observation_count,
                        comparison.second_name: _padded_values(
                            second_rows,
                            "predicted_steam_g_h",
                            observation_count,
                        ) if steam_comparable else [None] * observation_count,
                    },
                    "Пар, г/ч",
                    font_name=font_name,
                ),
            ]
        ),
    ]
    conclusion = Paragraph(
        _regime_conclusion(comparison, item),
        styles["conclusion"],
    )
    if item.first_stage_metrics and item.second_stage_metrics:
        result.append(
            KeepTogether(
                [
                    Paragraph(
                        "Начало, середина и конец режима",
                        styles["h2"],
                    ),
                    LineChartFlowable(
                        [0.0, 1.0, 2.0],
                        list(STAGES),
                        {
                            f"{comparison.first_name}: топливо": [
                                first_value if first_count and second_count else None
                                for first_value, first_count, second_count in _stage_pair(item.first_stage_metrics, item.second_stage_metrics, "mae_fuel", "mae_fuel_count")
                            ],
                            f"{comparison.second_name}: топливо": [
                                second_value if first_count and second_count else None
                                for second_value, first_count, second_count in _stage_pair(item.first_stage_metrics, item.second_stage_metrics, "mae_fuel", "mae_fuel_count", second=True)
                            ],
                            f"{comparison.first_name}: пар": [
                                first_value if first_count and second_count else None
                                for first_value, first_count, second_count in _stage_pair(item.first_stage_metrics, item.second_stage_metrics, "mae_steam", "mae_steam_count")
                            ],
                            f"{comparison.second_name}: пар": [
                                second_value if first_count and second_count else None
                                for second_value, first_count, second_count in _stage_pair(item.first_stage_metrics, item.second_stage_metrics, "mae_steam", "mae_steam_count", second=True)
                            ],
                        },
                        "MAE, г/ч",
                        font_name=font_name,
                    ),
                    conclusion,
                ]
            )
        )
    else:
        result.append(conclusion)
    return result


def _metrics_row(label: str, metrics: MetricSet) -> list[object]:
    return [
        label,
        metrics.record_count,
        _fmt_mae(metrics.mae_fuel, metrics.mae_fuel_count),
        _fmt_mape_metric(metrics.mape_fuel, metrics.mae_fuel_count, metrics.mape_fuel_count),
        _fmt_mae(metrics.mae_steam, metrics.mae_steam_count),
        _fmt_mape_metric(metrics.mape_steam, metrics.mae_steam_count, metrics.mape_steam_count),
    ]


def _padded_values(
    rows: list[PredictionRow],
    attribute: str,
    size: int,
) -> list[float | None]:
    values = [
        float(value) if value is not None else None
        for row in rows
        for value in (getattr(row, attribute),)
    ]
    return values + [None] * (size - len(values))


def _regime_conclusion(
    comparison: ComparisonAnalysis,
    item: CommonRegimeComparison,
) -> str:
    results = []
    for label, mae, mape in (
        ("топливо", "mae_fuel", "mape_fuel"),
        ("пар", "mae_steam", "mape_steam"),
    ):
        first_mape = getattr(item.first_metrics, mape)
        second_mape = getattr(item.second_metrics, mape)
        if first_mape is not None and second_mape is not None:
            winner = metric_winner(first_mape, second_mape, comparison.first_name, comparison.second_name)
            metric_name = "MAPE"
        else:
            first_mae = getattr(item.first_metrics, mae)
            second_mae = getattr(item.second_metrics, mae)
            winner = metric_winner(first_mae, second_mae, comparison.first_name, comparison.second_name)
            metric_name = "MAE"
        results.append(f"{label}: {metric_name} — {escape(winner)}")
    return ". ".join(results) + "."


def _paired_values(first: float | None, second: float | None) -> list[float | None]:
    return [first, second] if first is not None and second is not None else [None, None]


def _paired_regime_values(
    comparison: ComparisonAnalysis,
    value_name: str,
    count_name: str,
    *,
    second: bool = False,
) -> list[float | None]:
    values: list[float | None] = []
    for regime in comparison.common_regimes:
        first_metrics, second_metrics = regime.first_metrics, regime.second_metrics
        if not getattr(first_metrics, count_name) or not getattr(second_metrics, count_name):
            values.append(None)
            continue
        metrics = second_metrics if second else first_metrics
        values.append(getattr(metrics, value_name))
    return values


def _stage_pair(first_stages, second_stages, value_name: str, count_name: str, *, second: bool = False):
    for first_metrics, second_metrics in zip(first_stages, second_stages):
        yield (
            getattr(second_metrics if second else first_metrics, value_name),
            getattr(first_metrics, count_name),
            getattr(second_metrics, count_name),
        )


def _progress(
    callback: ProgressCallback | None,
    value: int,
    message: str,
) -> None:
    if callback is not None:
        callback(value, message)


__all__ = ["generate_comparison_report"]

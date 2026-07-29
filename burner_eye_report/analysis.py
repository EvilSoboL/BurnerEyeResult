from __future__ import annotations

import math
import statistics
from collections.abc import Iterable

from .models import (
    AnalysisResult,
    ExperimentData,
    MetricSet,
    PredictionRow,
    SelectedFrame,
    StageResult,
)


STAGES = ("Начало", "Середина", "Конец")


def analyze_experiment(
    experiment: ExperimentData,
    threshold_percent: float = 5.0,
) -> AnalysisResult:
    if experiment.blocking_issues:
        messages = "; ".join(issue.message for issue in experiment.blocking_issues)
        raise ValueError(f"Эксперимент не готов к отчёту: {messages}")
    if not experiment.rows or not experiment.regimes:
        raise ValueError("Нет данных для анализа.")
    if any(regime.final_status is None for regime in experiment.regimes):
        raise ValueError("Не для всех режимов выбран итоговый статус обученности.")
    if not math.isfinite(threshold_percent) or threshold_percent < 0:
        raise ValueError("Порог существенного изменения должен быть неотрицательным числом.")

    regime_lookup = {regime.regime_id: regime for regime in experiment.regimes}
    row_metrics = {row.frame_id: _row_metrics(row) for row in experiment.rows}
    overall_metrics = calculate_metrics(experiment.rows)
    trained_untrained: dict[bool, MetricSet] = {}
    for status in (True, False):
        rows = [
            row
            for row in experiment.rows
            if regime_lookup[row.regime_id].final_status is status
        ]
        if rows:
            trained_untrained[status] = calculate_metrics(rows)

    regime_metrics = {
        regime.regime_id: calculate_metrics(regime.rows) for regime in experiment.regimes
    }
    regime_stages: dict[str, list[StageResult]] = {}
    regime_conclusions: dict[str, str] = {}
    for regime in experiment.regimes:
        stages = _stage_results("regime", regime.regime_id, regime.rows)
        regime_stages[regime.regime_id] = stages
        regime_conclusions[regime.regime_id] = _quality_conclusion(
            stages, threshold_percent
        )

    overall_stages = _stage_results("experiment", "all", experiment.rows)
    overall_conclusion = _quality_conclusion(overall_stages, threshold_percent)
    group_stages: dict[bool, list[StageResult]] = {}
    for status in (True, False):
        rows = [
            row
            for row in experiment.rows
            if regime_lookup[row.regime_id].final_status is status
        ]
        if rows:
            group_stages[status] = _stage_results(
                "trained" if status else "untrained",
                "trained" if status else "untrained",
                rows,
            )

    selected, selection_notes = _select_frames(experiment.rows, row_metrics)
    error_conclusions = {
        "overall_fuel": _direction_conclusion(
            [metrics["signed_error_fuel"] for metrics in row_metrics.values()]
        ),
        "overall_steam": _direction_conclusion(
            [metrics["signed_error_steam"] for metrics in row_metrics.values()]
        ),
    }
    for regime in experiment.regimes:
        error_conclusions[f"{regime.regime_id}_fuel"] = _direction_conclusion(
            [row_metrics[row.frame_id]["signed_error_fuel"] for row in regime.rows]
        )
        error_conclusions[f"{regime.regime_id}_steam"] = _direction_conclusion(
            [row_metrics[row.frame_id]["signed_error_steam"] for row in regime.rows]
        )

    processing_notes = list(selection_notes)
    zero_fuel = sum(row.expected_fuel_g_h == 0 for row in experiment.rows)
    zero_steam = sum(row.expected_steam_g_h == 0 for row in experiment.rows)
    if zero_fuel:
        processing_notes.append(
            f"{zero_fuel} строк(и) исключены из MAPE топлива из-за нулевого фактического значения."
        )
    if zero_steam:
        processing_notes.append(
            f"{zero_steam} строк(и) исключены из MAPE пара из-за нулевого фактического значения."
        )
    return AnalysisResult(
        experiment=experiment,
        overall_metrics=overall_metrics,
        trained_untrained_metrics=trained_untrained,
        regime_metrics=regime_metrics,
        regime_stages=regime_stages,
        overall_stages=overall_stages,
        group_stages=group_stages,
        row_metrics=row_metrics,
        selected_frames=selected,
        regime_conclusions=regime_conclusions,
        overall_conclusion=overall_conclusion,
        error_conclusions=error_conclusions,
        threshold_percent=threshold_percent,
        processing_notes=processing_notes,
    )


def calculate_metrics(rows: Iterable[PredictionRow]) -> MetricSet:
    materialized = list(rows)
    fuel_abs = [
        abs(row.predicted_fuel_g_h - row.expected_fuel_g_h) for row in materialized
    ]
    steam_abs = [
        abs(row.predicted_steam_g_h - row.expected_steam_g_h) for row in materialized
    ]
    fuel_pct = [
        abs(row.predicted_fuel_g_h - row.expected_fuel_g_h)
        / abs(row.expected_fuel_g_h)
        * 100.0
        for row in materialized
        if row.expected_fuel_g_h != 0
    ]
    steam_pct = [
        abs(row.predicted_steam_g_h - row.expected_steam_g_h)
        / abs(row.expected_steam_g_h)
        * 100.0
        for row in materialized
        if row.expected_steam_g_h != 0
    ]
    return MetricSet(
        record_count=len(materialized),
        mae_fuel=_mean_or_none(fuel_abs),
        mape_fuel=_mean_or_none(fuel_pct),
        mape_fuel_count=len(fuel_pct),
        mae_steam=_mean_or_none(steam_abs),
        mape_steam=_mean_or_none(steam_pct),
        mape_steam_count=len(steam_pct),
    )


def split_three(rows: Iterable[PredictionRow]) -> list[list[PredictionRow]]:
    ordered = sorted(rows, key=lambda row: row.timestamp)
    quotient, remainder = divmod(len(ordered), 3)
    sizes = [quotient + (1 if index < remainder else 0) for index in range(3)]
    result: list[list[PredictionRow]] = []
    cursor = 0
    for size in sizes:
        result.append(ordered[cursor : cursor + size])
        cursor += size
    return result


def _stage_results(
    scope: str,
    scope_id: str,
    rows: Iterable[PredictionRow],
) -> list[StageResult]:
    materialized = list(rows)
    if len(materialized) < 3:
        return []
    results: list[StageResult] = []
    for stage, chunk in zip(STAGES, split_three(materialized)):
        results.append(
            StageResult(
                scope=scope,
                scope_id=scope_id,
                stage=stage,
                metrics=calculate_metrics(chunk),
                start_time=chunk[0].timestamp if chunk else None,
                end_time=chunk[-1].timestamp if chunk else None,
            )
        )
    return results


def _row_metrics(row: PredictionRow) -> dict[str, float | None]:
    signed_fuel = row.predicted_fuel_g_h - row.expected_fuel_g_h
    signed_steam = row.predicted_steam_g_h - row.expected_steam_g_h
    return {
        "signed_error_fuel": signed_fuel,
        "absolute_error_fuel": abs(signed_fuel),
        "ape_fuel": (
            abs(signed_fuel) / abs(row.expected_fuel_g_h) * 100.0
            if row.expected_fuel_g_h != 0
            else None
        ),
        "signed_error_steam": signed_steam,
        "absolute_error_steam": abs(signed_steam),
        "ape_steam": (
            abs(signed_steam) / abs(row.expected_steam_g_h) * 100.0
            if row.expected_steam_g_h != 0
            else None
        ),
    }


def _quality_conclusion(stages: list[StageResult], threshold_percent: float) -> str:
    if len(stages) != 3:
        return "Вывод невозможен из-за недостатка данных (требуется не менее трёх записей)."
    scores: list[float | None] = []
    for stage in stages:
        pct_values = [
            value
            for value in (stage.metrics.mape_fuel, stage.metrics.mape_steam)
            if value is not None
        ]
        if pct_values:
            scores.append(statistics.fmean(pct_values))
            continue
        mae_values = [
            value
            for value in (stage.metrics.mae_fuel, stage.metrics.mae_steam)
            if value is not None
        ]
        scores.append(statistics.fmean(mae_values) if mae_values else None)
    if any(score is None for score in scores):
        return "Вывод невозможен из-за недостатка данных."
    numeric = [float(score) for score in scores if score is not None]
    best_index = min(range(3), key=numeric.__getitem__)
    best = numeric[best_index]
    worst = max(numeric)
    if best == 0:
        significant = worst > 0
    else:
        significant = (worst - best) / abs(best) * 100.0 >= threshold_percent
    if not significant:
        return (
            "Качество существенно не меняется: различие не превышает "
            f"{threshold_percent:g}% относительно лучшего результата."
        )
    return f"Качество выше на участке «{STAGES[best_index].casefold()}»."


def _select_frames(
    rows: list[PredictionRow],
    row_metrics: dict[str, dict[str, float | None]],
) -> tuple[list[SelectedFrame], list[str]]:
    available = [row for row in rows if row.image_path is not None]
    if not available:
        return [], ["Нет доступных кадров для ранжирования."]

    positive_fuel = [abs(row.expected_fuel_g_h) for row in rows if row.expected_fuel_g_h != 0]
    positive_steam = [
        abs(row.expected_steam_g_h) for row in rows if row.expected_steam_g_h != 0
    ]
    fuel_scale = statistics.median(positive_fuel) if positive_fuel else 1.0
    steam_scale = statistics.median(positive_steam) if positive_steam else 1.0

    scored: list[tuple[float, str, PredictionRow]] = []
    fallback_count = 0
    for row in available:
        metrics = row_metrics[row.frame_id]
        components: list[float] = []
        methods: list[str] = []
        if metrics["ape_fuel"] is None:
            components.append(float(metrics["absolute_error_fuel"]) / fuel_scale * 100.0)
            methods.append(
                "нормализованная абсолютная ошибка топлива "
                f"(масштаб - медиана {fuel_scale:.6g} г/ч)"
            )
        else:
            components.append(float(metrics["ape_fuel"]))
        if metrics["ape_steam"] is None:
            components.append(float(metrics["absolute_error_steam"]) / steam_scale * 100.0)
            methods.append(
                "нормализованная абсолютная ошибка пара "
                f"(масштаб - медиана {steam_scale:.6g} г/ч)"
            )
        else:
            components.append(float(metrics["ape_steam"]))
        method = "Среднее процентных ошибок"
        if methods:
            fallback_count += 1
            method = "Комбинированная оценка: " + ", ".join(methods)
        scored.append((statistics.fmean(components), method, row))
    scored.sort(key=lambda item: (item[0], item[2].timestamp, item[2].frame_id))

    count = len(scored)
    best_count = min(2, max(1, (count + 1) // 2))
    worst_count = min(2, count - best_count)
    chosen: list[tuple[str, int, tuple[float, str, PredictionRow]]] = []
    for rank, item in enumerate(scored[:best_count], start=1):
        chosen.append(("Лучший кадр", rank, item))
    if worst_count:
        worst_items = list(reversed(scored[-worst_count:]))
        for rank, item in enumerate(worst_items, start=1):
            chosen.append(("Худший кадр", rank, item))

    selected: list[SelectedFrame] = []
    for label, rank, (score, method, row) in chosen:
        metrics = row_metrics[row.frame_id]
        selected.append(
            SelectedFrame(
                row=row,
                label=label,
                rank=rank,
                score=score,
                score_method=method,
                abs_error_fuel=float(metrics["absolute_error_fuel"]),
                ape_fuel=metrics["ape_fuel"],
                abs_error_steam=float(metrics["absolute_error_steam"]),
                ape_steam=metrics["ape_steam"],
            )
        )
    notes: list[str] = []
    if count < 4:
        notes.append(
            f"Доступно менее четырёх корректных кадров ({count}); в отчёт включены все без повторов."
        )
    if fallback_count:
        notes.append(
            f"Для {fallback_count} кадр(а/ов) оценка ранжирования использует "
            "нормализованную абсолютную ошибку из-за нулевого фактического расхода."
        )
    return selected, notes


def _direction_conclusion(values: Iterable[float | None]) -> str:
    numeric = [float(value) for value in values if value is not None]
    non_zero = [value for value in numeric if value != 0]
    if not non_zero:
        return "Выраженного направления ошибки не обнаружено."
    positive_share = sum(value > 0 for value in non_zero) / len(non_zero)
    negative_share = sum(value < 0 for value in non_zero) / len(non_zero)
    if positive_share >= 0.6:
        return "Модель преимущественно завышает предсказание."
    if negative_share >= 0.6:
        return "Модель преимущественно занижает предсказание."
    return "Выраженного направления ошибки не обнаружено."


def _mean_or_none(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None

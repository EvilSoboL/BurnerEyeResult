from __future__ import annotations

import math
from dataclasses import dataclass, field

from .analysis import STAGES, calculate_metrics, split_three
from .models import ExperimentData, MetricSet, PredictionRow, Regime


IGNORED_COMPARISON_BLOCKERS = {
    "frames_missing",
    "no_usable_frames",
}


@dataclass(slots=True)
class CommonRegimeComparison:
    key: tuple[float, float]
    first_regime: Regime
    second_regime: Regime
    first_metrics: MetricSet
    second_metrics: MetricSet
    first_stage_metrics: list[MetricSet] = field(default_factory=list)
    second_stage_metrics: list[MetricSet] = field(default_factory=list)

    @property
    def display_name(self) -> str:
        return self.first_regime.display_name


@dataclass(slots=True)
class ComparisonAnalysis:
    first: ExperimentData
    second: ExperimentData
    first_name: str
    second_name: str
    common_regimes: list[CommonRegimeComparison]
    first_only_regimes: list[Regime]
    second_only_regimes: list[Regime]
    first_overall_metrics: MetricSet
    second_overall_metrics: MetricSet

    @property
    def first_common_rows(self) -> list[PredictionRow]:
        return [
            row
            for comparison in self.common_regimes
            for row in comparison.first_regime.rows
        ]

    @property
    def second_common_rows(self) -> list[PredictionRow]:
        return [
            row
            for comparison in self.common_regimes
            for row in comparison.second_regime.rows
        ]


def analyze_comparison(
    first: ExperimentData,
    second: ExperimentData,
) -> ComparisonAnalysis:
    _validate_for_comparison(first, "первой")
    _validate_for_comparison(second, "второй")

    first_lookup = {_regime_key(regime): regime for regime in first.regimes}
    second_lookup = {_regime_key(regime): regime for regime in second.regimes}
    common_keys = sorted(
        first_lookup.keys() & second_lookup.keys(),
        key=lambda item: (item[1], item[0]),
    )
    if not common_keys:
        raise ValueError(
            "В выбранных папках нет общих режимов по фактическим расходам "
            "топлива и пара."
        )

    common_regimes: list[CommonRegimeComparison] = []
    for key in common_keys:
        first_regime = first_lookup[key]
        second_regime = second_lookup[key]
        common_regimes.append(
            CommonRegimeComparison(
                key=key,
                first_regime=first_regime,
                second_regime=second_regime,
                first_metrics=calculate_metrics(first_regime.rows),
                second_metrics=calculate_metrics(second_regime.rows),
                first_stage_metrics=_stage_metrics(first_regime.rows),
                second_stage_metrics=_stage_metrics(second_regime.rows),
            )
        )

    first_name, second_name = _model_names(first, second)
    first_rows = [
        row for comparison in common_regimes for row in comparison.first_regime.rows
    ]
    second_rows = [
        row for comparison in common_regimes for row in comparison.second_regime.rows
    ]
    return ComparisonAnalysis(
        first=first,
        second=second,
        first_name=first_name,
        second_name=second_name,
        common_regimes=common_regimes,
        first_only_regimes=[
            first_lookup[key]
            for key in sorted(
                first_lookup.keys() - second_lookup.keys(),
                key=lambda item: (item[1], item[0]),
            )
        ],
        second_only_regimes=[
            second_lookup[key]
            for key in sorted(
                second_lookup.keys() - first_lookup.keys(),
                key=lambda item: (item[1], item[0]),
            )
        ],
        first_overall_metrics=calculate_metrics(first_rows),
        second_overall_metrics=calculate_metrics(second_rows),
    )


def metric_winner(
    first_value: float | None,
    second_value: float | None,
    first_name: str,
    second_name: str,
) -> str:
    if first_value is None and second_value is None:
        return "Нет данных"
    if first_value is None:
        return second_name
    if second_value is None:
        return first_name
    if math.isclose(first_value, second_value, rel_tol=1e-9, abs_tol=1e-9):
        return "Одинаково"
    return first_name if first_value < second_value else second_name


def _regime_key(regime: Regime) -> tuple[float, float]:
    return round(regime.fuel_g_h, 6), round(regime.steam_g_h, 6)


def _stage_metrics(rows: list[PredictionRow]) -> list[MetricSet]:
    if len(rows) < 3:
        return []
    return [calculate_metrics(chunk) for chunk in split_three(rows)]


def _validate_for_comparison(experiment: ExperimentData, ordinal: str) -> None:
    blocking = [
        issue
        for issue in experiment.blocking_issues
        if issue.code not in IGNORED_COMPARISON_BLOCKERS
    ]
    if blocking:
        messages = "; ".join(issue.message for issue in blocking)
        raise ValueError(f"Ошибка в {ordinal} папке: {messages}")
    if not experiment.rows or not experiment.regimes:
        raise ValueError(f"В {ordinal} папке нет корректных данных для сравнения.")


def _model_names(
    first: ExperimentData,
    second: ExperimentData,
) -> tuple[str, str]:
    first_name = first.root.name.strip() or "Модель 1"
    second_name = second.root.name.strip() or "Модель 2"
    if first_name.casefold() == second_name.casefold():
        return f"{first_name} (1)", f"{second_name} (2)"
    return first_name, second_name


__all__ = [
    "ComparisonAnalysis",
    "CommonRegimeComparison",
    "STAGES",
    "analyze_comparison",
    "metric_winner",
]
